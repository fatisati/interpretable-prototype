"""THE TARGET CONFIGURATION: cohesion + off-diagonal ONLY, no containment diagonal.

WHAT EACH TERM IS FOR, AND WHY THE DIAGONAL IS GONE
----------------------------------------------------
  cohesion      within-group quality.  NO merge bias, NO size term, scale-free in size.
  off-diagonal  between-group separation -- the anti-overlap force cohesion is blind to.
  containment   DROPPED.

The diagonal is dropped because it has an UNCONDITIONAL merge bias:

    containment[merged] = ( within_k + within_j + between_kj ) / ( volume_k + volume_j )
                                                  ^^^^^^^^^^^^ pure gain, no offsetting cost

The numerator picks up the between-affinity for free while the denominator merely adds, so
the merged prototype's containment is ALWAYS above the volume-weighted average of its
parents -- regardless of the data. That is the merge bias, and it is the reason the
published loss collapses on spatial graphs (15/200 survivors, decisions.md 1-3).

Cohesion has NO such bias. Merging k and j changes it by

    n_k * n_j * ( 2*cross_similarity - cohesion_k - cohesion_j ) / ( n_k + n_j )

so merging is rewarded ONLY IF the two groups are genuinely more alike to each other than
they are internally -- a real, data-dependent test that well-separated groups fail.
Measured on the six most strongly-coupled Leiden communities on bk02 (the actual merge
candidates): cohesion DROPS on every one of them, by 0.3-0.7% of its value.

COHESION == SEACells' OBJECTIVE  (verified numerically to 1.8e-12)
------------------------------------------------------------------
SEACells minimizes || M - M B A ||_F^2 on the kernel matrix. In the hard-assignment limit
that expands to

    sum over cells of ||row[i]||^2            <- constant, partition-free
  - sum over k of within_dot[k] / size[k]     <- exactly our cohesion numerator

so minimizing SEACells' Frobenius reconstruction error IS maximizing our cohesion with
similarity = plain dot product of kernel rows. Our term is a soft, differentiable,
minibatch form of the objective SEACells itself optimizes.

THE TWO LAMBDAS, BOTH ANCHORED ON MEASUREMENT -- NOT SWEPT
-----------------------------------------------------------
OFF-DIAGONAL. nassoc_diag=False keeps ONLY the off-diagonal, and then its weight is
exactly nassoc_alpha (lambda_nassoc is pinned to 1). No new config key was needed --
'nassoc_diag' already existed for precisely this. The value is carried over from the one
configuration measured to work (alpha=2000 under the old 'pair' normalizer took dead
169 -> 30 and coverage 0.667 -> 0.944):

    'pair' value at K=200 = 8.451e-05  ->  alpha 2000 contributed 0.1690
    'pclu' value at K=200 = 0.01682    ->  equivalent alpha = 0.1690/0.01682 = 10.05

So alpha=10 under [D16] 'pclu' reproduces the contribution of the arm that worked, while
being portable across K (alpha drifts 9.5x over K=16..1774 under 'pclu' vs 1130x under
'pair').

COHESION. Measured at the real batch size (1024) and K=200 on bk02, 'glob'-scaled:

    Leiden-modularity 0.2627    RANDOM K=200 0.1937    ALL-ONE 0.0026

lambda_cohesion=1 therefore contributes ~0.26, the same order as the off-diagonal's 0.169
and ~10% of the umap term's ~2.5. This is a bracket anchored on a working configuration,
NOT gradient matching (decisions.md 20), and it is stated as such.

THE FORK THESE ARMS RESOLVE: cohesion_self
-------------------------------------------
Keeping the i==j self-pairs is SEACells-exact. But measured at batch 1024 / K=200 the self
part equals K_live/n_batch = 0.1875 -- 71% of the Leiden value and 97% of the random
value. For HARD assignments it is a constant with no gradient; for SOFT ones
sum_i assign^2 / sum_i assign is a SHARPNESS measure, so it has a live gradient rewarding
"sharpen assignments, keep more slots alive" rather than "group similar cells". That
pressure is arguably useful with K fixed at 200, but it may swamp the signal. Both sides
are run rather than guessed.

ARMS (all: off-diagonal only, alpha=10 'pclu', usage OFF, reinit ON [D13]+[D15],
latent 16, K=200, bk02)
    A  cohesion_self=1  lambda_cohesion=1     SEACells-exact
    B  cohesion_self=0  lambda_cohesion=1     signal only
    C  cohesion_self=0  lambda_cohesion=10    signal only, stronger

REFERENCES (s28nsc / bk02, K=200, LD16)
    nassoc alpha=2000+min  w_purity 0.8286  niche 0.4543  mod 0.5936  dead  30  K_live 170
    nassoc alpha=1         w_purity 0.7834  niche 0.4313  mod 0.6189  dead 169  K_live  31
    cohesion lambda=1      w_purity 0.7498  niche 0.4357  mod 0.6285  dead 174  K_live  26
    cohesion lambda=10     w_purity 0.7822  niche 0.4278  mod 0.6706  dead 163  K_live  37
    modularity lambda=10   w_purity 0.8686  niche 0.4939  mod 0.3087  dead   0  K_live 200

READ K_live FIRST. Modularity/purity on a collapsed run are coarseness artifacts -- 0.6706
on 37 live metacells says nothing. A result only counts if K_live is near 200.

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/train_ratio_assoc.py
Cost: ~13 min per arm on a T4, run in parallel. 3 arms ~ $0.39.
"""
import time
import modal

app = modal.App("scproto-repro")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "numpy", "scipy", "pandas", "anndata", "scanpy",
        "torch", "scarches", "wandb", "tqdm", "statsmodels", "diptest",
        "palantir", "faiss-cpu", "scikit-learn", "matplotlib", "scib-metrics",
        "ipython",
    )
    .pip_install("git+https://github.com/dpeerlab/SEACells.git", extra_options="--no-deps")
    .run_commands(*[
        f"mkdir -p /content/drive/MyDrive/codes/Islander/meta/{name} && "
        f"echo '{{}}' > /content/drive/MyDrive/codes/Islander/meta/{name}/cell2cat.json && "
        f"echo '{{}}' > /content/drive/MyDrive/codes/Islander/meta/{name}/batch2cat.json"
        for name in [
            "lung", "lung_fetal_donor", "lung_fetal_organoid", "brain", "breast",
            "heart", "eye", "gut_fetal", "skin", "COVID", "pancreas",
        ]
    ])
)

volume = modal.Volume.from_name("scproto-data")

ALPHA = 10.0   # off-diagonal weight under 'pclu'; = the working alpha=2000 under 'pair'

ARMS = [
    ("rat_A_l5",  0, 5.0),
    ("rat_B_l20", 0, 20.0),
]


@app.function(image=image, gpu="T4", timeout=7200, volumes={"/vol": volume})
def run(name: str, self_terms: int, lam_coh: float):
    import os, sys
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["HOME_DIR"] = "/vol/"
    os.environ["DATA_DIR"] = "/vol/data/"
    os.environ["MODEL_DIR"] = "/vol/models/"
    os.environ["CODE_DIR"] = "/vol/code/"
    sys.path.insert(0, "/vol/code")
    os.chdir("/vol/code")
    import h5py
    from anndata._io.specs.registry import _REGISTRY, IOSpec
    try:
        @_REGISTRY.register_read(h5py.Dataset, IOSpec("null", "0.1.0"))
        def _rn(e, *a, **kw):
            return None
    except TypeError:
        pass

    from interpretable_ssl.experiments.tasks import run_mc_task, LAMBDA_PROTO_UMAP_PRECON

    lc = dict(LAMBDA_PROTO_UMAP_PRECON) | {
        # --- off-diagonal ONLY: nassoc_diag=False drops the containment term, so the
        #     off-diagonal weight is exactly nassoc_alpha (lambda_nassoc pinned to 1).
        "lambda_nassoc": 1.0,
        "nassoc_diag": False,
        "nassoc_alpha": ALPHA,
        "nassoc_offdiag_agg": "pclu",     # [D16] per-cluster, portable across K
        "nassoc_offdiag_norm": "min",     # bills leakage to the SMALLER prototype
        # --- cohesion ---
        "lambda_cohesion": lam_coh,
        "cohesion_sim": "aff",            # [D18] RATIO ASSOCIATION: affinity_within/size
        "cohesion_row_norm": "none",      # irrelevant for 'aff' -- no row transform applied
        "cohesion_self": self_terms,      # moot for 'aff' (affinity has a zero diagonal)
        # --- no modularity, no usage, reinit on ---
        "lambda_modularity": 0.0,
        "lambda_proto_usage": 0.0,
        "proto_reinit_every": 2,
        "proto_reinit_rel": 0.1,          # [D15] relative cut
        "latent_dims": 16,
    }
    assert lc["nassoc_diag"] is False and lc["lambda_proto_usage"] == 0.0
    assert lc["lambda_modularity"] == 0.0 and lc["lambda_cohesion"] > 0

    print(f"\n{'='*78}\n=== {name}\n"
          f"    OFF-DIAGONAL ONLY (nassoc_diag=False), alpha={ALPHA} 'pclu'+'min'\n"
          f"    cohesion lambda={lam_coh}, row_norm=glob, self_terms={self_terms}\n"
          f"    usage OFF, reinit every 2 [D13/D15], latent 16, K=200, bk02\n"
          f"    WATCH: K_live at the end -- a collapsed run's modularity means nothing\n"
          f"{'='*78}")

    t0 = time.time()
    trainer, metrics, _ = run_mc_task(
        "s28nsc", cvae_epochs=50, train_epochs=20, eval_freq=2, patience=4,
        batch_size=1024, affinity_type="bk02", num_prototypes=200,
        lambda_config=lc, trainer_kwargs={"experiment_name": name})
    print(f"\n[{name}] done in {time.time()-t0:.0f}s  model={trainer.get_model_name()}")
    print(f"[{name}] >>> metrics: {metrics}")
    volume.commit()
    return name, metrics


@app.local_entrypoint()
def main():
    results = list(run.starmap(ARMS))
    print("\n" + "=" * 100)
    print("COHESION + OFF-DIAGONAL, no containment diagonal -- s28nsc/bk02 K=200 LD16")
    print("=" * 100)
    print(f"{'arm':<20}{'purity':>10}{'niche_pur':>11}{'modularity':>12}{'coverage':>10}")
    for name, m in results:
        def g(k):
            v = m.get(k)
            return f"{v:.4f}" if isinstance(v, (int, float)) else "--"
        print(f"{name:<20}{g('purity'):>10}{g('niche_purity'):>11}"
              f"{g('modularity'):>12}{g('coverage'):>10}")
    print("-" * 100)
    print(f"{'nassoc a=2000+min':<20}{'--':>10}{0.4543:>11.4f}{0.5936:>12.4f}{'--':>10}"
          "   dead 30, K_live 170  <- best so far")
    print(f"{'cohesion l=10 alone':<20}{'--':>10}{0.4278:>11.4f}{0.6706:>12.4f}{'--':>10}"
          "   dead 163, K_live 37  <- collapsed, do not read its modularity")
    print("\nK_live must be read from purity_per_mc.csv row count, not from 'dead'.")
