"""[D26] COMMUNITY + COHESION ONLY. Two arms: gram alone, vs gram + the 1-hop companion.

WHAT IS ON AND WHAT IS OFF
--------------------------
ON    community (umap)          the embedding / batch-correction term, untouched
      cohesion (gram, 2-hop)    within-prototype quality
      cohesion (aff, 1-hop)     arm B only
      reinit 'splt' every 2     [D13]+[D20] the ONLY dead-slot repair in this config
OFF   nassoc diagonal + off-diagonal, modularity, proto_usage, sim_recon

THE DIAGONAL: cohesion_diag = 1.0
----------------------------------
The affinity is an RBF kernel, so K[c][c] = exp(0) = 1 BY CONSTRUCTION. setdiag(0) is the
artificial step, made only because the graph-cut consumers (degree/sampling/UMAP/nassoc)
need a no-self-loop convention -- scproto.py:1100-1107 already restores 1 for the
sim_recon target under the 'SEACells RSS convention', and cohesion-with-gram IS a
reconstruction objective (kernel k-means on kernel rows), so it belongs on that side.

Applied BEFORE the L2 row norm, so the Gram is the genuine cosine between CLOSED
neighbourhood vectors v[c] = aff_row[c] + 1*e_c. Verified numerically against an explicit
cos(closed-neighbourhood) construction. The defect this fixes: at delta = 0 two DIRECTLY
ADJACENT cells in a locally triangle-free region get Gram[c][j] = 0 EXACTLY -- measured,
0.0 -> 0.9756 at delta = 1.

A SEPARATE diagonal: cohesion_form='pcex' still removes the Gram's OWN diagonal, so a
cell's similarity to itself never counts toward its prototype's cohesion (verified:
poisoning the Gram diagonal with 999 moves the value by < 1e-6). Two different diagonals,
1 for building the descriptor, 0 inside the loss.

HOW THE LAMBDAS WERE CHOSEN -- measured on the real graph, not swept
--------------------------------------------------------------------
The loss VALUE is the wrong thing to match. cohesion_aff sits near 0 (most aff[c][j] are
0), so 1 - cohesion_aff ~ 0.965 is nearly a CONSTANT: at equal lambda the aff term looks
comparable in the loss while contributing far less gradient. CLAUDE.md 9b, and the
nassoc_alpha error in decisions.md (parity was at 3806x, so a recommendation of 3 would
have changed nothing).

The right quantity is DYNAMIC RANGE -- how much the term can actually move:

    span[term] = value(good partition) - value(random partition)

Measured on s28nsc/bk02, N=58,423, at the exact settings these runs use (pcex, l2,
cohesion_diag as stated), hard assignments on the FULL graph, Leiden-CPM gamma=0.0069
(203 communities) against a random K=200 partition:

    term            leiden      bad30      random       SPAN
    gram delta=0   0.080057   0.042534   0.001492    0.078566
    gram delta=1   0.078437   0.041548   0.001419    0.077017
    aff            0.035154   0.018151   0.000448    0.034706

Both terms rank leiden > bad30 > random, so both carry real signal (a term that ranked
them otherwise would be disqualified before any training -- that is how every off-diagonal
form was ruled out, objective_selection_study.csv).

    span_gram / span_aff = 0.077017 / 0.034706 = 2.219

so lambda_aff = 2.219 * lambda_gram gives the two terms EQUAL authority over the answer.

TOTAL authority, anchored on the best measured arm rather than guessed: modularity +
cohesion at lambda_cohesion=20 (commit 0a7c244) was the best arm on niche purity and size
evenness, and gram's span here (0.077) is within 12% of the span the lambda=1..20 bracket
was anchored on (0.2627 - 0.1937 = 0.069 under the old 'glob' scaling). So the bracket
carries over unchanged and 20 stays the driver-end value -- which is the end we want,
because cohesion is now the ONLY partition-shaping term.

    arm A   gram 20.0                  authority 20.0 * 0.077017 = 1.540
    arm B   gram 10.0 + aff 22.2       authority 10.0 * 0.077017 = 0.770
                                                + 22.2 * 0.034706 = 0.770
                                                             total  1.540

Arm B holds the TOTAL cohesion authority identical to arm A and splits it 50/50 between
2-hop and 1-hop. That is deliberate: it makes the comparison "does mixing in proximity
help" and NOT "is more cohesion better", which two free lambdas would confound.

NOTE cohesion_diag=1 means the gram descriptor ALREADY carries first-order affinity
(setdiag(delta) adds exactly 2*delta*aff[c][j] to the off-diagonal Gram, verified to
2.4e-7). So arm A is not "zero first order"; the aff term in arm B is ADDITIONAL 1-hop
weight on top. The spans above were measured at delta=1, so the ratio already accounts
for it.

WHAT TO READ FIRST: K_live
---------------------------
Cohesion WITHOUT modularity collapsed before: "cohesion l=10 alone -- dead 163,
K_live 37" (train_modu_cohesion.py). Cohesion has no anti-collapse signal for DEAD slots
-- under the per-cell forms an empty prototype contributes exactly 0 and is never
penalised (scproto.py:2451, [D25]). It does strongly reject merging everything into one
(all-one scores 0.0014 vs leiden 0.078 here, a 55x penalty), but that is a different
failure mode from the one that actually happens.

So reinit is LOAD-BEARING here, not optional, and [D13] records reinit silently doing
NOTHING in every run whose folder name carries 'prie' (the EMA it reads was only written
inside the lambda_proto_usage > 0 branch). Confirm the revival count is non-zero in the
log before reading any metric. A modularity/purity number on a collapsed run is a
coarseness artifact.

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/train_cohesion_d26.py
Cost: ~13 min per arm on a T4, run in parallel. 2 arms ~ $0.26.
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
        "leidenalg", "python-igraph",
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

# measured spans, s28nsc/bk02 K=200, pcex + l2 + cohesion_diag=1 (see the header)
SPAN_GRAM = 0.077017
SPAN_AFF = 0.034706
AUTHORITY = 20.0 * SPAN_GRAM        # 1.5403 -- the best measured arm's total

# (name, lambda_cohesion [gram], lambda_cohesion_aff)
ARMS = [
    ("d26_A_gram",  20.0, 0.0),
    ("d26_B_both",  10.0, round(10.0 * SPAN_GRAM / SPAN_AFF, 1)),   # 22.2
]


@app.function(image=image, gpu="T4", timeout=7200, volumes={"/vol": volume})
def run(name: str, lam_gram: float, lam_aff: float):
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
        # --- everything except community + cohesion is OFF ---
        "lambda_nassoc": 0.0,
        "lambda_modularity": 0.0,
        "lambda_proto_usage": 0.0,
        # --- cohesion: 2-hop, and 1-hop in arm B ---
        "lambda_cohesion": lam_gram,
        "lambda_cohesion_aff": lam_aff,
        "cohesion_sim": "gram",
        "cohesion_form": "pcex",          # [D23] cell c out of its own numerator AND divisor
        "cohesion_row_norm": "l2",        # degree-neutral (+0.044 vs +0.464 raw)
        "cohesion_diag": 1.0,             # [D26] closed neighbourhood -- see the header
        "cohesion_self": 0,               # moot under pcex (it zeroes the Gram diagonal)
        # --- reinit is the ONLY dead-slot repair in this configuration ---
        "proto_reinit_every": 2,
        "proto_reinit_rel": 0.1,          # [D15] relative cut, correct for the n_k EMA
        "proto_reinit_target": "splt",    # [D20] seed from overloaded prototypes' worst fits
        "latent_dims": 16,
    }
    assert lc["lambda_modularity"] == 0.0 and lc["lambda_nassoc"] == 0.0
    assert lc["lambda_proto_usage"] == 0.0 and lc["lambda_cohesion"] > 0
    assert lc["cohesion_diag"] == 1.0 and lc["cohesion_form"] == "pcex"

    auth_g = lam_gram * SPAN_GRAM
    auth_a = lam_aff * SPAN_AFF
    print(f"\n{'='*78}\n=== {name}\n"
          f"    community + cohesion ONLY. modularity/nassoc/usage OFF, reinit splt/2.\n"
          f"    cohesion_diag=1.0 (closed neighbourhood), form=pcex, row_norm=l2\n"
          f"    lambda gram={lam_gram}  aff={lam_aff}\n"
          f"    AUTHORITY (lambda * measured span): gram {auth_g:.4f} + aff {auth_a:.4f}"
          f" = {auth_g + auth_a:.4f}  (target {AUTHORITY:.4f})\n"
          f"    WATCH 1: [reinit] revived N -- must be NON-ZERO ([D13])\n"
          f"    WATCH 2: K_live at the end -- a collapsed run's metrics are artifacts\n"
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
    print("\n" + "=" * 104)
    print("[D26] COMMUNITY + COHESION ONLY -- s28nsc/bk02 K=200 LD16, cohesion_diag=1, pcex")
    print("=" * 104)
    print(f"{'arm':<16}{'purity':>10}{'niche_pur':>11}{'modularity':>12}{'coverage':>10}")
    for name, m in results:
        def g(k):
            v = m.get(k)
            return f"{v:.4f}" if isinstance(v, (int, float)) else "--"
        print(f"{name:<16}{g('purity'):>10}{g('niche_purity'):>11}"
              f"{g('modularity'):>12}{g('coverage'):>10}")
    print("-" * 104)
    print("REFERENCES (same dataset/graph/K)")
    print(f"{'modularity l=10':<16}{0.8686:>10.4f}{0.4939:>11.4f}{0.3087:>12.4f}"
          f"{'--':>10}   K_live 200, median 288  <- baseline")
    print(f"{'modu + coh l=20':<16}{0.8710:>10.4f}{0.4963:>11.4f}{0.3063:>12.4f}"
          f"{'--':>10}   K_live 200, median 293  <- best so far")
    print(f"{'coh l=10 ALONE':<16}{'--':>10}{0.4278:>11.4f}{0.6706:>12.4f}"
          f"{'--':>10}   K_live 37  <- COLLAPSED, its modularity is an artifact")
    print("\nRead K_live first (row count of purity_per_mc.csv), then the reinit count.")
