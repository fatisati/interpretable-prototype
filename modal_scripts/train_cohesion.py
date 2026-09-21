"""[D14] AFFINITY-VECTOR COHESION as the global term, replacing nassoc/CPM/modularity.

THE ARGUMENT
------------
CPM's, modularity's and DMoN's charges are all functions of the assignment COLUMN SUMS
only:

    d(CPM charge)/d assign[i,k]        = gamma * (size[k] - assign[i,k])
    d(modularity charge)/d assign[i,k] = res * degree[i] * volume[k] / total_degree
    d(DMoN collapse)/d assign[i,k]     proportional to size[k] / norm(sizes)

None contains a term relating cell i to prototype k, so none can move a cell for
SIMILARITY reasons -- only for size reasons. And all three minimise a quadratic in group
size, so all three are size-UNIFORMITY terms (DMoN's is CPM's charge under a square root
-- same minimiser). The partition we want is UNEVEN, which is why raising their lambda
always made the metrics worse while lowering it caused collapse: one knob, two jobs, and
we only ever wanted one of them.

THE TERM
--------
    similarity(i,j) = cos(affinity_row[i], affinity_row[j])       Salton index
    cohesion[k]     = sum_ij assign[i,k]*assign[j,k]*similarity(i,j) / size[k]^2
    loss            = 1 - sum_k size[k]*cohesion[k] / n_cells

    d(loss)/d assign[i,k] = -( mean similarity of i to k's members - cohesion[k] )

Cell i joins k exactly when it is MORE similar to k's members than they are to each
other -- i.e. when adding it would IMPROVE k. No size, volume or degree appears.
cohesion[k] is a ratio of two quadratics, hence intensive and scale-free: a 20-cell
state is as viable as a 2000-cell one.

Gram = affinity @ affinity.T is the TWO-HOP matrix, so this asks the graph a question one
hop deeper than the community loss, which only ever sees direct edges. Two cells can be
adjacent yet sit across a boundary; identical affinity vectors cannot.

Weighted kernel k-means. Dhillon, Guan & Kulis (KDD 2004) prove that family equals
normalized cut / normalized association -- the same family as the published nassoc loss.

VERIFIED BEFORE SPENDING ANYTHING (CLAUDE.md 9b), on bk02 at K=200
------------------------------------------------------------------
    Leiden-modularity   cohesion 0.17858    loss 0.82142
    RANDOM K=200        cohesion 0.10116    loss 0.89884
    ALL-ONE (collapse)  cohesion 0.00201    loss 0.99799   <- 89x penalty, NO anti-collapse term
    splitting the largest community: 0.17858 -> 0.17907 (improves)

The ordering is right, collapse is punished hard by the similarity structure alone, and
the k-means splitting property holds empirically. The value is bounded in [0,1] BY
CONSTRUCTION and is asserted on the first step.

WHY THESE LAMBDAS
-----------------
Because the term is bounded in [0,1], lambda IS directly the maximum contribution it can
make to the total loss -- something gamma and resolution never offered. The community
(umap) term runs at ~2.5, so:

    lambda =  1    term contributes at most 1.0 against umap 2.5  -- a regulariser
    lambda = 10    term can dominate                              -- the driver

That brackets an order of magnitude. The per-epoch `cohesion_term` value is logged, so
the gradient behaviour is measured rather than assumed. decisions.md 20 says to set
weights by gradient matching, not value matching; this is a bracket, not a final answer,
and the two arms are what make the gradient-matching probe interpretable afterwards.

CONFIG: no nassoc, no modularity, usage OFF, reinit ON ([D13] -- the first runs in which
reinit can actually fire), latent 16, K=200, bk02 -- matching the modularity sweep.

REFERENCES TO BEAT (s28nsc / bk02, K=200, latent 16)
----------------------------------------------------
    modularity lambda=10   DGE 0.822   modularity 0.3087   dead   0
    modularity lambda=1    DGE 0.672   modularity 0.3893   dead  42
    scProto best (CPM)     DGE 0.842   int_frac 0.3338
    Leiden-modularity      DGE 0.856   int_frac 0.5147
    SEACells bk02          DGE 0.849   int_frac 0.4465

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/train_cohesion.py
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

ARMS = [("coh_L16_l1_reinit", 1.0), ("coh_L16_l10_reinit", 10.0)]


@app.function(image=image, gpu="T4", timeout=7200, volumes={"/vol": volume})
def run(name: str, lam: float):
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
        "lambda_cohesion": lam,        # [D14] the only global term
        "cohesion_row_norm": "l2",     # cosine -- removes the 17.8x degree spread
        "lambda_nassoc": 0.0,          # no nassoc
        "lambda_modularity": 0.0,      # no modularity
        "lambda_proto_usage": 0.0,     # usage OFF
        "proto_reinit_every": 2,       # [D13] can finally fire
        "proto_reinit_thresh": 0.01,
        "latent_dims": 16,
    }
    assert lc["lambda_nassoc"] == 0.0 and lc["lambda_modularity"] == 0.0
    assert lc["lambda_proto_usage"] == 0.0 and lc["lambda_cohesion"] > 0

    print(f"\n{'='*78}\n=== {name}\n"
          f"    lambda_cohesion={lam}, row_norm=l2, NO nassoc, NO modularity\n"
          f"    usage OFF, reinit every 2 epochs [D13], latent 16, K=200, bk02\n"
          f"    WATCH: '[D14] cohesion epoch-1 value' must be in [0,1]\n"
          f"           '[reinit] revived N' -- if absent, [D13] did not take\n{'='*78}")

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
    print("\n" + "=" * 96)
    print("[D14] COHESION as the sole global term -- s28nsc/bk02 K=200 LD16")
    print("=" * 96)
    print(f"{'arm':<26}{'purity':>10}{'niche_pur':>11}{'modularity':>12}{'coverage':>10}")
    for name, m in results:
        def g(k):
            v = m.get(k)
            return f"{v:.4f}" if isinstance(v, (int, float)) else "--"
        print(f"{name:<26}{g('purity'):>10}{g('niche_purity'):>11}"
              f"{g('modularity'):>12}{g('coverage'):>10}")
    print("-" * 96)
    print(f"{'modularity lambda=10':<26}{'--':>10}{'--':>11}{0.3087:>12.4f}{'--':>10}"
          "   dead 0, DGE 0.822")
    print(f"{'modularity lambda=1':<26}{'--':>10}{'--':>11}{0.3893:>12.4f}{'--':>10}"
          "   dead 42, DGE 0.672")
