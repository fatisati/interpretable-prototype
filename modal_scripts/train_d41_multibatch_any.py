"""[D41] Multi-batch cohesion across datasets: pancreas, immune, lung, crcx.

THE DECISION, WRITTEN BEFORE RUNNING
====================================
WHAT IS BEING TESTED
On pancreas, coh_batch='locl' produced metacells that mix batches (batch entropy
1.38 vs SEACells 0.18) at near-parity cell-type purity, and on the paper's own
rare-cell metrics it roughly DOUBLED cross-batch homogeneity against its own
identical except coh_batch, so the batch integration comes from the LOSS.

This run asks whether that replicates on three more datasets, two of them
non-spatial multi-batch (immune, lung) and one multi-batch SPATIAL (crcx, which
carries a niche annotation so covered states and joint purity are computable).

WHAT WOULD CHANGE THE CONCLUSION
    integration is NOT coming from the loss on that dataset and [D41] is a no-op
    there -- that is the finding, and the next target is the latent space;
  - if ct_purity collapses (say below 0.85) while batch entropy rises, the term
    is merging across cell types rather than across batches, and the trade-off
    is not worth it;
  - rare PRECISION is expected to drop (it did on pancreas, -0.20): merging rare
    cells across batches makes their metacells less pure. If recall/coverage do
    NOT rise to compensate, the trade is bad.

VERDICT (fill in per dataset after reading):
  pancreas  [ ] works  [ ] fails  [ ] inconclusive because ___
  immune    [ ] works  [ ] fails  [ ] inconclusive because ___
  lung      [ ] works  [ ] fails  [ ] inconclusive because ___
  crcx      [ ] works  [ ] fails  [ ] inconclusive because ___

CONFIGURATION
=============
cohesion_self = 0.0 -- REQUIRED with coh_batch, not a preference. The self-pair
    is 1/size_in_batch, which a batch-SPREAD prototype maximises, so at 1 the
    term pays the model to smear prototypes thinly across batches. Measured on
    pancreas: collapses the span 0.107 -> 0.0006, implying lambda 2396.
cohesion_diag = 1.0 -- defines the closed-neighbourhood gram descriptor; for the
    aff term it is added then removed again by cohesion_self=0, so it is a no-op
    there.
Community/nassoc/modularity/usage/swav/kl/recon all ZERO: cohesion is the only
    objective, so nothing can outvote the term under test.

LAMBDAS are measured per dataset, never swept and never carried across datasets
-- spans are not portable (CLAUDE.md 9b). lambda = AUTHORITY/span with
AUTHORITY = 1.5403 held fixed from the s28nsc arms, spans from
modal_scripts/measure_cohesion_spans_any.py on the MASKED graph with the
per-batch denominator.

COST: 4 arms x 4 datasets on T4. Pancreas (16k cells) took ~25 min/arm including
a fresh 50-epoch pretrain. Check `modal app list` before assuming it is cheap.

Run:
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach \
      modal_scripts/train_d41_multibatch_any.py --datasets pbmc-immune
"""
import time

import modal

app = modal.App("scproto-d41-any")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "numpy", "scipy", "pandas", "anndata", "scanpy",
        "torch", "scarches", "wandb", "tqdm", "statsmodels", "diptest",
        "palantir", "faiss-cpu", "scikit-learn", "matplotlib", "scib-metrics",
        "leidenalg", "python-igraph", "ipython",
    )
    .pip_install("git+https://github.com/dpeerlab/SEACells.git",
                 extra_options="--no-deps")
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

AUTHORITY = 1.5403

# dataset -> (affinity_type, K, span_aff, span_gram, latent_dims)
# spans measured by measure_cohesion_spans_any.py -- see the header.
# Measured 2026-09-23 by measure_cohesion_spans_any.py, on the MASKED graph with
# the per-batch denominator and cohesion_self=0, at each dataset's own K.
# span_gram/span_aff comes out 2.05-2.37 on all four independent graphs -- the
# same relationship between the two similarities everywhere, which is a good
# sign the calibration is measuring something real rather than noise.
# latent_dims = 8 throughout: the dataset default, what the paper's stored
# baselines use, and what the cached pretrain checkpoints were built at.
SPANS = {
    "pancreas":    ("arbf", 220, 0.108016, 0.256143, 8),
    "pbmc-immune": ("arbf", 300, 0.048287, 0.124487, 8),
    "lung":        ("arbf", 300, 0.067165, 0.181939, 8),
    "crcx":        ("arbf", 670, 0.029727, 0.061007, 8),
}


def arms_for(ds):
    aff_t, K, s_aff, s_gram, ld = SPANS[ds]
    if s_aff is None or s_gram is None:
        raise ValueError(f"{ds}: spans not measured yet -- run "
                         f"measure_cohesion_spans_any.py {ds} and fill SPANS")
    a1, g1 = AUTHORITY / s_aff, AUTHORITY / s_gram
    ah, gh = 0.5 * a1, 0.5 * g1
    return [
        (ds, "A_aff",  0.0, a1,  aff_t, K, ld, s_aff, s_gram),
        (ds, "B_gram", g1,  0.0,  aff_t, K, ld, s_aff, s_gram),
        (ds, "C_both", gh,  ah,   aff_t, K, ld, s_aff, s_gram),
    ]


@app.function(image=image, gpu="T4", timeout=14400, volumes={"/vol": volume})
def run(ds: str, arm: str, lam_gram: float, lam_aff: float, coh_batch: str,
        aff_type: str, K: int, latent_dims: int,
        span_aff: float, span_gram: float):
    import os
    import sys

    os.environ["WANDB_MODE"] = "disabled"
    os.environ["HOME_DIR"] = "/vol/"
    os.environ["DATA_DIR"] = "/vol/data/"
    os.environ["MODEL_DIR"] = "/vol/models/"
    os.environ["CODE_DIR"] = "/vol/code/"
    sys.path.insert(0, "/vol/code")
    os.chdir("/vol/code")        # get_affinity_path's './graphs' is CWD-relative

    # Older anndata encodes Python None as an HDF5 dataset with
    # `encoding-type: null`, for which current anndata has NO reader at all
    # (CLAUDE.md 11). Register one before reading anything.
    import h5py
    from anndata._io.specs.registry import _REGISTRY, IOSpec
    try:
        @_REGISTRY.register_read(h5py.Dataset, IOSpec("null", "0.1.0"))
        def _read_null(elem, *a, **kw):
            return None
    except TypeError:
        pass

    from interpretable_ssl.experiments.tasks import (run_mc_task,
                                                     LAMBDA_PROTO_UMAP_PRECON)

    name = f"d41_{ds}_{arm}"
    lc = dict(LAMBDA_PROTO_UMAP_PRECON) | {
        "lambda_umap": 0.0,
        "lambda_proto_recon": 0.0,
        "lambda_nassoc": 0.0,
        "lambda_modularity": 0.0,
        "lambda_proto_usage": 0.0,
        "lambda_swav": 0.0,
        "lambda_kl": 0.0,
        "lambda_recon": 0.0,
        "lambda_cohesion": lam_gram,
        "lambda_cohesion_aff": lam_aff,
        
        "cohesion_sim": "gram",
        "cohesion_form": "perc",
        "cohesion_row_norm": "l2",
        "cohesion_diag": 1.0,
        "cohesion_self": 0.0,
        "cohesion_global": 2,
        "coh_mem_init": "full",
        "coh_mem_hlife": 500.0,
        "proto_reinit_every": 0,
        "coh_diag_every": 50,
        "latent_dims": latent_dims,
    }
    auth = lam_gram * span_gram + lam_aff * span_aff
    assert abs(auth - AUTHORITY) < 0.03, f"authority drifted: {auth}"
    assert lc["cohesion_self"] == 0.0, "cohesion_self MUST be 0 with coh_batch"
    assert lc["lambda_umap"] == 0.0, "cohesion-only: nothing may outvote it"

    print("\n" + "=" * 78)
    print(f"=== {name}   K={K}  aff={aff_type}  LD={latent_dims}")
    print(f"    lambda_gram={lam_gram:.2f}  lambda_aff={lam_aff:.2f}  "
          f"authority={auth:.4f}")
    print(f"    coh_batch={coh_batch!r}  cohesion_self=0.0  cohesion_diag=1.0")
    print(f"    community OFF, no reinit, no early stop, 20 fixed epochs")
    print("=" * 78)

    t0 = time.time()
    trainer, metrics, _ = run_mc_task(
        ds, cvae_epochs=50, train_epochs=20, eval_freq=2, patience=4,
        batch_size=1024, affinity_type=aff_type, num_prototypes=K,
        lambda_config=lc, trainer_kwargs={"experiment_name": name},
        early_stop=False,
    )
    print(f"\n[{name}] model: {trainer.get_model_name()}")
    print(f"[{name}] done in {time.time()-t0:.0f}s")
    print(f"[{name}] >>> {metrics}")
    volume.commit()
    return name, trainer.get_model_name(), metrics


@app.local_entrypoint()
def main(datasets: str = "pbmc-immune"):
    jobs = []
    for ds in [d.strip() for d in datasets.split(",") if d.strip()]:
        jobs.extend(arms_for(ds))
    print(f"launching {len(jobs)} arms: "
          + ", ".join(f"{j[0]}/{j[1]}" for j in jobs))
    results = list(run.starmap(jobs))
    print("\n" + "=" * 92)
    print("[D41] training finished -- run the eval for the real scoreboard")
    print("=" * 92)
    for name, model_name, m in results:
        nd = m.get("n_dead")
        pur = m.get("purity")
        print(f"  {name:<24} n_dead={nd if nd is not None else '--':<5} "
              f"purity={pur if pur is None else round(pur, 4)}")
        print(f"      {model_name}")
