"""[D41] Multi-batch cohesion on PANCREAS (9 techs), cohesion-only, vs SEACells.

NOTE ON THE FILENAME: modal_scripts/train_cohesion_d41_sweep.py is a DIFFERENT,
pre-existing script whose contents are [D40] (bk04 on s28nsc). This file is the
[D41] work. Don't confuse them.

THE DECISION, WRITTEN BEFORE RUNNING
====================================
WHAT WE'RE TESTING
[D39] established on s28nsc (ONE batch) that cohesion alone is a working
objective -- gram+aff 50/50 gave joint_purity 0.4576, dge_r 0.8105, 0 dead
prototypes, close to SEACells. Pancreas is the first MULTI-batch test, and
multi-batch aggregation is the thing the whole method has to get right.

Premise ([D41]): cross-batch affinity is a batch effect, not a similarity, so a
cell's cohesion is computed only against its own bio batch (obs['tech']), with
BOTH the numerator and the denominator restricted to that batch.

WHAT WE EXPECT, AND WHAT WOULD CHANGE THE DECISION
  - arms A/B/C (coh_batch='locl') should hold batch entropy near SEACells'
    while reaching comparable ct_purity. That is the claim: we preserve good
    WITHIN-batch communities AND still combine batches.
  - arm D is the control: identical to C but coh_batch='off'. Its denominator
    carries a measured ~5.9x reward for batch-PURE prototypes, so it should show
    HIGHER ct_purity and MUCH LOWER batch entropy -- metacells that are really
    just "cell type within one tech". If D does NOT show that, the batch-purity
    pressure is coming from the encoder rather than the loss, and [D41] is a
    no-op -- that would be the finding, and the next target would be the latent
    space, not the cohesion term.
  - the rare-cell metrics are where the method is supposed to earn its keep: the
    per-batch denominator stops a rare type from being scored against a global
    size it can never fill. Expect rare_f1 / rare_homogeneity to favour 'locl'.
  - the honest risk: masking costs rare types most of their edges (t_cell keeps
    28% of its neighbours, macrophage 26%, vs ~5% lost for alpha/beta). If the
    rare metrics get WORSE, that cost outweighs the denominator fix, and the
    answer is the per-cell denominator variant (keep cross-batch edges in the
    numerator) rather than abandoning the idea.

VERDICT (fill in after reading):
  [ ] works   [ ] fails   [ ] inconclusive because ___   [ ] abandoned

CONFIGURATION, AND WHY EACH VALUE
=================================
cohesion_self = 0.0  -- REQUIRED with coh_batch, not a preference. The self-pair
    contributes 1/size_in_batch per cell; a batch-SPREAD prototype has
    size_in_batch = size/n_bio, so at cohesion_self=1 the term pays the model to
    smear prototypes thinly across batches. Measured on this exact graph at
    K=220: random-220 scores 0.12007 on the self part vs leiden's 0.01343,
    cancelling a pair signal that discriminates 12x and collapsing the span from
    0.107 to 0.0006 (lambda would come out 2396). The s28nsc arms used 1 and
    were right to -- there n_bio = 1 and the self part carries no batch
    preference at all.

cohesion_diag = 1.0  -- kept. It defines the CLOSED-neighbourhood gram
    descriptor (the 1-hop/2-hop mixture knob), and for the aff term it is added
    and then removed again by cohesion_self=0, so it is a no-op there.

LAMBDAS -- measured, not swept (modal_scripts/measure_cohesion_spans_pancreas.py,
    CPU, ~15s). span = value(leiden K=220) - value(random K=220) on the MASKED
    graph with the per-batch denominator; lambda = AUTHORITY / span with
    AUTHORITY = 1.5403 held fixed from the s28nsc arms so that "does mixing the
    two similarities help" is not confounded with "is more cohesion better".

        term   span       lambda alone    lambda at 50/50
        aff    0.107284      14.36            7.18
        gram   0.254654       6.05            3.02

    span_gram/span_aff = 2.374 here vs 2.219 on s28nsc/bk02 -- the same
    relationship between the two similarities on an independent graph.

REFERENCE VALUES on this graph, K=220, hard assignments, full graph, the
objective as configured here (cohesion_self=0, cohesion_diag=1, per-batch):

        partition          aff        gram
        leiden K=220     0.116908   0.282367   <- practical ceiling
        celltype (14)    0.073253   0.186007
        ct x tech (112)  0.073253   0.186007   <- identical: batch-neutral, as designed
        random-220       0.009624   0.027713   <- FLOOR
        one group        0.011614   0.032876

ARMS
====
  d41_A_aff    aff only,  lambda_cohesion_aff = 14.36
  d41_B_gram   gram only, lambda_cohesion     = 6.05
  d41_C_both   both 50/50, gram 3.02 + aff 7.18   <- the s28nsc winner's recipe
  d41_D_ctrl   identical to C but coh_batch='off'  <- isolates the multi-batch change

Everything else zero: umap, proto_recon, nassoc, modularity, proto_usage, swav,
kl, recon. Cohesion is the ONLY objective, so nothing can outvote it and the
arms differ in exactly one thing each.

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach \
       modal_scripts/train_d41_pancreas_multibatch.py
Cost: 4 arms in parallel on a T4, pancreas is 16,382 cells -- well under the
s28nsc runtime of ~13 min/arm.
"""
import time

import modal

app = modal.App("scproto-d41-pancreas")

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
    # Data_Handler.py eagerly opens these eleven at import time regardless of
    # which dataset is in use (CLAUDE.md 11). Stub them or nothing imports.
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
SPAN_AFF, SPAN_GRAM = 0.107284, 0.254654

LAM_AFF_ONLY = AUTHORITY / SPAN_AFF              # 14.36
LAM_GRAM_ONLY = AUTHORITY / SPAN_GRAM            # 6.05
LAM_AFF_HALF = 0.5 * AUTHORITY / SPAN_AFF        # 7.18
LAM_GRAM_HALF = 0.5 * AUTHORITY / SPAN_GRAM      # 3.02

# (name, lambda_gram, lambda_aff, coh_batch)
ARMS = [
    ("d41_A_aff",  0.0,            LAM_AFF_ONLY,  "locl"),
    ("d41_B_gram", LAM_GRAM_ONLY,  0.0,           "locl"),
    ("d41_C_both", LAM_GRAM_HALF,  LAM_AFF_HALF,  "locl"),
    ("d41_D_ctrl", LAM_GRAM_HALF,  LAM_AFF_HALF,  "off"),
]


@app.function(image=image, gpu="T4", timeout=7200, volumes={"/vol": volume})
def run(name: str, lam_gram: float, lam_aff: float, coh_batch: str):
    import os
    import sys

    os.environ["WANDB_MODE"] = "disabled"
    os.environ["HOME_DIR"] = "/vol/"
    os.environ["DATA_DIR"] = "/vol/data/"
    os.environ["MODEL_DIR"] = "/vol/models/"
    os.environ["CODE_DIR"] = "/vol/code/"
    sys.path.insert(0, "/vol/code")
    os.chdir("/vol/code")          # get_affinity_path's './graphs' is CWD-relative

    # Files written by an older anndata encode Python None as an HDF5 dataset
    # with `encoding-type: null`, which current anndata has no reader for at all
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

    lc = dict(LAMBDA_PROTO_UMAP_PRECON) | {
        # --- everything that is not cohesion is OFF -------------------------
        "lambda_umap": 0.0,
        "lambda_proto_recon": 0.0,
        "lambda_nassoc": 0.0,
        "lambda_modularity": 0.0,
        "lambda_proto_usage": 0.0,
        "lambda_swav": 0.0,
        "lambda_kl": 0.0,
        "lambda_recon": 0.0,
        # --- the two knobs that define the arm ------------------------------
        "lambda_cohesion": lam_gram,        # drives the GRAM (2-hop) path
        "lambda_cohesion_aff": lam_aff,     # independent 1-hop term
        "coh_batch": coh_batch,             # [D41] THE variable under test
        # --- the cohesion configuration -------------------------------------
        "cohesion_sim": "gram",
        "cohesion_form": "perc",
        "cohesion_row_norm": "l2",
        "cohesion_diag": 1.0,               # closed-neighbourhood descriptor
        "cohesion_self": 0.0,               # REQUIRED with coh_batch -- see header
        "cohesion_global": 2,               # [D29] remembered full-graph profiles
        "coh_mem_init": "full",             # [D32]
        "coh_mem_hlife": 500.0,             # [D31]
        "proto_reinit_every": 0,            # NO reinit
        "coh_diag_every": 25,
        "latent_dims": 16,
    }

    # Guard the authority so an arm cannot silently drift off the calibration.
    # NOTE the spans were measured under the PER-BATCH denominator, so for arm D
    # (coh_batch='off') this is not "the calibrated lambda for 'off'" -- it is
    # deliberately the SAME lambda as arm C, because D exists to isolate the
    # coh_batch change and nothing else. Holding lambda fixed is the point.
    auth = lam_gram * SPAN_GRAM + lam_aff * SPAN_AFF
    assert abs(auth - AUTHORITY) < 0.02, f"authority drifted: {auth} != {AUTHORITY}"
    assert lc["cohesion_self"] == 0.0, "cohesion_self MUST be 0 with coh_batch"
    assert lc["lambda_umap"] == 0.0 and lc["lambda_proto_recon"] == 0.0, \
        "cohesion-only: nothing may outvote the term under test"

    print("\n" + "=" * 78)
    print(f"=== {name}   (pancreas, arbf, K=220, 9 techs)")
    print(f"    lambda_gram={lam_gram:.2f}  lambda_aff={lam_aff:.2f}  "
          f"authority={auth:.4f}")
    print(f"    coh_batch={coh_batch!r}  cohesion_self=0.0  cohesion_diag=1.0")
    print(f"    community OFF, no reinit, no early stop, 20 fixed epochs")
    print(f"    REFERENCE (K=220, per-batch, self=0): "
          f"aff random 0.0096 / leiden 0.1169 | gram random 0.0277 / leiden 0.2824")
    print("=" * 78)

    t0 = time.time()
    trainer, metrics, _ = run_mc_task(
        "pancreas",
        cvae_epochs=50, train_epochs=20, eval_freq=2, patience=4,
        batch_size=1024, affinity_type="arbf", num_prototypes=220,
        lambda_config=lc,
        trainer_kwargs={"experiment_name": name},
        early_stop=False,
    )
    print(f"\n[{name}] model name: {trainer.get_model_name()}")
    print(f"[{name}] done in {time.time()-t0:.0f}s")
    print(f"[{name}] >>> metrics: {metrics}")
    volume.commit()
    return name, trainer.get_model_name(), metrics


@app.local_entrypoint()
def main():
    results = list(run.starmap(ARMS))
    print("\n" + "=" * 96)
    print("[D41] pancreas multi-batch cohesion -- raw training metrics")
    print("(the deciding metrics -- batch entropy, ct purity, rare f1/homogeneity/")
    print(" coverage, and cohesion on the HELD-OUT techs -- come from the")
    print(" post-hoc eval script, which also runs SEACells on the same graph)")
    print("=" * 96)
    print(f"{'arm':<14}{'n_dead':>9}{'purity':>10}{'coh_aff':>10}{'coh_gram':>11}")
    for name, model_name, m in results:
        def g(k, f="{:.4f}"):
            v = m.get(k)
            return f.format(v) if isinstance(v, (int, float)) else "--"
        print(f"{name:<14}{g('n_dead','{:.0f}'):>9}{g('purity'):>10}"
              f"{g('coh_aff'):>10}{g('coh_gram'):>11}")
    print("-" * 96)
    print("\nmodel names (for the eval script):")
    for name, model_name, _ in results:
        print(f"  {name:<14} {model_name}")
