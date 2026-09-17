"""[D9] CORRECTED CPM on PANCREAS / PCA affinity (arbf), K=220, usage loss OFF.

WHY PANCREAS IS THE REAL TEST OF THE [D9] FIX
---------------------------------------------
The estimator bias removed by nassoc_ht=1 is size-dependent: measured over-charge of
possible_pairs is 2.39x below 25 cells, 1.45x at 25-50, 1.13x at 50-100, and only
~1.0x at 200-900 (modal_scripts/measure_estimator_bias.py). On s28nsc every prototype
sits at ~292 cells, so the fix was worth ~8% and -- as predicted in advance -- moved
nothing (ht_L16 DGE 0.833 vs 0.842, inside noise).

Pancreas is the opposite regime. Measured on the real cached graph + the stored
SEACells run:

    SEACells pancreas K=220 on X_pca:
      median size 58,  min 8,  max 443,  max/median 7.6
      176 of 220 metacells are UNDER 100 CELLS
      internal_fraction 0.5684
      per-metacell density: median 0.2019, aggregate 0.1100

    N/K = 16382/220 = 74.5 cells per metacell

So four fifths of the partition we are chasing sits exactly where the old estimator
over-charged by 1.2x-2.4x.

COUNTER-EVIDENCE, STATED UP FRONT
---------------------------------
On s28nsc the corrected loss produced MORE uniform sizes, not less (max/median 1.7 vs
2.7 before, min size 150, zero prototypes under 100). So the size rigidity is NOT caused
by the estimator bias -- consistent with EXPERIMENTS.md section 5, which localized it to
CPM's intrinsic quadratic charge. scProto may therefore produce ~74-cell UNIFORM
metacells here against SEACells' 8-to-443 spread, and the comparison will turn on
whether uniform metacells can match an uneven partition at this scale.

GAMMA IS MEASURED, NOT SWEPT, AND IT IS DATASET-SPECIFIC
--------------------------------------------------------
gamma is the minimum AVERAGE affinity a cell needs to a group to be worth joining it,
so it is a property of THIS graph -- s28nsc's 0.028 does not transfer. Measured on
pancreas/arbf: SEACells' own partition achieves aggregate density 0.1100 and median
per-metacell density 0.2019, and the true celltype partition reaches internal_fraction
0.9668 (vs 0.4549 on s28nsc -- pancreas is a far cleaner graph). Using gamma = 0.11,
chosen at the aggregate density exactly as 0.028 was chosen for s28nsc.

NOTHING IS RECOMPUTED
---------------------
The arbf affinity is already cached (load_or_build_affinity checks get_affinity_path
first), the pancreas cvae_e50 pretrain checkpoint is reused, and the SEACells baseline
is read from the stored run -- seacell_X_pca_ksweepE8_K220 -- never refit.

latent_dims is left at the default 8, matching the published pancreas main-table config,
so this is a one-variable diff against it. (16 would key a different pretrain checkpoint
that does not exist for pancreas and would cost a full pretrain.)

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/train_pancreas_ht_nousage.py
Cost: one ~8 min T4 arm (16,382 cells, 3.6x smaller than s28nsc).
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

GAMMA = 0.11            # measured on pancreas/arbf, see module docstring
LAMBDA_NASSOC = 200     # same as the s28nsc ht run, for comparability across the two
SEACELL_REF = dict(     # stored run seacell_X_pca_ksweepE8_K220/metrics.json -- NOT refit
    mean_cell_type_purity=0.9563, weighted_mean_cell_type_purity=0.9653,
    mean_batch_entropy=0.1916, weighted_mean_batch_entropy=0.2023,
    modularity=0.7377, mean_modularity_batch=0.6729, std_modularity_batch=0.0518,
    coverage=0.8571, n_unused_protos=0,
    median_size=58, min_size=8, max_size=443, n_under_100=176,
    internal_fraction=0.5684, aggregate_density=0.1100,
)


@app.function(image=image, gpu="T4", timeout=5400, volumes={"/vol": volume})
def train_pancreas_ht():
    import os
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["HOME_DIR"] = "/vol/"
    os.environ["DATA_DIR"] = "/vol/data/"
    os.environ["MODEL_DIR"] = "/vol/models/"
    os.environ["CODE_DIR"] = "/vol/code/"

    import sys
    sys.path.insert(0, "/vol/code")
    os.chdir("/vol/code")

    import h5py
    from anndata._io.specs.registry import _REGISTRY, IOSpec
    try:
        @_REGISTRY.register_read(h5py.Dataset, IOSpec("null", "0.1.0"))
        def _read_null(elem, _reader):
            return None
    except TypeError:
        pass

    import numpy as np, pandas as pd, pickle, scipy.sparse as sp
    from interpretable_ssl.experiments.tasks import run_mc_task, LAMBDA_PROTO_UMAP_PRECON

    print("=" * 78)
    print(f"PANCREAS / arbf (PCA) / K=220 / gamma={GAMMA} / usage OFF / [D9] corrected CPM")
    print("=" * 78)

    lc = dict(LAMBDA_PROTO_UMAP_PRECON) | {
        'nassoc_diag_norm': 'pair', 'nassoc_diag_loss': 'cpm', 'nassoc_cpm_free': 1,
        'nassoc_exact': 1,
        'nassoc_ht': 1,                 # [D9] debiased, batch-only CPM
        'nassoc_cpm_norm': 'btsz',      # [D9] divide by the batch's edge count
        'nassoc_gamma': GAMMA,
        'lambda_nassoc': LAMBDA_NASSOC,
        'lambda_proto_usage': 0.0,      # usage OFF
    }

    t0 = time.time()
    trainer, metrics, _ = run_mc_task(
        "pancreas",
        cvae_epochs=50, train_epochs=20, eval_freq=2, patience=4,
        batch_size=1024, affinity_type="arbf", num_prototypes=220,
        lambda_config=lc,
        trainer_kwargs={"experiment_name": "panc_ht_g011_uOFF"},
    )
    el = time.time() - t0
    print(f"\n[panc_ht_g011_uOFF] done in {el:.0f}s  model={trainer.get_model_name()}")
    volume.commit()

    # ---- size distribution of the resulting partition, vs SEACells' -----------
    try:
        import scanpy as sc
        from interpretable_ssl.datasets.dataset_configs import DATASETS
        from interpretable_ssl.evaluation.collapse_diagnostics import labels_from_trainer
        adata = sc.read_h5ad(DATASETS["pancreas"]["path"])
        labels = labels_from_trainer(trainer, adata).dropna()
        with open("/vol/code/graphs/"
                  "affinity_pancreas16382_ncomp50_kneighbors50_arbf.pkl", "rb") as f:
            _A = pickle.load(f).tocsr()
        _A = _A.tocsr(copy=True); _A.setdiag(0); _A.eliminate_zeros()
        _T = sp.triu(_A, k=1).tocoo(); _m = float(_T.sum())
        lab = pd.Series(labels).reindex(adata.obs_names).to_numpy()
        li = np.full(len(lab), -1, np.int64)
        ok = ~pd.isna(lab); li[ok] = lab[ok].astype(np.int64)
        same = (li[_T.row] == li[_T.col]) & (li[_T.row] >= 0)
        e = float(_T.data[same].sum())
        sizes = np.bincount(li[li >= 0]); sizes = sizes[sizes > 0].astype(float)
        pairs = float((sizes * (sizes - 1) / 2).sum())
        shape = dict(n_alive=int(len(sizes)), median_size=float(np.median(sizes)),
                     min_size=float(sizes.min()), max_size=float(sizes.max()),
                     max_over_median=float(sizes.max() / np.median(sizes)),
                     n_under_100=int((sizes < 100).sum()),
                     internal_fraction=e / _m,
                     aggregate_density=e / pairs if pairs else float("nan"))
    except Exception as ex:
        import traceback; traceback.print_exc()
        shape = {"error": str(ex)}

    def g(d, k):
        v = d.get(k)
        return f"{v:.4f}" if isinstance(v, (int, float)) else "--"

    print("\n" + "=" * 96)
    print("PANCREAS -- corrected CPM vs SEACells (stored run, not refit)")
    print("=" * 96)
    print(f"{'metric':<34}{'scProto [D9]':>16}{'SEACells X_pca K220':>24}")
    for k in ("mean_cell_type_purity", "weighted_mean_cell_type_purity",
              "mean_batch_entropy", "weighted_mean_batch_entropy",
              "modularity", "mean_modularity_batch", "coverage", "n_unused_protos"):
        print(f"{k:<34}{g(metrics, k):>16}{g(SEACELL_REF, k):>24}")
    print(f"{'--- partition shape ---':<34}")
    for k in ("median_size", "min_size", "max_size", "max_over_median",
              "n_under_100", "internal_fraction", "aggregate_density"):
        print(f"{k:<34}{g(shape, k):>16}{g(SEACELL_REF, k):>24}")
    print("\nNOTE: purity HIGHER is better; batch_entropy HIGHER is better (mixing);")
    print("      modularity HIGHER is better. n_under_100 is the [D9] question:")
    print("      SEACells puts 176/220 metacells under 100 cells -- can we?")

    volume.commit()
    return {"elapsed": el, "metrics": metrics, "shape": shape,
            "model_name": trainer.get_model_name()}


@app.local_entrypoint()
def main():
    print(train_pancreas_ht.remote())
