"""Rerun of the H4_g015 arm (my-tasks/context/decisions.md sections 5-7) with the [D7]
fix applied: `nassoc_exact=1` + `nassoc_size_scale='full'` previously rescaled
`sum_i assign[i,p]^2` by `cell_scale**2` instead of `cell_scale`, which could collapse
`possible_pairs` back to 0 for rare prototypes -- exactly the failure mode `nassoc_exact`
was written to prevent. See decisions.md section 7 for the full derivation and the
real-data check that found it.

Same config as the existing (buggy) run already on the volume:
  models/s28nsc/lf_h4_g015_..._nagm0.15_nass-full_nacn-gful_nacf1_naex1_..._v31
i.e. decisions.md's "Best config to use" (dict(LAMBDA_PROTO_UMAP_PRECON) | nassoc_diag_norm=
'pair', nassoc_diag_loss='cpm', nassoc_cpm_free=1, nassoc_size_scale='full',
nassoc_cpm_norm='gful', nassoc_gamma=0.15, lambda_nassoc=1000) plus nassoc_exact=1 --
except this run's code has the [D7] fix, and the folder gets a distinct
'lf_h4_g015_d7fix' experiment_name so it does NOT overwrite the buggy baseline (needed for
the before/after diff).

Epoch schedule: CLAUDE.md's CURRENT standard (cvae_epochs=50, train_epochs=20,
eval_freq=2, patience=4). The original H4_g015 run's folder name does not encode epoch
count, so this is the best-available match, not a confirmed exact reproduction -- noted
in decisions.md.

After training, computes (in the SAME function, one cold start):
  - full run_mc_task eval (purity, niche purity, batch entropy, modularity) -- same as the
    existing metrics.json, for direct before/after diff
  - joint (cell-type x niche) purity + collapse shape (largest_share, dip stat) via
    interpretable_ssl.evaluation.collapse_diagnostics -- decisions.md's "joint purity" gap
  - the niche-program-recovery DGE pipeline (interpretable_ssl.evaluation.niche_dge),
    reusing the cached ground truth already on the volume
    (code/files/celltype_niches_full_s28nsc.csv) -- directly comparable to
    notebooks/s28nsc_dge_lite/tables/H4_g015_summary.csv

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/train_h4_g015_d7fix.py
Cost: ~12-13 min training on a T4 (CLAUDE.md section 11) + a few minutes of CPU-side eval.
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


@app.function(image=image, gpu="T4", timeout=3600, volumes={"/vol": volume})
def train_h4_g015_d7fix():
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

    import scanpy as sc
    from interpretable_ssl.experiments.tasks import run_mc_task, LAMBDA_PROTO_UMAP_PRECON
    from interpretable_ssl.datasets.dataset_configs import DATASETS
    from interpretable_ssl.evaluation import niche_dge as nd
    from interpretable_ssl.evaluation.collapse_diagnostics import (
        labels_from_trainer, joint_purity_per_metacell, collapse_summary, metacell_sizes,
    )

    t0 = time.time()
    lc = dict(LAMBDA_PROTO_UMAP_PRECON) | {
        'nassoc_diag_norm': 'pair',
        'nassoc_diag_loss': 'cpm',
        'nassoc_cpm_free': 1,
        'nassoc_size_scale': 'full',
        'nassoc_cpm_norm': 'gful',
        'nassoc_gamma': 0.15,
        'lambda_nassoc': 1000,
        'nassoc_exact': 1,   # the arm under test, now with the [D7] fix in scproto.py
    }

    trainer, metrics, _ = run_mc_task(
        "s28nsc",
        cvae_epochs=50, train_epochs=20, eval_freq=2, patience=4,
        batch_size=1024, affinity_type="bk02", num_prototypes=200,
        label_key="celltypes", niche_key="niches_2D",
        lambda_config=lc,
        trainer_kwargs={"experiment_name": "lf_h4_g015_d7fix"},
    )
    train_elapsed = time.time() - t0
    print("=== TRAINING DONE ===")
    print("elapsed seconds:", train_elapsed)
    print("model name:", trainer.get_model_name())
    print("run_mc_task metrics:", metrics)

    volume.commit()  # persist checkpoints/metrics before spending more time on eval

    # ---- joint purity + collapse shape (decisions.md's "coverage/joint purity" gap) ----
    t1 = time.time()
    ds_conf = DATASETS["s28nsc"]
    adata = sc.read_h5ad(ds_conf["path"])
    labels = labels_from_trainer(trainer, adata).dropna()
    n_states_found = int(labels.nunique())

    purity = joint_purity_per_metacell(labels, adata, "celltypes", "niches_2D")
    collapse_row = collapse_summary(labels, name="H4_g015_d7fix", purity=purity)
    collapse_row["n_metacells"] = int(labels.nunique())
    collapse_row["median_size"] = float(metacell_sizes(labels).median())
    print("collapse/purity row:", collapse_row)

    # ---- DGE recovery, same pipeline + cached ground truth as H4_g015_summary.csv ----
    GT_PATH = os.path.join("/vol/code", "files", "celltype_niches_full_s28nsc.csv")
    ground_truth = nd.get_ground_truth(adata, GT_PATH, ct_key="celltypes", niche_key="niches_2D")
    print(f"{len(ground_truth)} (cell type, niche) ground-truth pairs")

    TABLE_DIR = "/vol/code/notebooks/s28nsc_dge_lite/tables"
    os.makedirs(TABLE_DIR, exist_ok=True)
    dge_result = nd.run_niche_dge(
        "H4_g015_d7fix", labels, adata, ground_truth, TABLE_DIR,
        ct_key="celltypes", niche_key="niches_2D", force=True,
    )
    eval_elapsed = time.time() - t1
    print("=== EVAL DONE ===")
    print("eval elapsed seconds:", eval_elapsed)
    print("DGE summary:\n", dge_result["summary"])

    volume.commit()

    return {
        "train_elapsed": train_elapsed,
        "eval_elapsed": eval_elapsed,
        "model_name": trainer.get_model_name(),
        "run_mc_task_metrics": metrics,
        "n_states_found": n_states_found,
        "collapse_row": collapse_row,
        "dge_summary": dge_result["summary"].to_dict(orient="records"),
    }


@app.local_entrypoint()
def main():
    t0 = time.time()
    result = train_h4_g015_d7fix.remote()
    print("total wall time incl. cold start (s):", time.time() - t0)
    import json
    print(json.dumps(result, indent=2, default=str))
