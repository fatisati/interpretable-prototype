"""Small gamma sweep for `nassoc_exact=1` under `nassoc_size_scale='btch'` +
`nassoc_cpm_norm='dpar'` (the OTHER half of decisions.md section 7's [D7] finding).

Why this sweep exists: the [D7] fix only touches the `nassoc_size_scale='full'` code
path (see decisions.md section 7's "Does 'btch'+'dpar' need this fix too? No"). But
`gamma=0.15` was calibrated for 'full'-scale density (CLAUDE.md 9c's
mean_weighted_degree/(target_size-1) formula, a FULL-DATASET density target). Under
'btch', density is a raw, unrescaled one-minibatch ratio -- a different, noisier
quantity with no reason to share the same break-even threshold. Reusing 0.15 there would
not be a like-for-like comparison.

Existing 'dpar' runs on the volume (lf_g1_bn_g01/g03, lf_g2_bn_g03) used gamma in
0.01-0.03 WITHOUT nassoc_exact (i.e. the old hard size*(size-1)/2 possible_pairs
formula) -- a real prior suggesting 'btch'-scale gamma sits an order of magnitude below
'full'-scale gamma. This sweep covers [0.01, 0.03, 0.05] WITH nassoc_exact=1, to find
where a target-size metacell actually sits once possible_pairs is computed exactly.

lambda_nassoc kept at 1000 (matches the H1/H4 'full'+'gful' sweep) so gamma is the only
variable that differs run to run, not two things at once.

All three gammas run in ONE container (one cold start, one pretrain-checkpoint load,
one dataset load) since each individual training run is short relative to Modal's
container startup cost.

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/train_dpar_exact_gamma_sweep.py
Cost: ~3 x 12-13 min training on a T4 (CLAUDE.md section 11) + eval -- roughly 40-45 min,
a few dollars total. Ask before scaling this sweep any wider.
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

GAMMAS = [0.01, 0.03, 0.05]


@app.function(image=image, gpu="T4", timeout=3600 * 2, volumes={"/vol": volume})
def train_dpar_exact_gamma_sweep():
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

    ds_conf = DATASETS["s28nsc"]
    adata = sc.read_h5ad(ds_conf["path"])
    GT_PATH = os.path.join("/vol/code", "files", "celltype_niches_full_s28nsc.csv")
    ground_truth = nd.get_ground_truth(adata, GT_PATH, ct_key="celltypes", niche_key="niches_2D")
    TABLE_DIR = "/vol/code/notebooks/s28nsc_dge_lite/tables"
    os.makedirs(TABLE_DIR, exist_ok=True)

    all_results = {}
    for gamma in GAMMAS:
        tag = f"g{str(gamma).replace('0.', '')}"  # 0.01 -> g01, 0.03 -> g03, 0.05 -> g05
        name = f"lf_i_dpar_exact_{tag}"
        print(f"\n{'='*70}\n=== {name}  (gamma={gamma}) ===\n{'='*70}")

        t0 = time.time()
        lc = dict(LAMBDA_PROTO_UMAP_PRECON) | {
            'nassoc_diag_norm': 'pair',
            'nassoc_diag_loss': 'cpm',
            'nassoc_cpm_free': 1,
            'nassoc_size_scale': 'btch',   # explicit -- this is the default, but stated
                                            # for clarity: NO full-dataset rescaling here
            'nassoc_cpm_norm': 'dpar',
            'nassoc_gamma': gamma,
            'lambda_nassoc': 1000,
            'nassoc_exact': 1,
        }

        trainer, metrics, _ = run_mc_task(
            "s28nsc",
            cvae_epochs=50, train_epochs=20, eval_freq=2, patience=4,
            batch_size=1024, affinity_type="bk02", num_prototypes=200,
            label_key="celltypes", niche_key="niches_2D",
            lambda_config=lc,
            trainer_kwargs={"experiment_name": name},
        )
        train_elapsed = time.time() - t0
        print(f"[{name}] training done in {train_elapsed:.0f}s, model={trainer.get_model_name()}")
        volume.commit()

        labels = labels_from_trainer(trainer, adata).dropna()
        purity = joint_purity_per_metacell(labels, adata, "celltypes", "niches_2D")
        collapse_row = collapse_summary(labels, name=name, purity=purity)
        collapse_row["n_metacells"] = int(labels.nunique())
        collapse_row["median_size"] = float(metacell_sizes(labels).median())

        dge_result = nd.run_niche_dge(
            name, labels, adata, ground_truth, TABLE_DIR,
            ct_key="celltypes", niche_key="niches_2D", force=True,
        )
        print(f"[{name}] DGE summary:\n{dge_result['summary']}")

        all_results[name] = {
            "gamma": gamma,
            "train_elapsed": train_elapsed,
            "model_name": trainer.get_model_name(),
            "run_mc_task_metrics": metrics,
            "collapse_row": collapse_row,
            "dge_summary": dge_result["summary"].to_dict(orient="records"),
        }
        volume.commit()

    return all_results


@app.local_entrypoint()
def main():
    t0 = time.time()
    result = train_dpar_exact_gamma_sweep.remote()
    print("total wall time incl. cold start (s):", time.time() - t0)
    import json
    print(json.dumps(result, indent=2, default=str))
