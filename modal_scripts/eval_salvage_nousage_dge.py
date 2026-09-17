"""Recover the DGE + density numbers for two runs that TRAINED FULLY but whose eval step
was lost when the local connection dropped and Modal reaped the ephemeral app:

    lf_j_nousage_g05    gamma=0.05,  usage OFF, gful   (from the first no-usage sweep)
    lf_k_hi_g046        gamma=0.046, usage OFF, gful   (from the high-gamma sweep)

Both have `umap_checkpoint.pth` + `metrics.json` on the volume, so NO RETRAINING is
needed -- this uses `find_metacells(load_umap=True, skip_eval=True)` (CLAUDE.md section 4's
documented "load model only, do not re-run eval or overwrite saved outputs" path) to
reload each checkpoint, encode the cells once, and compute the missing numbers.

This matters because these are the only two HIGHER-gamma + usage-OFF points we have, and
the open question is exactly whether higher gamma buys coverage (n_pairs). Their partial
metrics.json already show modularity falling monotonically with gamma
(0.01 -> 0.370, 0.024 -> 0.295, 0.046 -> 0.212, 0.05 -> 0.241), which predicts they sit
LOWER on the DGE/coverage frontier -- this run tests that prediction directly.

Run with --detach so a local connection drop cannot reap the app (the lesson from the
two sweeps above):
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach modal_scripts/eval_salvage_nousage_dge.py
Cost: no training, just a checkpoint load + one forward pass per run. Minutes, cents.
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

# (experiment_name, gamma) -- must match the ORIGINAL runs exactly so get_model_name()
# resolves to the same folder on the volume.
RUNS = [("lf_j_nousage_g05", 0.05), ("lf_k_hi_g046", 0.046)]


@app.function(image=image, gpu="T4", timeout=1800, volumes={"/vol": volume})
def eval_salvage():
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

    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    import pickle
    import scanpy as sc
    from interpretable_ssl.experiments.tasks import find_metacells, LAMBDA_PROTO_UMAP_PRECON
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

    with open("/vol/code/graphs/affinity_s28nsc58423_ncomp50_kneighbors50_bk02.pkl", "rb") as f:
        _A = pickle.load(f).tocsr()
    _A = _A.tocsr(copy=True); _A.setdiag(0); _A.eliminate_zeros()
    _Atri = sp.triu(_A, k=1).tocoo()
    _m = float(_Atri.sum())

    def density_report(labels, obs_names):
        lab = pd.Series(labels).reindex(obs_names).to_numpy()
        ok = ~pd.isna(lab)
        li = np.full(len(lab), -1, dtype=np.int64)
        li[ok] = lab[ok].astype(np.int64)
        same = (li[_Atri.row] == li[_Atri.col]) & (li[_Atri.row] >= 0)
        e = float(_Atri.data[same].sum())
        sizes = np.bincount(li[li >= 0], minlength=max(li.max() + 1, 1)).astype(float)
        alive = sizes[sizes > 0]
        pairs = float((alive * (alive - 1) / 2).sum())
        return {
            "n_alive": int(len(alive)),
            "internal_fraction": e / _m if _m else float("nan"),
            "aggregate_density": e / pairs if pairs else float("nan"),
            "median_size": float(np.median(alive)) if len(alive) else float("nan"),
            "max_size": float(alive.max()) if len(alive) else float("nan"),
        }

    out = {}
    for name, gamma in RUNS:
        print(f"\n{'='*72}\n=== SALVAGE {name}  gamma={gamma}  (load checkpoint, no training)\n{'='*72}")
        lc = dict(LAMBDA_PROTO_UMAP_PRECON) | {
            'nassoc_diag_norm': 'pair',
            'nassoc_diag_loss': 'cpm',
            'nassoc_cpm_free': 1,
            'nassoc_size_scale': 'full',
            'nassoc_cpm_norm': 'gful',
            'nassoc_gamma': gamma,
            'lambda_nassoc': 1000,
            'nassoc_exact': 1,
            'lambda_proto_usage': 0.0,
        }
        try:
            # NOTE: find_metacells returns THREE values (trainer, metrics, _); its
            # docstring claims two, which is stale -- use *_ so it cannot bite again.
            trainer, *_ = find_metacells(
                "s28nsc",
                cvae_epochs=50, train_epochs=20, eval_freq=2, patience=4,
                batch_size=1024, affinity_type="bk02", num_prototypes=200,
                label_key="celltypes", niche_key="niches_2D",
                lambda_config=lc,
                load_pretrain=True,
                load_umap=True,      # <-- load the trained checkpoint
                skip_eval=True,      # <-- do NOT re-run eval or overwrite saved outputs
                trainer_kwargs={"experiment_name": name},
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[{name}] LOAD FAILED: {type(e).__name__}: {e}")
            out[name] = {"gamma": gamma, "error": f"{type(e).__name__}: {e}"}
            continue

        print(f"[{name}] loaded model: {trainer.get_model_name()}")
        labels = labels_from_trainer(trainer, adata).dropna()
        dens = density_report(labels, adata.obs_names)
        # explicit prototype-usage check: with lambda_proto_usage=0 nothing forces all
        # 200 slots to stay populated, so an empty slot here is a REAL result, not a bug.
        n_unused = 200 - dens["n_alive"]
        dens["n_unused"] = int(n_unused)
        print(f"[{name}] >>> PROTOTYPES USED: {dens['n_alive']}/200   UNUSED: {n_unused}")
        print(f"[{name}] density check: {dens}")

        purity = joint_purity_per_metacell(labels, adata, "celltypes", "niches_2D")
        crow = collapse_summary(labels, name=name, purity=purity)
        crow["n_metacells"] = int(labels.nunique())
        crow["median_size"] = float(metacell_sizes(labels).median())

        dge = nd.run_niche_dge(
            name, labels, adata, ground_truth, TABLE_DIR,
            ct_key="celltypes", niche_key="niches_2D", force=True,
        )
        s = dge["summary"].iloc[0]
        print(f"[{name}] >>> DGE={s['pearson_r']:.3f}  n_pairs={s['n_pairs']:.0f}  "
              f"density={dens['aggregate_density']:.5f}  median_size={dens['median_size']:.0f}  "
              f"joint_purity={crow['purity_micro']:.3f}")

        out[name] = {
            "gamma": gamma,
            "density_check": dens,
            "collapse_row": crow,
            "dge_summary": dge["summary"].to_dict(orient="records"),
        }
        volume.commit()

    print("\n===== SALVAGE SUMMARY =====")
    print(f"{'run':<20}{'gamma':>7}{'DGE':>8}{'n_pairs':>9}{'density':>10}{'med_sz':>8}"
          f"{'used/200':>10}{'joint_pur':>11}{'int_frac':>10}")
    for nm, r in out.items():
        if "error" in r:
            print(f"{nm:<20}{r['gamma']:>7}   FAILED")
            continue
        d = r["density_check"]; s = r["dge_summary"][0]
        print(f"{nm:<20}{r['gamma']:>7}{s['pearson_r']:>8.3f}{s['n_pairs']:>9.0f}"
              f"{d['aggregate_density']:>10.5f}{d['median_size']:>8.0f}"
              f"{str(d['n_alive'])+'/200':>10}"
              f"{r['collapse_row']['purity_micro']:>11.3f}{d['internal_fraction']:>10.4f}")
    print("\nReference (same run_niche_dge pipeline, same ground truth, majority-vote labels):")
    print(f"{'SEACells_lam0.2':<20}{'--':>7}{0.849:>8.3f}{19:>9.0f}{0.02952:>10.5f}{258:>8.0f}"
          f"{'200/200':>10}{0.504:>11.3f}{0.4465:>10.4f}")
    print(f"{'lf_j_nousage_g024':<20}{0.024:>7}{0.821:>8.3f}{13:>9.0f}{0.02449:>10.5f}{268:>8.0f}"
          f"{'200/200':>10}{float('nan'):>11.3f}{0.3000:>10.4f}")
    return out


@app.local_entrypoint()
def main():
    t0 = time.time()
    r = eval_salvage.remote()
    print("wall time (s):", time.time() - t0)
    import json
    print(json.dumps(r, indent=2, default=str))
