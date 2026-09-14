"""Gamma sweep with the USAGE LOSS TURNED OFF -- does gamma alone control how many
prototypes stay alive, the way it controls community count in real Leiden?

Motivation (decisions.md sections 6, 11): in Leiden, gamma is an INPUT and the number of
communities is an emergent OUTPUT. In scProto, K is a fixed architectural input AND
`lambda_proto_usage` actively forces every slot to stay populated, so gamma has nothing
to prune -- section 6 measured that gamma could move metacell SIZE but never COUNT.
Setting `lambda_proto_usage=0` removes that constraint for the first time, letting a
prototype gamma judges as "not worth it" actually die. If gamma then controls the
surviving count, scProto's gamma recovers its real Leiden-like role.

CONFIG: the fully-corrected gful arm -- nassoc_exact=1 (exact soft possible_pairs +
de-duplicated edges_inside, decisions.md section 5) WITH the [D7] cell_scale fix
(section 7), nassoc_size_scale='full', nassoc_cpm_norm='gful', nassoc_cpm_free=1.

GAMMA RANGE -- this is NOT the originally-requested 0.15/0.5/0.7/0.9. A pre-flight
check against the real bk02 graph and the real trained d7fix assignments showed those
are all far outside the reachable range:
```
MEASURED on the real d7fix run (bk02, K=200):
  aggregate density = sum_p edges_inside / sum_p possible_pairs = 0.0238
  per-prototype density: min 0.0072 | median 0.0193 | max 0.0690
  prototypes with density > 0.15:  0 of 200
```
i.e. not one prototype in 200 reaches even half of gamma=0.15. Independently derived
from CLAUDE.md section 9c's own formula with the MEASURED internal_fraction:
  predicted density = mean_wdeg * internal_frac / (size-1)
                    = 26.374 * 0.2661 / (292.1-1) = 0.0241   vs measured 0.0238
Inverting it shows what size each gamma implicitly demands:
  gamma=0.024 -> 293-cell metacells -> K=199   (matches K=200, self-consistent)
  gamma=0.15  ->  48-cell metacells -> K=1223  (unsatisfiable at K=200)
  gamma=0.9   ->   9-cell metacells -> K=6641  (far unsatisfiable)
So the sweep spans BELOW / AT / ABOVE the real equilibrium 0.024 instead.

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/train_gful_nousage_gamma_sweep.py
Cost: budgeted for ~$1 on a T4. Function timeout caps GPU time; volume.commit() runs
after EVERY gamma so a timeout loses at most the in-flight run, not the whole sweep.
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

# below / at / above the MEASURED equilibrium density (0.0238)
GAMMAS = [0.01, 0.024, 0.05, 0.10]


@app.function(image=image, gpu="T4", timeout=5400, volumes={"/vol": volume})
def train_gful_nousage_gamma_sweep():
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

    # real affinity, for the [9b] post-hoc density check on each run's OWN assignments
    with open("/vol/code/graphs/affinity_s28nsc58423_ncomp50_kneighbors50_bk02.pkl", "rb") as f:
        _A = pickle.load(f).tocsr()
    _A = _A.tocsr(copy=True); _A.setdiag(0); _A.eliminate_zeros()
    _Atri = sp.triu(_A, k=1).tocoo()
    _m = float(_Atri.sum())

    def density_report(labels, adata_obs_names):
        """[9b] measure the ACHIEVED density of this run, same way the pre-flight check did."""
        lab = pd.Series(labels).reindex(adata_obs_names).to_numpy()
        ok = ~pd.isna(lab)
        lab_i = np.full(len(lab), -1, dtype=np.int64)
        lab_i[ok] = lab[ok].astype(np.int64)
        same = (lab_i[_Atri.row] == lab_i[_Atri.col]) & (lab_i[_Atri.row] >= 0)
        edges_in = float(_Atri.data[same].sum())
        sizes = np.bincount(lab_i[lab_i >= 0], minlength=max(lab_i.max() + 1, 1)).astype(float)
        alive = sizes[sizes > 0]
        pairs = float((alive * (alive - 1) / 2).sum())
        return {
            "n_alive": int(len(alive)),
            "sum_edges_inside": edges_in,
            "sum_possible_pairs": pairs,
            "internal_fraction": edges_in / _m if _m else float("nan"),
            "aggregate_density": edges_in / pairs if pairs else float("nan"),
            "median_size": float(np.median(alive)) if len(alive) else float("nan"),
            "max_size": float(alive.max()) if len(alive) else float("nan"),
        }

    all_results = {}
    for gamma in GAMMAS:
        tag = f"g{str(gamma).replace('0.', '').replace('.', '')}"
        name = f"lf_j_nousage_{tag}"
        print(f"\n{'='*72}\n=== {name}   gamma={gamma}   lambda_proto_usage=0 (OFF)\n{'='*72}")

        t0 = time.time()
        lc = dict(LAMBDA_PROTO_UMAP_PRECON) | {
            'nassoc_diag_norm': 'pair',
            'nassoc_diag_loss': 'cpm',
            'nassoc_cpm_free': 1,
            'nassoc_size_scale': 'full',
            'nassoc_cpm_norm': 'gful',
            'nassoc_gamma': gamma,
            'lambda_nassoc': 1000,
            'nassoc_exact': 1,
            'lambda_proto_usage': 0.0,   # <-- THE experiment: usage pressure OFF
        }

        try:
            trainer, metrics, _ = run_mc_task(
                "s28nsc",
                cvae_epochs=50, train_epochs=20, eval_freq=2, patience=4,
                batch_size=1024, affinity_type="bk02", num_prototypes=200,
                label_key="celltypes", niche_key="niches_2D",
                lambda_config=lc,
                trainer_kwargs={"experiment_name": name},
            )
        except Exception as e:
            print(f"[{name}] FAILED: {type(e).__name__}: {e}")
            all_results[name] = {"gamma": gamma, "error": f"{type(e).__name__}: {e}"}
            volume.commit()
            continue

        train_elapsed = time.time() - t0
        print(f"[{name}] trained in {train_elapsed:.0f}s  model={trainer.get_model_name()}")
        volume.commit()

        labels = labels_from_trainer(trainer, adata).dropna()
        dens = density_report(labels, adata.obs_names)
        print(f"[{name}] [9b] DENSITY CHECK: {dens}")
        print(f"[{name}] >>> PROTOTYPES ALIVE: {dens['n_alive']} / 200   "
              f"(gamma={gamma}, achieved density={dens['aggregate_density']:.5f})")

        purity = joint_purity_per_metacell(labels, adata, "celltypes", "niches_2D")
        collapse_row = collapse_summary(labels, name=name, purity=purity)
        collapse_row["n_metacells"] = int(labels.nunique())
        collapse_row["median_size"] = float(metacell_sizes(labels).median())

        dge = nd.run_niche_dge(
            name, labels, adata, ground_truth, TABLE_DIR,
            ct_key="celltypes", niche_key="niches_2D", force=True,
        )
        print(f"[{name}] DGE:\n{dge['summary']}")

        all_results[name] = {
            "gamma": gamma,
            "train_elapsed": train_elapsed,
            "model_name": trainer.get_model_name(),
            "run_mc_task_metrics": metrics,
            "density_check": dens,
            "collapse_row": collapse_row,
            "dge_summary": dge["summary"].to_dict(orient="records"),
        }
        volume.commit()

    return all_results


@app.local_entrypoint()
def main():
    t0 = time.time()
    result = train_gful_nousage_gamma_sweep.remote()
    print("total wall time incl. cold start (s):", time.time() - t0)
    import json
    print(json.dumps(result, indent=2, default=str))
