"""HIGH-gamma sweep, usage loss OFF -- can pushing density ABOVE SEACells' fix coverage?

THE HYPOTHESIS (user's): coverage (n_pairs: scProto 12-14 vs SEACells 19) is the one gap
that has resisted every fix. SEACells achieves aggregate density 0.0295; scProto's best
gful arm sits at 0.0245. Since the previous sweep PROVED density now converges to ~= gamma
(gamma=0.024 -> density 0.0245, 0.4% off), we can now aim density directly: pick
gamma > 0.0295 and scProto should overshoot SEACells' density. Higher density means
tighter, smaller metacells, which should mean finer resolution and MORE cell states found.

Supporting evidence already in decisions.md:
  section 6  (usage ON, gful): gamma 0.15 -> 15 states, 0.5 -> 15, 0.7 -> 18 states
  section 11 (usage ON, dpar): gamma 0.03 -> 17 states (best yet) but ct purity fell to 0.634
Both say higher gamma finds more states. Neither was run with the usage loss OFF, which
the previous sweep showed independently improves internal_fraction (0.266 -> 0.300/0.379).

GAMMAS: 0.046 / 0.07 / 0.10 / 0.15, spanning SEACells-matched to far above.
  0.046  the SEACells-matched value (derived: mean_wdeg*internal_frac/(median_size-1)
          measured on SEACells' own bk02 assignments)
  0.07   ~2.4x SEACells density
  0.10   ~3.4x
  0.15   ~5x -- also tests whether density still TRACKS gamma this high or saturates.
          Note the size formula says gamma=0.15 implies 48-cell metacells, i.e. K=1223,
          which K=200 cannot supply on average (N/K forces 292). With usage OFF sizes are
          free to be uneven, so this run answers whether the model reaches high density by
          making MANY small metacells plus a few huge ones, or simply fails to track.

WATCH: n_pairs (coverage -- the actual hypothesis), aggregate_density vs gamma (does it
still track?), median/max size, and cell-type purity (section 11 saw it crash to 0.634
when gamma was pushed -- if that repeats, higher density is not free).

Config otherwise identical to the previous sweep: nassoc_exact=1 (+[D7] fix),
size_scale='full', cpm_norm='gful', cpm_free=1, diag_norm='pair', diag_loss='cpm',
lambda_nassoc=1000, lambda_proto_usage=0, s28nsc/bk02/K=200.

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/train_gful_nousage_highgamma.py
Cost: ~$0.8-1.0 on a T4 (prior runs: ~1150-1200s each). Timeout caps GPU spend;
volume.commit() after EVERY gamma so a stop loses at most the in-flight run.
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

SEACELLS_DENSITY = 0.02952          # measured on SEACells' own bk02 assignments
GAMMAS = [0.046, 0.07, 0.10, 0.15]  # SEACells-matched -> far above


@app.function(image=image, gpu="T4", timeout=5400, volumes={"/vol": volume})
def train_gful_nousage_highgamma():
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
            "min_size": float(alive.min()) if len(alive) else float("nan"),
        }

    all_results = {}
    for gamma in GAMMAS:
        tag = "g" + str(gamma).replace("0.", "").replace(".", "")
        name = f"lf_k_hi_{tag}"
        print(f"\n{'='*74}\n=== {name}   gamma={gamma}   usage OFF   "
              f"(SEACells density={SEACELLS_DENSITY})\n{'='*74}")

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
            'lambda_proto_usage': 0.0,
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

        el = time.time() - t0
        print(f"[{name}] trained in {el:.0f}s")
        volume.commit()

        labels = labels_from_trainer(trainer, adata).dropna()
        dens = density_report(labels, adata.obs_names)
        beat = dens["aggregate_density"] > SEACELLS_DENSITY
        print(f"[{name}] [9b] {dens}")
        print(f"[{name}] >>> density={dens['aggregate_density']:.5f} vs SEACells "
              f"{SEACELLS_DENSITY} -> {'ABOVE' if beat else 'below'} | "
              f"alive={dens['n_alive']}/200 | median_size={dens['median_size']:.0f}")

        purity = joint_purity_per_metacell(labels, adata, "celltypes", "niches_2D")
        crow = collapse_summary(labels, name=name, purity=purity)
        crow["n_metacells"] = int(labels.nunique())
        crow["median_size"] = float(metacell_sizes(labels).median())

        dge = nd.run_niche_dge(
            name, labels, adata, ground_truth, TABLE_DIR,
            ct_key="celltypes", niche_key="niches_2D", force=True,
        )
        n_pairs = float(dge["summary"]["n_pairs"].iloc[0])
        print(f"[{name}] >>> COVERAGE n_pairs = {n_pairs:.0f}  (SEACells=19, best so far=17)")
        print(f"[{name}] DGE:\n{dge['summary']}")

        all_results[name] = {
            "gamma": gamma,
            "train_elapsed": el,
            "model_name": trainer.get_model_name(),
            "run_mc_task_metrics": metrics,
            "density_check": dens,
            "beats_seacells_density": bool(beat),
            "collapse_row": crow,
            "dge_summary": dge["summary"].to_dict(orient="records"),
        }
        volume.commit()

    print("\n\n===== SWEEP SUMMARY =====")
    print(f"{'run':<16}{'gamma':>7}{'density':>10}{'vs SEA':>8}{'alive':>7}"
          f"{'med_sz':>8}{'int_frac':>10}{'DGE':>8}{'n_pairs':>9}{'ct_pur':>8}")
    for nm, r in all_results.items():
        if "error" in r:
            print(f"{nm:<16}{r['gamma']:>7}  FAILED: {r['error']}")
            continue
        d = r["density_check"]; s = r["dge_summary"][0]
        print(f"{nm:<16}{r['gamma']:>7}{d['aggregate_density']:>10.5f}"
              f"{('ABOVE' if r['beats_seacells_density'] else 'below'):>8}"
              f"{d['n_alive']:>7}{d['median_size']:>8.0f}{d['internal_fraction']:>10.4f}"
              f"{s['pearson_r']:>8.3f}{s['n_pairs']:>9.0f}"
              f"{r['run_mc_task_metrics'].get('purity', float('nan')):>8.3f}")
    return all_results


@app.local_entrypoint()
def main():
    t0 = time.time()
    result = train_gful_nousage_highgamma.remote()
    print("total wall time incl. cold start (s):", time.time() - t0)
    import json
    print(json.dumps(result, indent=2, default=str))
