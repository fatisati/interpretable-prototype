"""latent_dims=16 (vs the standard 8), gful + nassoc_exact=1, gamma x usage 2x2.

HYPOTHESIS: 8 latent dims for 18 cell types leaves little room for WITHIN-cell-type niche
substructure. decisions.md section 9c measured that z (8-d) retains 91% of the niche signal
while the prototype assignment retains only 54% -- so the encoder is fine but the 200-way
assignment in 8-d space is where the signal dies. More latent room may let the prototypes
separate niche substructure that currently collapses.

CAVEAT, stated up front: section 9c ALSO estimated that raising 8 -> 16 could gain at most
+0.014 lift (~9%) on niche separability, and concluded "not the fix". This run tests that
estimate directly rather than trusting it -- it is a prediction, not a measurement.

2x2: gamma in {0.15, 0.2} x lambda_proto_usage in {0.1 (ON), 0.0 (OFF)}.
gamma range chosen to sit near SEACells' own good operating range.

NOTE ON PRETRAIN: latent_dims=16 does NOT match the cached 8-d pretrain checkpoint, so the
FIRST run pays a fresh cvae_epochs=50 pretrain; runs 2-4 reuse it via load_pretrain=True.

Run (detached, so a local connection drop cannot reap the app -- learned the hard way):
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach modal_scripts/train_latent16_gamma_usage.py
Budget: ~$1 on a T4. volume.commit() after EVERY run so a stop loses at most one.
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

# (gamma, usage_lambda, tag)
ARMS = [
    (0.15, 0.1, "g015_uON"),
    (0.15, 0.0, "g015_uOFF"),
    (0.20, 0.1, "g02_uON"),
    (0.20, 0.0, "g02_uOFF"),
]


@app.function(image=image, gpu="T4", timeout=7200, volumes={"/vol": volume})
def train_latent16():
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

    import numpy as np, pandas as pd, scipy.sparse as sp, pickle
    import scanpy as sc
    from interpretable_ssl.experiments.tasks import find_metacells, LAMBDA_PROTO_UMAP_PRECON
    from interpretable_ssl.datasets.dataset_configs import DATASETS
    from interpretable_ssl.evaluation import niche_dge as nd
    from interpretable_ssl.evaluation.collapse_diagnostics import (
        labels_from_trainer, joint_purity_per_metacell, collapse_summary, metacell_sizes,
    )

    ds_conf = DATASETS["s28nsc"]
    adata = sc.read_h5ad(ds_conf["path"])
    GT = os.path.join("/vol/code", "files", "celltype_niches_full_s28nsc.csv")
    ground_truth = nd.get_ground_truth(adata, GT, ct_key="celltypes", niche_key="niches_2D")
    TABLE_DIR = "/vol/code/notebooks/s28nsc_dge_lite/tables"
    os.makedirs(TABLE_DIR, exist_ok=True)

    with open("/vol/code/graphs/affinity_s28nsc58423_ncomp50_kneighbors50_bk02.pkl", "rb") as f:
        _A = pickle.load(f).tocsr()
    _A = _A.tocsr(copy=True); _A.setdiag(0); _A.eliminate_zeros()
    _Atri = sp.triu(_A, k=1).tocoo(); _m = float(_Atri.sum())

    def dens_report(labels, obs_names):
        lab = pd.Series(labels).reindex(obs_names).to_numpy()
        ok = ~pd.isna(lab)
        li = np.full(len(lab), -1, dtype=np.int64); li[ok] = lab[ok].astype(np.int64)
        same = (li[_Atri.row] == li[_Atri.col]) & (li[_Atri.row] >= 0)
        e = float(_Atri.data[same].sum())
        sizes = np.bincount(li[li >= 0], minlength=max(li.max() + 1, 1)).astype(float)
        alive = sizes[sizes > 0]; pairs = float((alive * (alive - 1) / 2).sum())
        return {"n_alive": int(len(alive)), "n_unused": int(200 - len(alive)),
                "internal_fraction": e / _m, "aggregate_density": e / pairs if pairs else float("nan"),
                "median_size": float(np.median(alive)), "max_size": float(alive.max()),
                "size_ratio_max_over_median": float(alive.max() / np.median(alive))}

    out = {}
    for gamma, usage, tag in ARMS:
        name = f"lf_L16_{tag}"
        print(f"\n{'='*74}\n=== {name}  latent_dims=16  gamma={gamma}  usage={usage}\n{'='*74}")
        t0 = time.time()
        lc = dict(LAMBDA_PROTO_UMAP_PRECON) | {
            'nassoc_diag_norm': 'pair', 'nassoc_diag_loss': 'cpm', 'nassoc_cpm_free': 1,
            'nassoc_size_scale': 'full', 'nassoc_cpm_norm': 'gful',
            'nassoc_gamma': gamma, 'lambda_nassoc': 1000, 'nassoc_exact': 1,
            'lambda_proto_usage': usage,
            'latent_dims': 16,
        }
        try:
            trainer, metrics, *_ = find_metacells(
                "s28nsc", cvae_epochs=50, train_epochs=20, eval_freq=2, patience=4,
                batch_size=1024, affinity_type="bk02", num_prototypes=200,
                label_key="celltypes", niche_key="niches_2D",
                lambda_config=lc, load_pretrain=True,
                skip_metrics=["task2", "task3"],
                trainer_kwargs={"experiment_name": name},
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[{name}] FAILED: {type(e).__name__}: {e}")
            out[name] = {"gamma": gamma, "usage": usage, "error": str(e)}
            volume.commit(); continue

        el = time.time() - t0
        print(f"[{name}] done in {el:.0f}s  model={trainer.get_model_name()}")
        volume.commit()

        labels = labels_from_trainer(trainer, adata).dropna()
        d = dens_report(labels, adata.obs_names)
        pur = joint_purity_per_metacell(labels, adata, "celltypes", "niches_2D")
        crow = collapse_summary(labels, name=name, purity=pur)
        dge = nd.run_niche_dge(name, labels, adata, ground_truth, TABLE_DIR,
                               ct_key="celltypes", niche_key="niches_2D", force=True)
        s = dge["summary"].iloc[0]
        print(f"[{name}] >>> DGE={s['pearson_r']:.3f} n_pairs={s['n_pairs']:.0f} "
              f"used={d['n_alive']}/200 density={d['aggregate_density']:.5f} "
              f"int_frac={d['internal_fraction']:.4f} med={d['median_size']:.0f} "
              f"max/med={d['size_ratio_max_over_median']:.1f} joint_pur={crow['purity_micro']:.3f}")
        out[name] = {"gamma": gamma, "usage": usage, "elapsed": el,
                     "metrics": metrics, "density": d, "collapse": crow,
                     "dge": dge["summary"].to_dict(orient="records")}
        volume.commit()

    print("\n===== latent_dims=16 SUMMARY =====")
    print(f"{'run':<18}{'gam':>6}{'usage':>7}{'DGE':>8}{'n_pairs':>9}{'used':>7}"
          f"{'density':>10}{'int_frac':>10}{'med':>6}{'max/med':>9}{'jointP':>8}")
    for nm, r in out.items():
        if "error" in r:
            print(f"{nm:<18}{r['gamma']:>6}{r['usage']:>7}   FAILED"); continue
        d = r["density"]; s = r["dge"][0]
        print(f"{nm:<18}{r['gamma']:>6}{r['usage']:>7}{s['pearson_r']:>8.3f}{s['n_pairs']:>9.0f}"
              f"{str(d['n_alive'])+'/200':>7}{d['aggregate_density']:>10.5f}"
              f"{d['internal_fraction']:>10.4f}{d['median_size']:>6.0f}"
              f"{d['size_ratio_max_over_median']:>9.1f}{r['collapse']['purity_micro']:>8.3f}")
    print(f"{'SEACells (ref)':<18}{'--':>6}{'--':>7}{0.849:>8.3f}{19:>9.0f}{'200/200':>7}"
          f"{0.02952:>10.5f}{0.4465:>10.4f}{258:>6.0f}{3.6:>9.1f}{0.504:>8.3f}")
    print(f"{'8-d best (ref)':<18}{0.024:>6}{0.0:>7}{0.821:>8.3f}{13:>9.0f}{'200/200':>7}"
          f"{0.02449:>10.5f}{0.3000:>10.4f}{268:>6.0f}{2.4:>9.1f}{float('nan'):>8.3f}")
    return out


@app.local_entrypoint()
def main():
    t0 = time.time()
    r = train_latent16.remote()
    print("wall (s):", time.time() - t0)
    import json; print(json.dumps(r, indent=2, default=str))
