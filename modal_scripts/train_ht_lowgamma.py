"""[D9] CORRECTED CPM at LOW GAMMA -- the combination that has never been run.

TWO INDEPENDENT REASONS TO RUN LOW GAMMA, BOTH FOUND TODAY
----------------------------------------------------------
1. THE EPSILON FINDING. gamma is compared against a SOFT density, but every gamma we
   have ever set was derived from HARD partition statistics. Measured in the ht_L16 run
   (gamma=0.028):

       logged soft density (nassoc diag)  plateaus at  0.011
       final HARD aggregate density                     0.02695
       ratio                                            2.45      (effk settles at 2.7)

   So gamma=0.028 sat 2.5x above anything the loss could ever reach -- every prototype
   scored negative for the whole run, the saturated regime EXPERIMENTS.md 3c blames for
   gamma=0.15. Soft-corrected: gamma ~ 0.028/2.45 ~ 0.011.

   Root cause: epsilon calibration FAILED and fell back to an arbitrary constant --
       [eps calibration] E[p_pos]=0.3357 unreachable (max E[q_pos]=0.2051),
                         falling back to effk=5.0
   `umap_proto_effk=5.0` is a hard-coded default; nothing chose it. And effk then drifts
   from 5.0 to 2.7 during training, because calibration runs once, before training.
   Reproduced independently on pancreas (max E[q_pos]=0.2992 vs p_pos=0.3397).

2. THE ESTIMATOR FIX APPLIES HARDEST HERE. [D9] removed an over-charge on SMALL
   prototypes (2.39x under 25 cells, 1.45x at 25-50, ~1.0x at 200-900). Low gamma is
   precisely the regime that PRODUCES small prototypes -- so every previous low-gamma run
   was fighting a penalty that no longer exists. gamma=0.01 (old estimator) gave the
   HIGHEST internal_fraction of any scProto run, 0.3788, with median size 172 and
   max/median 6.7 -- but DGE only 0.746.

WHAT WOULD COUNT AS A RESULT
----------------------------
The trade so far has always been: more unevenness -> more containment -> worse DGE.
    gamma 0.01  max/med 6.7   int_frac 0.3788   DGE 0.746
    gamma 0.020 max/med 2.7   int_frac 0.3338   DGE 0.842
    gamma 0.028 max/med 1.7   int_frac 0.3137   DGE 0.833   (today, corrected)
A RESULT is int_frac above 0.38 with DGE holding near 0.83 and coverage above 14 --
i.e. the first time the trade stops being a trade. Anything else is another point on
the same frontier.

CAVEAT: the information-gap hypothesis (section 14) is NOT established -- EXPERIMENTS.md
section 8 explicitly withdraws the strong reading, because the probe froze z and fit a
linear head to SEACells' label identities, making every number a LOWER bound. So a
loss-side gain here is not ruled out by it.

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/train_ht_lowgamma.py
Cost: 2 x ~21 min on a T4. Commits after EVERY arm.
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

# gamma is now given on the MEASURED HARD/graph scale and converted by [D10].
# 0.028 = the value measure_gamma_from_graph.py derived; 0.020 = the low end
# of that window. Expected to convert to ~0.011 and ~0.008 in soft units.
GAMMAS = [0.028, 0.020]
LAMBDA_NASSOC = 200    # 'btsz' divides by ~1024 where 'gful' divided by ~1.5e6;
                       # estimated so lambda*diag lands near the umap term (~2.5).
                       # VERIFY on epoch 1 -- this is the one number not measured.
LATENT = 16


@app.function(image=image, gpu="T4", timeout=7200, volumes={"/vol": volume})
def train_ht_lowgamma():
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
        labels_from_trainer, joint_purity_per_metacell, collapse_summary,
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
        alive = sizes[sizes > 0]
        pairs = float((alive * (alive - 1) / 2).sum())
        return {"n_alive": int(len(alive)), "n_unused": int(200 - len(alive)),
                "internal_fraction": e / _m,
                "aggregate_density": e / pairs if pairs else float("nan"),
                "median_size": float(np.median(alive)), "max_size": float(alive.max()),
                "min_size": float(alive.min()),
                "n_under_100": int((alive < 100).sum()),
                "size_ratio_max_over_median": float(alive.max() / np.median(alive))}

    out = {}
    for GAMMA in GAMMAS:
        dim = LATENT
        name = f"d10_L16_gh{str(GAMMA).replace('0.','')}_uOFF"
        print(f"\n{'='*74}\n=== {name}   latent_dims={dim}  gamma={GAMMA}  "
              f"lambda_nassoc={LAMBDA_NASSOC}  usage=OFF  [D9 corrected CPM]\n{'='*74}")
        t0 = time.time()
        lc = dict(LAMBDA_PROTO_UMAP_PRECON) | {
            'nassoc_diag_norm': 'pair', 'nassoc_diag_loss': 'cpm', 'nassoc_cpm_free': 1,
            'nassoc_exact': 1,
            'nassoc_ht': 1,                  # [D9] debiased, batch-only CPM
            'nassoc_cpm_norm': 'btsz',       # [D9] divide by the batch's edge count
            'nassoc_gamma': GAMMA,
            'nassoc_gamma_scale': 'soft',   # [D10] convert hard -> soft scale
            'lambda_nassoc': LAMBDA_NASSOC,
            'lambda_proto_usage': 0.0,       # usage OFF
            'latent_dims': dim,
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
            out[name] = {"dim": dim, "gamma": GAMMA, "error": str(e)}
            volume.commit()
            continue

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
              f"alive={d['n_alive']}/200 density={d['aggregate_density']:.5f} "
              f"int_frac={d['internal_fraction']:.4f} med={d['median_size']:.0f} "
              f"min={d['min_size']:.0f} max/med={d['size_ratio_max_over_median']:.1f} "
              f"n<100={d['n_under_100']} joint_pur={crow['purity_micro']:.3f}")
        out[name] = {"dim": dim, "gamma": GAMMA, "elapsed": el, "density": d, "collapse": crow,
                     "dge": dge["summary"].to_dict(orient="records")}
        volume.commit()

    print(f"\n{'='*104}\nSUMMARY -- [D9] corrected CPM, usage OFF, gamma={GAMMA}\n{'='*104}")
    hdr = (f"{'run':<22}{'dim':>4}{'DGE':>7}{'pairs':>7}{'jointP':>8}{'int_frac':>10}"
           f"{'density':>10}{'med':>6}{'min':>6}{'max/med':>9}{'n<100':>7}")
    print(hdr)
    for k, v in out.items():
        if "error" in v:
            print(f"{k:<22}{v['dim']:>4}  ERROR: {v['error'][:60]}")
            continue
        d, c = v["density"], v["collapse"]
        s = v["dge"][0]
        print(f"{k:<22}{v['dim']:>4}{s['pearson_r']:>7.3f}{s['n_pairs']:>7.0f}"
              f"{c['purity_micro']:>8.3f}{d['internal_fraction']:>10.4f}"
              f"{d['aggregate_density']:>10.5f}{d['median_size']:>6.0f}"
              f"{d['min_size']:>6.0f}{d['size_ratio_max_over_median']:>9.1f}"
              f"{d['n_under_100']:>7}")
    print(f"{'--- references ---':<22}")
    print(f"{'SEACells bk02':<22}{'--':>4}{0.849:>7.3f}{19:>7}{0.504:>8.3f}"
          f"{0.4465:>10.4f}{0.02952:>10.5f}{258:>6}{'--':>6}{3.6:>9.1f}{'--':>7}")
    print(f"{'lf_L16_cg02_uOFF':<22}{16:>4}{0.842:>7.3f}{13:>7}{0.437:>8.3f}"
          f"{0.3338:>10.4f}{0.02593:>10.5f}{259:>6}{'--':>6}{2.7:>9.1f}{'--':>7}")
    print(f"{'d7fix (8d)':<22}{8:>4}{0.758:>7.3f}{14:>7}{0.423:>8.3f}"
          f"{0.2661:>10.4f}{0.02381:>10.5f}{294:>6}{'--':>6}{1.2:>9.1f}{'--':>7}")
    print(f"{'GROUND-TRUTH ceiling':<22}{'--':>4}{'--':>7}{'--':>7}{'--':>8}{0.4549:>10.4f}")
    volume.commit()
    return out


@app.local_entrypoint()
def main():
    train_ht_lowgamma.remote()
