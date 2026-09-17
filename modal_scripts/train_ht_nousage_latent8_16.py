"""[D9] CORRECTED CPM (nassoc_ht=1) -- latent_dims 8 and 16, usage loss OFF.

WHAT CHANGED IN THE LOSS
------------------------
The CPM diagonal is now computed LITERALLY ON THE BATCH, with both sampling biases
removed (defaults.py [D9]/[D9b], scproto.py, measured in measure_estimator_bias.py):

  reward[p] = sum over the B sampled edges of  S[head,p]*S[tail,p]      NO w factor
              (edges are already drawn with probability proportional to w; multiplying
               by w again made the objective effectively w^2, and the UMAP term does
               not do that -- the two losses were weighting the same batch differently)

  q_i       = p_i / mean(p_i)          survey weight, p_i = 1-(1-2*d_i/W)^B
  u[i,p]    = S[i,p] / q_i
  pairs[p]  = 0.5*((sum u)^2 - sum u^2)  * (2B/W) * N(N-1)/(n_b(n_b-1))
              (the survey weight removes the degree bias -- cells in sparse regions
               appear in only 2% of batches vs 21% for dense ones; the trailing factor
               keeps gamma on the GRAPH's density scale, independent of batch size)

  score[p]  = reward[p] - gamma*pairs[p]
  loss      = -(sum_p score[p]) / B          nassoc_cpm_norm='btsz'

GAMMA IS MEASURED, NOT SWEPT
----------------------------
measure_gamma_from_graph.py: gamma is the minimum AVERAGE affinity a cell needs to a
group to be worth joining it. On s28nsc/bk02 a perfect 292-cell same-state group gives
0.041, SEACells' own metacells achieve 0.0449 median / 0.0295 aggregate => 0.026-0.030.
Using 0.028.

WHAT TO EXPECT -- SET EXPECTATIONS BEFORE READING THE RESULT
------------------------------------------------------------
In the 200-900 cell band where every current prototype sits, the estimator fix is worth
only ~8% (effective gamma 0.92x), well inside the 0.020-0.032 window already swept.
So this is NOT expected to move DGE or coverage. It matters below ~100 cells. The
interesting question is whether the removed size-equalizing pressure lets sizes spread
(max/median above the rigid 1.2-1.3) and whether that buys any coverage.

References: SEACells bk02 = DGE 0.849, n_pairs 19, joint purity 0.504, int_frac 0.4465,
median 258, max/med 3.6. Best scProto so far = lf_L16_cg02_uOFF, DGE 0.842, 13 pairs.
Ceiling: the ground-truth celltype x niche partition itself only reaches int_frac 0.4549.

SCALE CHECK (CLAUDE.md 9b) -- read the first epoch before trusting anything:
  reward/B ~ 0.30,  gamma*pairs/B ~ 0.31,  they nearly cancel at a correct gamma.
  lambda_nassoc*diag should land in the same range as umap (~2.5). If nassoc is 10x
  umap, lambda is wrong -- stop and retune rather than reading the metrics.

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach modal_scripts/train_ht_nousage_latent8_16.py
Cost: 2 x ~13 min on a T4. Commits after EVERY arm.
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

GAMMA = 0.028          # measured, see measure_gamma_from_graph.py
LAMBDA_NASSOC = 200    # 'btsz' divides by ~1024 where 'gful' divided by ~1.5e6;
                       # estimated so lambda*diag lands near the umap term (~2.5).
                       # VERIFY on epoch 1 -- this is the one number not measured.
ARMS = [(16, "L16"), (8, "L8")]      # 16 first: the better arm, so a timeout keeps it


@app.function(image=image, gpu="T4", timeout=7200, volumes={"/vol": volume})
def train_ht():
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
    for dim, tag in ARMS:
        name = f"ht_{tag}_g028_uOFF"
        print(f"\n{'='*74}\n=== {name}   latent_dims={dim}  gamma={GAMMA}  "
              f"lambda_nassoc={LAMBDA_NASSOC}  usage=OFF  [D9 corrected CPM]\n{'='*74}")
        t0 = time.time()
        lc = dict(LAMBDA_PROTO_UMAP_PRECON) | {
            'nassoc_diag_norm': 'pair', 'nassoc_diag_loss': 'cpm', 'nassoc_cpm_free': 1,
            'nassoc_exact': 1,
            'nassoc_ht': 1,                  # [D9] debiased, batch-only CPM
            'nassoc_cpm_norm': 'btsz',       # [D9] divide by the batch's edge count
            'nassoc_gamma': GAMMA,
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
            out[name] = {"dim": dim, "error": str(e)}
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
        out[name] = {"dim": dim, "elapsed": el, "density": d, "collapse": crow,
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
    train_ht.remote()
