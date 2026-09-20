"""MODULARITY AS A REGULARIZER: lambda = 1 on s28nsc.

WHY lambda=1, and not a tuned value. Four data points across two objectives say the
graph term works as a REGULARIZER, not a driver -- every good result has it at or
below the community loss, and every time it dominates, results degrade:

    published pancreas   lambda_nassoc=1 (volume+MSE)          -> the paper result
    scProto best s28nsc  contribution ~1.5 vs umap ~2.5 = 0.6x -> DGE 0.842
    modularity lambda=10 contribution ~10  vs umap ~2   = 5x   -> DGE 0.822
    modularity lambda=100 contribution ~100 vs umap ~2  = 50x  -> DGE 0.769

The modularity term measures |loss_modularity| ~ 1.0-1.35 and umap ~ 1.8-2.5, so
lambda = 1 reproduces the 0.6x ratio of the best run. Nothing invented -- it is read
off the configuration that actually worked.

PREDICTION THIS TESTS: with lambda in {1, 10, 100} spanning ratios 0.6x/5x/50x, DGE
should be HIGHEST at lambda=1 and fall monotonically. If it does not, the regularizer
reading is wrong.

Original header follows.

STEPS 3-5 of my-tasks/todo/modularity-loss-plan.txt -- soft modularity in scProto.

Steps 1 and 2 already PASSED, free, before any GPU spend:
  1. our modularity formula matches igraph to 1e-16 at four resolutions (K=9..573),
     and the two degenerate references land exactly on the algebra
     (one blob -> 1-resolution = -18.43; uniform soft -> (1-res)/K = -0.0922).
  2. the estimators are unbiased and FLAT across prototype size (400 simulated batches):
     affinity_inside 1.001, volume 0.998, volume^2-via-EMA 0.999 -- against
     volume^2-by-squaring at 1.095 overall but 1.923 at 25-50 cells and 3.245 under 25.

CONFIG. lambda_nassoc = 0 -- modularity REPLACES nassoc, it is not a diagonal inside it.
Usage loss OFF (worth +0.015-0.040 DGE under CPM). Resolutions derived two independent
ways that agree: Leiden K-match, cross-checked against the resolution limit.

  s28nsc / bk02   resolution 19.431  (Leiden K=200; limit -> 6.2 cells)
  pancreas / arbf resolution 36.517  (Leiden K=220; limit -> 2.5 cells)

LAMBDA IS THE ONE NUMBER NOT DERIVED. modularity runs from -0.092 (uniform) toward
~+0.39 (Leiden's value at res 19.43), so |loss_modularity| ~ 0.1-0.4 against a UMAP term
of ~3.2. lambda = 10 puts them the same order; 100 brackets it from above. Step 3 is to
read the printed terms on epoch 1 and confirm.

TARGETS
  s28nsc:   Leiden-modularity  DGE 0.856  18 pairs  jointP 0.534  int_frac 0.5147
            SEACells           DGE 0.849  19 pairs  jointP 0.504  int_frac 0.4465
            scProto best (CPM) DGE 0.842  13 pairs  jointP 0.437  int_frac 0.3338
  pancreas: reproduced paper   modularity 0.6924  batch_ent 0.4220  purity 0.9041

STOP RULE from the plan: if s28nsc lands at ~0.84 / 13 pairs -- where CPM already is --
the objective was never the binding constraint and the problem is the search, not the loss.

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/train_modularity.py
Cost: 2 x ~21 min (s28nsc) + ~13 min (pancreas) on a T4, about $0.55.
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

RES_S28 = 19.431
RES_PANC = 36.517
LAMBDAS_S28 = [1.0]
LAMBDA_PANC = 10.0


def _env():
    import os, sys
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["HOME_DIR"] = "/vol/"
    os.environ["DATA_DIR"] = "/vol/data/"
    os.environ["MODEL_DIR"] = "/vol/models/"
    os.environ["CODE_DIR"] = "/vol/code/"
    sys.path.insert(0, "/vol/code")
    os.chdir("/vol/code")
    import h5py
    from anndata._io.specs.registry import _REGISTRY, IOSpec
    try:
        @_REGISTRY.register_read(h5py.Dataset, IOSpec("null", "0.1.0"))
        def _rn(e, *a, **kw):
            return None
    except TypeError:
        pass


@app.function(image=image, gpu="T4", timeout=9000, volumes={"/vol": volume})
def run_s28nsc():
    _env()
    import os, pickle, numpy as np, pandas as pd, scipy.sparse as sp, scanpy as sc
    from scipy.sparse.csgraph import connected_components
    from interpretable_ssl.experiments.tasks import find_metacells, LAMBDA_PROTO_UMAP_PRECON
    from interpretable_ssl.datasets.dataset_configs import DATASETS
    from interpretable_ssl.evaluation import niche_dge as nd
    from interpretable_ssl.evaluation.collapse_diagnostics import (
        labels_from_trainer, joint_purity_per_metacell, collapse_summary)

    adata = sc.read_h5ad(DATASETS["s28nsc"]["path"])
    gt = nd.get_ground_truth(adata, "/vol/code/files/celltype_niches_full_s28nsc.csv",
                             ct_key="celltypes", niche_key="niches_2D")
    TBL = "/vol/code/notebooks/s28nsc_dge_lite/tables"
    os.makedirs(TBL, exist_ok=True)
    with open("/vol/code/graphs/affinity_s28nsc58423_ncomp50_kneighbors50_bk02.pkl",
              "rb") as f:
        A = pickle.load(f).tocsr()
    A = A.tocsr(copy=True); A.setdiag(0); A.eliminate_zeros()
    T = sp.triu(A, k=1).tocoo(); m = float(T.sum())

    out = {}
    for lam in LAMBDAS_S28:
        name = f"modu_L16_r19_l{lam:g}_uOFF"
        lc = dict(LAMBDA_PROTO_UMAP_PRECON) | {
            "lambda_nassoc": 0.0,               # modularity REPLACES nassoc
            "lambda_modularity": lam,
            "modularity_res": RES_S28,
            "lambda_proto_usage": 0.0,
            "latent_dims": 16,
        }
        assert lc["lambda_nassoc"] == 0.0
        print(f"\n{'='*74}\n=== {name}: modularity res={RES_S28} lambda={lam} "
              f"usage OFF latent 16, nassoc OFF\n{'='*74}")
        t0 = time.time()
        try:
            trainer, metrics, *_ = find_metacells(
                "s28nsc", cvae_epochs=50, train_epochs=20, eval_freq=2, patience=4,
                batch_size=1024, affinity_type="bk02", num_prototypes=200,
                label_key="celltypes", niche_key="niches_2D",
                lambda_config=lc, load_pretrain=True, skip_metrics=["task2", "task3"],
                trainer_kwargs={"experiment_name": name})
        except Exception as e:
            import traceback; traceback.print_exc()
            out[name] = {"error": str(e)}; volume.commit(); continue
        print(f"[{name}] done in {time.time()-t0:.0f}s  model={trainer.get_model_name()}")
        volume.commit()

        labels = labels_from_trainer(trainer, adata).dropna()
        lab = pd.Series(labels).reindex(adata.obs_names).to_numpy()
        li = np.full(len(lab), -1, np.int64); ok = ~pd.isna(lab)
        li[ok] = lab[ok].astype(np.int64)
        same = (li[T.row] == li[T.col]) & (li[T.row] >= 0)
        e = float(T.data[same].sum())
        sizes = np.bincount(li[li >= 0]); sizes = sizes[sizes > 0].astype(float)
        pairs = float((sizes * (sizes - 1) / 2).sum())
        # connected components per prototype -- the CPM run had median 13
        ncomp = []
        for c in np.unique(li[li >= 0]):
            idx = np.where(li == c)[0]
            if len(idx) < 2:
                ncomp.append(1); continue
            k, _ = connected_components(A[idx][:, idx], directed=False)
            ncomp.append(k)
        pur = joint_purity_per_metacell(labels, adata, "celltypes", "niches_2D")
        crow = collapse_summary(labels, name=name, purity=pur)
        dge = nd.run_niche_dge(name, labels, adata, gt, TBL, ct_key="celltypes",
                               niche_key="niches_2D", force=True)
        s = dge["summary"].iloc[0]
        r = dict(lam=lam, DGE=float(s["pearson_r"]), n_pairs=int(s["n_pairs"]),
                 jointP=float(crow["purity_micro"]), int_frac=e / m,
                 density=e / pairs, K=int(len(sizes)),
                 med=float(np.median(sizes)), mn=float(sizes.min()),
                 mx=float(sizes.max()), n_under100=int((sizes < 100).sum()),
                 med_components=float(np.median(ncomp)),
                 pct_connected=100.0 * float(np.mean(np.array(ncomp) == 1)))
        print(f"[{name}] >>> {r}")
        out[name] = r
        volume.commit()
    return out


@app.function(image=image, gpu="T4", timeout=9000, volumes={"/vol": volume})
def run_pancreas():
    _env()
    from interpretable_ssl.experiments.tasks import run_mc_task, LAMBDA_PROTO_UMAP_PRECON
    name = f"modu_panc_r36_l{int(LAMBDA_PANC)}_uOFF"
    lc = dict(LAMBDA_PROTO_UMAP_PRECON) | {
        "lambda_nassoc": 0.0,
        "lambda_modularity": LAMBDA_PANC,
        "modularity_res": RES_PANC,
        "lambda_proto_usage": 0.0,
    }
    assert lc["lambda_nassoc"] == 0.0 and "latent_dims" not in lc   # paper default = 8
    print(f"\n{'='*74}\n=== {name}: modularity res={RES_PANC} lambda={LAMBDA_PANC} "
          f"usage OFF latent default(8), nassoc OFF\n{'='*74}")
    t0 = time.time()
    try:
        trainer, metrics, _ = run_mc_task(
            "pancreas", cvae_epochs=50, train_epochs=20, eval_freq=2, patience=4,
            batch_size=1024, affinity_type="arbf", num_prototypes=220,
            lambda_config=lc, trainer_kwargs={"experiment_name": name})
    except Exception as e:
        import traceback; traceback.print_exc()
        volume.commit(); return {"error": str(e)}
    print(f"[{name}] done in {time.time()-t0:.0f}s  model={trainer.get_model_name()}")
    print(f"[{name}] >>> metrics: {metrics}")
    volume.commit()
    return metrics


@app.local_entrypoint()
def main():
    s = run_s28nsc.remote()
    p = None   # pancreas already running in the previous job

    print("\n" + "=" * 112)
    print("STEP 4 -- s28nsc / bk02 / K=200, modularity replaces nassoc, usage OFF")
    print("=" * 112)
    print(f"{'run':<26}{'DGE':>7}{'pairs':>7}{'jointP':>8}{'int_frac':>10}{'density':>10}"
          f"{'K':>5}{'med':>6}{'min':>6}{'n<100':>7}{'comps':>7}{'%conn':>7}")
    if isinstance(s, dict):
        for k, v in s.items():
            if "error" in v:
                print(f"{k:<26}  ERROR: {v['error'][:60]}"); continue
            print(f"{k:<26}{v['DGE']:>7.3f}{v['n_pairs']:>7}{v['jointP']:>8.3f}"
                  f"{v['int_frac']:>10.4f}{v['density']:>10.5f}{v['K']:>5}"
                  f"{v['med']:>6.0f}{v['mn']:>6.0f}{v['n_under100']:>7}"
                  f"{v['med_components']:>7.0f}{v['pct_connected']:>7.1f}")
    print(f"{'--- references ---':<26}")
    print(f"{'Leiden-modularity':<26}{0.856:>7.3f}{18:>7}{0.534:>8.3f}{0.5147:>10.4f}"
          f"{0.03786:>10.5f}{200:>5}{282:>6}{15:>6}{'--':>7}{1:>7}{100.0:>7.1f}")
    print(f"{'SEACells bk02':<26}{0.849:>7.3f}{19:>7}{0.504:>8.3f}{0.4465:>10.4f}"
          f"{0.02952:>10.5f}{200:>5}{258:>6}{33:>6}{'--':>7}{'--':>7}{'--':>7}")
    print(f"{'scProto best (CPM)':<26}{0.842:>7.3f}{13:>7}{0.437:>8.3f}{0.3338:>10.4f}"
          f"{0.02593:>10.5f}{200:>5}{259:>6}{'--':>6}{'--':>7}{'--':>7}{'--':>7}")
    print(f"{'scProto CPM at opt':<26}{0.605:>7.3f}{15:>7}{0.425:>8.3f}{0.5501:>10.4f}"
          f"{0.01246:>10.5f}{195:>5}{42:>6}{1:>6}{135:>7}{13:>7}{6.7:>7.1f}")

    print("\n" + "=" * 112)
    print("STEP 5 -- pancreas / arbf / K=220, modularity replaces nassoc, usage OFF")
    print("=" * 112)
    print(f"{'run':<30}{'purity':>10}{'batch_ent':>12}{'modularity':>12}{'coverage':>10}")
    if isinstance(p, dict) and "error" not in p:
        print(f"{'modu_panc':<30}{p.get('purity', float('nan')):>10.4f}"
              f"{p.get('batch_entropy', float('nan')):>12.4f}"
              f"{p.get('modularity', float('nan')):>12.4f}"
              f"{p.get('coverage', float('nan')):>10.4f}")
    else:
        print(f"{'modu_panc':<30}  ERROR")
    print(f"{'repro_paper_LDdef':<30}{0.9041:>10.4f}{0.4220:>12.4f}"
          f"{0.6924:>12.4f}{1.0000:>10.4f}")
    print(f"{'CPM (gamma 0.042, l=50)':<30}{0.9030:>10.4f}{1.0907:>12.4f}"
          f"{0.4652:>12.4f}{0.9286:>10.4f}")
