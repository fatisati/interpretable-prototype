"""CPM at the K-MATCHED gamma, on both datasets. Can scProto reach SEACells?

WHY THIS GAMMA
--------------
leiden_cpm_control.py ran OUR OWN CPM objective with a real optimizer (leidenalg) on
the same graphs, and searched the resolution until it produced the same K we use:

    s28nsc / bk02, K=201    gamma = 0.00686   int_frac 0.5501  jointP 0.529  states 52
    pancreas / arbf, K=221  gamma = 0.04209   int_frac 0.6464  ctP 0.978

Both BEAT SEACells (0.4465 and 0.5684). So the CPM objective is not the problem -- the
graph contains far more structure than anyone has been extracting, and the gap is
downstream of the objective.

It also corrects the gamma derivation twice over. Deriving gamma from SEACells' achieved
density gave 0.028 (s28nsc) and 0.11 (pancreas); at those values hard CPM returns K=1043
and K=687, not 200/220. The K-matched values are ~4x and ~2.6x lower.

THE MOST CONCEPTUALLY CORRECT CPM WE HAVE
-----------------------------------------
    nassoc_diag_loss='cpm'      score = edges_inside - gamma*possible_pairs (Traag 2011)
    nassoc_cpm_free=1           the published SUBTRACTION form, never a ratio
    nassoc_diag_norm='pair'     density, not containment -- fixes the merge bias [D1]
    nassoc_exact=1              exact soft pair count (size^2 - sumsq)/2 [D6]
    nassoc_ht=1                 both sampling biases removed [D9]: no w^2 double-count,
                                Horvitz-Thompson survey weight on the cells
    nassoc_cpm_norm='btsz'      computed literally on the batch, divided by its edge count
    nassoc_gamma_scale='soft'   gamma given on the HARD scale Leiden measured it on,
                                converted by the code to the soft scale the loss uses [D10]

TWO ARMS, EACH CONTROLLED AGAINST SOMETHING WE ALREADY HAVE
-----------------------------------------------------------
  s28nsc   vs SEACells bk02 (DGE 0.849, 19 pairs, jointP 0.504, int_frac 0.4465) and
           vs our own best lf_L16_cg02_uOFF (0.842, 13, 0.437, 0.3338).
           latent 16, usage OFF, lambda_nassoc=200 -- same as the other btsz runs, so
           the ONLY change is gamma.

  pancreas vs the paper row (modularity 0.61, batch entropy 1.04, purity 0.98) and vs
           our own reproduction repro_paper_LDdef (0.6109 / 0.8640 / 0.9724).
           EVERYTHING matches that reproduction -- default latent, usage ON at 0.1,
           lambda_nassoc=1 as published, same epochs -- and the ONLY change is the
           nassoc diagonal. lambda stays at 1 deliberately: with 'btsz' the CPM term is
           O(0.1-0.3), the same order as the published (M_kk-1)^2 term, so lambda=1
           preserves the paper's balance. The earlier pancreas failure used 200, which
           is the prime suspect for that degeneracy and is NOT repeated here.

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/train_cpm_kmatched_gamma.py
Cost: ~21 min (s28nsc) + ~13 min (pancreas) on a T4, about $0.34.
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

CPM = {
    "nassoc_diag_norm": "pair", "nassoc_diag_loss": "cpm", "nassoc_cpm_free": 1,
    "nassoc_exact": 1, "nassoc_ht": 1, "nassoc_cpm_norm": "btsz",
    "nassoc_gamma_scale": "soft",
}
GAMMA_S28 = 0.00686     # Leiden-CPM K=201 on bk02
GAMMA_PANC = 0.04209    # Leiden-CPM K=221 on arbf
# lambda_nassoc for pancreas, chosen from the MEASURED scale rather than guessed.
# From the earlier pancreas run at lambda=200: nassoc=0.0432, 0.01*proto_recon=24.37,
# loss=36.67 -> umap ~ 3.66, so 200*nassoc = 8.64 is 2.4x umap. On s28nsc the same
# lambda gives 20x, i.e. pancreas' CPM term is ~8x smaller. lambda=1 (the published
# value, for the volume+MSE term) would put CPM at 0.3% of the loss -- inert. So 200
# is the balanced choice here; 50 brackets it from below.
PANC_LAMBDAS = [200, 50]


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


@app.function(image=image, gpu="T4", timeout=7200, volumes={"/vol": volume})
def run_s28nsc():
    _env()
    import os, pickle, numpy as np, pandas as pd, scipy.sparse as sp, scanpy as sc
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

    name = "cpmk_L16_gk00686_uOFF"
    lc = dict(LAMBDA_PROTO_UMAP_PRECON) | CPM | {
        "nassoc_gamma": GAMMA_S28, "lambda_nassoc": 200,
        "lambda_proto_usage": 0.0, "latent_dims": 16,
    }
    print(f"\n=== {name}: K-matched gamma={GAMMA_S28} (Leiden K=201), usage OFF")
    t0 = time.time()
    trainer, metrics, *_ = find_metacells(
        "s28nsc", cvae_epochs=50, train_epochs=20, eval_freq=2, patience=4,
        batch_size=1024, affinity_type="bk02", num_prototypes=200,
        label_key="celltypes", niche_key="niches_2D",
        lambda_config=lc, load_pretrain=True, skip_metrics=["task2", "task3"],
        trainer_kwargs={"experiment_name": name})
    print(f"[{name}] done in {time.time()-t0:.0f}s")
    volume.commit()

    labels = labels_from_trainer(trainer, adata).dropna()
    lab = pd.Series(labels).reindex(adata.obs_names).to_numpy()
    li = np.full(len(lab), -1, np.int64); ok = ~pd.isna(lab)
    li[ok] = lab[ok].astype(np.int64)
    same = (li[T.row] == li[T.col]) & (li[T.row] >= 0)
    e = float(T.data[same].sum())
    sizes = np.bincount(li[li >= 0]); sizes = sizes[sizes > 0].astype(float)
    pairs = float((sizes * (sizes - 1) / 2).sum())
    pur = joint_purity_per_metacell(labels, adata, "celltypes", "niches_2D")
    crow = collapse_summary(labels, name=name, purity=pur)
    dge = nd.run_niche_dge(name, labels, adata, gt, TBL, ct_key="celltypes",
                           niche_key="niches_2D", force=True)
    s = dge["summary"].iloc[0]
    res = dict(DGE=float(s["pearson_r"]), n_pairs=int(s["n_pairs"]),
               jointP=float(crow["purity_micro"]), int_frac=e / m,
               density=e / pairs, K=int(len(sizes)),
               med=float(np.median(sizes)), mn=float(sizes.min()),
               mx=float(sizes.max()), n_under100=int((sizes < 100).sum()))
    print(f"[{name}] >>> {res}")
    volume.commit()
    return res


@app.function(image=image, gpu="T4", timeout=7200, volumes={"/vol": volume})
def run_pancreas():
    _env()
    from interpretable_ssl.experiments.tasks import run_mc_task, LAMBDA_PROTO_UMAP_PRECON

    out = {}
    for lam in PANC_LAMBDAS:
      name = f"cpmk_panc_gk042_l{lam}_uOFF"
      # Matches repro_paper_LDdef -- latent 8 (the default, confirmed against the paper
      # appendix: "hidden dimension 64, latent dimension 8") -- EXCEPT: the CPM diagonal,
      # the K-matched gamma, usage OFF, and lambda_nassoc from the MEASURED scale.
      lc = dict(LAMBDA_PROTO_UMAP_PRECON) | CPM | {
          "nassoc_gamma": GAMMA_PANC, "lambda_nassoc": lam,
          "lambda_proto_usage": 0.0,
      }
      assert lc["lambda_proto_usage"] == 0.0 and lc["lambda_nassoc"] == lam
      assert "latent_dims" not in lc, "pancreas must use the default latent 8"
      print(f"\n=== {name}: CPM diagonal, gamma={GAMMA_PANC} (Leiden K=221), "
            f"lambda_nassoc={lam}, usage OFF, latent=default(8)")
      t0 = time.time()
      try:
          trainer, metrics, _ = run_mc_task(
              "pancreas", cvae_epochs=50, train_epochs=20, eval_freq=2, patience=4,
              batch_size=1024, affinity_type="arbf", num_prototypes=220,
              lambda_config=lc, trainer_kwargs={"experiment_name": name})
      except Exception as e:
          import traceback; traceback.print_exc()
          out[name] = {"error": str(e)}
          volume.commit()
          continue
      print(f"[{name}] done in {time.time()-t0:.0f}s  model={trainer.get_model_name()}")
      print(f"[{name}] >>> metrics: {metrics}")
      out[name] = metrics
      volume.commit()
    return out


@app.local_entrypoint()
def main():
    # s28nsc already ran this session: DGE 0.605, 15 pairs, jointP 0.425,
    # int_frac 0.5501 (= Leiden's exactly), K=195, med 42, max 1928. Not repeated.
    s = None
    p = run_pancreas.remote()

    print("\n" + "=" * 104)
    print("CPM AT THE K-MATCHED GAMMA -- s28nsc / bk02 / K=200, usage OFF")
    print("=" * 104)
    print(f"{'run':<26}{'DGE':>7}{'pairs':>7}{'jointP':>8}{'int_frac':>10}"
          f"{'density':>10}{'med':>6}{'min':>6}{'n<100':>7}")
    if isinstance(s, dict) and "DGE" in s:
        print(f"{'cpmk_L16_gk00686_uOFF':<26}{s['DGE']:>7.3f}{s['n_pairs']:>7}"
              f"{s['jointP']:>8.3f}{s['int_frac']:>10.4f}{s['density']:>10.5f}"
              f"{s['med']:>6.0f}{s['mn']:>6.0f}{s['n_under100']:>7}")
    print(f"{'--- references ---':<26}")
    print(f"{'SEACells bk02':<26}{0.849:>7.3f}{19:>7}{0.504:>8.3f}{0.4465:>10.4f}"
          f"{0.02952:>10.5f}{258:>6}{'--':>6}{'--':>7}")
    print(f"{'scProto best (cg02)':<26}{0.842:>7.3f}{13:>7}{0.437:>8.3f}{0.3338:>10.4f}"
          f"{0.02593:>10.5f}{259:>6}{'--':>6}{'--':>7}")
    print(f"{'Leiden-CPM K=201':<26}{'--':>7}{52:>7}{0.529:>8.3f}{0.5501:>10.4f}"
          f"{0.03203:>10.5f}{309:>6}{1:>6}{'--':>7}")

    print("\n" + "=" * 104)
    print("PANCREAS -- paper config, ONLY the nassoc diagonal changed")
    print("=" * 104)
    print(f"{'run':<30}{'purity':>10}{'batch_ent':>12}{'modularity':>12}{'coverage':>10}")
    if isinstance(p, dict):
        for k, v in p.items():
            if not isinstance(v, dict) or "error" in v:
                print(f"{k:<30}  ERROR"); continue
            print(f"{k:<30}"
                  f"{v.get('purity', float('nan')):>10.4f}"
                  f"{v.get('batch_entropy', float('nan')):>12.4f}"
                  f"{v.get('modularity', float('nan')):>12.4f}"
                  f"{v.get('coverage', float('nan')):>10.4f}")
    print(f"{'--- references (same keys) ---':<30}")
    print(f"{'repro_paper_LDdef':<30}{0.9041:>10.4f}{0.4220:>12.4f}"
          f"{0.6924:>12.4f}{1.0000:>10.4f}")
    print(f"{'earlier CPM attempt (g=0.11,':<30}")
    print(f"{'  lambda=200, usage OFF)':<30}{'--':>10}{'--':>12}{0.3127:>12.4f}"
          f"{0.9286:>10.4f}")
    print("\nNOTE: these are run_mc_task keys (unweighted). The PAPER table uses the")
    print("      weighted_* keys -- read metrics.json on the volume for those.")
