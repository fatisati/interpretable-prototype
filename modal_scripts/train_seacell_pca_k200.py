"""SEACells on PCA, s28nsc, K=200 -- the like-for-like baseline for every K=200 run.

WHY THIS RUN EXISTS
-------------------
Every s28nsc K=200 comparison on record uses `seacell_X_bklam02_K200` /
`seacell_X_bklam04_K200` as the SEACells reference: SEACells fitted on a BANKSY
embedding, i.e. on features that already mix each cell's expression with its spatial
neighbours'. decisions.md section 14 showed that this single fact accounts for most of
the gap (`BANKSY X_bklam02` reaches internal_fraction 0.4262 vs `pca50` 0.2790 through
the same linear head), and scProto's encoder never receives spatial features.

So the BANKSY SEACells run is NOT an information-matched baseline. The one that is --
SEACells on the same PCA of the same expression matrix, at the same K=200 -- has never
been run. The only SEACells(PCA) run on s28nsc is K=800 (`models/s28nsc/seacell`,
n_SEACells=800), which is not K-matched either.

This script produces `models/s28nsc/seacell_K200`, i.e.
get_seacell_model_dir('s28nsc', 'X_pca', num_prototypes=200), and evaluates it with the
SAME metric set used for the other arms so its numbers drop straight into the tables.

WHAT IT WRITES  (all under /vol/models/s28nsc/seacell_K200/)
------------------------------------------------------------
  seacell_sc.h5ad / seacell_agg.h5ad   the fit itself (delta against the dataset's
                                       shared seacell_sc_base.h5ad -- see save_seacell)
  metrics.json                         task1 + task2 + task3 scalars, each with its std
  *_per_mc.csv, modularity*.csv/json, coverage.csv, scgraph.csv, umap_*.csv,
  cell_assignments.csv, proto_vectors.npy
and under /vol/code/files/tables/s28nsc_seacell_pca_k200/:
  partition_stats.csv     internal_fraction / density / size stats on bk02, bk04, bk08
  collapse_summary.csv    largest_share, dip, tightness gap, purity_micro (bk02)
  dge_summary.csv         niche-program recovery (pearson_r etc.), same ground truth
                          file the other K=200 arms are scored against

Task1's affinity-based compactness is computed on 'bk02' because that is the graph the
K=200 scProto arms are judged on; internal_fraction is additionally reported on bk04 and
bk08 so this one baseline is usable against every K=200 run regardless of its graph.

Run (two steps, both resumable -- training is skipped if seacell_sc.h5ad already exists):
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/train_seacell_pca_k200.py
CPU-only; no GPU is needed for SEACells or for any of this eval.
"""
import modal

app = modal.App("scproto-seacell-pca-k200")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "numpy", "scipy", "pandas", "anndata", "scanpy",
        "torch", "scarches", "wandb", "tqdm", "statsmodels", "diptest",
        "palantir", "faiss-cpu", "scikit-learn", "matplotlib", "scib-metrics",
        "ipython", "umap-learn", "igraph", "leidenalg",
    )
    .pip_install("git+https://github.com/dpeerlab/SEACells.git", extra_options="--no-deps")
    # Data_Handler.py eagerly open()s these eleven at import time -- see CLAUDE.md §11
    .run_commands(
        "for d in lung lung_fetal_donor lung_fetal_organoid brain breast heart eye "
        "gut_fetal skin COVID pancreas; do "
        "mkdir -p /content/drive/MyDrive/codes/Islander/meta/$d && "
        "echo '{}' > /content/drive/MyDrive/codes/Islander/meta/$d/cell2cat.json && "
        "echo '{}' > /content/drive/MyDrive/codes/Islander/meta/$d/batch2cat.json; done"
    )
)

volume = modal.Volume.from_name("scproto-data")

DS_ID = "s28nsc"
K = 200
CT_KEY, NICHE_KEY = "celltypes", "niches_2D"
TABLE_DIR = "/vol/code/files/tables/s28nsc_seacell_pca_k200"
FIG_DIR = "/vol/code/files/figures/s28nsc_seacell_pca_k200"


def _bootstrap():
    """env vars + sys.path + the anndata null-reader patch. See CLAUDE.md §11."""
    import os
    import sys

    os.environ["HOME_DIR"] = "/vol/"
    os.environ["DATA_DIR"] = "/vol/data/"
    os.environ["MODEL_DIR"] = "/vol/models/"
    os.environ["CODE_DIR"] = "/vol/code/"
    sys.path.insert(0, "/vol/code")
    os.chdir("/vol/code")  # get_affinity_path's graph_dir='./graphs' is CWD-relative

    import h5py
    from anndata._io.specs.registry import _REGISTRY, IOSpec

    try:
        @_REGISTRY.register_read(h5py.Dataset, IOSpec("null", "0.1.0"))
        def _read_null(elem, _reader):
            return None
    except TypeError:
        pass  # already registered


@app.function(image=image, cpu=8, memory=32768, timeout=10800, volumes={"/vol": volume})
def train():
    """SEACells archetypal analysis on X_pca at K=200. Load-if-exists (CLAUDE.md §4):
    mode='eval' makes train_seacell skip the fit when seacell_sc.h5ad is already there."""
    import os
    _bootstrap()

    from seacell_train import train_seacell
    from interpretable_ssl.configs.paths import get_seacell_model_dir

    save_dir = get_seacell_model_dir(DS_ID, "X_pca", num_prototypes=K)
    print(f"[target] {save_dir}")
    if os.path.exists(os.path.join(save_dir, "seacell_sc.h5ad")):
        print("loaded from disk: seacell_sc.h5ad already exists, skipping the fit")
        return {"status": "loaded", "dir": save_dir}

    print("computed: no saved fit found, running SEACells on X_pca ...")
    train_seacell(DS_ID, mode="train", build_kernel_on="X_pca", num_prototypes=K)
    volume.commit()
    return {"status": "computed", "dir": save_dir}


@app.function(image=image, cpu=8, memory=32768, timeout=10800, volumes={"/vol": volume})
def evaluate():
    """task1 + task2 + task3 -> metrics.json, then the K=200-arm comparison metrics."""
    import json
    import os
    import pickle

    _bootstrap()

    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    import scanpy as sc
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from seacell_train import eval_seacell_task1, eval_seacell_task2, eval_seacell_task3
    from interpretable_ssl.configs.paths import get_seacell_model_dir
    from interpretable_ssl.datasets.dataset_configs import DATASETS

    save_dir = get_seacell_model_dir(DS_ID, "X_pca", num_prototypes=K)
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # ------------------------------------------------------------------ task 1/2/3
    # affinity_type='bk02': the aff-DC compactness is measured on the same graph the
    # K=200 scProto arms are trained and judged on, so the numbers are comparable.
    print("\n=== task1 ===")
    print(eval_seacell_task1(DS_ID, build_kernel_on="X_pca", num_prototypes=K,
                             affinity_type="bk02", graph_dir="./graphs"))
    volume.commit()

    print("\n=== task2 ===")
    print(eval_seacell_task2(DS_ID, build_kernel_on="X_pca", num_prototypes=K,
                             compute_dge=True))
    volume.commit()

    print("\n=== task3 ===")
    print(eval_seacell_task3(DS_ID, build_kernel_on="X_pca", num_prototypes=K))
    volume.commit()

    with open(os.path.join(save_dir, "metrics.json")) as f:
        metrics = json.load(f)
    print("\n=== metrics.json ===")
    for k_, v in metrics.items():
        if isinstance(v, (int, float)):
            print(f"  {k_:<40} {v}")

    # --------------------------------------------- K=200-arm comparison metrics
    adata = sc.read_h5ad(DATASETS[DS_ID]["path"])
    assign = pd.read_csv(os.path.join(save_dir, "cell_assignments.csv"))
    labels = assign.set_index("cell_id")["metacell_id"].reindex(adata.obs_names.to_numpy())
    n_na = int(labels.isna().sum())
    if n_na:
        print(f"{n_na} cells unassigned, excluded from the comparison metrics")
    labels = labels.dropna()

    def to_ints(labels_series):
        lab = pd.Series(labels_series).reindex(adata.obs_names).to_numpy()
        ok = ~pd.isna(lab)
        li = np.full(len(lab), -1, dtype=np.int64)
        li[ok] = pd.Categorical(pd.Series(lab[ok]).astype(str)).codes
        return li

    li = to_ints(labels)

    # internal_fraction = (summed affinity weight of edges whose two endpoints landed in
    # the SAME metacell) / (total edge weight in the graph). Same definition every other
    # K=200 arm is reported with (diag_and_train_correct_gamma.py::partition_stats).
    rows = []
    for aff_tag in ("bk02", "bk04", "bk08"):
        gp = f"/vol/code/graphs/affinity_{DS_ID}{adata.n_obs}_ncomp50_kneighbors50_{aff_tag}.pkl"
        if not os.path.exists(gp):
            print(f"[{aff_tag}] no cached affinity at {gp}, skipping")
            continue
        with open(gp, "rb") as f:
            A = pickle.load(f).tocsr(copy=True)
        A.setdiag(0)
        A.eliminate_zeros()
        tri = sp.triu(A, k=1).tocoo()
        m = float(tri.sum())
        same = (li[tri.row] == li[tri.col]) & (li[tri.row] >= 0)
        e = float(tri.data[same].sum())
        sizes = np.bincount(li[li >= 0], minlength=max(int(li.max()) + 1, 1)).astype(float)
        alive = sizes[sizes > 0]
        pairs = float((alive * (alive - 1) / 2).sum())
        rows.append({
            "affinity": aff_tag,
            "edges_inside": e,
            "total_edge_weight": m,
            "internal_fraction": e / m,
            "total_possible_pairs": pairs,
            "aggregate_density": e / pairs if pairs else float("nan"),
            "n_alive": int(len(alive)),
            "median_size": float(np.median(alive)),
            "max_size": float(alive.max()),
            "size_ratio_max_over_median": float(alive.max() / np.median(alive)),
        })
        print(f"[{aff_tag}] internal_fraction={e / m:.4f}  n_alive={len(alive)}  "
              f"median_size={np.median(alive):.0f}  max_size={alive.max():.0f}")
        if aff_tag == "bk02":
            aff_bk02 = A
    part_df = pd.DataFrame(rows).set_index("affinity")
    part_df.to_csv(f"{TABLE_DIR}/partition_stats.csv")

    print("\n=== collapse summary (on bk02) ===")
    from interpretable_ssl.evaluation.collapse_diagnostics import (
        joint_purity_per_metacell, collapse_summary, plot_collapse, metacell_sizes,
    )
    name = "SEACells_pca_K200"
    pur = joint_purity_per_metacell(labels, adata, CT_KEY, NICHE_KEY)
    row = collapse_summary(labels, affinity=aff_bk02, name=name, purity=pur)
    collapse_df = pd.DataFrame([row]).set_index("run")
    collapse_df["n_metacells"] = labels.nunique()
    collapse_df["median_size"] = metacell_sizes(labels).median()
    collapse_df.to_csv(f"{TABLE_DIR}/collapse_summary.csv")
    print(collapse_df.round(4).T)
    plot_collapse(labels, aff_bk02, name=name)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{name}.png", dpi=120, bbox_inches="tight")
    plt.close()

    print("\n=== niche DGE ===")
    from interpretable_ssl.evaluation import niche_dge as nd
    ground_truth = nd.get_ground_truth(
        adata, "/vol/code/files/celltype_niches_full_s28nsc.csv",
        ct_key=CT_KEY, niche_key=NICHE_KEY)
    print(f"{len(ground_truth)} (cell type, niche) pairs")
    dge_results = {name: nd.run_niche_dge(
        name, labels, adata, ground_truth, TABLE_DIR, ct_key=CT_KEY, niche_key=NICHE_KEY)}
    dge_summary = nd.summary_table(dge_results)
    dge_summary.to_csv(f"{TABLE_DIR}/dge_summary.csv")
    print(dge_summary.round(4))

    volume.commit()
    print("\nDone. Wrote:", save_dir, TABLE_DIR, FIG_DIR)
    return {
        "metrics": {k_: v for k_, v in metrics.items() if isinstance(v, (int, float))},
        "partition_stats": part_df.to_dict(),
        "collapse": collapse_df.to_dict(),
        "dge_summary": dge_summary.to_dict(),
    }


@app.local_entrypoint()
def main():
    print(train.remote())
    print(evaluate.remote())
