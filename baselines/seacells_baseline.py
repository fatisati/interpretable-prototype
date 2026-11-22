"""SEACells baseline adapter for the rare-cell-preservation task.

Thin wrapper around the SEACells library (we call it, we don't reimplement
it). Follows docs/architecture.md: reads the split, both affinity graphs,
and the target cluster count from evaluation/prepare_eval.py's output --
creates none of them itself. Run prepare_eval.py for the dataset first.

Three SEACells-specific points:

1. Kernel = the shared affinity graph. SEACells is the one baseline that
   needs the affinity graph as *input* (its kernel), not only for later
   scoring. Instead of letting the library build its own kernel (a slow
   per-row Python loop in its build_graph.py), we hand it the saved graph
   through the library's own `add_precomputed_kernel_matrix` hook. Same
   recipe, same diagonal (1), so this is the same input SEACells would have
   built for itself -- see affinity.py.

2. Cluster count = cluster_config.json's k_train / k_full, passed straight in
   as SEACells' own n_SEACells. Clustering is built into SEACells (each
   metacell is a cluster), so there's no Leiden sweep here, unlike scVI.

3. SEACells also needs the PCA itself, not just the graph:
   `initialize_archetypes` picks its starting cells from adata.obsm["X_pca"]
   directly. That PCA must match the one the saved graph was built from, so
   n_pca_comps is read from affinity_config.json (not a separate flag that
   could drift), and computed with the same common.compute_pca.

Runs twice, per tasks/rare_cell_preservation.md's SEACells section:
  - train-only: same cells scVI trains on (fair comparison of cohesion).
  - full dataset: train + test pooled (the practical ceiling number).

Output: assignments.csv (cell_id, split, label, batch, cluster_train_run,
cluster_full_run -- same cluster column names as scvi_baseline.py) +
config.json (with a run_history entry, see run_record.py). No metrics here;
compute_metrics.py's seacells adapter scores this file. Unlike scVI/PCA,
the metrics use BOTH columns: one job per run (see that adapter).

Usage:
    python evaluation/prepare_eval.py \\
        --h5ad datasets/processed/heart_cell_atlas_hvg.h5ad \\
        --batch-key donor --label-key cell_type --layer counts \\
        --out-dir runs/rare_cell_preservation/heart_cell_atlas
    python baselines/seacells_baseline.py \\
        --h5ad datasets/processed/heart_cell_atlas_hvg.h5ad \\
        --batch-key donor --label-key cell_type --layer counts \\
        --out-dir runs/rare_cell_preservation/heart_cell_atlas/seacells
"""
import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import load_npz

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))  # shared pipeline code
from run_record import record_run  # first local import: records the script's start time
from common import compute_pca, environment_versions, load_cluster_config, load_split

import SEACells

# On top of common.environment_versions(): palantir runs SEACells' waypoint
# initialization, numba compiles its inner loops.
SEACELLS_PACKAGES = ["SEACells", "palantir", "numba", "scipy", "scikit-learn"]


def seacells_environment_versions():
    versions = environment_versions()
    for pkg in SEACELLS_PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
    return versions


def load_affinity(dataset_dir):
    """Read-only, same rule as load_split/load_cluster_config: error if
    prepare_eval.py hasn't built the graphs, never build them here.
    """
    paths = [dataset_dir / name for name in
             ("affinity_train.npz", "affinity_full.npz", "affinity_config.json")]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"missing {missing} -- run `python evaluation/prepare_eval.py` for this "
            f"dataset first. Baselines only read the affinity graph, they don't build it."
        )
    affinity_config = json.loads(paths[2].read_text())
    return load_npz(paths[0]), load_npz(paths[1]), affinity_config


def run_seacells(adata, kernel_matrix, n_seacells, n_waypoint_eigs, convergence_epsilon,
                  min_iter, max_iter):
    """Fit one SEACells model on `adata` with a precomputed kernel. Returns
    each cell's hard metacell assignment, indexed by cell id.
    """
    model = SEACells.core.SEACells(
        adata,
        build_kernel_on="X_pca",
        n_SEACells=int(n_seacells),
        n_waypoint_eigs=n_waypoint_eigs,
        convergence_epsilon=convergence_epsilon,
    )
    model.add_precomputed_kernel_matrix(kernel_matrix)
    model.initialize_archetypes()
    model.fit(min_iter=min_iter, max_iter=max_iter)
    return model.get_hard_assignments()["SEACell"]


def run(
    h5ad_path,
    batch_key,
    label_key,
    out_dir,
    layer=None,
    test_size=0.2,
    seed=0,
    n_waypoint_eigs=10,
    convergence_epsilon=1e-3,
    min_iter=10,
    max_iter=100,
):
    adata = sc.read_h5ad(h5ad_path)
    out_dir = Path(out_dir)
    dataset_dir = out_dir.parent

    # All read-only -- created once by prepare_eval.py, shared by every
    # baseline on this dataset (see docs/architecture.md).
    split_path = dataset_dir / "split.csv"
    adata.obs["split"] = load_split(adata, label_key, batch_key, test_size, seed, split_path)
    train_mask = (adata.obs["split"] == "train").to_numpy()
    cluster_config = load_cluster_config(dataset_dir / "cluster_config.json")
    affinity_train, affinity_full, affinity_config = load_affinity(dataset_dir)
    n_pca_comps = affinity_config["n_pca_comps"]

    train_adata = adata[train_mask].copy()
    full_adata = adata.copy()
    # Same PCA recipe and size as the saved graphs (see point 3 above).
    train_adata.obsm["X_pca"] = compute_pca(train_adata, layer, n_pca_comps)
    full_adata.obsm["X_pca"] = compute_pca(full_adata, layer, n_pca_comps)

    print(f"[seacells] train-only run: {train_adata.n_obs:,} cells -> "
          f"{cluster_config['k_train']} metacells")
    # SEACells draws its starting cells from numpy's GLOBAL random generator and
    # takes no seed of its own. Unseeded, two identical runs gave different
    # metacells (agreement ARI 0.58, measured 2026-11-22), so seed it here.
    np.random.seed(seed)
    train_assign = run_seacells(train_adata, affinity_train, cluster_config["k_train"],
                                n_waypoint_eigs, convergence_epsilon, min_iter, max_iter)

    print(f"[seacells] full-dataset run: {full_adata.n_obs:,} cells -> "
          f"{cluster_config['k_full']} metacells")
    np.random.seed(seed)  # re-seed, so the full run doesn't depend on the train run
    full_assign = run_seacells(full_adata, affinity_full, cluster_config["k_full"],
                               n_waypoint_eigs, convergence_epsilon, min_iter, max_iter)

    result = pd.DataFrame({
        "cell_id": adata.obs_names,
        "split": adata.obs["split"].to_numpy(),
        label_key: adata.obs[label_key].to_numpy(),
        batch_key: adata.obs[batch_key].to_numpy(),
    })
    # cluster_train_run: only train cells were in that run -> test cells NA.
    # cluster_full_run: every cell was in that run -> every cell has a label.
    result["cluster_train_run"] = pd.NA
    result.loc[train_mask, "cluster_train_run"] = (
        train_assign.reindex(result.loc[train_mask, "cell_id"]).to_numpy()
    )
    result["cluster_full_run"] = full_assign.reindex(result["cell_id"]).to_numpy()

    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "assignments.csv", index=False)

    config = {
        "h5ad_path": str(h5ad_path),
        "split_path": str(split_path),
        "affinity_dir": str(dataset_dir),
        "batch_key": batch_key,
        "label_key": label_key,
        "layer": layer,
        "test_size": test_size,
        "seed": seed,
        "numpy_global_seed_set": True,
        "n_pca_comps": n_pca_comps,
        "n_waypoint_eigs": n_waypoint_eigs,
        "convergence_epsilon": convergence_epsilon,
        "min_iter": min_iter,
        "max_iter": max_iter,
        "n_train": int(train_mask.sum()),
        "n_full": int(adata.n_obs),
        "k_train_target": cluster_config["k_train"],
        "k_full_target": cluster_config["k_full"],
        "n_clusters_train": int(train_assign.nunique()),
        "n_clusters_full": int(full_assign.nunique()),
        "environment": seacells_environment_versions(),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"[seacells] wrote {out_dir / 'assignments.csv'}")
    return out_dir


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5ad", required=True, help="path to a processed h5ad file")
    p.add_argument("--batch-key", required=True, help="adata.obs column with batch/study id")
    p.add_argument("--label-key", required=True, help="adata.obs column with cell type label")
    p.add_argument("--out-dir", required=True, help="where to write assignments.csv + config.json")
    p.add_argument("--layer", default=None, help="adata.layers key used for PCA (must match prepare_eval.py's)")
    p.add_argument("--test-size", type=float, default=0.2, help="must match prepare_eval.py's (checked)")
    p.add_argument("--seed", type=int, default=0, help="must match prepare_eval.py's (checked)")
    p.add_argument("--n-waypoint-eigs", type=int, default=10)
    p.add_argument("--convergence-epsilon", type=float, default=1e-3)
    p.add_argument("--min-iter", type=int, default=10)
    p.add_argument("--max-iter", type=int, default=100)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        h5ad_path=args.h5ad,
        batch_key=args.batch_key,
        label_key=args.label_key,
        out_dir=args.out_dir,
        layer=args.layer,
        test_size=args.test_size,
        seed=args.seed,
        n_waypoint_eigs=args.n_waypoint_eigs,
        convergence_epsilon=args.convergence_epsilon,
        min_iter=args.min_iter,
        max_iter=args.max_iter,
    )
    record_run(Path(args.out_dir) / "config.json")
