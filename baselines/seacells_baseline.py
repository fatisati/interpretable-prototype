"""SEACells baseline adapter for the rare-cell-preservation task.

Thin wrapper around the SEACells library (we call it, we don't reimplement
it) -- except for one piece: instead of letting SEACells build its own
cell-cell kernel (SEACellGraph.rbf in the library's build_graph.py, a Python
loop over every row of the full n x n distance matrix), we hand it the same
graph evaluation/prepare_eval.py already saved for this dataset (built with
affinity.py's sparse-native version of the identical recipe), through the
library's own `add_precomputed_kernel_matrix` hook. Like every baseline, this
script only reads the split and graphs -- run prepare_eval.py first.

SEACells still needs the PCA embedding itself (not just the graph) --
`initialize_archetypes` picks starting points directly from
`adata.obsm["X_pca"]`, independent of whichever kernel was injected. That PCA
must be the exact same one the affinity graph was built from, or the kernel
and the starting-point selection would be working from two different
descriptions of the data. `common.compute_pca` is reused for both, so they
can't drift apart.

Per the task's SEACells section (tasks/rare_cell_preservation.md), run twice:
  - train-only: same cells scVI trains on, for a fair cohesion comparison.
  - full dataset (train + test pooled): the practical "how well does this
    actually do" number.

The number of metacells (n_SEACells) is decided once per dataset and cached
in `seacells_config.json`, next to split.csv/affinity_config.json -- not
recomputed per run. A fixed per-dataset count (derived from cells_per_seacell,
separately for the train-only and full cell counts) keeps every future rerun,
and any other metacell-based baseline added later, comparable; recomputing it
fresh each time would let it silently drift between runs.

No purity/cohesion metrics computed here -- same minimal-output convention
scvi_baseline.py follows (save raw assignments, score them in a separate
step).

Usage:
    python baselines/seacells_baseline.py \\
        --h5ad datasets/processed/heart_cell_atlas_hvg.h5ad \\
        --batch-key donor --label-key cell_type --layer counts \\
        --out-dir runs/rare_cell_preservation/heart_cell_atlas/seacells
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import scanpy as sc
from scipy.sparse import load_npz

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))  # shared pipeline code
from run_record import record_run  # first local import: records the script's start time
from common import compute_pca, load_split

import SEACells


def load_or_create_seacells_counts(n_train, n_full, cells_per_seacell, config_path):
    """Decide n_SEACells once per dataset and cache it, instead of
    recomputing it fresh every run -- two runs (or two different
    metacell-based baselines, later) must land on the same count for the
    same dataset, or any comparison between them is meaningless.
    """
    config_path = Path(config_path)
    requested = {"cells_per_seacell": cells_per_seacell}

    if config_path.exists():
        saved = json.loads(config_path.read_text())
        if saved["cells_per_seacell"] != cells_per_seacell:
            raise ValueError(
                f"{config_path} was created with cells_per_seacell={saved['cells_per_seacell']}, "
                f"but this call asked for {cells_per_seacell} -- delete {config_path} to "
                f"regenerate with the new ratio, or pass the ratio already on disk."
            )
        if saved["n_train"] != n_train or saved["n_full"] != n_full:
            raise ValueError(
                f"{config_path} was computed for n_train={saved['n_train']}/n_full={saved['n_full']}, "
                f"but this run has n_train={n_train}/n_full={n_full} -- delete {config_path} to "
                f"regenerate, or check this is the same split as before."
            )
        print(f"[seacells] loaded cached metacell counts from {config_path}")
        return saved["n_seacells_train"], saved["n_seacells_full"]

    n_seacells_train = max(2, round(n_train / cells_per_seacell))
    n_seacells_full = max(2, round(n_full / cells_per_seacell))
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({
        **requested, "n_train": n_train, "n_full": n_full,
        "n_seacells_train": n_seacells_train, "n_seacells_full": n_seacells_full,
    }, indent=2))
    print(f"[seacells] wrote {config_path} (n_seacells_train={n_seacells_train}, "
          f"n_seacells_full={n_seacells_full})")
    return n_seacells_train, n_seacells_full


def run_seacells(adata, kernel_matrix, n_seacells, n_waypoint_eigs, convergence_epsilon,
                  max_iter, min_iter):
    """Build + fit one SEACells model, using a precomputed kernel instead of
    letting the library build its own (see module docstring). Returns the
    hard per-cell metacell assignment as a pandas Series indexed by cell id.
    """
    model = SEACells.core.SEACells(
        adata,
        build_kernel_on="X_pca",
        n_SEACells=n_seacells,
        n_waypoint_eigs=n_waypoint_eigs,
        convergence_epsilon=convergence_epsilon,
    )
    model.add_precomputed_kernel_matrix(kernel_matrix)
    model.initialize_archetypes()
    model.fit(min_iter=min_iter, max_iter=max_iter)
    return model.get_hard_assignments()["SEACell"]


def validate_affinity_config(affinity_config_path, n_pca_comps, k, graph_construction):
    saved = json.loads(Path(affinity_config_path).read_text())
    requested = {"n_pca_comps": n_pca_comps, "k": k, "graph_construction": graph_construction}
    saved_subset = {key: saved[key] for key in requested}
    if saved_subset != requested:
        raise ValueError(
            f"{affinity_config_path} was built with {saved_subset}, but this call asked for "
            f"{requested} -- they must match, since SEACells needs the exact PCA/graph that "
            f"produced the cached affinity files. Re-run evaluation/prepare_eval.py --force "
            f"to rebuild with the new parameters, or pass the ones already there."
        )


def run(
    h5ad_path,
    batch_key,
    label_key,
    out_dir,
    layer=None,
    test_size=0.2,
    seed=0,
    n_pca_comps=50,
    k=15,
    graph_construction="union",
    cells_per_seacell=75,
    n_waypoint_eigs=10,
    convergence_epsilon=1e-3,
    max_iter=100,
    min_iter=10,
):
    out_dir = Path(out_dir)
    dataset_dir = out_dir.parent

    adata = sc.read_h5ad(h5ad_path)
    split_path = dataset_dir / "split.csv"
    adata.obs["split"] = load_split(adata, label_key, batch_key, test_size, seed, split_path)
    train_mask = (adata.obs["split"] == "train").to_numpy()

    # Read-only, like the split: built once by evaluation/prepare_eval.py.
    affinity_config_path = dataset_dir / "affinity_config.json"
    if not affinity_config_path.exists():
        raise FileNotFoundError(
            f"{affinity_config_path} does not exist -- run `python evaluation/prepare_eval.py` "
            f"for this dataset first. Baselines only read the affinity graphs, they don't build them."
        )
    validate_affinity_config(dataset_dir / "affinity_config.json", n_pca_comps, k, graph_construction)
    affinity_train = load_npz(dataset_dir / "affinity_train.npz")
    affinity_full = load_npz(dataset_dir / "affinity_full.npz")
    print(f"[seacells] loaded {dataset_dir / 'affinity_train.npz'} and "
          f"{dataset_dir / 'affinity_full.npz'}")

    train_adata = adata[train_mask].copy()
    full_adata = adata.copy()
    # Same PCA that built the loaded affinity graphs -- not a fresh, merely
    # similarly-configured, recomputation (see module docstring).
    train_adata.obsm["X_pca"] = compute_pca(train_adata, layer, n_pca_comps)
    full_adata.obsm["X_pca"] = compute_pca(full_adata, layer, n_pca_comps)

    n_seacells_train, n_seacells_full = load_or_create_seacells_counts(
        n_train=int(train_adata.n_obs), n_full=int(full_adata.n_obs),
        cells_per_seacell=cells_per_seacell, config_path=dataset_dir / "seacells_config.json",
    )

    print(f"[seacells] train-only run: {train_adata.n_obs:,} cells -> {n_seacells_train} metacells")
    train_assignments = run_seacells(
        train_adata, affinity_train, n_seacells_train, n_waypoint_eigs,
        convergence_epsilon, max_iter, min_iter,
    )

    print(f"[seacells] full-dataset run: {full_adata.n_obs:,} cells -> {n_seacells_full} metacells")
    full_assignments = run_seacells(
        full_adata, affinity_full, n_seacells_full, n_waypoint_eigs,
        convergence_epsilon, max_iter, min_iter,
    )

    result = pd.DataFrame({
        "cell_id": adata.obs_names,
        "split": adata.obs["split"].to_numpy(),
        label_key: adata.obs[label_key].to_numpy(),
        batch_key: adata.obs[batch_key].to_numpy(),
    })
    result["SEACell_full"] = full_assignments.reindex(result["cell_id"]).to_numpy()
    result["SEACell_train"] = pd.NA
    result.loc[train_mask, "SEACell_train"] = (
        train_assignments.reindex(result.loc[train_mask, "cell_id"]).to_numpy()
    )

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
        "n_pca_comps": n_pca_comps,
        "k": k,
        "graph_construction": graph_construction,
        "cells_per_seacell": cells_per_seacell,
        "n_seacells_train": n_seacells_train,
        "n_seacells_full": n_seacells_full,
        "n_waypoint_eigs": n_waypoint_eigs,
        "convergence_epsilon": convergence_epsilon,
        "max_iter": max_iter,
        "min_iter": min_iter,
        "n_train": int(train_adata.n_obs),
        "n_full": int(full_adata.n_obs),
        "n_unique_metacells_train": int(train_assignments.nunique()),
        "n_unique_metacells_full": int(full_assignments.nunique()),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"[seacells] wrote {out_dir / 'assignments.csv'}")
    return out_dir


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad", required=True, help="path to a processed h5ad file")
    p.add_argument("--batch-key", required=True, help="adata.obs column with batch/study id")
    p.add_argument("--label-key", required=True, help="adata.obs column with cell type label")
    p.add_argument("--out-dir", required=True, help="where to write assignments.csv + config.json")
    p.add_argument("--layer", default=None, help="adata.layers key to use as input (default: X)")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-pca-comps", type=int, default=50, help="must match the affinity graph's own PCA")
    p.add_argument("--k", type=int, default=15, help="must match the affinity graph's own k")
    p.add_argument("--graph-construction", default="union", choices=["union", "intersection"])
    p.add_argument("--cells-per-seacell", type=int, default=75,
                    help="metacell sizing ratio -- only used the first time a dataset is run; "
                         "cached in seacells_config.json after that")
    p.add_argument("--n-waypoint-eigs", type=int, default=10)
    p.add_argument("--convergence-epsilon", type=float, default=1e-3)
    p.add_argument("--max-iter", type=int, default=100)
    p.add_argument("--min-iter", type=int, default=10)
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
        n_pca_comps=args.n_pca_comps,
        k=args.k,
        graph_construction=args.graph_construction,
        cells_per_seacell=args.cells_per_seacell,
        n_waypoint_eigs=args.n_waypoint_eigs,
        convergence_epsilon=args.convergence_epsilon,
        max_iter=args.max_iter,
        min_iter=args.min_iter,
    )
    record_run(Path(args.out_dir) / "config.json")
