"""Build the evaluation scaffolding every baseline on a dataset shares:
the train/test split, the two ground-truth affinity graphs, and the
target cluster count.

Run this once per dataset, before any baseline. Baseline scripts (e.g.
scvi_baseline.py) only ever *read* what this script writes -- they error
out if it's missing, rather than silently creating their own copy. This is
the same rule datasets/prepare.py already follows for preprocessing (see
that file's own docstring): decide something about the data once, save it,
have every consumer read the same saved file. See docs/architecture.md for
the full pipeline design -- every baseline must follow it, no exceptions.

Writes, into datasets/frozen/<dataset>/ (committed to git, see evaluation/frozen.py):
  split.csv, split.json                -- train/test division. Only written when the
                                          dataset has none yet, or with --refresh-split;
                                          otherwise the committed one is read.
Writes, into --out-dir:
  affinity_train.npz, affinity_full.npz -- RBF graphs, train-only and all-cells PCA
  affinity_config.json
  cluster_config.json                   -- target cluster count K (n_cells // cells_per_cluster)

Usage:
    python evaluation/prepare_eval.py \\
        --h5ad datasets/processed/heart_cell_atlas_hvg.h5ad \\
        --batch-key donor --label-key cell_type \\
        --out-dir runs/rare_cell_preservation/heart_cell_atlas \\
        [--layer counts]
"""
import argparse
import json
from pathlib import Path

import scanpy as sc
from scipy.sparse import save_npz

from run_record import record_run  # first local import: records the script's start time
from affinity import build_affinity_graph
import frozen
from common import compute_pca, load_split, stratified_split


def create_split(adata, dataset, label_key, batch_key, test_size, seed):
    """Creates the dataset's frozen split -- this script is the one place
    allowed to, and only when none exists yet or --refresh-split is given.
    Everything else (baselines, metrics) only reads it.
    """
    train_pos, test_pos = stratified_split(adata, label_key, batch_key, test_size, seed)
    split = ["train"] * adata.n_obs
    for i in test_pos:
        split[i] = "test"
    frozen.write_split(dataset, list(adata.obs_names), split, {
        "seed": seed, "test_size": test_size, "label_key": label_key, "batch_key": batch_key,
    })
    return split


def create_cluster_config(n_train, n_full, cells_per_cluster, cluster_config_path):
    """Target cluster count K, decided once per dataset and saved -- every
    baseline must hit the same K for a comparison between them to be fair
    (more clusters makes purity trivially easier). See docs/architecture.md's
    Clustering section: this is the one shared number; how each baseline
    reaches it is that baseline's own business.
    """
    k_train = max(2, n_train // cells_per_cluster)
    k_full = max(2, n_full // cells_per_cluster)
    cluster_config_path.parent.mkdir(parents=True, exist_ok=True)
    cluster_config_path.write_text(json.dumps({
        "cells_per_cluster": cells_per_cluster,
        "n_train": n_train, "n_full": n_full,
        "k_train": k_train, "k_full": k_full,
    }, indent=2))
    print(f"[prepare_eval] wrote {cluster_config_path} (k_train={k_train}, k_full={k_full})")
    return k_train, k_full


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
    cells_per_cluster=75,
    force=False,
    refresh_split=False,
):
    out_dir = Path(out_dir)
    dataset = frozen.dataset_id(h5ad_path)
    split_path = frozen.split_path(dataset)
    train_path = out_dir / "affinity_train.npz"
    full_path = out_dir / "affinity_full.npz"
    cluster_config_path = out_dir / "cluster_config.json"

    if (split_path.exists() and train_path.exists() and full_path.exists()
            and cluster_config_path.exists() and not force and not refresh_split):
        print(f"[prepare_eval] {out_dir} already has split + affinity graphs + cluster "
              f"config, skipping (--force to rebuild)")
        return out_dir

    adata = sc.read_h5ad(h5ad_path)
    frozen.check_genes(adata, dataset)
    if split_path.exists() and not refresh_split:
        adata.obs["split"] = load_split(adata, label_key, batch_key, test_size, seed, split_path)
    else:
        adata.obs["split"] = create_split(adata, dataset, label_key, batch_key, test_size, seed)
        split_path = frozen.split_path(dataset)

    train_adata = adata[adata.obs["split"] == "train"]
    print(f"[prepare_eval] train-only affinity graph: {train_adata.n_obs:,} cells")
    train_pca = compute_pca(train_adata, layer, n_pca_comps)
    affinity_train = build_affinity_graph(train_pca, k=k, graph_construction=graph_construction)

    print(f"[prepare_eval] full affinity graph: {adata.n_obs:,} cells")
    full_pca = compute_pca(adata, layer, n_pca_comps)
    affinity_full = build_affinity_graph(full_pca, k=k, graph_construction=graph_construction)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_npz(train_path, affinity_train)
    save_npz(full_path, affinity_full)

    create_cluster_config(train_adata.n_obs, adata.n_obs, cells_per_cluster, cluster_config_path)

    # Row/column order for affinity_train is "cells where split=='train', in
    # adata's original order" -- reconstructable from split.csv plus this
    # ordering rule, so not saved separately.
    config = {
        "h5ad_path": str(h5ad_path),
        "split_path": str(split_path),
        "frozen": frozen.fingerprint(dataset),
        "n_pca_comps": n_pca_comps,
        "k": k,
        "graph_construction": graph_construction,
        "n_train": int(train_adata.n_obs),
        "n_full": int(adata.n_obs),
    }
    (out_dir / "affinity_config.json").write_text(json.dumps(config, indent=2))
    for sidecar in (out_dir / "affinity_config.json", cluster_config_path):
        record_run(sidecar)

    print(f"[prepare_eval] wrote {train_path} (nnz={affinity_train.nnz:,}) and {full_path} (nnz={affinity_full.nnz:,})")
    return out_dir


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad", required=True)
    p.add_argument("--batch-key", required=True)
    p.add_argument("--label-key", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--layer", default=None)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-pca-comps", type=int, default=50)
    p.add_argument("--k", type=int, default=15, help="neighborhood size for the affinity kernel (same meaning as SEACells' k)")
    p.add_argument("--graph-construction", default="union", choices=["union", "intersection"])
    p.add_argument("--cells-per-cluster", type=int, default=75,
                    help="target cluster count = n_cells // this; same ratio SEACells uses for metacell sizing")
    p.add_argument("--force", action="store_true",
                   help="rebuild affinity/cluster files even if they exist (the frozen split is kept)")
    p.add_argument("--refresh-split", action="store_true",
                   help="recompute the committed split in datasets/frozen/ (then commit the diff); "
                        "makes every later stage stale")
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
        cells_per_cluster=args.cells_per_cluster,
        force=args.force,
        refresh_split=args.refresh_split,
    )
