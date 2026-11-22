"""Build the ground-truth affinity graphs every baseline's cohesion metric is
measured against.

Built once from the dataset's own PCA -- not from any baseline's output
embedding. If a baseline's cohesion were measured against a graph built from
its own embedding, that would be circular: the baseline would always look
consistent with "its own notion of similarity," regardless of whether it
preserved anything real. "The original affinity graph" (task doc's wording)
means one fixed, shared ground truth every baseline is compared against.

Two graphs, matching the task's own split (see tasks/rare_cell_preservation.md):
  - train-only: PCA fit on train cells only, affinity graph built from that --
    no information from test cells leaks in.
  - full: PCA fit on all cells (train + test) together.

Usage:
    python baselines/build_affinity.py \\
        --h5ad datasets/processed/heart_cell_atlas_hvg.h5ad \\
        --batch-key donor --label-key cell_type \\
        --out-dir runs/rare_cell_preservation/heart_cell_atlas \\
        [--layer counts]

Shares out-dir with scvi_baseline.py's --out-dir's parent (one level up from
its own <out-dir>/scvi/) so both read/write the same split.csv -- run either
script first, the other will reuse its split rather than recompute one.
"""
import argparse
import json
from pathlib import Path

import scanpy as sc
from scipy.sparse import save_npz

from affinity import build_affinity_graph
from run_record import record_run  # first local import: records the script's start time
from common import compute_pca, load_or_create_split


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
    force=False,
):
    out_dir = Path(out_dir)
    train_path = out_dir / "affinity_train.npz"
    full_path = out_dir / "affinity_full.npz"
    if train_path.exists() and full_path.exists() and not force:
        print(f"[affinity] {train_path} and {full_path} already exist, skipping (--force to rebuild)")
        return out_dir

    adata = sc.read_h5ad(h5ad_path)
    split_path = out_dir / "split.csv"
    adata.obs["split"] = load_or_create_split(adata, label_key, batch_key, test_size, seed, split_path)

    train_adata = adata[adata.obs["split"] == "train"]
    print(f"[affinity] train-only: {train_adata.n_obs:,} cells")
    train_pca = compute_pca(train_adata, layer, n_pca_comps)
    affinity_train = build_affinity_graph(train_pca, k=k, graph_construction=graph_construction)

    print(f"[affinity] full: {adata.n_obs:,} cells")
    full_pca = compute_pca(adata, layer, n_pca_comps)
    affinity_full = build_affinity_graph(full_pca, k=k, graph_construction=graph_construction)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_npz(train_path, affinity_train)
    save_npz(full_path, affinity_full)

    # Row/column order for affinity_train is "cells where split=='train', in
    # adata's original order" -- not saved separately, since split.csv (same
    # out_dir) plus this ordering rule is enough to reconstruct it exactly.
    config = {
        "h5ad_path": str(h5ad_path),
        "split_path": str(split_path),
        "n_pca_comps": n_pca_comps,
        "k": k,
        "graph_construction": graph_construction,
        "n_train": int(train_adata.n_obs),
        "n_full": int(adata.n_obs),
    }
    (out_dir / "affinity_config.json").write_text(json.dumps(config, indent=2))
    record_run(out_dir / "affinity_config.json")

    print(f"[affinity] wrote {train_path} (nnz={affinity_train.nnz:,}) and {full_path} (nnz={affinity_full.nnz:,})")
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
    p.add_argument("--force", action="store_true", help="rebuild even if affinity_train.npz/affinity_full.npz already exist")
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
        force=args.force,
    )
