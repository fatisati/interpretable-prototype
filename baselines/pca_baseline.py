"""PCA baseline (no batch correction) -- the "unintegrated" control.

Same pipeline as scVI with one change: the embedding is plain PCA instead of
a batch-corrected latent space. Fit on train cells only, then test cells are
projected with that fit -- mirrors scVI (learn from train, embed test), so the
train/test setup matches too. Then the same shared clustering
(common.cluster_embedding_file) and the same metrics.

Why: tells apart "integration lost the rare type" from "our clustering (fixed
K, ~75-cell clusters) lost it". If PCA + Leiden also loses a rare type, the
loss isn't the correction's fault.

Caveat: the ground-truth affinity graph is built from uncorrected PCA of all
cells (prepare_eval.py), so this baseline's cohesion / neighbor-affinity are
close to a ceiling by construction. Its cell-type metrics (agreement, purity,
neighbor same-type share) are the fair comparison.

Usage:
    python baselines/pca_baseline.py --h5ad datasets/processed/heart_cell_atlas_hvg.h5ad \\
        --batch-key donor --label-key cell_type --layer counts \\
        --out-dir runs/rare_cell_preservation/heart_cell_atlas/pca
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))  # shared pipeline code
from run_record import record_run  # first local import: records the script's start time
from common import cluster_embedding_file, environment_versions, load_cluster_config, load_split, looks_like_raw_counts


def log_normalized(adata, layer):
    """Same normalization as common.compute_pca (the affinity graph's PCA):
    normalize + log1p only if the values look like raw counts."""
    ad = adata.copy()
    if layer:
        ad.X = ad.layers[layer].copy()
    if looks_like_raw_counts(ad.X):
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)
    return ad.X.toarray() if issparse(ad.X) else np.asarray(ad.X)


def run(h5ad_path, batch_key, label_key, out_dir, layer=None, test_size=0.2, seed=0, n_comps=50):
    adata = sc.read_h5ad(h5ad_path)
    dataset_dir = Path(out_dir).parent
    split_path = dataset_dir / "split.csv"
    adata.obs["split"] = load_split(adata, label_key, batch_key, test_size, seed, split_path)
    train_mask = (adata.obs["split"] == "train").to_numpy()
    load_cluster_config(dataset_dir / "cluster_config.json")  # fail early if missing

    X = log_normalized(adata, layer)
    pca = PCA(n_components=n_comps, random_state=seed).fit(X[train_mask])
    emb = pca.transform(X)
    print(f"[pca] {adata.n_obs:,} cells x {adata.n_vars:,} genes -> {n_comps} PCs "
          f"(fit on {train_mask.sum():,} train cells)")

    result = pd.DataFrame({
        "cell_id": adata.obs_names,
        "split": adata.obs["split"].to_numpy(),
        label_key: adata.obs[label_key].to_numpy(),
        batch_key: adata.obs[batch_key].to_numpy(),
    })
    for d in range(n_comps):
        result[f"latent_{d}"] = emb[:, d]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "embedding.csv", index=False)
    (out_dir / "config.json").write_text(json.dumps({
        "h5ad_path": str(h5ad_path), "split_path": str(split_path),
        "batch_key": batch_key, "label_key": label_key, "layer": layer,
        "test_size": test_size, "seed": seed, "n_comps": n_comps, "pca_fit_on": "train",
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "n_train": int(train_mask.sum()), "n_test": int((~train_mask).sum()),
        "environment": {**environment_versions(), "scikit-learn": __import__("sklearn").__version__},
    }, indent=2))
    cluster_embedding_file(out_dir)  # same shared clustering as scVI
    return out_dir


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5ad", required=True)
    p.add_argument("--batch-key", required=True)
    p.add_argument("--label-key", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--layer", default=None)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-comps", type=int, default=50)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(a.h5ad, a.batch_key, a.label_key, a.out_dir, a.layer, a.test_size, a.seed, a.n_comps)
    record_run(Path(a.out_dir) / "config.json")
