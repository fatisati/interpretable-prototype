"""Fingerprint one dataset's pipeline outputs, so two machines can be compared.

Reads files only (plus one PCA, a few seconds). Trains nothing, changes nothing.
Writes one JSON with a short hash or summary for every stage output it finds:

    processed   the stage-1 h5ad: cell order, gene order, matrix values
    split       split.csv
    cluster_config
    pca         the PCA the affinity graph is built from (recomputed here)
    affinity_full / affinity_train
    baselines   every subfolder holding embedding.csv or assignments.csv
    environment package versions, platform, linear-algebra library

Hashes are over CONTENT, never over file bytes: file checksums differ between
machines for reasons that don't matter (anndata's file format version, Windows
writing "\\r\\n" line endings in CSVs). Two machines with the same content get
the same hash here.

Compare two of these with evaluation/compare_fingerprints.py.
Plan and background: docs/reproducibility-plan.md.

Usage (from the repo root):
    python evaluation/fingerprint.py \\
        --run-dir runs/rare_cell_preservation/heart_cell_atlas \\
        --h5ad datasets/processed/heart_cell_atlas_hvg.h5ad \\
        --layer counts --machine mac-studio
Writes repro/fingerprints/<dataset>/<machine>.json (dataset = run-dir's name).
"""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse, load_npz

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def short_hash(*chunks):
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(chunk if isinstance(chunk, (bytes, bytearray, memoryview)) else str(chunk).encode("utf-8"))
    return h.hexdigest()[:16]


def names_hash(names):
    return short_hash("\n".join(str(n) for n in names))


def matrix_hash(matrix):
    """Hash of a matrix's values, independent of storage format: converted to
    CSR with sorted column indices, explicit zeros removed, fixed dtypes."""
    m = csr_matrix(matrix) if not issparse(matrix) else matrix.tocsr().copy()
    m.sum_duplicates()
    m.eliminate_zeros()
    m.sort_indices()
    return short_hash(
        np.asarray(m.shape, dtype=np.int64).tobytes(),
        m.indptr.astype(np.int64).tobytes(),
        m.indices.astype(np.int64).tobytes(),
        m.data.astype(np.float64).tobytes(),
    )


def read_table(folder, name):
    """Read <folder>/<name> or <folder>/<name>.gz (evidence packs are gzipped)."""
    for candidate in (Path(folder) / name, Path(folder) / (name + ".gz")):
        if candidate.exists():
            return pd.read_csv(candidate)
    return None


def rounded(values, digits=6):
    return [round(float(v), digits) for v in values]


def drop_per_batch(value):
    """Keep dataset-level metric numbers only; the per-batch breakdown stays in
    the metrics folder (compare it with compare_runs.py)."""
    if isinstance(value, dict):
        return {k: drop_per_batch(v) for k, v in value.items() if not str(k).endswith("per_batch")}
    return value


def partition_hash(labels):
    """Hash of a clustering that ignores what the clusters are called: labels
    are renumbered by order of first appearance."""
    codes = pd.factorize(pd.Series(labels).astype(str))[0]
    return short_hash(codes.astype(np.int64).tobytes())


def fingerprint_processed(h5ad_path, split_meta):
    import anndata as ad
    adata = ad.read_h5ad(h5ad_path)
    out = {
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "cells_hash": names_hash(adata.obs_names),
        "genes_hash_in_order": names_hash(adata.var_names),
        "genes_hash_sorted": names_hash(sorted(map(str, adata.var_names))),
        "matrix_hash": {"X": matrix_hash(adata.X)},
    }
    for layer in sorted(adata.layers.keys()):
        out["matrix_hash"]["layers/" + layer] = matrix_hash(adata.layers[layer])
    for role in ("label_key", "batch_key"):
        key = (split_meta or {}).get(role)
        if key and key in adata.obs:
            out[role + "_values_hash"] = names_hash(adata.obs[key].astype(str))
    return out, adata


def fingerprint_split(run_dir):
    split = read_table(run_dir, "split.csv")
    if split is None:
        return None
    pairs = sorted(split["cell_id"].astype(str) + ":" + split["split"].astype(str))
    return {
        "hash": names_hash(pairs),
        "n_train": int((split["split"] == "train").sum()),
        "n_test": int((split["split"] == "test").sum()),
    }


def fingerprint_affinity(path):
    if not Path(path).exists():
        return None
    a = load_npz(path).tocoo()
    n = a.shape[0]
    order = np.lexsort((a.col, a.row))
    edge_ids = (a.row[order].astype(np.int64) * n + a.col[order].astype(np.int64))
    degree = np.bincount(a.row, minlength=n)
    return {
        "n": int(n),
        "nnz": int(a.nnz),
        "weight_sum": round(float(a.data.sum()), 6),
        "edge_hash": short_hash(edge_ids.tobytes()),
        "weights_hash_6dp": short_hash(np.round(a.data[order].astype(np.float64), 6).tobytes()),
        "degree_min_median_max": [int(degree.min()), float(np.median(degree)), int(degree.max())],
    }


def fingerprint_pca(adata, layer, n_comps):
    from common import compute_pca
    pca = np.asarray(compute_pca(adata, layer, n_comps), dtype=np.float64)
    return {
        "layer": layer,
        "shape": list(pca.shape),
        "column_std_first10": rounded(pca.std(axis=0)[:10]),
        "abs_column_sum_first10": rounded(np.abs(pca).sum(axis=0)[:10], 4),
        "values_hash_4dp_abs": short_hash(np.round(np.abs(pca), 4).tobytes()),
    }


def fingerprint_baseline(folder):
    table = read_table(folder, "embedding.csv")
    kind = "embedding"
    if table is None:
        table, kind = read_table(folder, "assignments.csv"), "assignments"
    if table is None:
        return None
    out = {"kind": kind, "n_rows": int(len(table)), "cells_hash": names_hash(table["cell_id"])}
    latent_cols = [c for c in table.columns if c.startswith("latent_")]
    if latent_cols:
        values = table[latent_cols].to_numpy(dtype=np.float64)
        out["latent_dims"] = len(latent_cols)
        out["latent_column_means"] = rounded(values.mean(axis=0))
        out["latent_column_stds"] = rounded(values.std(axis=0))
        out["latent_hash_4dp"] = short_hash(np.round(values, 4).tobytes())
    for col in ("cluster_train_run", "cluster_full_run"):
        if col in table:
            present = table[col].dropna()
            out[col] = {"n_clusters": int(present.astype(str).nunique()), "partition_hash": partition_hash(present)}
    config_path = Path(folder) / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        keep = ("seed", "accelerator", "device_used", "epochs_trained", "gene_likelihood", "n_latent",
                "resolution_train", "resolution_full", "n_clusters_train", "n_clusters_full", "environment")
        out["config"] = {k: config[k] for k in keep if k in config}
    metrics = {}
    for metrics_dir in sorted(Path(folder).glob("metrics*")):
        summary_path = metrics_dir / "metrics_summary.json"
        if summary_path.exists():
            metrics[metrics_dir.name] = drop_per_batch(json.loads(summary_path.read_text()))
    if metrics:
        out["metrics"] = metrics
    return out


def environment():
    from importlib.metadata import PackageNotFoundError, version
    packages = ["scvi-tools", "torch", "lightning", "scanpy", "anndata", "numpy", "scipy", "pandas",
                "scikit-learn", "scikit-misc", "leidenalg", "igraph", "SEACells", "palantir", "numba", "h5py"]
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
    out = {"python": platform.python_version(), "platform": platform.platform(),
           "machine_arch": platform.machine(), "packages": versions}
    try:
        from threadpoolctl import threadpool_info
        out["linear_algebra"] = [
            {k: lib.get(k) for k in ("user_api", "internal_api", "version", "num_threads", "threading_layer")}
            for lib in threadpool_info()
        ]
    except Exception as exc:  # threadpoolctl missing: say so instead of leaving a silent gap
        out["linear_algebra"] = f"unavailable ({type(exc).__name__})"
    return out


def run(run_dir, h5ad_path, layer, machine, out_path=None, skip_pca=False):
    run_dir = Path(run_dir)
    dataset = run_dir.name
    split_json = run_dir / "split.json"
    split_meta = json.loads(split_json.read_text()) if split_json.exists() else None
    affinity_json = run_dir / "affinity_config.json"
    affinity_config = json.loads(affinity_json.read_text()) if affinity_json.exists() else {}

    result = {"dataset": dataset, "machine": machine, "run_dir": run_dir.as_posix()}
    processed, adata = fingerprint_processed(h5ad_path, split_meta)
    result["processed"] = processed
    result["split"] = fingerprint_split(run_dir)
    if split_meta:
        result["split_params"] = {k: split_meta.get(k) for k in ("seed", "test_size", "label_key", "batch_key")}
    cluster_json = run_dir / "cluster_config.json"
    if cluster_json.exists():
        cluster = json.loads(cluster_json.read_text())
        result["cluster_config"] = {k: cluster.get(k) for k in ("cells_per_cluster", "k_train", "k_full")}
    if not skip_pca:
        result["pca"] = fingerprint_pca(adata, layer, affinity_config.get("n_pca_comps", 50))
    result["affinity_params"] = {k: affinity_config.get(k) for k in ("n_pca_comps", "k", "graph_construction")}
    result["affinity_full"] = fingerprint_affinity(run_dir / "affinity_full.npz")
    result["affinity_train"] = fingerprint_affinity(run_dir / "affinity_train.npz")
    result["baselines"] = {}
    for folder in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        baseline = fingerprint_baseline(folder)
        if baseline is not None:
            result["baselines"][folder.name] = baseline
    result["environment"] = environment()

    out_path = Path(out_path) if out_path else REPO_ROOT / "repro" / "fingerprints" / dataset / f"{machine}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="\n", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, default=str) + "\n")
    print(f"[fingerprint] wrote {out_path}")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="runs/<task>/<dataset> (holds split.csv, affinity_*.npz, baselines)")
    p.add_argument("--h5ad", required=True, help="the processed stage-1 file this run-dir was built from")
    p.add_argument("--layer", default=None, help="same --layer value the stages were run with (heart: counts)")
    p.add_argument("--machine", required=True, help="short tag for this computer, e.g. win-main, mac-studio")
    p.add_argument("--out", default=None, help="default: repro/fingerprints/<dataset>/<machine>.json")
    p.add_argument("--skip-pca", action="store_true", help="don't recompute the PCA (skips the pca block)")
    a = p.parse_args()
    run(a.run_dir, a.h5ad, a.layer, a.machine, a.out, a.skip_pca)
