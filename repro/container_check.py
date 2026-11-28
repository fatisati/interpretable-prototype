"""Does the reference container give the same Leiden clusters on every machine?

Experiment tool for docs/decisions/0003, meant to run inside the image built from
repro/docker/Dockerfile. It starts from one committed embedding (the Mac's PCA of heart)
and saves Leiden seed-0 labels for the train and full jobs, two ways:

    <job>_own_graph      neighbour graph built here (sc.pp.neighbors, as evaluation/common.py)
    <job>_windows_graph  the graph saved by Windows (repro/evidence/windows/graph_leiden_check/)

With --expect it then compares the labels, cell by cell, with a labels file from another
machine. "identical" in every column means the two machines produce the same clusters in
this container. It also prints how close each column is to the Windows-native clusters.

Usage (from the repo root, see docs/runbook.md section 0b):
    docker run --rm -v "$PWD:/repo" clean-v2-ref python repro/container_check.py \
        --expect repro/evidence/mac-studio/container_check/labels_linux_arm64.csv.gz

Writes runs/container_check/labels.csv and labels.json (versions, platform, sha256 of
the labels, the comparison, run record).
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_record import record_run  # first local import: records the script's start time

warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.metrics import adjusted_rand_score

import graph_leiden_check as glc

PACKAGES = ("scanpy", "anndata", "pynndescent", "umap-learn", "leidenalg", "igraph",
            "numpy", "scipy", "scikit-learn", "pandas", "numba", "llvmlite")
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")
MAC_PCA = "repro/evidence/mac-studio/heart_cell_atlas/pca"


def versions():
    from importlib.metadata import version
    return {p: version(p) for p in PACKAGES}


def make_labels(embedding, config, windows_dir):
    df, latent, jobs = glc.load(embedding, config)
    labels = pd.DataFrame({"cell_id": df["cell_id"].astype(str)})
    for job, (mask, resolution) in jobs.items():
        ids = df.loc[mask, "cell_id"].astype(str).to_numpy()
        windows_graph, saved = glc.ad_from_saved(Path(windows_dir) / f"graph_{job}.npz", ids)
        assert list(saved["cell_ids"]) == list(ids), "cell order differs from the embedding file"
        own_graph = glc.build_graph(df, latent, mask)
        for tag, graph in (("own_graph", own_graph), ("windows_graph", windows_graph)):
            labels[f"{job}_{tag}"] = pd.NA
            labels.loc[mask, f"{job}_{tag}"] = glc.leiden(graph, resolution)
    return labels


def compare(mine, other, columns):
    """Per column: are the labels the same for every cell, and the adjusted Rand index."""
    out = {}
    for mine_col, other_col in columns:
        a, b = mine[mine_col], other[other_col]
        keep = (a.notna() & b.notna()).to_numpy()
        a, b = a[keep].astype(str), b[keep].astype(str)
        out[mine_col] = {
            "identical": bool((a.to_numpy() == b.to_numpy()).all()),
            "ari": round(float(adjusted_rand_score(a, b)), 4),
            "n_clusters": [int(a.nunique()), int(b.nunique())],
        }
    return out


def show(title, result):
    print(f"\n{title}")
    for col, r in result.items():
        verdict = "identical" if r["identical"] else "DIFFERENT"
        print(f"   {col:20s} {verdict:9s}  ARI {r['ari']:.4f}  "
              f"({r['n_clusters'][0]} vs {r['n_clusters'][1]} clusters)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embedding", default=f"{MAC_PCA}/embedding.csv.gz")
    p.add_argument("--config", default=f"{MAC_PCA}/config.json")
    p.add_argument("--windows-dir", default="repro/evidence/windows/graph_leiden_check")
    p.add_argument("--out", default="runs/container_check/labels.csv")
    p.add_argument("--expect", help="labels file from another machine to compare with")
    a = p.parse_args()

    labels = make_labels(a.embedding, a.config, a.windows_dir)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = labels.to_csv(index=False, lineterminator="\n")
    out.write_text(text, newline="\n")
    columns = [c for c in labels.columns if c != "cell_id"]

    result = {
        "labels_file": out.name,
        "labels_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "inputs": {"embedding": a.embedding, "config": a.config, "windows_dir": a.windows_dir},
        "n_clusters": {c: int(labels[c].nunique()) for c in columns},
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "packages": versions(),
            "threads": {v: os.environ.get(v) for v in THREAD_VARS},
        },
    }
    print(f"[container_check] {platform.platform()} ({platform.machine()}), python {platform.python_version()}")
    print(f"[container_check] wrote {out}  sha256 {result['labels_sha256'][:16]}...")

    native = pd.read_csv(Path(a.windows_dir) / "clusters_seed0.csv", dtype=str)
    result["vs_windows_native"] = compare(labels, native, [(c, "cluster_" + c.split("_")[0]) for c in columns])
    show("Against the Windows-native clusters (clusters_seed0.csv):", result["vs_windows_native"])

    if a.expect:
        other = pd.read_csv(a.expect, dtype=str)
        assert list(other["cell_id"]) == list(labels["cell_id"]), "cell order differs from the expected file"
        result["vs_expected"] = {"file": a.expect, "columns": compare(labels, other, [(c, c) for c in columns])}
        show(f"Against {a.expect}:", result["vs_expected"]["columns"])
        same = all(r["identical"] for r in result["vs_expected"]["columns"].values())
        print("\n[container_check] " + ("SAME CLUSTERS in every column" if same else "DIFFERENT clusters in at least one column"))

    sidecar = out.with_suffix(".json")
    sidecar.write_text(json.dumps(result, indent=2))
    record_run(sidecar)
