"""Compare the actual files of two runs of the same dataset (not just hashes).

A fingerprint says THAT two machines differ; this says HOW MUCH and WHERE.
Give it two folders laid out like runs/<task>/<dataset>/ -- a live run folder
or an evidence pack under repro/evidence/<machine>/<dataset>/ (gzipped CSVs are
read directly). It compares whatever both folders have:

    hvg_genes.txt     genes kept: shared / only in A / only in B, and order
    split.csv         cells on the same side, per split
    cluster_config    K
    affinity_full     shared edges (Jaccard), weight difference on shared edges
    <baseline>/embedding.csv
                      largest coordinate difference; for each cell, how many of
                      its 15 nearest neighbours are the same in A and B
    <baseline> cluster labels
                      agreement between the two clusterings (adjusted Rand
                      index, 1 = identical grouping)
    <baseline>/metrics/metrics_summary.json
                      every metric, A vs B

Usage:
    python evaluation/compare_runs.py \\
        --a runs/rare_cell_preservation/heart_cell_atlas \\
        --b repro/evidence/mac-studio/heart_cell_atlas
    --genes-a / --genes-b take a processed .h5ad instead of hvg_genes.txt.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz


def read_table(folder, name):
    for candidate in (Path(folder) / name, Path(folder) / (name + ".gz")):
        if candidate.exists():
            return pd.read_csv(candidate)
    return None


def read_genes(folder, h5ad=None):
    if h5ad:
        import anndata as ad
        return [str(g) for g in ad.read_h5ad(h5ad, backed="r").var_names]
    path = Path(folder) / "hvg_genes.txt"
    if path.exists():
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # A live run folder has no gene list of its own: use the processed file it was built from.
    config = Path(folder) / "affinity_config.json"
    if config.exists():
        h5ad_path = json.loads(config.read_text()).get("h5ad_path")
        if h5ad_path and Path(h5ad_path).exists():
            return read_genes(folder, h5ad_path)
    return None


def section(title):
    print(f"\n== {title}")


def compare_genes(ga, gb):
    section("genes kept after stage 1")
    if ga is None or gb is None:
        print("   skipped (need hvg_genes.txt on both sides, or --genes-a / --genes-b)")
        return
    sa, sb = set(ga), set(gb)
    print(f"   A: {len(ga):,} genes   B: {len(gb):,} genes   shared: {len(sa & sb):,}   "
          f"only A: {len(sa - sb):,}   only B: {len(sb - sa):,}   Jaccard: {len(sa & sb) / max(1, len(sa | sb)):.4f}")
    print(f"   same order: {ga == gb}")
    if sa != sb:
        print(f"   first only-A: {sorted(sa - sb)[:8]}\n   first only-B: {sorted(sb - sa)[:8]}")


def compare_split(a_dir, b_dir):
    section("split.csv")
    a, b = read_table(a_dir, "split.csv"), read_table(b_dir, "split.csv")
    if a is None or b is None:
        print("   skipped (missing on one side)")
        return None
    a, b = a.set_index("cell_id")["split"], b.set_index("cell_id")["split"]
    shared = a.index.intersection(b.index)
    print(f"   cells A: {len(a):,}   B: {len(b):,}   shared ids: {len(shared):,}")
    same = (a.loc[shared] == b.loc[shared])
    print(f"   same side: {int(same.sum()):,} of {len(shared):,} ({same.mean():.4%})")
    print(f"   test cells A: {(a == 'test').sum():,}   B: {(b == 'test').sum():,}   "
          f"in both test sets: {int(((a.loc[shared] == 'test') & (b.loc[shared] == 'test')).sum()):,}")
    return shared


def compare_json(a_dir, b_dir, name, keys):
    section(name)
    pa, pb = Path(a_dir) / name, Path(b_dir) / name
    if not (pa.exists() and pb.exists()):
        print("   skipped (missing on one side)")
        return
    ja, jb = json.loads(pa.read_text()), json.loads(pb.read_text())
    for key in keys:
        mark = "same" if ja.get(key) == jb.get(key) else "DIFFERENT"
        print(f"   {key}: A={ja.get(key)}  B={jb.get(key)}  {mark}")


def compare_affinity(a_dir, b_dir, name="affinity_full.npz"):
    section(name)
    pa, pb = Path(a_dir) / name, Path(b_dir) / name
    if not (pa.exists() and pb.exists()):
        print("   skipped (missing on one side)")
        return
    a, b = load_npz(pa).tocsr(), load_npz(pb).tocsr()
    if a.shape != b.shape:
        print(f"   shapes differ: {a.shape} vs {b.shape} -- row order can't be assumed equal, stopping here")
        return
    print("   (assumes both graphs use the same cell order = the processed file's order; check 'cells' first)")
    ea, eb = (a != 0), (b != 0)
    both = ea.multiply(eb)
    union = ea.nnz + eb.nnz - both.nnz
    print(f"   edges A: {ea.nnz:,}   B: {eb.nnz:,}   shared: {both.nnz:,}   Jaccard: {both.nnz / max(1, union):.6f}")
    diff = abs(a.multiply(both) - b.multiply(both))
    print(f"   weight on shared edges: max |diff| = {diff.max():.3g}   "
          f"mean |diff| = {diff.sum() / max(1, both.nnz):.3g}")


def neighbour_overlap(xa, xb, k=15):
    from sklearn.neighbors import NearestNeighbors
    ia = NearestNeighbors(n_neighbors=k + 1).fit(xa).kneighbors(xa, return_distance=False)[:, 1:]
    ib = NearestNeighbors(n_neighbors=k + 1).fit(xb).kneighbors(xb, return_distance=False)[:, 1:]
    return np.array([len(set(r) & set(s)) / k for r, s in zip(ia, ib)])


def compare_baseline(name, fa, fb):
    section(f"baseline: {name}")
    ta = read_table(fa, "embedding.csv")
    tb = read_table(fb, "embedding.csv")
    if ta is None or tb is None:
        ta, tb = read_table(fa, "assignments.csv"), read_table(fb, "assignments.csv")
    if ta is None or tb is None:
        print("   skipped (no embedding.csv / assignments.csv on one side)")
        return
    ta, tb = ta.set_index("cell_id"), tb.set_index("cell_id")
    shared = ta.index.intersection(tb.index)
    print(f"   cells A: {len(ta):,}   B: {len(tb):,}   shared ids: {len(shared):,}   "
          f"same row order: {list(ta.index) == list(tb.index)}")
    ta, tb = ta.loc[shared], tb.loc[shared]
    latent = [c for c in ta.columns if c.startswith("latent_") and c in tb.columns]
    if latent:
        xa, xb = ta[latent].to_numpy(float), tb[latent].to_numpy(float)
        flipped = np.minimum(np.abs(xa - xb).max(axis=0), np.abs(xa + xb).max(axis=0))
        print(f"   coordinates: max |diff| = {np.abs(xa - xb).max():.3g}   "
              f"(allowing each axis to flip sign: {flipped.max():.3g})")
        overlap = neighbour_overlap(xa, xb)
        print(f"   15 nearest neighbours shared per cell: mean {overlap.mean():.4f}   "
              f"1st percentile {np.percentile(overlap, 1):.4f}   cells with all 15 equal: {(overlap == 1).mean():.2%}")
    from sklearn.metrics import adjusted_rand_score
    for col in ("cluster_train_run", "cluster_full_run"):
        if col in ta and col in tb:
            keep = ta[col].notna() & tb[col].notna()
            if keep.any():
                ca, cb = ta.loc[keep, col].astype(str), tb.loc[keep, col].astype(str)
                print(f"   {col}: clusters A {ca.nunique()}  B {cb.nunique()}   "
                      f"agreement (adjusted Rand index) {adjusted_rand_score(ca, cb):.4f}")
    for metrics_dir in sorted(Path(fa).glob("metrics*")):
        sa, sb = metrics_dir / "metrics_summary.json", Path(fb) / metrics_dir.name / "metrics_summary.json"
        if sa.exists() and sb.exists():
            rows = []
            flat_a = dict(_flatten(json.loads(sa.read_text())))
            flat_b = dict(_flatten(json.loads(sb.read_text())))
            for key in sorted(set(flat_a) & set(flat_b)):
                va, vb = flat_a[key], flat_b[key]
                if (isinstance(va, (int, float)) and isinstance(vb, (int, float)) and key.endswith("mean")
                        and "per_batch" not in key and va == va and vb == vb):  # x == x drops NaN
                    rows.append((abs(va - vb), key, va, vb))
            rows.sort(reverse=True)
            print(f"   {metrics_dir.name}: {len(rows)} dataset-level metric means compared; largest differences:")
            for diff, key, va, vb in rows[:8]:
                print(f"      {key:58s} A={va:.4f}  B={vb:.4f}  |diff|={diff:.2e}")


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _flatten(child, f"{prefix}.{key}" if prefix else str(key))
    else:
        yield prefix, value


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--genes-a", default=None, help="processed .h5ad for side A (instead of A/hvg_genes.txt)")
    p.add_argument("--genes-b", default=None)
    p.add_argument("--map", nargs="*", default=[], metavar="A_NAME=B_NAME",
                   help="compare baseline folders with different names, e.g. scvi=scvi_cpu")
    args = p.parse_args()
    a_dir, b_dir = Path(args.a), Path(args.b)
    print(f"A = {a_dir}\nB = {b_dir}")

    compare_genes(read_genes(a_dir, args.genes_a), read_genes(b_dir, args.genes_b))
    compare_split(a_dir, b_dir)
    compare_json(a_dir, b_dir, "cluster_config.json", ["cells_per_cluster", "k_train", "k_full"])
    compare_affinity(a_dir, b_dir)

    pairs = dict(item.split("=", 1) for item in args.map)
    for folder in sorted(x for x in a_dir.iterdir() if x.is_dir()):
        other = b_dir / pairs.get(folder.name, folder.name)
        if other.is_dir():
            compare_baseline(f"{folder.name} vs {other.name}", folder, other)


if __name__ == "__main__":
    main()
