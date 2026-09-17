"""MEASURE GAMMA FROM THE AFFINITY GRAPH -- no sweeping, no guessing.

WHY
---
CLAUDE.md 9b says gamma must be DERIVED from measured properties of the real graph,
and the CLAUDE.md edit of 2026-09-17 marks that derivation as UNSETTLED. This script
measures the quantity gamma is actually compared against.

THE MOVE RULE IS THE DERIVATION
-------------------------------
The CPM gradient wrt cell i's membership in prototype p is exactly Leiden's local move:

    delta = (affinity from i to members of p) - gamma * size[p]
          = size[p] * ( MEAN affinity from i to the members of p  -  gamma )

so                cell i joins p  iff  mean_affinity(i -> members of p) > gamma

That mean is over ALL members of p, including the ones i has no edge to. So gamma is
"the minimum AVERAGE affinity a cell must have to a group to be worth joining it",
and it is measurable without training anything.

WHAT IS MEASURED
----------------
1. graph basics: N, m, mean nonzero weight, mean weighted degree.
2. PAIR level -- mean affinity of an edge whose endpoints share (celltype x niche) vs
   one whose endpoints differ. This is gamma in the "a pair earns A_ij and is charged
   gamma" reading.
3. GROUP level (THE ONE THAT MATCHES THE MOVE RULE) -- split each cell's weighted
   degree into same-label and different-label mass, then divide by (target_size - 1).
   This is the mean-affinity-to-group a perfectly label-pure metacell would support.
4. The same group-level quantity on SEACELLS' OWN PARTITION instead of labels -- an
   apples-to-apples "what gamma does a partition we believe in support?". SEACells
   metacells are spatially local; a (celltype x niche) LABEL CLASS is not, so 3 and 4
   answer different questions and both are reported.

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run --detach modal_scripts/measure_gamma_from_graph.py
Cost: CPU only, no GPU, a few minutes.
"""
import modal

app = modal.App("scproto-repro")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("numpy", "scipy", "pandas", "anndata", "scanpy", "h5py",
                 "scikit-learn", "statsmodels", "diptest", "matplotlib",
                 "palantir", "ipython", "scib-metrics", "torch")
    # metacell_metrics.py hard-imports SEACells at module level, and SEACells
    # hard-imports IPython.display at ITS module level (CLAUDE.md 11).
    .pip_install("git+https://github.com/dpeerlab/SEACells.git",
                 extra_options="--no-deps")
    # Data_Handler.py eagerly open()s these eleven at module import time (CLAUDE.md 11)
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

AFF = "/vol/code/graphs/affinity_s28nsc58423_ncomp50_kneighbors50_bk02.pkl"
TARGET_SIZES = [100, 150, 200, 292, 400]
LABEL_KEY, NICHE_KEY = "celltypes", "niches_2D"


@app.function(image=image, cpu=4, memory=32768, timeout=3600, volumes={"/vol": volume})
def measure():
    import os, sys, pickle
    import numpy as np, pandas as pd, scipy.sparse as sp, h5py

    os.environ["WANDB_MODE"] = "disabled"
    os.environ["HOME_DIR"] = "/vol/"
    os.environ["DATA_DIR"] = "/vol/data/"
    os.environ["MODEL_DIR"] = "/vol/models/"
    os.environ["CODE_DIR"] = "/vol/code/"
    sys.path.insert(0, "/vol/code")
    os.chdir("/vol/code")

    import scanpy as sc
    from anndata._io.specs.registry import _REGISTRY, IOSpec
    try:
        @_REGISTRY.register_read(h5py.Dataset, IOSpec("null", "0.1.0"))
        def _read_null(elem, _reader):
            return None
    except TypeError:
        pass

    # ---------------------------------------------------------------- graph
    with open(AFF, "rb") as f:
        A = pickle.load(f).tocsr()
    A = A.tocsr(copy=True)
    A.setdiag(0)
    A.eliminate_zeros()
    A = A.tocoo()
    N = A.shape[0]
    row, col, dat = A.row, A.col, A.data
    upper = row < col
    m = float(dat[upper].sum())
    n_edges = int(upper.sum())
    d_all = np.bincount(row, weights=dat, minlength=N)
    all_pairs = N * (N - 1) / 2.0

    print("=" * 78)
    print("1. GRAPH BASICS  (bk02)")
    print("=" * 78)
    print(f"  cells N                        = {N:,}")
    print(f"  undirected edges               = {n_edges:,}")
    print(f"  total edge weight m            = {m:,.0f}")
    print(f"  mean weight of a NONZERO edge  = {m / n_edges:.4f}")
    print(f"  max edge weight                = {dat.max():.4f}")
    print(f"  mean weighted degree           = {d_all.mean():.4f}")
    print(f"  median weighted degree         = {np.median(d_all):.4f}")
    print(f"  all pairs N(N-1)/2             = {all_pairs:,.0f}")
    print(f"  fraction of pairs with an edge = {n_edges / all_pairs:.7f}")

    def group_stats(li, tag, note=""):
        """li: int label per cell, -1 = unassigned."""
        ok = li >= 0
        both = ok[row] & ok[col]
        same = both & (li[row] == li[col])
        su = same & upper
        du = (~same) & both & upper
        print(f"\n  [{tag}] {note}")
        print(f"    edges WITHIN  a group : n={int(su.sum()):>9,}  "
              f"mean weight={dat[su].mean():.4f}")
        print(f"    edges BETWEEN groups  : n={int(du.sum()):>9,}  "
              f"mean weight={dat[du].mean():.4f}")
        print(f"    internal_fraction (share of edge WEIGHT kept inside) "
              f"= {dat[su].sum() / m:.4f}")

        d_same = np.bincount(row[same], weights=dat[same], minlength=N)
        d_diff = d_all - d_same
        sizes = np.bincount(li[ok])
        sizes = sizes[sizes > 0]
        print(f"    groups={len(sizes):>5}   median size={np.median(sizes):.0f}   "
              f"max size={sizes.max():,}")
        print(f"    mean same-group weighted degree = {d_same[ok].mean():.4f}")
        print(f"    mean diff-group weighted degree = {d_diff[ok].mean():.4f}")
        print(f"    {'target_size':>12} {'gamma_signal':>13} {'gamma_noise':>12} {'ratio':>7}")
        for S in TARGET_SIZES:
            g_sig = d_same[ok].mean() / (S - 1)
            g_noi = d_diff[ok].mean() / (S - 1)
            print(f"    {S:>12} {g_sig:>13.4f} {g_noi:>12.4f} "
                  f"{g_sig / max(g_noi, 1e-9):>7.2f}")

        n_lab = int(li[ok].max()) + 1
        w_in = np.bincount(li[row[same]], weights=dat[same], minlength=n_lab) / 2.0
        cnt = np.bincount(li[ok], minlength=n_lab).astype(float)
        pairs = cnt * (cnt - 1) / 2.0
        dens = np.where(pairs > 0, w_in / np.maximum(pairs, 1e-9), np.nan)
        print(f"    per-group density: median={np.nanmedian(dens):.5f}  "
              f"mean={np.nanmean(dens):.5f}  "
              f"aggregate={w_in.sum() / max(pairs.sum(), 1e-9):.5f}")

    from interpretable_ssl.datasets.dataset_configs import DATASETS
    adata = sc.read_h5ad(DATASETS["s28nsc"]["path"])
    print(f"\n[data] {adata.shape}")
    print(f"[data] obs columns: {list(adata.obs.columns)}")

    def codes(values):
        return pd.Categorical(pd.Series(values).astype(str)).codes.astype(np.int64)

    print("\n" + "=" * 78)
    print("2+3. WHAT GAMMA THE GROUND-TRUTH LABELS SUPPORT")
    print("=" * 78)
    ct = adata.obs[LABEL_KEY].astype(str).values
    ni = adata.obs[NICHE_KEY].astype(str).values
    group_stats(codes(ct), "celltype only")
    group_stats(codes(ni), "niche only")
    joint = codes(np.array([f"{a}|{b}" for a, b in zip(ct, ni)]))
    group_stats(joint, "celltype x niche",
                "<- label classes are NOT spatially local, so this UNDERSTATES gamma")

    print("\n" + "=" * 78)
    print("4. WHAT GAMMA SEACELLS' OWN PARTITION SUPPORTS  (spatially local)")
    print("=" * 78)
    mdir = "/vol/models/s28nsc"
    cands = sorted(d for d in os.listdir(mdir)
                   if d.startswith("seacell")
                   and os.path.exists(os.path.join(mdir, d, "seacell_sc.h5ad")))
    print(f"  seacell dirs found: {cands}")
    pick = next((d for d in cands if "bklam02" in d or "bk02" in d),
                cands[0] if cands else None)
    if pick:
        # Read obs['SEACell'] DIRECTLY. Deliberately NOT importing metacell_metrics --
        # its module chain pulls SEACells -> scib_metrics -> scvi, none of which this
        # script needs. The per-tag assignment is exactly what differs between tags, so
        # it always lives in the delta file itself, never only in the shared base.
        sad = sc.read_h5ad(os.path.join(mdir, pick, "seacell_sc.h5ad"))
        print(f"  delta file: obs columns = {list(sad.obs.columns)}")
        lab = pd.Series(sad.obs["SEACell"].astype(str).values,
                        index=sad.obs_names).reindex(adata.obs_names).to_numpy()
        li = np.full(N, -1, dtype=np.int64)
        okm = ~pd.isna(lab)
        li[okm] = codes(lab[okm])
        group_stats(li, f"SEACells [{pick}]", "<- a partition we believe in")
    else:
        print("  no SEACells run found on the volume")

    print("\n" + "=" * 78)
    print("HOW TO READ THIS")
    print("=" * 78)
    print("  gamma_signal at target_size S = the mean affinity a cell has to a PERFECT")
    print("    group of size S. gamma must be BELOW this or nothing ever joins.")
    print("  gamma_noise = the same for cells that should NOT be grouped. gamma must be")
    print("    ABOVE this or everything joins everything.")
    print("  A usable gamma lives between the two; the ratio is how much room there is.")


@app.local_entrypoint()
def main():
    measure.remote()
