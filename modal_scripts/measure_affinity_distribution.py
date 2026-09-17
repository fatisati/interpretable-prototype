"""AFFINITY DISTRIBUTION FOR SAME-STATE vs DIFFERENT-STATE PAIRS.

THE QUESTION (Fatemeh, 2026-09-17)
---------------------------------
For pairs of cells in the SAME state (same celltype AND same niche): what is the mean
and median of the affinity between them? And -- the important part -- the zero pairs
must be counted too, because CPM charges gamma for EVERY pair inside a community,
connected or not.

WHY BOTH VERSIONS ARE NEEDED
----------------------------
    mean over NONZERO pairs only  = "when there IS an edge, how strong is it?"
    mean over ALL pairs incl. 0s  = exactly the DENSITY, which is what gamma is
                                    compared against in the CPM move rule

and they factor:

    density = (fraction of pairs that have an edge) * (mean weight of those edges)

so this also says WHERE the state signal lives -- in which pairs are connected at all,
or in how strongly they are connected.

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/measure_affinity_distribution.py
Cost: CPU only, a few minutes.
"""
import modal

app = modal.App("scproto-repro")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("numpy", "scipy", "pandas", "anndata", "scanpy", "h5py")
)

volume = modal.Volume.from_name("scproto-data")

AFF = "/vol/code/graphs/affinity_s28nsc58423_ncomp50_kneighbors50_bk02.pkl"
QUANTILES = [1, 5, 25, 50, 75, 95, 99]


@app.function(image=image, cpu=4, memory=32768, timeout=3600, volumes={"/vol": volume})
def measure():
    import os, sys, pickle
    import numpy as np, pandas as pd, h5py

    os.environ["WANDB_MODE"] = "disabled"
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

    with open(AFF, "rb") as f:
        A = pickle.load(f).tocsr()
    A = A.tocsr(copy=True); A.setdiag(0); A.eliminate_zeros()
    A = A.tocoo()
    N = A.shape[0]
    row, col, dat = A.row, A.col, A.data
    up = row < col
    urow, ucol, udat = row[up], col[up], dat[up]
    m = float(udat.sum())
    print(f"[graph] N={N:,}  undirected edges={len(udat):,}  m={m:,.0f}")

    def report(li, tag):
        ok = li >= 0
        both = ok[urow] & ok[ucol]
        same = both & (li[urow] == li[ucol])
        diff = both & ~same

        cnt = np.bincount(li[ok])
        cnt = cnt[cnt > 0].astype(float)
        n_ok = float(ok.sum())
        pairs_same = float((cnt * (cnt - 1) / 2).sum())
        pairs_all = n_ok * (n_ok - 1) / 2.0
        pairs_diff = pairs_all - pairs_same

        print("\n" + "=" * 78)
        print(f"{tag}   ({len(cnt)} groups, median size {np.median(cnt):.0f}, "
              f"max {cnt.max():,.0f})")
        print("=" * 78)
        for name, mask, npairs in (("SAME state", same, pairs_same),
                                   ("DIFF state", diff, pairs_diff)):
            w = udat[mask]
            n_nz = len(w)
            q = np.percentile(w, QUANTILES) if n_nz else np.zeros(len(QUANTILES))
            frac_nz = n_nz / npairs
            mean_nz = w.mean() if n_nz else 0.0
            mean_all = w.sum() / npairs
            print(f"\n  --- {name} pairs ---")
            print(f"    total pairs (incl. zeros)   = {npairs:>15,.0f}")
            print(f"    pairs WITH an edge          = {n_nz:>15,}")
            print(f"    fraction connected          = {frac_nz:>15.6f}")
            print(f"    NONZERO ONLY : mean={mean_nz:.4f}  median={np.median(w):.4f}"
                  if n_nz else "    NONZERO ONLY : n/a")
            print(f"    NONZERO quantiles {QUANTILES} =")
            print(f"        {np.array2string(q, precision=4, floatmode='fixed')}")
            print(f"    ALL PAIRS (zeros counted) : mean = {mean_all:.6f}   <- THE DENSITY")
            print(f"    ALL PAIRS median            = "
                  f"{0.0 if frac_nz < 0.5 else np.median(w):.6f}"
                  f"   ({'>50% of pairs are zero' if frac_nz < 0.5 else 'majority connected'})")
            print(f"    check: frac_connected * mean_nonzero = "
                  f"{frac_nz * mean_nz:.6f}")

    from interpretable_ssl.datasets.dataset_configs import DATASETS
    adata = sc.read_h5ad(DATASETS["s28nsc"]["path"])

    def codes(v):
        return pd.Categorical(pd.Series(v).astype(str)).codes.astype(np.int64)

    ct = adata.obs["celltypes"].astype(str).values
    ni = adata.obs["niches_2D"].astype(str).values
    report(codes(np.array([f"{a}|{b}" for a, b in zip(ct, ni)])),
           "STATE = celltype x niche  (the ground-truth label classes)")

    mdir = "/vol/models/s28nsc"
    cands = sorted(d for d in os.listdir(mdir)
                   if d.startswith("seacell")
                   and os.path.exists(os.path.join(mdir, d, "seacell_sc.h5ad")))
    pick = next((d for d in cands if "bklam02" in d or "bk02" in d), None)
    if pick:
        sad = sc.read_h5ad(os.path.join(mdir, pick, "seacell_sc.h5ad"))
        lab = pd.Series(sad.obs["SEACell"].astype(str).values,
                        index=sad.obs_names).reindex(adata.obs_names).to_numpy()
        li = np.full(N, -1, dtype=np.int64)
        okm = ~pd.isna(lab)
        li[okm] = codes(lab[okm])
        report(li, f"GROUP = SEACells metacell  [{pick}]  (spatially local)")

    print("\n" + "=" * 78)
    print("WHAT TO LOOK AT")
    print("=" * 78)
    print("  If mean_nonzero is nearly the SAME for same-state and diff-state pairs,")
    print("  then the affinity WEIGHTS carry almost no state information, and all the")
    print("  signal is in WHICH pairs have an edge at all (the kNN structure).")
    print("  gamma is a threshold on a mean that includes the zeros, so it is driven")
    print("  by the connected FRACTION, not by the weights.")


@app.local_entrypoint()
def main():
    measure.remote()
