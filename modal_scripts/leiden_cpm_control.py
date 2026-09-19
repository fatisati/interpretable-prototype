"""THE DISCRIMINATING CONTROL: run OUR OWN objective with a REAL optimizer.

THE QUESTION
------------
scProto reaches internal_fraction 0.29-0.36 on s28nsc/bk02. SEACells reaches 0.4465,
and the ground-truth celltype x niche partition itself only reaches 0.4549 -- so
SEACells sits at 98% of the ceiling and we sit at ~70%.

Two hypotheses explain that, and they need OPPOSITE fixes:

  (a) the OBJECTIVE cannot express the target -- the encoder sees expression only,
      never space, so no optimizer would find the partition.
  (b) the OBJECTIVE is fine and OUR OPTIMIZER is weak -- we do simultaneous gradient
      steps on soft memberships where Leiden does sequential greedy moves plus
      multi-level aggregation, and we have no analogue of the aggregation step that
      is most of where Louvain's power comes from.

Every experiment so far has added another point to the DGE/coverage frontier. This one
splits the hypothesis space instead:

    Leiden-CPM reaches ~0.45   ->  the objective is fine, (b): our optimizer is the gap
    Leiden-CPM lands near 0.31 ->  the objective cannot express it, (a): features are

THREE OPTIMIZERS, ALL ON THE SAME GRAPH
---------------------------------------
  Leiden + CPM            leidenalg.CPMVertexPartition       <- OUR objective exactly
  Leiden + modularity     leidenalg.RBConfigurationVertexPartition
  Louvain + modularity    igraph community_multilevel

MATCHED K IS ESSENTIAL. CPM chooses its own number of communities, and internal_fraction
rises trivially as communities get smaller -- a partition into singletons would score 0
and one giant community would score 1.0. So each method's resolution is searched until
it yields ~200 communities, matching scProto and SEACells, and the derived gamma=0.028
is ALSO reported on its own so we can see what K our own gamma actually implies.

Run LOCALLY (free, CPU, both inputs are on this machine):
    python modal_scripts/leiden_cpm_control.py
"""
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp

AFF = (r"G:\.shortcut-targets-by-id\1nUSrrvX0QPGoRflBV8uoFtis4P4zyk_3\codes"
       r"\interpretable-prototype\graphs"
       r"\affinity_s28nsc58423_ncomp50_kneighbors50_bk02.pkl")
H5AD = (r"G:\.shortcut-targets-by-id\1AWk3HED5vjpJuz_mnw4pundDJHzdL99K\data"
        r"\spatial\NSCLC_3D_section_28.h5ad")
TARGET_K = 200
SEED = 42


def main():
    import igraph as ig
    import leidenalg as la
    import anndata as ad
    import h5py
    from anndata._io.specs.registry import _REGISTRY, IOSpec
    try:
        @_REGISTRY.register_read(h5py.Dataset, IOSpec("null", "0.1.0"))
        def _rn(e, *a, **kw):   # signature varies across anndata versions
            return None
    except TypeError:
        pass

    A = pickle.load(open(AFF, "rb")).tocsr()
    A = A.tocsr(copy=True); A.setdiag(0); A.eliminate_zeros()
    T = sp.triu(A, k=1).tocoo()
    m = float(T.sum())
    N = A.shape[0]
    print(f"[graph] N={N:,}  undirected edges={T.nnz:,}  m={m:,.0f}")

    g = ig.Graph(n=N, edges=list(zip(T.row.tolist(), T.col.tolist())), directed=False)
    g.es["weight"] = T.data.tolist()
    print(f"[igraph] {g.vcount():,} vertices, {g.ecount():,} edges")

    obs = ad.read_h5ad(H5AD, backed="r").obs
    ct = obs["celltypes"].astype(str).values
    ni = obs["niches_2D"].astype(str).values
    joint = np.array([f"{a}|{b}" for a, b in zip(ct, ni)])
    jcodes = pd.Categorical(joint).codes.astype(np.int64)
    ccodes = pd.Categorical(ct).codes.astype(np.int64)
    n_states = jcodes.max() + 1
    print(f"[labels] {ccodes.max()+1} cell types, {n_states} celltype x niche states\n")

    def evaluate(memb, name, extra=""):
        li = np.asarray(memb, dtype=np.int64)
        same = li[T.row] == li[T.col]
        int_frac = float(T.data[same].sum()) / m
        sizes = np.bincount(li)
        sizes = sizes[sizes > 0].astype(float)
        pairs = float((sizes * (sizes - 1) / 2).sum())
        dens = float(T.data[same].sum()) / pairs if pairs else float("nan")

        # joint purity, micro (each CELL counts equally) -- matches purity_micro elsewhere
        def purity(codes):
            tot = 0
            for c in np.unique(li):
                sel = codes[li == c]
                if len(sel):
                    tot += np.bincount(sel).max()
            return tot / len(li)

        # how many distinct states are the MAJORITY label of at least one community
        maj = set()
        for c in np.unique(li):
            sel = jcodes[li == c]
            if len(sel):
                maj.add(int(np.bincount(sel).argmax()))
        return dict(name=name, K=len(sizes), int_frac=int_frac, density=dens,
                    med=float(np.median(sizes)), mn=float(sizes.min()),
                    mx=float(sizes.max()),
                    mxmed=float(sizes.max() / np.median(sizes)),
                    jointP=purity(jcodes), ctP=purity(ccodes),
                    states=len(maj), extra=extra)

    rows = []

    def search(fn, lo, hi, label, n_iter=18):
        """Bisect a resolution parameter until the partition has ~TARGET_K communities."""
        best = None
        for _ in range(n_iter):
            mid = (lo + hi) / 2
            part = fn(mid)
            k = len(set(part.membership)) if hasattr(part, "membership") else len(set(part))
            memb = part.membership if hasattr(part, "membership") else part
            if best is None or abs(k - TARGET_K) < abs(best[1] - TARGET_K):
                best = (memb, k, mid)
            if k > TARGET_K:
                hi = mid          # too many communities -> lower the resolution
            elif k < TARGET_K:
                lo = mid
            else:
                break
        print(f"  {label}: settled at resolution={best[2]:.5f} -> K={best[1]}")
        return best

    # ---------------- Leiden + CPM, at OUR derived gamma -----------------------
    print("[1/4] Leiden + CPM at the derived gamma = 0.028 (K is whatever it wants)")
    p = la.find_partition(g, la.CPMVertexPartition, weights="weight",
                          resolution_parameter=0.028, seed=SEED, n_iterations=2)
    rows.append(evaluate(p.membership, "Leiden-CPM gamma=0.028", "our derived gamma"))
    print(f"  -> K={rows[-1]['K']}")

    # ---------------- Leiden + CPM, resolution matched to K=200 ----------------
    print(f"[2/4] Leiden + CPM, gamma searched for K~{TARGET_K}")
    memb, k, res = search(
        lambda r: la.find_partition(g, la.CPMVertexPartition, weights="weight",
                                    resolution_parameter=r, seed=SEED, n_iterations=2),
        1e-5, 0.05, "Leiden-CPM")
    rows.append(evaluate(memb, f"Leiden-CPM K~{TARGET_K}", f"gamma={res:.5f}"))

    # ---------------- Leiden + modularity --------------------------------------
    print(f"[3/4] Leiden + modularity (RBConfiguration), resolution for K~{TARGET_K}")
    memb, k, res = search(
        lambda r: la.find_partition(g, la.RBConfigurationVertexPartition,
                                    weights="weight", resolution_parameter=r,
                                    seed=SEED, n_iterations=2),
        0.1, 400.0, "Leiden-modularity")
    rows.append(evaluate(memb, f"Leiden-modularity K~{TARGET_K}", f"res={res:.3f}"))

    # ---------------- Louvain ---------------------------------------------------
    print(f"[4/4] Louvain (igraph multilevel), resolution for K~{TARGET_K}")
    memb, k, res = search(
        lambda r: g.community_multilevel(weights="weight", resolution=r).membership,
        0.1, 400.0, "Louvain")
    rows.append(evaluate(memb, f"Louvain K~{TARGET_K}", f"res={res:.3f}"))

    # ---------------- report ----------------------------------------------------
    print("\n" + "=" * 116)
    print("HARD-OPTIMIZER CONTROL on s28nsc / bk02 -- can a real optimizer do better "
          "with the SAME objective?")
    print("=" * 116)
    h = (f"{'method':<28}{'K':>6}{'int_frac':>10}{'density':>10}{'jointP':>8}"
         f"{'ctP':>8}{'states':>8}{'med':>7}{'min':>6}{'max':>7}{'max/med':>9}  note")
    print(h)
    for r in rows:
        print(f"{r['name']:<28}{r['K']:>6}{r['int_frac']:>10.4f}{r['density']:>10.5f}"
              f"{r['jointP']:>8.3f}{r['ctP']:>8.3f}{r['states']:>8}{r['med']:>7.0f}"
              f"{r['mn']:>6.0f}{r['mx']:>7.0f}{r['mxmed']:>9.1f}  {r['extra']}")
    print("-" * 116)
    print(f"{'SEACells bk02':<28}{200:>6}{0.4465:>10.4f}{0.02952:>10.5f}"
          f"{0.504:>8.3f}{0.823:>8.3f}{19:>8}{258:>7}{'--':>6}{925:>7}{3.6:>9.1f}"
          f"  stored run")
    print(f"{'scProto best (lf_L16_cg02)':<28}{200:>6}{0.3338:>10.4f}{0.02593:>10.5f}"
          f"{0.437:>8.3f}{'--':>8}{13:>8}{259:>7}{'--':>6}{'--':>7}{2.7:>9.1f}"
          f"  gradient descent")
    print(f"{'GROUND TRUTH ct x niche':<28}{175:>6}{0.4549:>10.4f}{'--':>10}"
          f"{1.000:>8.3f}{1.000:>8.3f}{'--':>8}{58:>7}{'--':>6}{'--':>7}{'--':>9}"
          f"  the ceiling")
    print("\nHOW TO READ IT")
    print("  Leiden-CPM at K~200 well ABOVE scProto's 0.3338 -> the OBJECTIVE is fine")
    print("     and our OPTIMIZER is the gap (simultaneous soft updates, no aggregation).")
    print("  Leiden-CPM near 0.33                            -> the objective cannot")
    print("     express the target on this graph; the gap is upstream, in the features.")
    print("  'states' = how many distinct celltype x niche states are the MAJORITY label")
    print("     of at least one community -- the closest free proxy for DGE n_pairs.")


if __name__ == "__main__":
    main()
