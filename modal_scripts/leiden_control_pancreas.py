"""THE OBJECTIVE CONTROL, ported to PANCREAS: CPM vs modularity, same optimizer, same K.

WHY
---
On s28nsc/bk02 the head-to-head at matched K=200 was decisive:

    Leiden + CPM          DGE 0.630   23 pairs   jointP 0.530   int_frac 0.5509
    Leiden + modularity   DGE 0.856   18 pairs   jointP 0.534   int_frac 0.5147

Same optimizer, same graph, same K -- so the ONLY difference is the objective, and
modularity's own optimum is a far better partition. This script asks the same question
on pancreas/arbf, which is non-spatial and multi-batch (9 technologies), i.e. the
setting where the two objectives could plausibly rank differently.

WHAT IS AND IS NOT COMPARABLE TO THE s28nsc RUN
-----------------------------------------------
The "DGE" column above is `pearson_r` from `evaluation/niche_dge.py`, which recovers
CELLTYPE x NICHE differential expression. **Pancreas has no niche labels** (obs is
`tech` + `celltype` only), so that exact metric DOES NOT EXIST here and is not reported.

What is reported instead is the metric set the pancreas table uses, computed with the
same formulas as `mc_metric_utils.py`:

  purity          per metacell, majority celltype fraction; mean and SIZE-WEIGHTED mean
                  (the paper table uses the weighted key -- see EXPERIMENTS 11g)
  batch_entropy   per metacell, -sum(p * log p) over `tech`; mean and size-weighted
  mod_conn        Newman weighted modularity on ad.obsp['connectivities'] at res 1,
                  which is what compute_modularity() reports -- NOT the arbf graph
  mod_arbf        modularity on the arbf affinity graph the LOSS actually uses, at
                  resolution 1, reported separately so the two graphs are never conflated
  coverage        fraction of the 14 celltypes that are the majority label of >=1 group
  int_frac        share of total affinity weight that falls inside communities
  sizes           min / median / max -- CPM's singleton problem shows up here

Run LOCALLY (free, CPU, both inputs are on this machine):
    python modal_scripts/leiden_control_pancreas.py
"""
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp

AFF = (r"G:\.shortcut-targets-by-id\1nUSrrvX0QPGoRflBV8uoFtis4P4zyk_3\codes"
       r"\interpretable-prototype\graphs"
       r"\affinity_pancreas16382_ncomp50_kneighbors50_arbf.pkl")
H5AD = (r"G:\.shortcut-targets-by-id\1AWk3HED5vjpJuz_mnw4pundDJHzdL99K\data"
        r"\pancreas_hvg.h5ad")
TARGET_K = 220          # scProto/SEACells K on pancreas
SEED = 42
GAMMA_DERIVED = 0.042   # the K-matched gamma from EXPERIMENTS 11i


def main():
    import igraph as ig
    import leidenalg as la
    import anndata as ad_io

    A = pickle.load(open(AFF, "rb")).tocsr()
    A = A.tocsr(copy=True)
    A.setdiag(0)
    A.eliminate_zeros()
    T = sp.triu(A, k=1).tocoo()
    m = float(T.sum())
    N = A.shape[0]
    print(f"[graph] N={N:,}  undirected edges={T.nnz:,}  total weight m={m:,.0f}")

    g = ig.Graph(n=N, edges=list(zip(T.row.tolist(), T.col.tolist())), directed=False)
    g.es["weight"] = T.data.tolist()
    print(f"[igraph] {g.vcount():,} vertices, {g.ecount():,} edges")

    adata = ad_io.read_h5ad(H5AD)
    assert adata.n_obs == N, f"graph N={N} != adata n_obs={adata.n_obs}"
    ct = adata.obs["celltype"].astype(str).values
    tech = adata.obs["tech"].astype(str).values
    ccodes = pd.Categorical(ct).codes.astype(np.int64)
    n_ct = int(ccodes.max()) + 1
    print(f"[labels] {n_ct} cell types, {len(set(tech))} technologies\n")

    # Newman modularity on the SAME graph compute_modularity() uses
    C = sp.csr_matrix(adata.obsp["connectivities"])
    C = (C + C.T) / 2
    c_deg = np.asarray(C.sum(axis=1)).ravel()
    c_two_m = c_deg.sum()

    def modularity_conn(li):
        Q = 0.0
        for c in np.unique(li):
            mask = li == c
            e_k = C[mask][:, mask].sum()
            d_k = c_deg[mask].sum()
            Q += (e_k - d_k * d_k / c_two_m) / c_two_m
        return float(Q)

    def evaluate(memb, name, extra=""):
        li = np.asarray(memb, dtype=np.int64)
        same = li[T.row] == li[T.col]
        int_frac = float(T.data[same].sum()) / m

        df = pd.DataFrame({"_mc": li, "celltype": ct, "tech": tech})
        grp = df.groupby("_mc")
        sizes = grp.size()
        pur = grp["celltype"].apply(lambda s: s.value_counts(normalize=True).iloc[0])

        def _ent(s):
            p = s.value_counts(normalize=True).values
            return float(-np.sum(p * np.log(p + 1e-10)))

        ent = grp["tech"].apply(_ent)
        w = sizes.reindex(pur.index).astype(float)
        wsum = w.sum()
        wpur = float((pur * w).sum() / wsum)
        went = float((ent * w).sum() / wsum)
        cov = grp["celltype"].agg(lambda x: x.mode()[0]).nunique() / n_ct

        return dict(name=name, K=int(len(sizes)), int_frac=int_frac,
                    pur=float(pur.mean()), wpur=wpur,
                    ent=float(ent.mean()), went=went,
                    mod_conn=modularity_conn(li),
                    mod_arbf=float(g.modularity(li, weights="weight")),
                    cov=cov, med=float(sizes.median()),
                    mn=int(sizes.min()), mx=int(sizes.max()), extra=extra)

    rows = []

    def search(fn, lo, hi, label, n_iter=20):
        best = None
        for _ in range(n_iter):
            mid = (lo + hi) / 2
            part = fn(mid)
            memb = part.membership if hasattr(part, "membership") else part
            k = len(set(memb))
            if best is None or abs(k - TARGET_K) < abs(best[1] - TARGET_K):
                best = (memb, k, mid)
            if k > TARGET_K:
                hi = mid
            elif k < TARGET_K:
                lo = mid
            else:
                break
        print(f"  {label}: settled at resolution={best[2]:.5f} -> K={best[1]}")
        return best

    print(f"[1/4] Leiden + CPM at the derived gamma = {GAMMA_DERIVED} (K is free)")
    p = la.find_partition(g, la.CPMVertexPartition, weights="weight",
                          resolution_parameter=GAMMA_DERIVED, seed=SEED, n_iterations=2)
    rows.append(evaluate(p.membership, f"Leiden-CPM g={GAMMA_DERIVED}", "derived gamma"))
    print(f"  -> K={rows[-1]['K']}")

    print(f"[2/4] Leiden + CPM, gamma searched for K~{TARGET_K}")
    memb, k, res = search(
        lambda r: la.find_partition(g, la.CPMVertexPartition, weights="weight",
                                    resolution_parameter=r, seed=SEED, n_iterations=2),
        1e-5, 0.5, "Leiden-CPM")
    rows.append(evaluate(memb, f"Leiden-CPM K~{TARGET_K}", f"gamma={res:.5f}"))

    print(f"[3/4] Leiden + modularity (RBConfiguration), resolution for K~{TARGET_K}")
    memb, k, res = search(
        lambda r: la.find_partition(g, la.RBConfigurationVertexPartition, weights="weight",
                                    resolution_parameter=r, seed=SEED, n_iterations=2),
        0.1, 400.0, "Leiden-modularity")
    rows.append(evaluate(memb, f"Leiden-modularity K~{TARGET_K}", f"res={res:.3f}"))

    print(f"[4/4] Louvain (igraph multilevel), resolution for K~{TARGET_K}")
    memb, k, res = search(
        lambda r: g.community_multilevel(weights="weight", resolution=r).membership,
        0.1, 400.0, "Louvain")
    rows.append(evaluate(memb, f"Louvain K~{TARGET_K}", f"res={res:.3f}"))

    print("\n" + "=" * 126)
    print("OBJECTIVE CONTROL on pancreas / arbf -- CPM vs modularity, same optimizer, "
          f"same K~{TARGET_K}")
    print("=" * 126)
    print(f"{'method':<28}{'K':>5}{'purity':>8}{'w_pur':>8}{'b_ent':>8}{'w_ent':>8}"
          f"{'mod_conn':>10}{'mod_arbf':>10}{'int_frac':>10}{'cov':>7}"
          f"{'med':>6}{'min':>5}{'max':>6}  note")
    for r in rows:
        print(f"{r['name']:<28}{r['K']:>5}{r['pur']:>8.4f}{r['wpur']:>8.4f}"
              f"{r['ent']:>8.4f}{r['went']:>8.4f}{r['mod_conn']:>10.4f}"
              f"{r['mod_arbf']:>10.4f}{r['int_frac']:>10.4f}{r['cov']:>7.3f}"
              f"{r['med']:>6.0f}{r['mn']:>5d}{r['mx']:>6d}  {r['extra']}")
    print("-" * 126)
    print("scProto references (from metrics.json; 'purity'/'batch_entropy' there are the")
    print("UNWEIGHTED means, so compare them to the purity / b_ent columns):")
    print(f"{'  published loss (repro)':<28}{220:>5}{0.9041:>8.4f}{'--':>8}"
          f"{0.4220:>8.4f}{'--':>8}{0.6924:>10.4f}{'--':>10}{'--':>10}{1.000:>7.3f}")
    print(f"{'  modularity GLOBAL l=10':<28}{220:>5}{0.9703:>8.4f}{'--':>8}"
          f"{0.2471:>8.4f}{'--':>8}{0.5070:>10.4f}{'--':>10}{'--':>10}{0.786:>7.3f}")
    print(f"{'  CPM g=0.042 l=50':<28}{220:>5}{0.9030:>8.4f}{'--':>8}"
          f"{1.0907:>8.4f}{'--':>8}{0.4652:>10.4f}{'--':>10}{'--':>10}{0.929:>7.3f}")
    print("\nHOW TO READ IT")
    print("  This compares the two OBJECTIVES' own optima. It says nothing about scProto's")
    print("  optimizer -- that is the separate question of whether scProto can reach them.")
    print("  NOTE: no DGE column. The s28nsc DGE metric is celltype x NICHE recovery and")
    print("  pancreas has no niche labels, so it is undefined here.")

    out = "my-tasks/context/leiden_control_pancreas.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
