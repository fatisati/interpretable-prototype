"""WHICH LOSS ACTUALLY PREDICTS WHAT WE WANT? -- scored on real partitions, no training.

THE QUESTION
------------
Every loss term we have is a proxy. We tune them and then judge by joint purity, state
coverage and DGE. EXPERIMENTS 11i already found the proxy and the goal can DISAGREE:
optimizing CPM harder made DGE monotonically WORSE. So before spending another GPU-hour
on a loss term, measure whether it correlates with the thing we actually want.

METHOD
------
Generate ~30 partitions of the SAME graph at a range of K by deliberately varied means --
good ones, coarse ones, corrupted ones, and two constructed pathologies. Score every one
under every candidate loss term AND under the target metrics. Then correlate.

    loss correlates with the target      -> optimizing it should help
    loss is flat or anti-correlated      -> optimizing it CANNOT help, whatever the weight

THE PATHOLOGIES MATTER MOST. A term that ranks good partitions well but cannot see the
specific failure we keep hitting (two prototypes on the same cells) is useless to us.
That is why 'overlap' (split every community in half at random, so the halves occupy the
SAME region) and 'coarse' (merge pairs) are included explicitly.

WHAT IS SCORED
--------------
targets     joint_purity   micro purity on celltype x niche -- the headline goal
            n_states       distinct celltype x niche states that are the majority
                           label of at least one community -- the coverage proxy
            int_frac       share of total affinity weight inside communities

candidates  ratio_assoc    sum_k affinity_within[k]/size[k]        HIGHER better
            containment    mean_k affinity_within[k]/volume[k]     HIGHER better  (published diagonal)
            cohesion_gram  2-hop within-similarity, per cell       HIGHER better
            modularity     igraph, resolution 1                    HIGHER better
            cpm_g028       edges_inside - gamma*pairs              HIGHER better
            offdiag_sumsq  mean_k sum_j leakage^2                  LOWER better  (current)
            offdiag_hinge  mean_k sum_j max(0,leakage-thr)^2       LOWER better  (proposed)
            offdiag_max    mean_k max_j leakage                    LOWER better

Run LOCALLY (free, CPU):  python modal_scripts/objective_selection_study.py
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
OUT = "my-tasks/context/objective_selection_study.csv"
SEED = 42


def main():
    import igraph as ig
    import leidenalg as la
    import anndata as ad_io
    import h5py
    from anndata._io.specs.registry import _REGISTRY, IOSpec
    from scipy.stats import spearmanr
    try:
        @_REGISTRY.register_read(h5py.Dataset, IOSpec("null", "0.1.0"))
        def _rn(e, *a, **kw):
            return None
    except TypeError:
        pass

    A = pickle.load(open(AFF, "rb")).tocsr()
    A = A.tocsr(copy=True); A.setdiag(0); A.eliminate_zeros(); A = A.astype(np.float64)
    N = A.shape[0]
    deg = np.asarray(A.sum(1)).ravel()
    T = sp.triu(A, k=1).tocoo()
    m_tot = float(T.sum())
    rng = np.random.default_rng(SEED)

    g = ig.Graph(n=N, edges=list(zip(T.row.tolist(), T.col.tolist())), directed=False)
    g.es["weight"] = T.data.tolist()
    print(f"[graph] N={N:,} edges={T.nnz:,}")

    obs = ad_io.read_h5ad(H5AD, backed="r").obs
    joint = np.array([f"{a}|{b}" for a, b in
                      zip(obs["celltypes"].astype(str), obs["niches_2D"].astype(str))])
    jcodes = pd.Categorical(joint).codes.astype(np.int64)
    print(f"[labels] {jcodes.max()+1} celltype x niche states")

    # ---- 2-hop Gram on a fixed edge-sampled subset (the real batch construction) ----
    nrm = np.sqrt(A.multiply(A).sum(1)).A.ravel(); nrm[nrm == 0] = 1.0
    An = sp.csr_matrix(sp.diags(1.0 / nrm) @ A)
    e = rng.choice(T.nnz, 2048, replace=False)
    sub = np.unique(np.concatenate([T.row[e], T.col[e]]))
    R = An[sub]
    Gram = np.asarray((R @ R.T).todense())
    print(f"[gram] {len(sub)} edge-sampled cells for the 2-hop term")

    # ---- partitions -------------------------------------------------------------
    parts = {}

    def leiden(part_cls, res, tag):
        p = la.find_partition(g, part_cls, weights="weight",
                              resolution_parameter=res, seed=SEED, n_iterations=2)
        parts[tag] = np.asarray(p.membership)

    for r in [8.0, 12.0, 15.0, 19.431, 24.0, 30.0, 40.0]:
        leiden(la.RBConfigurationVertexPartition, r, f"leiden_mod_r{r:g}")
    for gam in [0.010, 0.020, 0.028, 0.040, 0.060]:
        leiden(la.CPMVertexPartition, gam, f"leiden_cpm_g{gam:g}")
    for r in [12.0, 19.431, 30.0]:
        parts[f"louvain_r{r:g}"] = np.asarray(
            g.community_multilevel(weights="weight", resolution=r).membership)

    base = parts["leiden_mod_r19.431"]
    for frac in [0.05, 0.10, 0.20, 0.40, 0.70]:
        c = base.copy()
        sel = rng.random(N) < frac
        c[sel] = rng.integers(0, base.max() + 1, sel.sum())
        parts[f"corrupt_{int(frac*100)}pct"] = c
    # PATHOLOGY 1: two prototypes on the SAME cells
    ov = base.copy()
    for k in np.unique(base):
        mem = np.where(base == k)[0]
        ov[rng.permutation(mem)[:len(mem) // 2]] = ov.max() + 1
    parts["PATHOLOGY_overlap"] = ov
    # PATHOLOGY 2: merged / too coarse
    mg = base.copy()
    for k in range(0, base.max() + 1, 2):
        mg[mg == k + 1] = k
    parts["PATHOLOGY_coarse"] = mg
    parts["random_K200"] = rng.integers(0, 200, N)
    parts["GROUND_TRUTH_ctxniche"] = jcodes.copy()
    print(f"[partitions] {len(parts)} generated")

    # hinge threshold: the p95 leakage of a GOOD partition = "normal adjacency"
    def M_of(l):
        lab = np.unique(l, return_inverse=True)[1]
        K = lab.max() + 1
        S = sp.csr_matrix((np.ones(N), (np.arange(N), lab)), shape=(N, K))
        AK = (S.T @ A @ S).toarray()
        vol = np.asarray(S.T @ deg).ravel()
        cnt = np.asarray(S.sum(0)).ravel()
        ok = vol > 0
        AKo, volo, cnto = AK[np.ix_(ok, ok)], vol[ok], cnt[ok]
        M = AKo / np.sqrt(np.outer(volo, volo))
        w = np.diag(AKo).copy()
        np.fill_diagonal(M, 0.0)
        return M, w, volo, cnto, lab
    Mb, *_ = M_of(base)
    THR = float(np.quantile(Mb[Mb > 0], 0.95))
    print(f"[hinge] threshold = p95 leakage of the reference partition = {THR:.5f}\n")

    rows = []
    for tag, l in parts.items():
        M, w, vol, cnt, lab = M_of(l)
        K = len(cnt)
        same = lab[T.row] == lab[T.col]
        int_frac = float(T.data[same].sum()) / m_tot
        # targets
        jp = sum(np.bincount(jcodes[lab == c]).max() for c in np.unique(lab)) / N
        states = len({int(np.bincount(jcodes[lab == c]).argmax()) for c in np.unique(lab)})
        # candidate losses
        sizes = cnt.astype(float)
        ratio_assoc = float((w / np.maximum(sizes, 1e-9)).sum() / N)
        containment = float((w / np.maximum(vol, 1e-9)).mean())
        modular = float(g.modularity(lab, weights="weight"))
        pairs = float((sizes * (sizes - 1) / 2).sum())
        cpm = float((T.data[same].sum() - 0.028 * pairs) / m_tot)
        od_sumsq = float((M ** 2).sum(1).mean())
        od_hinge = float((np.maximum(M - THR, 0) ** 2).sum(1).mean())
        od_max = float(M.max(1).mean())
        ls = lab[sub]
        Ks = ls.max() + 1
        S = np.zeros((len(sub), Ks)); S[np.arange(len(sub)), ls] = 1.0
        sz = S.sum(0); keep = sz > 0
        coh = float((((Gram @ S) * S).sum(0)[keep] / sz[keep]).sum() / len(sub))
        rows.append(dict(partition=tag, K=K, joint_purity=jp, n_states=states,
                         int_frac=int_frac, ratio_assoc=ratio_assoc,
                         containment=containment, cohesion_gram=coh, modularity=modular,
                         cpm_g028=cpm, offdiag_sumsq=od_sumsq, offdiag_hinge=od_hinge,
                         offdiag_max=od_max))
        print(f"  {tag:<26} K={K:>5} jointP={jp:.3f} states={states:>3}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    # ---------------- correlations ----------------
    LOSSES = [("ratio_assoc", +1), ("containment", +1), ("cohesion_gram", +1),
              ("modularity", +1), ("cpm_g028", +1), ("offdiag_sumsq", -1),
              ("offdiag_hinge", -1), ("offdiag_max", -1)]
    real = df[~df.partition.str.startswith(("PATHOLOGY", "GROUND"))]
    print("\n" + "=" * 92)
    print("Does each loss PREDICT what we want?  spearman(loss oriented so HIGHER=better, target)")
    print("=" * 92)
    print(f"{'loss term':<18}{'joint_purity':>14}{'n_states':>11}{'int_frac':>11}"
          f"{'K':>9}   {'detects OVERLAP?':>18}")
    base_row = df[df.partition == "leiden_mod_r19.431"].iloc[0]
    ov_row = df[df.partition == "PATHOLOGY_overlap"].iloc[0]
    for name, sign in LOSSES:
        v = real[name] * sign
        c1 = spearmanr(v, real.joint_purity).statistic
        c2 = spearmanr(v, real.n_states).statistic
        c3 = spearmanr(v, real.int_frac).statistic
        c4 = spearmanr(v, real.K).statistic
        better = (ov_row[name] - base_row[name]) * sign
        verdict = "YES, penalises it" if better < 0 else "NO, prefers it"
        print(f"{name:<18}{c1:>14.3f}{c2:>11.3f}{c3:>11.3f}{c4:>9.3f}   {verdict:>18}")
    print("\nHOW TO READ")
    print("  joint_purity column is the one that matters -- it is the headline goal.")
    print("  A loss with a correlation near 0 or negative CANNOT help however it is weighted.")
    print("  The K column flags a loss that is really just counting clusters.")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
