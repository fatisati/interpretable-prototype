"""MEASURE THE MINIBATCH ESTIMATOR BIAS IN nassoc -- by simulation, not algebra.

WHY THIS EXISTS
---------------
The nassoc loss estimates three full-graph quantities from one minibatch:

    size[p]            = sum over all cells of S[i,p]
    possible_pairs[p]  = sum over pairs i<j of S[i,p]*S[j,p]
    edges_inside[p]    = sum over edges of w_ij * S[i,p]*S[j,p]

CLAUDE.md 9b says a minibatch estimate must be checked for a scale mismatch against
its numerator. Two attempts to settle this algebraically both failed, because the
sampling is NOT uniform:

    * a batch is 1024 EDGES, drawn with replacement PROPORTIONAL TO WEIGHT
    * the cells of the batch are the deduplicated ENDPOINTS of those edges, so a
      cell's inclusion probability grows with its weighted degree
    * nassoc then multiplies by w AGAIN (A_part = S_h.T @ (w_b * S_t)), so the
      effective edge weighting is w^2, while the UMAP term uses w^1

So: simulate the real sampler and compare estimates against ground truth.

METHOD
------
Use SEACells' K=200 partition as a KNOWN hard assignment S (one-hot), so every true
value is computable exactly. Then draw many real minibatches exactly as the trainer
does, apply the trainer's own rescaling, and compare.

Reports bias per prototype against prototype SIZE and MEAN DEGREE, because the
question that matters is not "is there a bias" but "does it hit small / sparse
prototypes differently" -- those are the rare states whose coverage we are missing.

Run LOCALLY (both inputs are on this machine):
    python modal_scripts/measure_estimator_bias.py
"""
import os
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp

AFF = (r"G:\.shortcut-targets-by-id\1nUSrrvX0QPGoRflBV8uoFtis4P4zyk_3\codes"
       r"\interpretable-prototype\graphs"
       r"\affinity_s28nsc58423_ncomp50_kneighbors50_bk02.pkl")
H5AD = (r"G:\.shortcut-targets-by-id\1AWk3HED5vjpJuz_mnw4pundDJHzdL99K\data"
        r"\spatial\NSCLC_3D_section_28.h5ad")

BATCH_EDGES = 1024      # trainer's batch_size, applied to EDGES
N_BATCHES = 400         # enough to average out noise and expose systematic bias
N_EPOCHS_THRESH = 200   # EdgeDataset's weight_threshold = max_w / n_epochs
SEED = 0


def main():
    rng = np.random.default_rng(SEED)

    # ---------------------------------------------------------------- graph
    with open(AFF, "rb") as f:
        A = pickle.load(f).tocsr()
    A = A.tocsr(copy=True)
    A.setdiag(0)
    A.eliminate_zeros()
    N = A.shape[0]
    coo = A.tocoo()
    head, tail, w = coo.row, coo.col, coo.data.astype(np.float64)

    # EdgeDataset drops very weak edges exactly this way
    keep = w >= (w.max() / float(N_EPOCHS_THRESH))
    head, tail, w = head[keep], tail[keep], w[keep]
    n_total_edges = len(w)
    W = w.sum()                                   # = 2m over kept directed entries
    print(f"[graph] N={N:,}  directed entries kept={n_total_edges:,}  "
          f"sum(w)=2m={W:,.0f}  mean w={w.mean():.4f}")

    d_all = np.bincount(head, weights=w, minlength=N)

    # ---------------------------------------------------------------- partition
    import anndata as ad
    obs = ad.read_h5ad(H5AD, backed="r").obs
    labels = _seacell_labels(obs.index)
    if labels is None:
        # fall back to celltype x niche so the script still runs
        lab = (obs["celltypes"].astype(str) + "|" + obs["niches_2D"].astype(str)).values
        print("[partition] SEACells labels unavailable -> using celltype x niche")
    else:
        lab = labels
        print("[partition] using SEACells K=200")
    li = pd.Categorical(pd.Series(lab).astype(str)).codes.astype(np.int64)
    K = int(li.max()) + 1

    # ---------------------------------------------------------------- truth
    size_true = np.bincount(li, minlength=K).astype(np.float64)
    pairs_true = size_true * (size_true - 1) / 2.0
    same = li[head] == li[tail]
    edges_true = np.bincount(li[head[same]], weights=w[same], minlength=K) / 2.0
    deg_mean = np.bincount(li, weights=d_all, minlength=K) / np.maximum(size_true, 1)
    print(f"[truth] K={K}  median size={np.median(size_true):.0f}  "
          f"total pairs={pairs_true.sum():,.0f}  edges_inside={edges_true.sum():,.0f}")

    # ---------------------------------------------------------------- simulate
    p_edge = w / W
    # Horvitz-Thompson inclusion probability. Each draw touches 2 cells; cell i is an
    # endpoint of directed entries totalling 2*d_i of the weight mass, so
    #   P(one draw touches i) = 2*d_i/W,  P(i in batch) = 1 - (1 - 2*d_i/W)^E
    p_in = 1.0 - np.power(1.0 - 2.0 * d_all / W, BATCH_EDGES)
    p_in = np.maximum(p_in, np.percentile(p_in[p_in > 0], 5))   # floor: cap 1/p variance
    print(f"[HT] inclusion prob: mean={p_in.mean():.4f} min={p_in.min():.4f} "
          f"max={p_in.max():.4f}  -> expected unique cells = {p_in.sum():.0f}")

    acc = {k: np.zeros(K) for k in
           ("size_code", "size_ht", "pairs_code", "pairs_c2", "pairs_pairscale",
            "pairs_ht", "edges_code", "edges_w1")}
    n_cells_seen = []

    for b in range(N_BATCHES):
        idx = rng.choice(n_total_edges, size=BATCH_EDGES, replace=True, p=p_edge)
        h_b, t_b, w_b = head[idx], tail[idx], w[idx]
        cells = np.unique(np.concatenate([h_b, t_b]))
        n_b = len(cells)
        n_cells_seen.append(n_b)
        c = N / n_b

        # --- size / sumsq over the batch's unique cells (hard S -> sumsq == size)
        size_b = np.bincount(li[cells], minlength=K).astype(np.float64)
        sumsq_b = size_b.copy()

        acc["size_code"] += size_b * c

        # what the code computes: size_eff^2 - c*sumsq_b
        acc["pairs_code"] += 0.5 * np.maximum((size_b * c) ** 2 - c * sumsq_b, 0)
        # the c^2 variant that [D7] replaced
        acc["pairs_c2"] += 0.5 * np.maximum((size_b * c) ** 2 - c * c * sumsq_b, 0)
        # pair-count rescale: batch-native pairs * N(N-1)/(n_b(n_b-1))
        kappa = (N * (N - 1.0)) / (n_b * (n_b - 1.0))
        acc["pairs_pairscale"] += 0.5 * np.maximum(size_b**2 - sumsq_b, 0) * kappa

        # --- Horvitz-Thompson: weight each batch cell by 1/P(it is in the batch).
        # Unbiased for the FULL-GRAPH sums under the real (degree-biased) sampling.
        inv = 1.0 / p_in[cells]
        u_sum = np.bincount(li[cells], weights=inv, minlength=K)
        u_sq = np.bincount(li[cells], weights=inv**2, minlength=K)
        acc["size_ht"] += u_sum
        acc["pairs_ht"] += 0.5 * np.maximum(u_sum**2 - u_sq, 0)

        # --- edges_inside
        same_b = li[h_b] == li[t_b]
        # what the code computes: sum of w over sampled same-proto edges, scaled by
        # EDGE COUNT ratio -- weight applied a second time on top of ∝w sampling
        e_w2 = np.bincount(li[h_b[same_b]], weights=w_b[same_b], minlength=K) / 2.0
        acc["edges_code"] += e_w2 * (n_total_edges / BATCH_EDGES)
        # unbiased under ∝w sampling: COUNT sampled same-proto edges, scale by W/n
        e_w1 = np.bincount(li[h_b[same_b]], minlength=K).astype(np.float64) / 2.0
        acc["edges_w1"] += e_w1 * (W / BATCH_EDGES)

    for k in acc:
        acc[k] /= N_BATCHES
    print(f"[sim] {N_BATCHES} batches of {BATCH_EDGES} edges -> "
          f"mean unique cells/batch = {np.mean(n_cells_seen):.0f}  "
          f"(c = N/n_b = {N / np.mean(n_cells_seen):.1f})")

    # ---------------------------------------------------------------- report
    def ratio(est, true):
        ok = true > 0
        return est[ok] / true[ok], ok

    print("\n" + "=" * 78)
    print("AGGREGATE BIAS  (estimate / truth, 1.00 = unbiased)")
    print("=" * 78)
    rows = [
        ("size      [what the code does]", acc["size_code"], size_true),
        ("size      [Horvitz-Thompson]", acc["size_ht"], size_true),
        ("pairs     [code: c^2*size^2 - c*sumsq]", acc["pairs_code"], pairs_true),
        ("pairs     [pre-D7: c^2 on both]", acc["pairs_c2"], pairs_true),
        ("pairs     [pair-count rescale]", acc["pairs_pairscale"], pairs_true),
        ("pairs     [Horvitz-Thompson]  <-- PROPOSED", acc["pairs_ht"], pairs_true),
        ("edges_in  [code: w applied twice]", acc["edges_code"], edges_true),
        ("edges_in  [count-only, W/2E]  <-- PROPOSED", acc["edges_w1"], edges_true),
    ]
    for name, est, true in rows:
        r, ok = ratio(est, true)
        print(f"  {name:<42} total={est.sum()/true.sum():6.3f}   "
              f"median={np.median(r):6.3f}   p10={np.percentile(r,10):6.3f}   "
              f"p90={np.percentile(r,90):6.3f}")

    print("\n" + "=" * 78)
    print("IS THE BIAS SIZE-DEPENDENT?  (bias by prototype-size quartile)")
    print("=" * 78)
    q = pd.qcut(size_true, 4, labels=["Q1 smallest", "Q2", "Q3", "Q4 largest"])
    for name, est, true in rows:
        line = f"  {name:<42}"
        for lvl in ["Q1 smallest", "Q2", "Q3", "Q4 largest"]:
            msk = (q == lvl) & (true > 0)
            line += f" {est[msk].sum()/true[msk].sum():6.3f}"
        print(line)
    print(f"  {'':<42} " + " ".join(f"{s:>6}" for s in ["Q1", "Q2", "Q3", "Q4"]))
    print(f"\n  median size per quartile: " +
          ", ".join(f"{lvl}={np.median(size_true[q==lvl]):.0f}"
                    for lvl in ["Q1 smallest", "Q2", "Q3", "Q4 largest"]))

    # Our real prototypes sit at ~292 cells (SEACells: median 258, max 925). The label
    # classes used here span 8..5856, so bin by ABSOLUTE size and read off that range.
    print("\n" + "=" * 78)
    print("BIAS BY ABSOLUTE PROTOTYPE SIZE  (our prototypes live in the 200-900 band)")
    print("=" * 78)
    edges_bin = [0, 25, 50, 100, 200, 300, 500, 1000, 10**9]
    names = ["<25", "25-50", "50-100", "100-200", "200-300", "300-500",
             "500-1k", ">1k"]
    sb = pd.cut(size_true, edges_bin, labels=names, right=False)
    print(f"  {'':<42}" + "".join(f"{n:>9}" for n in names))
    print(f"  {'n prototypes':<42}" +
          "".join(f"{int((sb == n).sum()):>9}" for n in names))
    for name, est, true in rows:
        line = f"  {name:<42}"
        for n in names:
            msk = (sb == n) & (true > 0)
            line += (f"{est[msk].sum()/true[msk].sum():>9.3f}" if msk.sum()
                     else f"{'-':>9}")
        print(line)

    print("\n" + "=" * 78)
    print("IS THE BIAS DEGREE-DEPENDENT?  (bias by prototype mean-degree quartile)")
    print("=" * 78)
    qd = pd.qcut(deg_mean, 4, labels=["D1 sparsest", "D2", "D3", "D4 densest"])
    for name, est, true in rows:
        line = f"  {name:<42}"
        for lvl in ["D1 sparsest", "D2", "D3", "D4 densest"]:
            msk = (qd == lvl) & (true > 0)
            line += f" {est[msk].sum()/true[msk].sum():6.3f}"
        print(line)
    print(f"\n  median mean-degree per quartile: " +
          ", ".join(f"{lvl}={np.median(deg_mean[qd==lvl]):.1f}"
                    for lvl in ["D1 sparsest", "D2", "D3", "D4 densest"]))

    print("\n" + "=" * 78)
    print("NET EFFECT ON THE LOSS: effective gamma = gamma * (pairs_est/pairs_true)")
    print("                                             / (edges_est/edges_true)")
    print("=" * 78)
    pe = acc["pairs_code"].sum() / pairs_true.sum()
    ee = acc["edges_code"].sum() / edges_true.sum()
    print(f"  pairs inflation  = {pe:.4f}")
    print(f"  edges inflation  = {ee:.4f}")
    print(f"  effective gamma multiplier = {pe/ee:.4f}")
    print(f"  -> gamma=0.026 actually behaves like {0.026*pe/ee:.4f}")


def _seacell_labels(obs_index):
    """Try the local mirror of MODEL_DIR for a SEACells K=200 run; None if absent."""
    import glob
    import anndata as ad
    base = r"G:\.shortcut-targets-by-id\1-0FnoSFlnEGxgKJQCAbspCuHc3osVs25\models\s28nsc"
    if not os.path.isdir(base):
        return None
    for d in sorted(glob.glob(os.path.join(base, "seacell*"))):
        f = os.path.join(d, "seacell_sc.h5ad")
        if not os.path.exists(f):
            continue
        if "bklam02" not in d and "bk02" not in d:
            continue
        try:
            sad = ad.read_h5ad(f, backed="r")
            s = pd.Series(sad.obs["SEACell"].astype(str).values, index=sad.obs_names)
            out = s.reindex(obs_index)
            if out.isna().any():
                return None
            return out.values
        except Exception as e:
            print(f"[partition] could not read {f}: {e}")
            return None
    return None


if __name__ == "__main__":
    main()
