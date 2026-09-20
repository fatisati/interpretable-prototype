"""STEP 2 of my-tasks/todo/modularity-loss-plan.txt -- are the modularity estimators
unbiased under OUR sampler?

This is the same gate that caught both CPM bugs ([D9]): a batch is 1024 EDGES drawn with
replacement PROPORTIONAL TO w, and its cells are their deduplicated endpoints, so nothing
in a batch is a uniform sample. Simulate the real sampler against a KNOWN partition and
compare each estimator to the truth, binned by prototype size.

THE THREE QUANTITIES (decisions.md 18)

  affinity_inside[k] = sum over ORDERED pairs of affinity[i,j]*assign[i,k]*assign[j,k]
      estimator: (total_degree / n_sampled_edges) * sum over sampled edges of
                 assign[head,k]*assign[tail,k]          <- NO affinity factor

  volume[k] = sum over ALL cells of weighted_degree[i]*assign[i,k]
      estimator: Horvitz-Thompson,
                 sum over batch cells of weighted_degree[i]*assign[i,k]/prob_in_batch[i]

  volume[k]^2
      NAIVE      : square the estimate         -- expected to be biased UP
      EMA-CORRECT: live estimate * running average from PREVIOUS batches (independent,
                   so the product is unbiased)

PASS = ratios near 1.00 AND flat across prototype size.

Run LOCALLY (free): python modal_scripts/measure_modularity_estimator.py
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
SEACELL = (r"C:\Users\USER\AppData\Local\Temp\claude"
           r"\G---shortcut-targets-by-id-1nUSrrvX0QPGoRflBV8uoFtis4P4zyk-3-codes-"
           r"interpretable-prototype-my-tasks-context"
           r"\7e28d1b0-c718-46ba-8d62-3e54edc33012\scratchpad\seacell_s28.h5ad")

BATCH_EDGES = 1024
N_BATCHES = 400
EMA_ALPHA = 0.9
SEED = 0


def main():
    import anndata as ad, h5py
    from anndata._io.specs.registry import _REGISTRY, IOSpec
    try:
        @_REGISTRY.register_read(h5py.Dataset, IOSpec("null", "0.1.0"))
        def _rn(e, *a, **kw):
            return None
    except TypeError:
        pass

    rng = np.random.default_rng(SEED)
    A = pickle.load(open(AFF, "rb")).tocsr()
    A = A.tocsr(copy=True); A.setdiag(0); A.eliminate_zeros()
    coo = A.tocoo()
    head, tail, w = coo.row, coo.col, coo.data.astype(np.float64)
    n_cells = A.shape[0]
    weighted_degree = np.asarray(A.sum(axis=1)).ravel()
    total_degree = float(weighted_degree.sum())
    print(f"[graph] n_cells={n_cells:,}  directed entries={len(w):,}  "
          f"total_degree={total_degree:,.0f}")

    prob_in_batch = 1.0 - np.power(
        1.0 - 2.0 * weighted_degree / total_degree, BATCH_EDGES)
    prob_in_batch = np.maximum(prob_in_batch, np.percentile(prob_in_batch, 5))
    print(f"[HT] prob_in_batch: mean={prob_in_batch.mean():.4f} "
          f"min={prob_in_batch.min():.4f} max={prob_in_batch.max():.4f}")

    obs = ad.read_h5ad(H5AD, backed="r").obs
    partitions = {}
    try:
        sad = ad.read_h5ad(SEACELL, backed="r")
        lab = pd.Series(sad.obs["SEACell"].astype(str).values,
                        index=sad.obs_names).reindex(obs.index).to_numpy()
        partitions["SEACells K=200 (median 258)"] = pd.Categorical(lab).codes.astype(np.int64)
    except Exception as e:
        print(f"[warn] SEACells labels unavailable: {e}")
    joint = (obs["celltypes"].astype(str) + "|" + obs["niches_2D"].astype(str))
    partitions["celltype x niche (median 58, min 1)"] = pd.Categorical(joint).codes.astype(np.int64)

    p_edge = w / total_degree

    for pname, comm in partitions.items():
        K = int(comm.max()) + 1
        assign = np.zeros((n_cells, K))
        assign[np.arange(n_cells), comm] = 1.0

        # ---- truth, full graph
        true_affinity_inside = np.einsum('ik,ik->k', assign, A @ assign)
        true_volume = assign.T @ weighted_degree
        true_volume_sq = true_volume ** 2
        size = np.bincount(comm, minlength=K).astype(float)

        # ---- simulate
        acc = {k: np.zeros(K) for k in
               ("affinity_inside", "volume", "vol_sq_naive", "vol_sq_ema")}
        volume_ema = None
        for _ in range(N_BATCHES):
            idx = rng.choice(len(w), size=BATCH_EDGES, replace=True, p=p_edge)
            h_b, t_b = head[idx], tail[idx]
            cells = np.unique(np.concatenate([h_b, t_b]))

            # one-hot assignments -> assign[head,k]*assign[tail,k] is 1 exactly when
            # both endpoints are in k, so this is "count same-community sampled edges"
            same = comm[h_b] == comm[t_b]
            acc["affinity_inside"] += (total_degree / BATCH_EDGES) * np.bincount(
                comm[h_b][same], minlength=K).astype(float)

            vol_est = np.bincount(
                comm[cells],
                weights=weighted_degree[cells] / prob_in_batch[cells],
                minlength=K)
            acc["volume"] += vol_est
            acc["vol_sq_naive"] += vol_est ** 2
            # EMA from PREVIOUS batches only -> independent of vol_est -> unbiased product
            acc["vol_sq_ema"] += vol_est * (volume_ema if volume_ema is not None else vol_est)
            volume_ema = (vol_est if volume_ema is None
                          else EMA_ALPHA * volume_ema + (1 - EMA_ALPHA) * vol_est)

        for k in acc:
            acc[k] /= N_BATCHES

        print("\n" + "=" * 96)
        print(f"{pname}   K={K}")
        print("=" * 96)
        bins = [0, 25, 50, 100, 200, 400, 10 ** 9]
        names = ["<25", "25-50", "50-100", "100-200", "200-400", ">400"]
        sb = pd.cut(size, bins, labels=names, right=False)
        rows = [("affinity_inside", acc["affinity_inside"], true_affinity_inside),
                ("volume  [Horvitz-Thompson]", acc["volume"], true_volume),
                ("volume^2 NAIVE (square est)", acc["vol_sq_naive"], true_volume_sq),
                ("volume^2 EMA  <-- PROPOSED", acc["vol_sq_ema"], true_volume_sq)]
        print(f"  {'estimator':<30}{'total':>8}" + "".join(f"{n:>9}" for n in names))
        print(f"  {'n prototypes':<30}{K:>8}" +
              "".join(f"{int((sb == n).sum()):>9}" for n in names))
        for nm, est, true in rows:
            line = f"  {nm:<30}{est.sum()/true.sum():>8.3f}"
            for n in names:
                m = (sb == n) & (true > 0)
                line += (f"{est[m].sum()/true[m].sum():>9.3f}" if m.sum() else f"{'-':>9}")
            print(line)


if __name__ == "__main__":
    main()
