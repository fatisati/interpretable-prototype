"""[D41] Derive lambda_cohesion / lambda_cohesion_aff for ANY dataset.

Spans are NOT portable across graphs (same reason nassoc_gamma is not,
CLAUDE.md 9b), so every dataset needs its own. CPU only.

    span[term]   = value(leiden at K) - value(random at K)     on the MASKED graph
                   with the PER-BATCH denominator and cohesion_self = 0
    lambda[term] = AUTHORITY / span[term]

AUTHORITY = 1.5403 is held fixed from the s28nsc arms so that "does mixing the
two similarities help" is never confounded with "is more cohesion better".

Weight by DYNAMIC RANGE, not by loss value: cohesion_aff sits near 0 (most
affinity entries are 0), so 1 - cohesion_aff is a near-CONSTANT and at equal
lambda it contributes almost nothing to the gradient while looking comparable in
the loss.

cohesion_self MUST be 0 with the per-batch denominator -- the self-pair is
1/size_in_batch, which a batch-SPREAD partition maximises, so at 1 the term pays
the model to smear prototypes thinly across batches. Measured on pancreas that
collapsed the span from 0.107 to 0.0006 and implied lambda = 2396.

Usage:
    python modal_scripts/measure_cohesion_spans_any.py pancreas
    python modal_scripts/measure_cohesion_spans_any.py pbmc-immune lung crcx
"""
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interpretable_ssl.evaluation.metacell_scoreboard import (  # noqa: E402
    cohesion, load_affinity, read_obs)

AUTHORITY = 1.5403

# dataset -> (affinity pkl, h5ad relative to DATA_DIR, batch_key, label_key,
#             niche_key, K)
DATASETS = {
    "pancreas": ("graphs/affinity_pancreas16382_ncomp50_kneighbors50_arbf.pkl",
                 "pancreas_hvg.h5ad", "tech", "celltype", None, 220),
    "pbmc-immune": ("graphs/affinity_pbmc-immune33506_ncomp50_kneighbors50_arbf.pkl",
                    "scpoli/Immune_ALL_human_hvg.h5ad", "study",
                    "final_annotation", None, 300),
    "lung": ("graphs/affinity_lung32472_ncomp50_kneighbors50_arbf.pkl",
             "lung_hvg.h5ad", "batch", "cell_type", None, 300),
    "crcx": ("graphs/affinity_crcx50130_ncomp50_kneighbors50_arbf.pkl",
             "spatial/crc_xenium/crc_xenium_prepped.h5ad", "patient_id",
             "celltype", "Niche", 670),
    # Node-holdout TRAIN graphs (the protocol pivot, 2026-09-24): spans are not
    # portable across graphs, so the node-holdout runs need their own. Run with
    # DATA_DIR pointing at the REPO ROOT -- the h5ads live in node_heldout_data/.
    "pancreas-nh": (
        "graphs/affinity_pancreas_arbf_nodeheldout20_seed0_train13105"
        "_ncomp50_kneighbors50_arbf.pkl",
        "node_heldout_data/pancreas_arbf_nodeheldout20_seed0_train.h5ad",
        "tech", "celltype", None, 220),
    "lung-nh": (
        "graphs/affinity_lung_arbf_nodeheldout20_seed0_train25977"
        "_ncomp50_kneighbors50_arbf.pkl",
        "node_heldout_data/lung_arbf_nodeheldout20_seed0_train.h5ad",
        "batch", "cell_type", None, 300),
    "crcx-bk05": (
        "graphs/affinity_crcx50130_ncomp50_kneighbors50_bk05.pkl",
        "spatial/crc_xenium/crc_xenium_prepped.h5ad", "patient_id",
        "celltype", "Niche", 670),
    "pbmc-immune-nh": (
        "graphs/affinity_pbmc-immune_arbf_nodeheldout20_seed0_train26806"
        "_ncomp50_kneighbors50_arbf.pkl",
        "node_heldout_data/pbmc-immune_arbf_nodeheldout20_seed0_train.h5ad",
        "study", "final_annotation", None, 300),
}


def leiden_at_k(A, K, seed=0, n_tries=18):
    """CPM-Leiden with the resolution tuned to land near K communities.

    The span is defined at the K the run will actually use -- cohesion depends
    on K, so measuring at a different one would give the wrong lambda.
    """
    import igraph as ig
    import leidenalg
    N = A.shape[0]
    src, dst = A.nonzero()
    m = src < dst
    g = ig.Graph(n=N, edges=list(zip(src[m].tolist(), dst[m].tolist())))
    g.es["weight"] = np.asarray(A[src[m], dst[m]]).ravel().tolist()
    lo, hi, best_lab, best = 1e-4, 5.0, None, None
    for _ in range(n_tries):
        mid = (lo * hi) ** 0.5
        lab = np.asarray(leidenalg.find_partition(
            g, leidenalg.CPMVertexPartition, weights="weight",
            resolution_parameter=mid, seed=seed).membership)
        nk = lab.max() + 1
        if best is None or abs(nk - K) < abs(best[1] - K):
            best, best_lab = (mid, nk), lab
        if nk < K:
            lo = mid
        else:
            hi = mid
    return best_lab, best


def run(ds):
    pkl, h5rel, bkey, lkey, nkey, K = DATASETS[ds]
    h5 = os.path.join(os.environ["DATA_DIR"], h5rel)
    t0 = time.time()
    A = load_affinity(pkl)
    N = A.shape[0]
    obs = read_obs(h5, [bkey, lkey] + ([nkey] if nkey else []))
    batches = obs[bkey].astype(str)
    import pandas as pd
    bcodes = pd.Categorical(batches).codes.astype(np.int64)
    n_bio = int(bcodes.max()) + 1

    coo = A.tocoo()
    same = bcodes[coo.row] == bcodes[coo.col]
    print(f"\n{'='*84}\n{ds}:  N={N:,}  edges={A.nnz//2:,}  bio batches={n_bio}  K={K}")
    print(f"  cross-batch edges dropped by [D41]: "
          f"{int((~same).sum()):,}/{A.nnz:,} ({100*(~same).mean():.1f}%)  "
          f"({time.time()-t0:.0f}s)")

    lab_leiden, best = leiden_at_k(A_masked(A, bcodes), K)
    print(f"  leiden: resolution={best[0]:.5f} -> {best[1]} communities "
          f"(target {K})   ({time.time()-t0:.0f}s)")

    rng = np.random.default_rng(0)
    parts = {
        "leiden": lab_leiden,
        "random-K": rng.integers(0, K, N),
        "celltype": obs[lkey].astype(str),
        "one group": np.zeros(N, np.int64),
    }
    if nkey and obs.get(nkey) is not None:
        parts["ct x niche"] = np.array(
            [f"{a}|{b}" for a, b in zip(obs[lkey].astype(str),
                                        obs[nkey].astype(str))])
    parts["celltype x batch"] = np.array(
        [f"{a}|{b}" for a, b in zip(obs[lkey].astype(str), batches)])

    # V[b] semantics = pass batch_codes; the old global denominator = None.
    # (cohesion_per_cell no longer takes per_batch -- multi-batch is the
    # behaviour, not an option.) Separate caches: the two denominators cache
    # different intermediates.
    cache_pb, cache_gl = {}, {}
    print(f"\n  {'partition':<18}{'K':>6}{'aff [D41]':>12}{'gram [D41]':>12}"
          f"{'aff global':>12}{'gram global':>13}")
    vals = {}
    for name, lab in parts.items():
        row = {}
        for s_ in ("aff", "gram"):
            row[f"{s_}_pb"] = cohesion(lab, A, bcodes, sim=s_, _cache=cache_pb)
            row[f"{s_}_gl"] = cohesion(lab, A, None, sim=s_, _cache=cache_gl)
        vals[name] = row
        nk = len(np.unique(np.asarray(lab)))
        print(f"  {name:<18}{nk:>6}{row['aff_pb']:>12.6f}{row['gram_pb']:>12.6f}"
              f"{row['aff_gl']:>12.6f}{row['gram_gl']:>13.6f}")

    # batch-neutrality check: 'celltype' (spread over batches) must score the
    # SAME as 'celltype x batch' (the same biology, batch-pure) under [D41].
    a, b = vals["celltype"], vals["celltype x batch"]
    for s_ in ("aff", "gram"):
        d = abs(a[f"{s_}_pb"] - b[f"{s_}_pb"])
        ratio = (b[f"{s_}_gl"] / a[f"{s_}_gl"]) if a[f"{s_}_gl"] > 0 else float("nan")
        flag = "OK" if d < 1e-9 else f"DIFFER by {d:.2e}"
        print(f"  batch-neutrality [{s_}]: {flag}   "
              f"(global denominator favours batch-pure {ratio:.1f}x)")

    spans = {}
    print()
    for term, key in (("aff", "aff_pb"), ("gram", "gram_pb")):
        sp_ = vals["leiden"][key] - vals["random-K"][key]
        spans[term] = sp_
        print(f"  {term:<5} span = {sp_:.6f}   lambda alone = {AUTHORITY/sp_:>8.2f}"
              f"   at 50/50 = {0.5*AUTHORITY/sp_:>8.2f}")
    print(f"  span_gram/span_aff = {spans['gram']/spans['aff']:.3f}")
    print(f"\n  ARMS for {ds} (K={K}):")
    print(f"    A_aff   lambda_cohesion_aff = {AUTHORITY/spans['aff']:.2f}")
    print(f"    B_gram  lambda_cohesion     = {AUTHORITY/spans['gram']:.2f}")
    print(f"    C_both  gram {0.5*AUTHORITY/spans['gram']:.2f} + "
          f"aff {0.5*AUTHORITY/spans['aff']:.2f}")
    print(f"  done in {time.time()-t0:.0f}s")
    return ds, spans, vals


def A_masked(A, bcodes):
    coo = A.tocoo()
    keep = bcodes[coo.row] == bcodes[coo.col]
    return sp.csr_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])),
                         shape=A.shape)


if __name__ == "__main__":
    targets = sys.argv[1:] or list(DATASETS)
    results = {}
    for ds in targets:
        try:
            name, spans, _ = run(ds)
            results[name] = spans
        except Exception as e:
            import traceback
            print(f"\n{ds}: FAILED -- {e}")
            traceback.print_exc()
    print("\n" + "=" * 84)
    print("SUMMARY -- lambdas to put in the run script")
    print(f"{'dataset':<14}{'span_aff':>11}{'span_gram':>11}"
          f"{'aff_only':>10}{'gram_only':>11}{'aff_half':>10}{'gram_half':>11}")
    for ds, s in results.items():
        print(f"{ds:<14}{s['aff']:>11.6f}{s['gram']:>11.6f}"
              f"{AUTHORITY/s['aff']:>10.2f}{AUTHORITY/s['gram']:>11.2f}"
              f"{0.5*AUTHORITY/s['aff']:>10.2f}{0.5*AUTHORITY/s['gram']:>11.2f}")
