"""The frozen metacell scoreboard: one implementation, used everywhere.

Scores a partition (cell -> metacell labels) against the claims the method makes.
Library code on purpose (CLAUDE.md 2) so the Modal eval, local checks and any
notebook all compute the SAME numbers -- a comparison against SEACells is only
meaningful if both sides go through identical code.

THE METRICS, and which claim each answers
-----------------------------------------
1. "we preserve good within-batch communities"
       coh_aff / coh_gram -- the training objective itself, on HARD assignments
       over the full graph. Reported BOTH per-batch ([D41]: cross-batch
       similarity zeroed, denominator = size within the cell's own batch) AND
       global, so no arm can look good merely because the metric was changed to
       match it.

2. "but also combine batches"
       batch_entropy -- UNNORMALISED Shannon entropy of the batch distribution
       inside each metacell, in nats. EXACTLY mc_metric_utils.calc_batch_entropy,
       which is what the paper's published numbers use; do NOT divide by
       log(n_batches). 0 = single batch, ceiling = log(n_batches). Higher better.

3. "a bio proof"
       ct_purity, coverage; plus joint_purity and n_covered_states against
       (cell type, niche) PAIRS where a niche annotation exists.

4. "why useful? for RARE cell types"
       rare_metrics_paper() -- a faithful port of the paper's own definitions.
       See that docstring; in particular cross_batch_homogeneity is the only
       one of them that can tell real cross-batch integration apart from each
       batch getting its own private cluster.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp


# --------------------------------------------------------------- rare types ---
def get_rare_types(cell_types, batches, quantile=0.25):
    """GLOBAL rare types (median per-batch frequency below its Q25).

    NOTE: this is NOT what the paper's rare table uses -- the paper defines rare
    types PER BATCH, see rare_metrics_paper(). Kept only for quick dataset
    summaries; never report it as "the" rare set.
    """
    freq = (pd.DataFrame({"ct": cell_types, "b": batches})
            .groupby(["b", "ct"]).size()
            .groupby(level="b", group_keys=False).apply(lambda s: s / s.sum())
            .unstack(level="b", fill_value=0.0))
    med = freq.median(axis=1)
    return sorted(med[med < med.quantile(quantile)].index)


def rare_metrics_paper(labels, cell_types, batches, rare_quantile=0.25):
    """The PAPER's batch-local rare-cell metrics.

    Ported faithfully from paper_figures._rare_table_one, the function that
    produced the published numbers. Do NOT "improve" the definitions here; if
    they should change, change them there and re-run the paper tables too.

    Rare types are LOCAL TO EACH BATCH: within a batch, a type is rare if its
    within-batch frequency is below mean(freq) - std(freq), falling back to the
    Q25 quantile when that set comes out empty. Every metric is computed per
    batch and aggregated as mean +/- std ACROSS BATCHES.

        homogeneity (= repr micro)
            for each rare cell, the fraction of its metacell carrying the SAME
            label as that cell (counting cells from every batch). Averaged over
            the batch's rare cells.

        repr_macro
            the same per-cell scores, averaged within each rare type first and
            then over types, so a 3-cell type counts as much as a 300-cell one.

        cross_batch_homogeneity   <-- THE ONE THAT TESTS BATCH INTEGRATION
            same denominator (metacell size) but the numerator counts only
            same-label mates from a DIFFERENT batch. Coverage, homogeneity and
            F1 can all be maximised by giving each batch's rare cells their own
            same-batch-only cluster -- the "no co-clustering" failure mode. This
            metric is blind to that loophole: same-batch same-label mates never
            count toward it.

        coverage
            fraction of this batch's locally-rare types for which at least one
            cell of that type IN THIS BATCH sits in a metacell whose majority
            label is that type.

        recall     fraction of a type's cells in THIS batch sitting in a
                   metacell dedicated to (majority-labelled as) that type.
        precision  the global average purity of the metacells dedicated to it.
        f1         harmonic mean of the two, averaged over the batch's rare types.
        purity     size-weighted purity of the metacells whose majority label is
                   a locally-rare type.
    """
    labels = np.asarray(labels)
    df = pd.DataFrame({"metacell_id": labels,
                       "ct": np.asarray(cell_types),
                       "b": np.asarray(batches)})

    mc_sizes = df.groupby("metacell_id").size()
    mc_label_counts = df.groupby(["metacell_id", "ct"]).size()
    mc_label_frac = mc_label_counts / mc_sizes
    mc_majority = (mc_label_counts.groupby(level="metacell_id").idxmax()
                   .map(lambda x: x[1]))

    mc_purity_series = mc_label_frac.reindex(
        pd.MultiIndex.from_arrays([mc_majority.index, mc_majority.values])
    ).fillna(0.0)
    mc_purity_series.index = mc_majority.index
    mc_purity_by_ct = mc_purity_series.groupby(mc_majority).mean().to_dict()

    mc_label_batch_counts = df.groupby(["metacell_id", "ct", "b"]).size()

    acc = {k: [] for k in ("coverage", "repr_macro", "homogeneity",
                           "cross_batch_homogeneity", "purity",
                           "recall", "precision", "f1")}
    rare_seen = set()

    for batch_val, bdf in df.groupby("b"):
        ct_freq = bdf.ct.value_counts(normalize=True)
        thresh = float(ct_freq.mean() - ct_freq.std())
        bl_rare = set(ct_freq[ct_freq < thresh].index)
        if not bl_rare:
            bl_rare = set(ct_freq[ct_freq < ct_freq.quantile(rare_quantile)].index)
        if not bl_rare:
            continue
        rare_seen |= bl_rare
        bl_cells = bdf[bdf.ct.isin(bl_rare)]
        if bl_cells.empty:
            continue

        acc["coverage"].append(sum(
            (bdf.loc[bdf.ct == c, "metacell_id"].map(mc_majority) == c).any()
            for c in bl_rare) / len(bl_rare))

        lookup = pd.MultiIndex.from_arrays([bl_cells.metacell_id, bl_cells.ct])
        scores = mc_label_frac.reindex(lookup).fillna(0.0)
        scores.index = bl_cells.index
        acc["repr_macro"].append(float(scores.groupby(bl_cells.ct).mean().mean()))
        acc["homogeneity"].append(float(scores.mean()))

        same_b = mc_label_batch_counts.reindex(
            pd.MultiIndex.from_arrays([bl_cells.metacell_id, bl_cells.ct,
                                       pd.Index([batch_val] * len(bl_cells))])
        ).fillna(0.0).values
        total = mc_label_counts.reindex(
            pd.MultiIndex.from_arrays([bl_cells.metacell_id, bl_cells.ct])
        ).fillna(0.0).values
        cross = (total - same_b) / mc_sizes.reindex(bl_cells.metacell_id).values
        acc["cross_batch_homogeneity"].append(float(np.mean(cross)))

        rare_mc = mc_majority.isin(bl_rare)
        pur = mc_purity_series[rare_mc].values
        szs = mc_sizes[rare_mc].values.astype(float)
        if len(pur):
            acc["purity"].append(float((pur * (szs / szs.sum())).sum()))

        f1s, recs, precs = [], [], []
        for c in bl_rare:
            cc = bdf[bdf.ct == c]
            if cc.empty:
                continue
            rec = float((cc.metacell_id.map(mc_majority) == c).mean())
            pre = mc_purity_by_ct.get(c, 0.0)
            den = pre + rec
            f1s.append(2 * pre * rec / den if den > 0 else 0.0)
            recs.append(rec)
            precs.append(pre)
        if f1s:
            acc["f1"].append(float(np.mean(f1s)))
            acc["recall"].append(float(np.mean(recs)))
            acc["precision"].append(float(np.mean(precs)))

    out = {}
    for k, v in acc.items():
        a = np.asarray(v, dtype=float)
        out[f"rare_{k}_mean"] = float(a.mean()) if a.size else float("nan")
        out[f"rare_{k}_std"] = float(a.std()) if a.size else float("nan")
    out["n_rare_types"] = len(rare_seen)
    out["rare_types_seen"] = ",".join(sorted(rare_seen))
    return out


# ------------------------------------------------------------- the cohesion ---
def _gram_rows(A, delta=1.0):
    """L2-normalised CLOSED-neighbourhood rows R, so Gram = R @ R.T (diag 1)."""
    cA = (A + delta * sp.eye(A.shape[0], format="csr", dtype=np.float32)).tocsr()
    n = np.sqrt(cA.multiply(cA).sum(axis=1)).A.ravel()
    n[n == 0] = 1.0
    return sp.csr_matrix(sp.diags(1.0 / n) @ cA)


def cohesion(labels, A, batch_codes=None, sim="aff", per_batch=False,
             cohesion_self=0.0, cohesion_diag=1.0, _cache=None):
    """The training objective on hard assignments, over the full graph.

        numer[c][p] = sum over j of similarity(c,j) * S[j][p]
        den[c][p]   = size_in_batch[bio(c)][p]  if per_batch else size[p]
        value       = (1/n_cells) * sum_c sum_p S[c][p]*numer[c][p]/max(den,1)

    cohesion_self defaults to 0: with per_batch=True a nonzero value rewards
    smearing prototypes thinly across batches (the self part is 1/size_in_batch,
    which a batch-SPREAD prototype maximises). Measured on pancreas, leaving it
    at 1 collapses the span from 0.107 to 0.0006.
    """
    N = A.shape[0]
    _, codes = np.unique(np.asarray(labels), return_inverse=True)
    K = int(codes.max()) + 1
    S = sp.csr_matrix((np.ones(N, np.float32), (np.arange(N), codes)),
                      shape=(N, K))
    Sd = S.toarray()

    M = A
    if per_batch:
        if batch_codes is None:
            raise ValueError("per_batch=True needs batch_codes")
        if _cache is not None and "A_mask" in _cache:
            M = _cache["A_mask"]
        else:
            coo = A.tocoo()
            keep = batch_codes[coo.row] == batch_codes[coo.col]
            M = sp.csr_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])),
                              shape=A.shape)
            if _cache is not None:
                _cache["A_mask"] = M

    if sim == "aff":
        numer = np.asarray((M @ S).todense()) + cohesion_diag * Sd
    else:
        rkey = f"R_{per_batch}"
        if _cache is not None and rkey in _cache:
            R = _cache[rkey]
        else:
            R = _gram_rows(M, cohesion_diag)
            if _cache is not None:
                _cache[rkey] = R
        numer = np.asarray((R @ (R.T @ S)).todense())
    # sim(c,c) == 1 in both modes (aff via cohesion_diag, gram via the L2 norm)
    numer = numer - (1.0 - cohesion_self) * Sd

    if per_batch:
        nb = int(batch_codes.max()) + 1
        size_bp = np.zeros((nb, K), np.float64)
        np.add.at(size_bp, batch_codes, Sd)
        den = size_bp[batch_codes]
    else:
        den = np.tile(Sd.sum(0), (N, 1))
    return float((Sd * (numer / np.maximum(den, 1.0))).sum() / N)


# ------------------------------------------------------------ the composite ---
def score_partition(labels, cell_types, batches, A=None, niches=None,
                    n_proto=None, _cache=None):
    """Every scoreboard metric for one partition. Returns a flat dict."""
    labels = np.asarray(labels)
    df = pd.DataFrame({"mc": labels, "ct": np.asarray(cell_types),
                       "b": np.asarray(batches)})
    out = {}

    g = df.groupby("mc")
    size = g.size()
    purity = g.ct.agg(lambda s: s.value_counts().iloc[0] / len(s))
    major = g.ct.agg(lambda s: s.value_counts().index[0])
    # UNNORMALISED Shannon entropy in nats == mc_metric_utils.calc_batch_entropy
    entropy = g.b.agg(lambda s: float(
        -(lambda p: (p * np.log(p + 1e-10)).sum())(
            s.value_counts(normalize=True).values)))
    w = size / size.sum()

    out["ct_purity_mean"] = float(purity.mean())
    out["ct_purity_std"] = float(purity.std())
    out["ct_purity_wmean"] = float((purity * w).sum())
    out["batch_entropy_mean"] = float(entropy.mean())
    out["batch_entropy_std"] = float(entropy.std())
    out["batch_entropy_wmean"] = float((entropy * w).sum())
    out["coverage"] = float(major.nunique() / df.ct.nunique())
    out["n_mc"] = int(len(size))
    out["median_size"] = float(size.median())
    if n_proto is not None:
        out["n_dead"] = int(n_proto - len(np.unique(labels)))

    if niches is not None:
        jdf = pd.DataFrame({
            "mc": labels,
            "joint": (pd.Series(np.asarray(cell_types)).astype(str) + "||"
                      + pd.Series(np.asarray(niches)).astype(str)).to_numpy()})
        jg = jdf.groupby("mc")
        out["joint_purity"] = float(
            jg.joint.agg(lambda s: s.value_counts().iloc[0]).sum() / len(jdf))
        jmaj = jg.joint.agg(lambda s: s.value_counts().index[0])
        out["n_covered_states"] = int(jmaj.nunique())
        out["n_states_total"] = int(jdf.joint.nunique())
        out["state_coverage"] = float(jmaj.nunique() / jdf.joint.nunique())

    out.update(rare_metrics_paper(labels, cell_types, batches))

    if A is not None:
        bcodes = pd.Categorical(np.asarray(batches)).codes.astype(np.int64)
        for s_ in ("aff", "gram"):
            out[f"coh_{s_}_perbatch"] = cohesion(labels, A, bcodes, sim=s_,
                                                 per_batch=True, _cache=_cache)
            out[f"coh_{s_}_global"] = cohesion(labels, A, bcodes, sim=s_,
                                               per_batch=False, _cache=_cache)
    return out


# ------------------------------------------------------------------- io ------
def load_affinity(path):
    """Load a cached affinity .pkl in canonical form: symmetric, non-negative,
    no diagonal, no explicit stored zeros (the pickles carry diagonal zeros)."""
    import pickle
    with open(path, "rb") as f:
        obj = pickle.load(f)
    A = obj["affinity"] if isinstance(obj, dict) and "affinity" in obj else obj
    if isinstance(obj, dict) and not sp.issparse(A):
        A = next(v for v in obj.values() if sp.issparse(v))
    A = sp.csr_matrix(A).astype(np.float32)
    A = ((A + A.T) / 2.0).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def read_obs(h5_path, keys):
    """Read obs columns straight from HDF5.

    Avoids anndata: these files predate the current reader and carry
    `encoding-type: null` entries it has no reader registered for (CLAUDE.md 11).
    Handles both categorical (group) and plain (dataset) encodings.
    """
    import h5py
    out = {}
    with h5py.File(h5_path, "r") as f:
        for k in keys:
            if k is None or f"obs/{k}" not in f:
                out[k] = None
                continue
            node = f[f"obs/{k}"]
            if isinstance(node, h5py.Group):
                cats = np.array([c.decode() if isinstance(c, bytes) else c
                                 for c in node["categories"][:]])
                out[k] = cats[node["codes"][:]]
            else:
                v = node[:]
                out[k] = (np.array([c.decode() if isinstance(c, bytes) else c
                                    for c in v])
                          if v.dtype.kind in "SO" else v)
    return out
