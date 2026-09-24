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
    # every rare cell's own homogeneity score, pooled across batches, so the
    # spread can be reported OVER CELLS as well as over batches. The paper
    # aggregates over batches; a per-cell std answers a different and equally
    # fair question ("how variable is a rare cell's experience") and is what the
    # table asks for.
    cell_scores, cell_cross = [], []
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
        cell_scores.append(np.asarray(scores.values, np.float64))

        same_b = mc_label_batch_counts.reindex(
            pd.MultiIndex.from_arrays([bl_cells.metacell_id, bl_cells.ct,
                                       pd.Index([batch_val] * len(bl_cells))])
        ).fillna(0.0).values
        total = mc_label_counts.reindex(
            pd.MultiIndex.from_arrays([bl_cells.metacell_id, bl_cells.ct])
        ).fillna(0.0).values
        cross = (total - same_b) / mc_sizes.reindex(bl_cells.metacell_id).values
        acc["cross_batch_homogeneity"].append(float(np.mean(cross)))
        cell_cross.append(np.asarray(cross, np.float64))

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
    # std OVER CELLS (pooled across batches), alongside the over-batches std
    for key, chunks in (("homogeneity", cell_scores),
                        ("cross_batch_homogeneity", cell_cross)):
        if chunks:
            allv = np.concatenate(chunks)
            out[f"rare_{key}_cellmean"] = float(allv.mean())
            out[f"rare_{key}_cellstd"] = float(allv.std())
            out[f"rare_{key}_ncells"] = int(allv.size)
    return out


# ------------------------------------------------------------- the cohesion ---
def _gram_rows(A, delta=1.0):
    """L2-normalised CLOSED-neighbourhood rows R, so Gram = R @ R.T (diag 1)."""
    cA = (A + delta * sp.eye(A.shape[0], format="csr", dtype=np.float32)).tocsr()
    n = np.sqrt(cA.multiply(cA).sum(axis=1)).A.ravel()
    n[n == 0] = 1.0
    return sp.csr_matrix(sp.diags(1.0 / n) @ cA)


def build_inV(A, batch_codes, n_bio):
    """V[b] = node set of bio_graph_b = all edges with >=1 endpoint in batch b.

        V[b] = {j : bio(j) == b}  U  {j : j has an edge into batch b}

    Returned as a dense (n_bio, n_cells) 0/1 array.
    """
    N = A.shape[0]
    onehot = sp.csr_matrix((np.ones(N, np.float32), (np.arange(N), batch_codes)),
                           shape=(N, n_bio))
    touch = np.asarray(((A > 0).astype(np.float32) @ onehot).todense()) > 0
    own = batch_codes[:, None] == np.arange(n_bio)[None, :]
    return (touch | own).T.astype(np.float64)          # (n_bio, N)


def cohesion_per_cell(labels, A, batch_codes=None, sim="aff",
                      cohesion_self=1.0, cohesion_diag=1.0, _cache=None):
    """The training objective, per cell, on HARD assignments.

    Multi-batch is the BEHAVIOUR, not an option -- exactly as the trainer does it.
    The only thing the batch structure changes is the SET OF CELLS each cell is
    scored against:

        V[b]        = node set of bio_graph_b (all edges with >=1 endpoint in b)
        numer[c][p] = sum over j in V[bio(c)] of  ass[j][p] * sim[c][j]
        denom[c][p] = sum over j in V[bio(c)] of  ass[j][p]
        cohesion_c  = sum over p of ass[c][p] * numer[c][p] / denom[c][p]

    With ONE batch (or batch_codes=None) V[b] is every cell, so the denominator
    is the global size and this reduces to the batch-agnostic formula exactly.

    Nothing is masked: cross-batch similarities keep their real values. What is
    excluded are the cells whose zeros carry no information -- a cross-batch pair
    with no edge is the batch effect, not measured dissimilarity, so it is absent
    from the numerator and must be absent from the denominator too.

    Returns a length-n_cells array; the caller takes mean and std OVER CELLS.
    """
    N = A.shape[0]
    _, codes = np.unique(np.asarray(labels), return_inverse=True)
    K = int(codes.max()) + 1
    S = sp.csr_matrix((np.ones(N, np.float32), (np.arange(N), codes)),
                      shape=(N, K))
    Sd = S.toarray().astype(np.float64)

    if batch_codes is None:
        batch_codes = np.zeros(N, np.int64)
    n_bio = int(batch_codes.max()) + 1

    if sim == "aff":
        # sim[c][j] nonzero only for c's neighbours, and c is in b, so every such
        # j is already in V[b]: the numerator restriction is AUTOMATIC.
        numer = np.asarray((A @ S).todense()).astype(np.float64)
        numer += cohesion_diag * Sd                    # sim(c,c)
    else:
        rk = "R" if _cache is None else "R_gram"
        R = (_cache.get(rk) if _cache is not None else None)
        if R is None:
            R = _gram_rows(A, cohesion_diag)
            if _cache is not None:
                _cache[rk] = R
        if n_bio == 1:
            numer = np.asarray((R @ (R.T @ S)).todense()).astype(np.float64)
        else:
            # gram: the restriction is NOT automatic -- a 2-hop neighbour can sit
            # OUTSIDE V[b] while having nonzero similarity. Restrict explicitly,
            # one prototype profile per batch.
            inV = (_cache.get("inV") if _cache is not None else None)
            if inV is None:
                inV = build_inV(A, batch_codes, n_bio)
                if _cache is not None:
                    _cache["inV"] = inV
            numer = np.zeros((N, K), np.float64)
            for b in range(n_bio):
                selb = batch_codes == b
                if not selb.any():
                    continue
                PV_b = R.T @ sp.csr_matrix(Sd * inV[b][:, None])
                numer[selb] = np.asarray((R[selb] @ PV_b).todense())
    numer = numer - (1.0 - cohesion_self) * Sd         # weight on the self-pair

    if n_bio == 1:
        den = np.tile(Sd.sum(0), (N, 1))
    else:
        inV = (_cache.get("inV") if _cache is not None else None)
        if inV is None:
            inV = build_inV(A, batch_codes, n_bio)
            if _cache is not None:
                _cache["inV"] = inV
        den = (inV @ Sd)[batch_codes]                  # (N, K)
    return (Sd * (numer / np.maximum(den, 1e-12))).sum(axis=1)


def cohesion(labels, A, batch_codes=None, sim="aff", **kw):
    """Mean over cells of cohesion_per_cell."""
    return float(cohesion_per_cell(labels, A, batch_codes, sim=sim, **kw).mean())


def modularity_per_batch(labels, A, batch_codes, node_mask=None):
    """Modularity computed inside each bio batch's own subgraph, per batch.

    Returns a dict {batch -> modularity}; the caller takes mean and std OVER
    BATCHES, which is the statistic the paper's Table 1 uses
    (mc_metric_utils.calc_modularity_per_batch).

    node_mask restricts to a set of nodes (e.g. test nodes). Per the definition
    used throughout, the subgraph of a node set is ALL EDGES WITH AT LEAST ONE
    ENDPOINT in the set -- not only edges with both endpoints inside.
    """
    labels = np.asarray(labels)
    _, codes = np.unique(labels, return_inverse=True)
    out = {}
    for b in np.unique(batch_codes):
        keep = batch_codes == b
        if node_mask is not None:
            keep = keep & node_mask
        if keep.sum() < 2:
            continue
        idx = np.where(keep)[0]
        # all edges with >=1 endpoint in this set
        Ab = A[idx]                                   # rows of the set
        coo = Ab.tocoo()
        w = coo.data
        if w.size == 0:
            continue
        ci = codes[idx][coo.row]
        cj = codes[coo.col]
        two_m = float(w.sum())                        # directed sum over these rows
        if two_m <= 0:
            continue
        inside = float(w[ci == cj].sum())
        deg = np.zeros(codes.max() + 1)
        np.add.at(deg, ci, w)
        out[str(b)] = inside / two_m - float((deg / two_m) ** 2 @ np.ones_like(deg))
    return out


# ------------------------------------------------------------ the composite ---
def _w_mean_std(x, w):
    """Size-weighted mean and its matching weighted std.

    w is normalised to sum 1, so
        wmean = sum_i w_i x_i
        wstd  = sqrt( sum_i w_i (x_i - wmean)^2 )
    i.e. the spread ACROSS METACELLS weighted by size -- not a standard error.
    """
    w = np.asarray(w, np.float64)
    w = w / w.sum()
    m = float((w * x).sum())
    return m, float(np.sqrt((w * (np.asarray(x, np.float64) - m) ** 2).sum()))


def score_partition(labels, cell_types, batches, A_train=None, A_test=None,
                    niches=None, n_proto=None, _cache=None, _cache_test=None):
    """Every scoreboard metric for one partition.

    A_train  the graph the model was fitted on (the visible 80%).
    A_test   the held-out edges -- cohesion/modularity on these is the
             de-circularised number: the model never had a training signal for
             any of them.
    """
    labels = np.asarray(labels)
    ct = np.asarray(cell_types).astype(str)
    bt = np.asarray(batches).astype(str)
    bcodes = pd.Categorical(bt).codes.astype(np.int64)
    df = pd.DataFrame({"mc": labels, "ct": ct, "b": bt})
    out = {}

    # --- per-metacell purity and batch entropy -----------------------------
    g = df.groupby("mc")
    size = g.size()
    purity = g.ct.agg(lambda s: s.value_counts().iloc[0] / len(s))
    major = g.ct.agg(lambda s: s.value_counts().index[0])
    # UNNORMALISED Shannon entropy in nats == mc_metric_utils.calc_batch_entropy
    entropy = g.b.agg(lambda s: float(
        -(lambda q: (q * np.log(q + 1e-10)).sum())(
            s.value_counts(normalize=True).values)))

    out["ct_purity_mean"] = float(purity.mean())
    out["ct_purity_std"] = float(purity.std())
    out["ct_purity_wmean"], out["ct_purity_wstd"] = _w_mean_std(purity.values,
                                                                size.values)
    out["batch_entropy_mean"] = float(entropy.mean())
    out["batch_entropy_std"] = float(entropy.std())
    out["batch_entropy_wmean"], out["batch_entropy_wstd"] = _w_mean_std(
        entropy.values, size.values)
    out["coverage"] = float(major.nunique() / df.ct.nunique())
    out["n_mc"] = int(len(size))
    out["median_size"] = float(size.median())
    if n_proto is not None:
        out["n_dead"] = int(n_proto - len(np.unique(labels)))

    # --- joint (cell type, niche) ------------------------------------------
    if niches is not None:
        jdf = pd.DataFrame({"mc": labels,
                            "joint": (pd.Series(ct) + "||"
                                      + pd.Series(np.asarray(niches).astype(str))
                                      ).to_numpy()})
        jg = jdf.groupby("mc")
        out["joint_purity"] = float(
            jg.joint.agg(lambda s: s.value_counts().iloc[0]).sum() / len(jdf))
        jmaj = jg.joint.agg(lambda s: s.value_counts().index[0])
        out["n_covered_states"] = int(jmaj.nunique())
        out["n_states_total"] = int(jdf.joint.nunique())

    # --- rare: the PAPER's batch-local definitions -------------------------
    out.update(rare_metrics_paper(labels, ct, bt))

    # --- the objective, mean +- std OVER CELLS -----------------------------
    for tag, Ax, cch in (("train", A_train, _cache), ("test", A_test, _cache_test)):
        if Ax is None:
            continue
        for sim in ("aff", "gram"):
            v = cohesion_per_cell(labels, Ax, bcodes, sim=sim, _cache=cch)
            out[f"coh_{sim}_{tag}_mean"] = float(v.mean())
            out[f"coh_{sim}_{tag}_std"] = float(v.std())

    # --- modularity per bio batch, mean +- std OVER BATCHES ----------------
    for tag, Ax in (("train", A_train), ("test", A_test)):
        if Ax is None:
            continue
        mods = modularity_per_batch(labels, Ax, bcodes)
        if mods:
            vals = np.array(list(mods.values()), np.float64)
            out[f"modularity_{tag}_mean"] = float(vals.mean())
            out[f"modularity_{tag}_std"] = float(vals.std())
            out[f"modularity_{tag}_nbatch"] = int(len(vals))
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
