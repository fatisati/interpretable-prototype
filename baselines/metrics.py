"""Generic metrics for the rare-cell-preservation task (tasks/rare_cell_preservation.md).

No knowledge of any specific baseline. Every function here takes a
DataFrame with exactly these columns -- cell_id, batch, true_type, cluster
-- plus, for cohesion, the matching affinity sparse matrix in the same row
order. baselines/compute_metrics.py's per-baseline adapters are the only
code responsible for reshaping a baseline's actual saved output into this
shape; nothing in this file knows scVI, Harmony, or any dataset exists.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_frequency(df):
    """Each cell's own (true_type, batch) frequency, within its own batch
    only. Continuous value -- used both for the rare/common split below and
    for the frequency-dependence check.
    """
    counts = df.groupby(["batch", "true_type"]).size().rename("n_type").reset_index()
    totals = df.groupby("batch").size().rename("n_batch").reset_index()
    merged = counts.merge(totals, on="batch")
    merged["freq"] = merged["n_type"] / merged["n_batch"]
    freq_lookup = merged.set_index(["batch", "true_type"])["freq"]
    key = pd.MultiIndex.from_arrays([df["batch"], df["true_type"]])
    return freq_lookup.reindex(key).to_numpy()


RARE_RULES = {
    "std": "frequency more than {k} std below its batch's mean type-frequency",
    "log_std": "log10(frequency) more than {k} std below its batch's mean log10 type-frequency (task file's rule)",
    "fixed": "frequency below {cutoff:.2%} of its batch",
}


def rare_group_table(df, rule="std", std_threshold=1.0, min_types_for_std=4, fixed_cutoff=0.01):
    """The rare/common decision, one row per group (batch, true_type):
    n_cells, freq, the batch's cutoff ("threshold", in frequency units), and
    is_rare. Saved as-is so a reader can see *why* each group is (not) rare.

    rule:
      "std"     -- task file's rule: freq < batch mean - k * batch std.
                   With one or two huge types per batch the std is so large
                   that the cutoff can fall below 0 -> nothing is rare.
      "log_std" -- same rule on log10(freq), so huge types don't dominate
                   the spread; threshold reported back in frequency units.
      "fixed"   -- freq < fixed_cutoff (e.g. 1% of the batch).
    Fallback for the two std rules (fewer than `min_types_for_std` types in
    a batch -- not in the task doc, our own call): lowest quartile, <=.
    """
    if rule not in RARE_RULES:
        raise ValueError(f"rule must be one of {sorted(RARE_RULES)}, got {rule!r}")
    t = df.groupby(["batch", "true_type"]).size().rename("n_cells").reset_index()
    t["freq"] = t["n_cells"] / t["batch"].map(df.groupby("batch").size())

    t["threshold"], t["is_rare"], t["fallback"] = np.nan, False, False
    for batch, idx in t.groupby("batch").groups.items():
        freqs = t.loc[idx, "freq"]
        if rule == "fixed":
            t.loc[idx, "threshold"] = fixed_cutoff
            t.loc[idx, "is_rare"] = freqs < fixed_cutoff
        elif len(freqs) < min_types_for_std:
            q1 = freqs.quantile(0.25)
            t.loc[idx, "threshold"], t.loc[idx, "fallback"] = q1, True
            t.loc[idx, "is_rare"] = freqs <= q1
        else:
            vals = np.log10(freqs) if rule == "log_std" else freqs
            cut = vals.mean() - std_threshold * vals.std()
            t.loc[idx, "threshold"] = 10 ** cut if rule == "log_std" else cut
            t.loc[idx, "is_rare"] = vals < cut
    t["is_rare"] = t["is_rare"].astype(bool)
    t["rule"] = rule
    return t


def compute_rare_mask(df, std_threshold=1.0, min_types_for_std=4, rule="std", fixed_cutoff=0.01):
    """Per batch: which (type, batch) pairs are locally rare -- see
    rare_group_table for the rules.

    Returns (is_rare: bool array aligned to df, rare_pairs: set of (batch, true_type)).
    """
    table = rare_group_table(df, rule, std_threshold, min_types_for_std, fixed_cutoff)
    rare = table[table["is_rare"]]
    rare_pairs = set(zip(rare["batch"], rare["true_type"]))
    pairs = pd.Series(list(zip(df["batch"], df["true_type"])), index=df.index)
    is_rare = pairs.isin(rare_pairs).to_numpy()
    return is_rare, rare_pairs


def cluster_majority_type(df):
    """Global majority true_type per cluster -- every cell in it, any
    batch. Clusters are dataset-wide objects, not batch-specific ones, so
    this is always computed this way (see docs/architecture.md discussion
    and tasks/rare_cell_preservation.md's Metrics section).
    """
    return df.groupby("cluster")["true_type"].agg(lambda s: s.value_counts().idxmax())


def cell_type_purity(df):
    """Global, no batch split: for each cluster, fraction matching its own
    majority type; mean +/- std across all clusters, dataset-wide.
    """
    majority = cluster_majority_type(df)
    is_majority = (df["true_type"].to_numpy() == df["cluster"].map(majority).to_numpy())
    per_cluster = pd.Series(is_majority, index=df.index).groupby(df["cluster"]).mean()
    return float(per_cluster.mean()), float(per_cluster.std())


def agreement(df, scope):
    """Per-cell ratio: fraction of a comparison group *within the cell's
    own cluster* that shares its type.

    scope:
      "global" -- every cell in the cluster (includes the cell itself).
      "within" -- only cells from the cell's own batch (still includes it).
      "cross"  -- only cells from other batches (excludes it by
                  construction). NaN if the cluster has no other-batch
                  cells at all.

    Returns a float array aligned to df, with NaN wherever undefined.
    """
    if scope not in ("global", "within", "cross"):
        raise ValueError(f"scope must be 'global', 'within', or 'cross', got {scope!r}")

    result = np.full(len(df), np.nan)
    positions = pd.Series(np.arange(len(df)), index=df.index)

    for cluster_id, idx in df.groupby("cluster").groups.items():
        pos = positions.loc[idx].to_numpy()
        batches = df.loc[idx, "batch"].to_numpy()
        types = df.loc[idx, "true_type"].to_numpy()
        cluster_size = len(idx)

        # Global: count of each type within this cluster.
        global_unique, global_counts = np.unique(types, return_counts=True)
        global_count_of = dict(zip(global_unique, global_counts))

        if scope == "global":
            result[pos] = [global_count_of[t] / cluster_size for t in types]
            continue

        # Within/cross both need per-(cluster, batch) type counts.
        for batch_val in np.unique(batches):
            batch_rows = batches == batch_val
            batch_size = int(batch_rows.sum())
            batch_types = types[batch_rows]
            within_unique, within_counts = np.unique(batch_types, return_counts=True)
            within_count_of = dict(zip(within_unique, within_counts))

            if scope == "within":
                result[pos[batch_rows]] = [within_count_of[t] / batch_size for t in batch_types]
            else:  # cross: global count minus this batch's own count, over the rest of the cluster
                cross_size = cluster_size - batch_size
                if cross_size == 0:
                    continue  # stays NaN -- no other-batch cells in this cluster at all
                result[pos[batch_rows]] = [
                    (global_count_of[t] - within_count_of.get(t, 0)) / cross_size
                    for t in batch_types
                ]

    return result


def summarize_per_cell(values, df, rare_mask=None):
    """Turn a per-cell array (e.g. from `agreement`) into the reporting
    shape used throughout this task: one overall number plus one number
    per batch, for all cells and (if rare_mask given) for rare cells only.

    `coverage` in each result is the fraction of that population with a
    defined (non-NaN) value -- reported explicitly rather than silently
    averaging over only the defined ones, since a low coverage number
    (e.g. most cells have no cross-batch cluster-mates) is itself
    meaningful, not just a detail to average away.
    """
    def agg(mask):
        vals = values[mask]
        if len(vals) == 0:
            return {"mean": float("nan"), "std": float("nan"), "coverage": float("nan"), "n": 0}
        defined = ~np.isnan(vals)
        defined_vals = vals[defined]
        return {
            "mean": float(defined_vals.mean()) if len(defined_vals) else float("nan"),
            "std": float(defined_vals.std()) if len(defined_vals) else float("nan"),
            "coverage": float(defined.mean()),
            "n": int(defined.sum()),
        }

    batches = df["batch"].to_numpy()
    all_mask = np.ones(len(df), dtype=bool)

    out = {
        "all_overall": agg(all_mask),
        "all_per_batch": {b: agg(batches == b) for b in np.unique(batches)},
    }
    if rare_mask is not None:
        out["rare_overall"] = agg(rare_mask)
        out["rare_per_batch"] = {b: agg(rare_mask & (batches == b)) for b in np.unique(batches)}
    return out


def coverage(df, rare_pairs):
    """Per batch: fraction of that batch's locally-rare types with at
    least one cell in a cluster whose global majority label is that type.
    Binary per-type check (present/absent), not a ratio -- catches a type
    dropped entirely, which `agreement` (always some number) can't.
    """
    majority = cluster_majority_type(df)
    correctly_placed = (df["true_type"].to_numpy() == df["cluster"].map(majority).to_numpy())

    by_batch = {}
    for batch, true_type in rare_pairs:
        by_batch.setdefault(batch, []).append(true_type)

    result = {}
    for batch, types in by_batch.items():
        batch_mask = (df["batch"] == batch).to_numpy()
        hits = 0
        for true_type in types:
            type_mask = batch_mask & (df["true_type"] == true_type).to_numpy()
            if type_mask.any() and correctly_placed[type_mask].any():
                hits += 1
        result[batch] = hits / len(types)
    return result


def rare_precision_recall_f1(df, rare_pairs):
    """Step 1 (once, globally): every cluster's own majority type -- fixed,
    reused below, never recomputed per batch.
    Step 2, per (rare type T, batch b) pair, using only batch b's own
    cells against that fixed labeling:
      TP = b's type-T cells in a cluster labeled T
      FN = b's type-T cells in a cluster labeled something else
      FP = b's non-T cells in a cluster labeled T
      precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = harmonic mean.

    Returns (per_pair DataFrame, (f1_mean, f1_std) across all pairs) --
    each pair is its own independent measurement (see task doc), so this
    mean/std isn't an artifact of the same type being rare in several batches.
    """
    majority = cluster_majority_type(df)
    cluster_label = df["cluster"].map(majority).to_numpy()
    true_type = df["true_type"].to_numpy()
    batch = df["batch"].to_numpy()

    rows = []
    for b, t in rare_pairs:
        batch_mask = batch == b
        is_type = true_type == t
        is_labeled_type = cluster_label == t

        tp = int((batch_mask & is_type & is_labeled_type).sum())
        fn = int((batch_mask & is_type & ~is_labeled_type).sum())
        fp = int((batch_mask & ~is_type & is_labeled_type).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        # F1 = 2TP / (2TP + FP + FN): same value as the harmonic mean whenever
        # both are defined, and 0 (not NaN) when the type is missed entirely.
        # A NaN here used to drop fully-scattered rare types from the F1
        # average -- inflating F1 for exactly the failure this task tests.
        # NaN only if the type has no cells in this batch and nothing is
        # labeled as it (can't happen for a rare pair present in the batch).
        denom = 2 * tp + fp + fn
        f1 = 2 * tp / denom if denom > 0 else float("nan")

        rows.append({
            "true_type": t, "batch": b, "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        })

    # Explicit columns so zero rare pairs gives an empty table, not a crash.
    per_pair = pd.DataFrame(rows, columns=["true_type", "batch", "tp", "fp", "fn",
                                           "precision", "recall", "f1"])
    return per_pair, (float(per_pair["f1"].mean()), float(per_pair["f1"].std()))


def cohesion(affinity, cluster_labels):
    """Per-cell cohesion: average sim(i, j) over every j in i's cluster,
    with sim(i, i) := 1.

    The stored diagonal is ignored and replaced by exactly 1, so this is
    correct whatever the matrix holds there. (affinity.py's graphs store 1
    on the diagonal; an earlier version of this function assumed 0 and
    added 1 on top, counting self twice -- e.g. two cells with sim 0.5
    scored 1.25 instead of 0.75.)
    """
    n = affinity.shape[0]
    result = np.zeros(n)
    for cluster_id in pd.unique(cluster_labels):
        idx = np.where(cluster_labels == cluster_id)[0]
        sub = affinity[idx][:, idx]
        row_sums = np.asarray(sub.sum(axis=1)).ravel() - sub.diagonal() + 1.0
        result[idx] = row_sums / len(idx)
    return result


def group_summary(metric_values, df, is_rare, freq_lookup):
    """One row per group = (batch, true_type): the unit frequency belongs to.

    Frequency is a property of a group, not of a cell -- every cell of type T
    in batch b shares the same frequency. So the frequency question ("does
    the metric drop as frequency drops?") is asked across groups, not cells
    (pseudobulk: average first, then test). Counting cells instead would
    (1) let big common groups outvote small rare ones, and (2) fake
    certainty: 18,000 cells are really ~100-150 groups.

    Per group: freq, n_cells (cells with a defined value -- e.g. cross-batch
    agreement is undefined for some cells), mean, std (ddof=1; NaN for one
    cell), is_rare. std is reported, not used in the test: a high std in a
    rare group means its cells split -- some kept together, some scattered
    -- which the mean alone hides.

    Also min, q25, q75, max (NaN-skipping, like mean): the real range and
    the middle half of the group's cells. Plots draw these, not mean +/- std
    -- for a bounded, skewed metric (most cells at 1, a few low) mean + std
    lands above 1, a value no cell can have.

    freq comes from `freq_lookup` ((batch, true_type) -> frequency over ALL
    cells), not from `df`: for a test job `df` holds only test cells, and a
    group's frequency shouldn't depend on which side of the split we score.
    """
    groups = pd.DataFrame({
        "batch": df["batch"].to_numpy(), "true_type": df["true_type"].to_numpy(),
        "freq": freq_lookup.reindex(pd.MultiIndex.from_arrays([df["batch"], df["true_type"]])).to_numpy(),
        "value": metric_values, "is_rare": is_rare,
    })
    out = groups.groupby(["batch", "true_type"], sort=True).agg(
        freq=("freq", "first"), n_cells=("value", "count"),
        mean=("value", "mean"), std=("value", "std"),
        min=("value", "min"), q25=("value", lambda v: v.quantile(0.25)),
        q75=("value", lambda v: v.quantile(0.75)), max=("value", "max"),
        is_rare=("is_rare", "first"),
    ).reset_index()
    return out


def group_trend(groups):
    """Across groups (from group_summary): does the group mean drop as
    frequency drops? Each group counts once, whatever its cell count.

    spearman: rank correlation of group mean vs frequency (> 0 = rarer
      groups score lower). Spearman, not a straight-line fit, because the
      task's question is "does it go down" (monotonic), and a sharp drop
      at the rare threshold counts as much as a smooth decline.
    slope: least-squares fit of group mean vs log10(frequency) -- how big
      the drop is, per 10x change in frequency. log10 because frequencies
      span orders of magnitude. Reported for size, not for the yes/no.
    Groups with an undefined mean are left out; n_groups says how many
    were used.
    """
    g = groups.dropna(subset=["mean", "freq"])
    n = len(g)
    if n < 3 or g["freq"].nunique() < 2:
        return {"n_groups": n, "spearman_corr": float("nan"),
                "spearman_pvalue": float("nan"), "slope_per_log10_freq": float("nan")}
    corr, pvalue = spearmanr(g["mean"], g["freq"])
    slope = float(np.polyfit(np.log10(g["freq"]), g["mean"], 1)[0])
    return {"n_groups": n, "spearman_corr": float(corr), "spearman_pvalue": float(pvalue),
            "slope_per_log10_freq": slope}


# ---------------------------------------------------------------------------
# Neighbor metrics: read straight off the embedding, no clustering involved.
# ---------------------------------------------------------------------------

def _knn(fit_points, query_points, k, exclude_self):
    """k nearest neighbors of each query row among fit_points (indices into
    fit_points). exclude_self: query_points ARE fit_points, drop each row's
    own index (asks for k+1, removes self, keeps the first k)."""
    from sklearn.neighbors import NearestNeighbors
    k_eff = min(k + (1 if exclude_self else 0), len(fit_points))
    _, idx = NearestNeighbors(n_neighbors=k_eff).fit(fit_points).kneighbors(query_points)
    if not exclude_self:
        return idx
    out = np.empty((len(idx), k_eff - 1), dtype=int)
    for r, row in enumerate(idx):
        row = row[row != r]
        out[r] = row[: k_eff - 1]
    return out


def knn_neighbors(embedding, df, k=15):
    """Each cell's k nearest neighbors in the embedding, three ways:
      "global" -- among all cells,
      "within" -- among cells of its own batch only,
      "cross"  -- among cells of other batches only.
    Returns {scope: list of index arrays (into df rows), one per cell}.
    """
    embedding = np.asarray(embedding, dtype=float)
    batches = df["batch"].to_numpy()
    n = len(df)
    out = {"global": list(_knn(embedding, embedding, k, exclude_self=True)),
           "within": [None] * n, "cross": [None] * n}
    for b in np.unique(batches):
        rows = np.where(batches == b)[0]
        others = np.where(batches != b)[0]
        within = _knn(embedding[rows], embedding[rows], k, exclude_self=True)
        for r, nb in zip(rows, within):
            out["within"][r] = rows[nb]
        if len(others):
            cross = _knn(embedding[others], embedding[rows], k, exclude_self=False)
            for r, nb in zip(rows, cross):
                out["cross"][r] = others[nb]
    return out


def knn_same_type(neighbors, df):
    """Per cell: share of its k nearest neighbors with the same cell type.
    The embedding-level version of `agreement` -- no clusters, so it says
    whether the embedding keeps a cell near its own kind, separately from
    whether clustering later keeps it in a cluster of its own kind."""
    types = df["true_type"].to_numpy()
    return np.array([np.nan if nb is None or len(nb) == 0 else float(np.mean(types[nb] == types[i]))
                     for i, nb in enumerate(neighbors)])


def knn_affinity(affinity, neighbors):
    """Per cell: mean ground-truth affinity to its k nearest embedding
    neighbors (use the within-batch neighbors -- the affinity graph is built
    on uncorrected data, so it barely links cells across batches). High =
    the embedding's neighbors are cells the affinity graph also calls similar."""
    affinity = affinity.tocsr()
    out = np.full(len(neighbors), np.nan)
    for i, nb in enumerate(neighbors):
        if nb is not None and len(nb):
            out[i] = float(np.asarray(affinity[i, nb].todense()).ravel().mean())
    return out
