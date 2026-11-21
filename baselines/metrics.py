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


def compute_rare_mask(df, std_threshold=1.0, min_types_for_std=4):
    """Per batch: a (type, batch) pair is rare if its frequency is more
    than `std_threshold` standard deviations below that batch's mean
    type-frequency. Fallback (fewer than `min_types_for_std` distinct
    types in that batch -- not specified in the task doc, our own
    judgment call, change it here if needed): lowest quartile instead.

    Returns (is_rare: bool array aligned to df, rare_pairs: set of (batch, true_type)).
    """
    type_freq = df.groupby(["batch", "true_type"]).size().rename("n").reset_index()
    batch_totals = df.groupby("batch").size().rename("n_batch")
    type_freq["freq"] = type_freq["n"] / type_freq["batch"].map(batch_totals)

    rare_pairs = set()
    for batch, sub in type_freq.groupby("batch"):
        freqs = sub.set_index("true_type")["freq"]
        if len(freqs) >= min_types_for_std:
            threshold = freqs.mean() - std_threshold * freqs.std()
            rare_types = freqs[freqs < threshold].index
        else:
            q1 = freqs.quantile(0.25)
            rare_types = freqs[freqs <= q1].index
        rare_pairs.update((batch, t) for t in rare_types)

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
        if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
            f1 = float("nan")
        else:
            f1 = 2 * precision * recall / (precision + recall)

        rows.append({
            "true_type": t, "batch": b, "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        })

    per_pair = pd.DataFrame(rows)
    return per_pair, (float(per_pair["f1"].mean()), float(per_pair["f1"].std()))


def cohesion(affinity, cluster_labels):
    """Per-cell cohesion: average sim(i, j) over every j in i's cluster,
    with sim(i, i) := 1 (the stored affinity matrix has a 0 diagonal, so
    that's added explicitly here rather than read off the matrix).
    """
    n = affinity.shape[0]
    result = np.zeros(n)
    for cluster_id in pd.unique(cluster_labels):
        idx = np.where(cluster_labels == cluster_id)[0]
        sub = affinity[idx][:, idx]
        row_sums = np.asarray(sub.sum(axis=1)).ravel() + 1.0
        result[idx] = row_sums / len(idx)
    return result


def frequency_dependence(metric_values, frequency_values, n_bins=10):
    """Spearman correlation between a per-cell metric and its (type, batch)
    frequency, plus a binned table of mean metric value per frequency bin
    -- checks the decline is smooth across the whole range, not just a
    jump at the rare/common threshold (see task doc's frequency-dependence
    check). NaNs in either array are dropped before computing.
    """
    mask = ~np.isnan(metric_values) & ~np.isnan(frequency_values)
    corr, pvalue = spearmanr(metric_values[mask], frequency_values[mask])

    bins = pd.qcut(frequency_values[mask], q=n_bins, duplicates="drop")
    table = pd.DataFrame({"metric": metric_values[mask], "bin": bins}).groupby("bin")["metric"].agg(
        ["mean", "std", "count"]
    )
    return float(corr), float(pvalue), table
