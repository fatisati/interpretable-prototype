"""Metrics driver for the rare-cell-preservation task.

Reads a baseline's saved embedding plus the shared ground-truth affinity
graphs (from prepare_eval.py), scores it with the generic math in
metrics.py, and saves the result. Rerunnable on its own -- a new metric
later means running this again, never retraining anything.

The only baseline-specific code is the ADAPTERS dict below: each adapter
reshapes that baseline's particular embedding.csv into one or more scoring
"jobs" (a cell_id/batch/true_type/cluster table, its matching affinity
matrix, and a mask saying which rows actually get scored -- see the scvi
adapter's own comments for why "which rows build the cluster structure"
and "which rows get scored" aren't always the same set). Everything after
that is generic and knows nothing about scVI specifically.

Usage:
    python baselines/compute_metrics.py --baseline scvi \\
        --embedding runs/rare_cell_preservation/heart_cell_atlas/scvi/embedding.csv \\
        --affinity-dir runs/rare_cell_preservation/heart_cell_atlas \\
        --label-key cell_type --batch-key donor

Output always goes to a metrics/ folder next to the embedding file (see
metrics_dir_for). There is no --out-dir flag on purpose: results can never
land somewhere stage 5 won't look.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from metrics import (
    agreement, cell_type_purity, cohesion, compute_frequency, compute_rare_mask,
    coverage, frequency_dependence, rare_precision_recall_f1, summarize_per_cell,
)


def scvi_adapter(embedding_path, label_key, batch_key, affinity_dir):
    """scVI is inductive: two jobs, train and test, per the task's own
    evaluation protocol.

    "train": population = train cells only, clustered on their own
    (cluster_train_run), scored against affinity_train.npz. Scored rows =
    every row (the job's whole population is exactly what gets scored).

    "test": population = ALL cells (train + test), clustered together
    (cluster_full_run), scored against affinity_full.npz -- cohesion for a
    test cell needs to know every other cell sharing its cluster,
    including train cells, from this same run. But *assignment* metrics
    (purity, agreement, coverage, rare F1) and the final cohesion average
    only count test cells, per the task doc -- hence score_mask, separate
    from which rows exist in the job's df.
    """
    df = pd.read_csv(embedding_path)
    df = df.rename(columns={label_key: "true_type", batch_key: "batch"})
    affinity_dir = Path(affinity_dir)

    train_rows = df[df["split"] == "train"].reset_index(drop=True).copy()
    train_rows["cluster"] = train_rows["cluster_train_run"]

    full_rows = df.reset_index(drop=True).copy()
    full_rows["cluster"] = full_rows["cluster_full_run"]
    test_score_mask = (full_rows["split"] == "test").to_numpy()

    return {
        "train": {
            "df": train_rows,
            "affinity": load_npz(affinity_dir / "affinity_train.npz"),
            "score_mask": np.ones(len(train_rows), dtype=bool),
        },
        "test": {
            "df": full_rows,
            "affinity": load_npz(affinity_dir / "affinity_full.npz"),
            "score_mask": test_score_mask,
        },
    }


ADAPTERS = {"scvi": scvi_adapter}

# Headline continuous metrics checked for frequency-dependence -- the same
# `frequency_dependence` helper works on any per-cell array, so extending
# this to other metrics later is just adding entries here.
FREQUENCY_CHECK_SCOPES = ["global"]


def score_job(df, affinity, score_mask, rare_pairs_global, std_threshold, min_types_for_std):
    """Score one job. `rare_pairs_global` is the single, fixed, dataset-
    wide rare-type list (computed once from the whole embedding, every
    cell) -- reused for every job so train and test numbers stay
    comparable, instead of each job re-deriving its own rare-type list
    from whichever cells happen to be in it.
    """
    scored_df = df[score_mask].reset_index(drop=True)

    # A rare pair with zero cells in this job's scored population isn't a
    # miss, it's not applicable to this job at all -- drop it rather than
    # let coverage/F1 count it as an automatic failure.
    pairs_present = set(zip(scored_df["batch"], scored_df["true_type"]))
    rare_pairs = {p for p in rare_pairs_global if p in pairs_present}
    is_rare_scored = pd.Series(
        list(zip(scored_df["batch"], scored_df["true_type"]))
    ).isin(rare_pairs).to_numpy()

    result = {}
    freq_rows = []

    purity_mean, purity_std = cell_type_purity(scored_df)
    result["purity"] = {"mean": purity_mean, "std": purity_std}

    for scope in ("global", "within", "cross"):
        values_full = agreement(df, scope)  # needs the FULL job population for cluster structure
        values_scored = values_full[np.asarray(score_mask)]
        result[f"agreement_{scope}"] = summarize_per_cell(values_scored, scored_df, rare_mask=is_rare_scored)
        if scope in FREQUENCY_CHECK_SCOPES:
            freq = compute_frequency(scored_df)
            corr, pvalue, table = frequency_dependence(values_scored, freq)
            freq_rows.append({"metric": f"agreement_{scope}", "spearman_corr": corr,
                               "spearman_pvalue": pvalue, "binned_table": table})

    result["coverage"] = coverage(scored_df, rare_pairs)

    per_pair, (f1_mean, f1_std) = rare_precision_recall_f1(scored_df, rare_pairs)
    result["rare_f1"] = {"mean": f1_mean, "std": f1_std, "n_pairs": len(per_pair)}

    cohesion_full = cohesion(affinity, df["cluster"].to_numpy())
    cohesion_scored = cohesion_full[np.asarray(score_mask)]
    rare_cohesion = cohesion_scored[is_rare_scored]
    result["cohesion"] = {
        "overall": {"mean": float(np.nanmean(cohesion_scored)), "std": float(np.nanstd(cohesion_scored))},
        "rare": {
            "mean": float(np.nanmean(rare_cohesion)) if rare_cohesion.size else float("nan"),
            "std": float(np.nanstd(rare_cohesion)) if rare_cohesion.size else float("nan"),
        },
    }
    freq = compute_frequency(scored_df)
    corr, pvalue, table = frequency_dependence(cohesion_scored, freq)
    freq_rows.append({"metric": "cohesion", "spearman_corr": corr,
                       "spearman_pvalue": pvalue, "binned_table": table})

    return result, per_pair, freq_rows


def flatten_per_batch(summary, job_name):
    """metrics_summary.json's nested per-batch dicts -> one row per
    (job, metric, batch) for a flat, spreadsheet-friendly CSV.
    """
    rows = []
    for metric_name, value in summary.items():
        if not (isinstance(value, dict) and "all_per_batch" in value):
            continue
        for population, per_batch in (("all", value["all_per_batch"]), ("rare", value.get("rare_per_batch", {}))):
            for batch, stats in per_batch.items():
                rows.append({"job": job_name, "metric": metric_name, "population": population,
                             "batch": batch, **stats})
    return rows


def metrics_dir_for(embedding_path):
    """The one place the metrics output location is decided:
    runs/<task>/<dataset>/<baseline>/metrics/. A subfolder of the baseline's
    own run folder, so each folder has one writer and re-scoring never
    touches the (expensive) baseline output."""
    return Path(embedding_path).parent / "metrics"


def run(baseline, embedding_path, label_key, batch_key, affinity_dir,
        std_threshold=1.0, min_types_for_std=4):
    if baseline not in ADAPTERS:
        raise ValueError(f"no adapter for baseline={baseline!r}; add one to ADAPTERS in this file")

    jobs = ADAPTERS[baseline](embedding_path, label_key, batch_key, affinity_dir)

    # One fixed, dataset-wide rare-type list, from every cell across every
    # job -- not re-derived per job (see score_job's docstring for why).
    # A cell can appear in more than one job (e.g. scVI's "test" job's
    # population is every cell, train and test together, same cells as
    # the "train" job's population too) -- dedup by cell_id so frequencies
    # aren't skewed by counting the same cell twice.
    all_rows = pd.concat([job["df"] for job in jobs.values()], ignore_index=True)
    all_rows = all_rows.drop_duplicates(subset="cell_id")
    _, rare_pairs_global = compute_rare_mask(all_rows, std_threshold, min_types_for_std)
    print(f"[metrics] {len(rare_pairs_global)} locally-rare (type, batch) pairs found, dataset-wide")

    summary, per_batch_rows, per_rare_type_rows, freq_rows_all = {}, [], [], []
    for job_name, job in jobs.items():
        print(f"[metrics] scoring job={job_name} ({job['score_mask'].sum():,} scored cells)")
        result, per_pair, freq_rows = score_job(
            job["df"], job["affinity"], job["score_mask"], rare_pairs_global,
            std_threshold, min_types_for_std,
        )
        summary[job_name] = result
        per_batch_rows += flatten_per_batch(result, job_name)
        per_pair["job"] = job_name
        per_rare_type_rows.append(per_pair)
        for row in freq_rows:
            row["job"] = job_name
            freq_rows_all.append(row)

    out_dir = metrics_dir_for(embedding_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    pd.DataFrame(per_batch_rows).to_csv(out_dir / "metrics_per_batch.csv", index=False)
    pd.concat(per_rare_type_rows, ignore_index=True).to_csv(out_dir / "metrics_per_rare_type.csv", index=False)

    freq_summary = pd.DataFrame([
        {"job": r["job"], "metric": r["metric"], "spearman_corr": r["spearman_corr"], "spearman_pvalue": r["spearman_pvalue"]}
        for r in freq_rows_all
    ])
    freq_summary.to_csv(out_dir / "frequency_dependence.csv", index=False)

    (out_dir / "metrics_config.json").write_text(json.dumps({
        "embedding_path": str(embedding_path), "affinity_dir": str(affinity_dir),
        "std_threshold": std_threshold, "min_types_for_std": min_types_for_std,
        "n_rare_pairs": len(rare_pairs_global),
    }, indent=2))

    print(f"[metrics] wrote {out_dir / 'metrics_summary.json'}")
    return out_dir


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", required=True, choices=list(ADAPTERS))
    p.add_argument("--embedding", required=True, help="path to that baseline's embedding.csv")
    p.add_argument("--label-key", required=True)
    p.add_argument("--batch-key", required=True)
    p.add_argument("--affinity-dir", required=True, help="dataset dir containing affinity_train/full.npz")
    p.add_argument("--std-threshold", type=float, default=1.0)
    p.add_argument("--min-types-for-std", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        baseline=args.baseline,
        embedding_path=args.embedding,
        label_key=args.label_key,
        batch_key=args.batch_key,
        affinity_dir=args.affinity_dir,
        std_threshold=args.std_threshold,
        min_types_for_std=args.min_types_for_std,
    )
