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
    python evaluation/compute_metrics.py --baseline scvi \\
        --embedding runs/rare_cell_preservation/heart_cell_atlas/scvi/embedding.csv \\
        --affinity-dir runs/rare_cell_preservation/heart_cell_atlas \\
        --label-key cell_type --batch-key donor

Output always goes to a metrics/ folder next to the embedding file (see
metrics_dir_for). There is no --out-dir flag on purpose: results can never
land somewhere stage 5 won't look.
"""
import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.sparse import load_npz

from run_record import record_run  # first local import: records the script's start time
from metrics import (
    RARE_RULES, agreement, cell_type_purity, cohesion, knn_affinity, knn_neighbors, knn_same_type,
    rare_group_table,
    coverage, group_summary, group_trend, rare_precision_recall_f1, summarize_per_cell,
)


def embedding_adapter(embedding_path, label_key, batch_key, affinity_dir):
    """For any baseline that saves an embedding (latent_* columns) plus a
    cluster label per cell (scVI, PCA, later Harmony/ComBat).

    ONE clustering of all cells (cluster_full_run) and the full affinity
    graph, for both jobs -- so train and test differ only in whether the
    model saw those cells, not in which clustering produced their labels.
      "train": scored rows = train cells.
      "test":  scored rows = test cells.
    Both jobs keep every cell in `df`: a cell's cluster-mates and embedding
    neighbors can be any cell, train or test.
    """
    df = pd.read_csv(embedding_path)
    df = df.rename(columns={label_key: "true_type", batch_key: "batch"}).reset_index(drop=True)
    df["cluster"] = df["cluster_full_run"]
    latent_cols = [c for c in df.columns if c.startswith("latent_")]
    embedding = df[latent_cols].to_numpy(dtype=float)
    affinity = load_npz(Path(affinity_dir) / "affinity_full.npz")
    split = df["split"].to_numpy()
    return {
        job: {"df": df, "affinity": affinity, "embedding": embedding, "score_mask": split == job}
        for job in ("train", "test")
    }


def seacells_adapter(embedding_path, label_key, batch_key, affinity_dir):
    """For seacells_baseline.py's assignments.csv. SEACells can't embed
    unseen cells, so there is no "test" job (task file, SEACells section).
    Instead, one job per SEACells run, each with its own clustering:
      "train": train cells only, cluster_train_run, affinity_train.npz --
               same cells scVI trains on (the fair cohesion comparison).
      "full":  every cell, cluster_full_run, affinity_full.npz -- the
               practical "how SEACells is really used" number.
    Every row in each job is scored. No embedding -> no neighbor metrics.
    """
    df = pd.read_csv(embedding_path)
    df = df.rename(columns={label_key: "true_type", batch_key: "batch"})
    train = df[df["split"] == "train"].reset_index(drop=True).copy()
    train["cluster"] = train["cluster_train_run"]
    full = df.reset_index(drop=True).copy()
    full["cluster"] = full["cluster_full_run"]
    affinity_dir = Path(affinity_dir)
    return {
        "train": {"df": train, "affinity": load_npz(affinity_dir / "affinity_train.npz"),
                  "score_mask": np.ones(len(train), dtype=bool)},
        "full": {"df": full, "affinity": load_npz(affinity_dir / "affinity_full.npz"),
                 "score_mask": np.ones(len(full), dtype=bool)},
    }


ADAPTERS = {"scvi": embedding_adapter, "pca": embedding_adapter, "seacells": seacells_adapter}



def score_job(df, affinity, score_mask, rare_pairs_global, freq_lookup, embedding=None, k_neighbors=15):
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
    # Frequency question, asked per group (see metrics.group_summary):
    # one group table + one trend row per per-cell metric.
    group_tables, trend_rows = [], []

    def frequency_check(metric_name, values_scored):
        groups = group_summary(values_scored, scored_df, is_rare_scored, freq_lookup)
        groups.insert(0, "metric", metric_name)
        group_tables.append(groups)
        trend_rows.append({"metric": metric_name, **group_trend(groups)})

    purity_mean, purity_std = cell_type_purity(scored_df)
    result["purity"] = {"mean": purity_mean, "std": purity_std}

    for scope in ("global", "within", "cross"):
        values_full = agreement(df, scope)  # needs the FULL job population for cluster structure
        values_scored = values_full[np.asarray(score_mask)]
        result[f"agreement_{scope}"] = summarize_per_cell(values_scored, scored_df, rare_mask=is_rare_scored)
        frequency_check(f"agreement_{scope}", values_scored)

    # Per-batch values plus one dataset-wide mean/std, saved here so stage 5
    # (results) never has to aggregate anything itself.
    coverage_per_batch = coverage(scored_df, rare_pairs)
    coverage_values = np.array(list(coverage_per_batch.values()), dtype=float)
    result["coverage"] = {
        "per_batch": coverage_per_batch,
        "mean": float(coverage_values.mean()) if coverage_values.size else float("nan"),
        "std": float(coverage_values.std()) if coverage_values.size else float("nan"),
        "n_batches": int(coverage_values.size),
    }

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
    frequency_check("cohesion", cohesion_scored)

    # Neighbor metrics: straight off the embedding, no clusters. Only for
    # baselines that have an embedding (not SEACells / BBKNN).
    if embedding is not None:
        mask = np.asarray(score_mask)
        neighbors = knn_neighbors(embedding, df, k_neighbors)
        for scope in ("global", "within", "cross"):
            values = knn_same_type(neighbors[scope], df)[mask]
            result[f"knn_same_type_{scope}"] = summarize_per_cell(values, scored_df, rare_mask=is_rare_scored)
            frequency_check(f"knn_same_type_{scope}", values)
        values = knn_affinity(affinity, neighbors["within"])[mask]
        result["knn_affinity_within"] = summarize_per_cell(values, scored_df, rare_mask=is_rare_scored)
        frequency_check("knn_affinity_within", values)

    return result, per_pair, pd.concat(group_tables, ignore_index=True), trend_rows


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


OFFICIAL_RARE_RULE = "log_std"  # task file + docs/decisions/0002-rare-cell-type-rule.md


def metrics_dir_for(embedding_path, rare_rule=OFFICIAL_RARE_RULE):
    """The one place the metrics output location is decided:
    runs/<task>/<dataset>/<baseline>/metrics/. A subfolder of the baseline's
    own run folder, so each folder has one writer and re-scoring never
    touches the (expensive) baseline output.

    The task file's rare rule (OFFICIAL_RARE_RULE) writes to metrics/. Any other rare rule
    writes to its own metrics__rare-<rule>/ next to it -- a what-if copy for
    comparing rules side by side, never overwriting the official numbers."""
    name = "metrics" if rare_rule == OFFICIAL_RARE_RULE else f"metrics__rare-{rare_rule}"
    return Path(embedding_path).parent / name


def validate_jobs(jobs, embedding_path):
    """Fail loudly on a stale or broken baseline output instead of scoring
    it. Checks every row in each job's df, not just scored rows: cohesion
    for a scored cell depends on every other cell in its cluster, so one
    unlabeled train cell silently corrupts test cohesion. (This exact case
    happened: an old scvi embedding.csv had no cluster_full_run label for
    any train cell, and metrics ran without complaint.)"""
    for job_name, job in jobs.items():
        df = job["df"]
        n_missing = int(df["cluster"].isna().sum())
        if n_missing:
            raise ValueError(
                f"job={job_name}: {n_missing:,} of {len(df):,} cells have no cluster label in "
                f"{embedding_path}. Likely a stale baseline output -- re-run the baseline."
            )
        n_aff = job["affinity"].shape[0]
        if n_aff != len(df):
            raise ValueError(
                f"job={job_name}: affinity graph has {n_aff:,} cells but the job has {len(df):,}. "
                f"Baseline output and stage 2 files don't match -- re-run whichever is older."
            )


def run(baseline, embedding_path, label_key, batch_key, affinity_dir,
        std_threshold=1.0, min_types_for_std=4, rare_rule=OFFICIAL_RARE_RULE, fixed_cutoff=0.01):
    if baseline not in ADAPTERS:
        raise ValueError(f"no adapter for baseline={baseline!r}; add one to ADAPTERS in this file")

    jobs = ADAPTERS[baseline](embedding_path, label_key, batch_key, affinity_dir)
    validate_jobs(jobs, embedding_path)

    # One fixed, dataset-wide rare-type list, from every cell across every
    # job -- not re-derived per job (see score_job's docstring for why).
    # A cell can appear in more than one job (e.g. scVI's "test" job's
    # population is every cell, train and test together, same cells as
    # the "train" job's population too) -- dedup by cell_id so frequencies
    # aren't skewed by counting the same cell twice.
    all_rows = pd.concat([job["df"] for job in jobs.values()], ignore_index=True)
    all_rows = all_rows.drop_duplicates(subset="cell_id")
    rare_table = rare_group_table(all_rows, rare_rule, std_threshold, min_types_for_std, fixed_cutoff)
    rare = rare_table[rare_table["is_rare"]]
    rare_pairs_global = set(zip(rare["batch"], rare["true_type"]))
    # Each group's frequency, from ALL cells -- frequency belongs to the
    # group, not to the train/test split, so train and test jobs use the
    # same value (and tiny groups don't get a noisy test-only frequency).
    freq_lookup = rare_table.set_index(["batch", "true_type"])["freq"]
    print(f"[metrics] rare rule={rare_rule}: {len(rare_pairs_global)} locally-rare (type, batch) "
          f"pairs found, dataset-wide")

    summary, per_batch_rows, per_rare_type_rows, group_tables_all, trend_rows_all = {}, [], [], [], []
    for job_name, job in jobs.items():
        print(f"[metrics] scoring job={job_name} ({job['score_mask'].sum():,} scored cells)")
        result, per_pair, groups, trend_rows = score_job(
            job["df"], job["affinity"], job["score_mask"], rare_pairs_global, freq_lookup,
            embedding=job.get("embedding"),
        )
        summary[job_name] = result
        per_batch_rows += flatten_per_batch(result, job_name)
        per_pair["job"] = job_name
        per_rare_type_rows.append(per_pair)
        groups.insert(0, "job", job_name)
        group_tables_all.append(groups)
        trend_rows_all += [{"job": job_name, **row} for row in trend_rows]

    out_dir = metrics_dir_for(embedding_path, rare_rule)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The rare/common decision for every group, with the cutoff it was
    # compared against -- so "why is this group rare?" is answerable.
    rare_table.to_csv(out_dir / "rare_groups.csv", index=False)

    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    pd.DataFrame(per_batch_rows).to_csv(out_dir / "metrics_per_batch.csv", index=False)
    pd.concat(per_rare_type_rows, ignore_index=True).to_csv(out_dir / "metrics_per_rare_type.csv", index=False)

    # Frequency question, per group (cell type x batch): the trend across
    # groups, one row per (job, metric), and the group table behind it.
    pd.DataFrame(trend_rows_all).to_csv(out_dir / "frequency_dependence.csv", index=False)
    pd.concat(group_tables_all, ignore_index=True).to_csv(out_dir / "frequency_groups.csv", index=False)

    (out_dir / "metrics_config.json").write_text(json.dumps({
        "embedding_path": str(embedding_path), "affinity_dir": str(affinity_dir),
        "rare_rule": rare_rule,
        "rare_rule_text": RARE_RULES[rare_rule].format(k=std_threshold, cutoff=fixed_cutoff),
        "std_threshold": std_threshold, "min_types_for_std": min_types_for_std,
        "fixed_cutoff": fixed_cutoff,
        "n_rare_pairs": len(rare_pairs_global),
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__},
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
    p.add_argument("--rare-rule", default=OFFICIAL_RARE_RULE, choices=["std", "log_std", "fixed"],
                   help="log_std = task file's rule (writes metrics/); others write metrics__rare-<rule>/")
    p.add_argument("--fixed-cutoff", type=float, default=0.01, help="for --rare-rule fixed: 0.01 = 1%%")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = run(
        baseline=args.baseline,
        embedding_path=args.embedding,
        label_key=args.label_key,
        batch_key=args.batch_key,
        affinity_dir=args.affinity_dir,
        std_threshold=args.std_threshold,
        min_types_for_std=args.min_types_for_std,
        rare_rule=args.rare_rule,
        fixed_cutoff=args.fixed_cutoff,
    )
    record_run(out_dir / "metrics_config.json")
