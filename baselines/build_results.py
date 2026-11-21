"""Stage 5 (results) for the rare-cell-preservation task.

Reads every baseline's saved metrics (stage 4's output, under
runs/<task>/<dataset>/<baseline>/metrics/) and writes:

  results/<task>_tables.md    comparison tables + one PASS/FAIL row per
                              expectation in the task file. GENERATED --
                              never hand-edit; re-run this script instead.
  results/<task>_summary.csv  every headline number, one row each (tidy
                              format, for plots or spreadsheets later).

The verdict itself (results/<task>.md) is written by hand, citing these
tables. This script only reads, joins, formats, and compares numbers
against rules fixed in advance -- it never computes a metric. Any new
number (a slope, an average across batches, ...) belongs in
compute_metrics.py, so there is exactly one place numbers come from
(see docs/architecture.md, stage 5's "must never" column).

Output is deterministic: same metrics files in -> byte-identical files
out (sorted order, fixed number formatting, no timestamps), so a diff
of the generated file shows exactly what changed between runs.

Usage:
    python baselines/build_results.py --task rare_cell_preservation
"""
import argparse
import json
import math
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Expectation rules. These mirror the "What I expect" section of
# tasks/rare_cell_preservation.md (plus the SEACells section's own
# expectation). They are fixed BEFORE any results exist. Changing one after
# seeing results is allowed only with a written note in results/<task>.md
# saying what changed and why -- otherwise the check just moves to fit the
# data.
# ---------------------------------------------------------------------------

# E1: a metric "drops as frequency falls" if its SLOPE (change per 10x
# change in frequency, saved by compute_metrics.py) is positive. The slope
# decides because it measures how BIG the drop is. Spearman only measures
# ORDER, so a tiny consistent difference can outvote a big one -- e.g.
# cohesion counts a cell's similarity to itself as 1, giving small clusters
# a ~0.01 bonus that Spearman can rank above a 0.7 loss for scattered rare
# cells (seen in tests/check_build_results.py). Spearman is still shown;
# when it disagrees (not positive with p < ALPHA), the row is flagged for a
# human look at the binned table instead of silently overruling it.
ALPHA = 0.05

# Job-name convention every adapter in compute_metrics.py must follow, so
# this script knows which runs are comparable:
#   scVI:                    train, test
#   SEACells:                train, full
#   Harmony / BBKNN / ComBat: full
REFERENCE_BASELINE = "seacells"
INDUCTIVE_BASELINE = "scvi"
EXPECTED_BASELINES = ["bbknn", "combat", "harmony", "scvi", "seacells"]

# E3/E4: which reference (SEACells) job each job is compared against. A job
# not listed here has no fair SEACells equivalent (scVI's test job: SEACells
# can't be evaluated on unseen cells -- task file, SEACells section).
REFERENCE_JOB_FOR = {"train": "train", "full": "full"}

# E4: assignment metrics where SEACells (full) is expected to beat every
# correction baseline (full). (metric, population) pairs; higher = better
# for all of them.
E4_METRICS = [
    ("purity", "all"),
    ("agreement_global", "rare"),
    ("agreement_within", "rare"),
    ("agreement_cross", "rare"),
    ("coverage", "rare"),
    ("rare_f1", "rare"),
]

AGREEMENT_METRICS = ["agreement_global", "agreement_within", "agreement_cross"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def find_metrics_dirs(runs_root, task):
    """Every runs/<task>/<dataset>/<baseline>/metrics/ that has a summary,
    sorted so output order never depends on filesystem order."""
    task_dir = Path(runs_root) / task
    found = sorted(task_dir.glob("*/*/metrics/metrics_summary.json"))
    return [(p.parent.parent.parent.name, p.parent.parent.name, p.parent) for p in found]


def summary_to_rows(summary, dataset, baseline):
    """metrics_summary.json -> tidy rows: one per (job, metric, population).
    Pure reshaping; every number is copied, none is computed."""
    rows = []

    def add(job, metric, population, stats, **extra):
        rows.append({
            "dataset": dataset, "baseline": baseline, "job": job,
            "metric": metric, "population": population,
            "mean": stats.get("mean", math.nan), "std": stats.get("std", math.nan),
            "coverage": stats.get("coverage", math.nan), "n": stats.get("n", math.nan),
            **extra,
        })

    for job, result in summary.items():
        add(job, "purity", "all", result["purity"])
        for metric in AGREEMENT_METRICS:
            add(job, metric, "all", result[metric]["all_overall"])
            add(job, metric, "rare", result[metric]["rare_overall"])
        cov = result["coverage"]
        if "mean" not in cov:
            raise ValueError(
                f"{dataset}/{baseline}: coverage has no dataset-wide mean -- metrics were made by "
                f"an older compute_metrics.py. Re-run compute_metrics.py for this baseline."
            )
        add(job, "coverage", "rare", {"mean": cov["mean"], "std": cov["std"], "n": cov["n_batches"]})
        add(job, "rare_f1", "rare", {"mean": result["rare_f1"]["mean"], "std": result["rare_f1"]["std"],
                                     "n": result["rare_f1"]["n_pairs"]})
        add(job, "cohesion", "all", result["cohesion"]["overall"])
        add(job, "cohesion", "rare", result["cohesion"]["rare"])
    return rows


def load_all(runs_root, task):
    """Returns (headline DataFrame, frequency-dependence DataFrame,
    frequency-bins DataFrame, list of run info dicts)."""
    headline, freq, bins, runs = [], [], [], []
    for dataset, baseline, mdir in find_metrics_dirs(runs_root, task):
        summary = json.loads((mdir / "metrics_summary.json").read_text())
        headline += summary_to_rows(summary, dataset, baseline)

        fd = pd.read_csv(mdir / "frequency_dependence.csv")
        if "slope_per_log10_freq" not in fd.columns:
            raise ValueError(
                f"{mdir}: frequency_dependence.csv has no slope column -- made by an older "
                f"compute_metrics.py. Re-run compute_metrics.py for this baseline."
            )
        fd.insert(0, "baseline", baseline)
        fd.insert(0, "dataset", dataset)
        freq.append(fd)

        bins_path = mdir / "frequency_bins.csv"
        if bins_path.exists():
            fb = pd.read_csv(bins_path)
            fb.insert(0, "baseline", baseline)
            fb.insert(0, "dataset", dataset)
            bins.append(fb)

        config_path = mdir / "metrics_config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        runs.append({"dataset": dataset, "baseline": baseline, "jobs": sorted(summary),
                     "path": mdir.as_posix(), "config": config})

    headline_df = pd.DataFrame(headline)
    freq_df = pd.concat(freq, ignore_index=True) if freq else pd.DataFrame(
        columns=["dataset", "baseline", "job", "metric", "spearman_corr", "spearman_pvalue", "slope_per_log10_freq"])
    bins_df = pd.concat(bins, ignore_index=True) if bins else pd.DataFrame()
    sort_cols = ["dataset", "baseline", "job", "metric"]
    if not headline_df.empty:
        headline_df = headline_df.sort_values(sort_cols + ["population"], kind="stable").reset_index(drop=True)
    freq_df = freq_df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    if not bins_df.empty:
        bins_df = bins_df.sort_values(sort_cols + ["freq_low"], kind="stable").reset_index(drop=True)
    return headline_df, freq_df, bins_df, runs


# ---------------------------------------------------------------------------
# Expectation checks -- comparisons of saved numbers only
# ---------------------------------------------------------------------------

def _is_num(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def lookup(headline, dataset, baseline, job, metric, population):
    hit = headline[(headline["dataset"] == dataset) & (headline["baseline"] == baseline)
                   & (headline["job"] == job) & (headline["metric"] == metric)
                   & (headline["population"] == population)]
    return float(hit["mean"].iloc[0]) if len(hit) else math.nan


def e1_status(slope, corr, pvalue):
    """Slope decides; Spearman disagreeing only adds a flag (see ALPHA's comment)."""
    if not _is_num(slope):
        return "N/A", "slope undefined"
    status = "PASS" if slope > 0 else "FAIL"
    reason = "" if slope > 0 else "metric does not fall with frequency"
    spearman_agrees = _is_num(corr) and _is_num(pvalue) and corr > 0 and pvalue < ALPHA
    if (slope > 0) != spearman_agrees:
        flag = "Spearman disagrees, check binned table"
        reason = f"{reason}; {flag}" if reason else flag
    return status, reason


def check_e1(freq):
    rows = []
    for r in freq.itertuples(index=False):
        status, reason = e1_status(r.slope_per_log10_freq, r.spearman_corr, r.spearman_pvalue)
        rows.append({"dataset": r.dataset, "baseline": r.baseline, "job": r.job, "metric": r.metric,
                     "spearman": r.spearman_corr, "p": r.spearman_pvalue,
                     "slope": r.slope_per_log10_freq, "status": status, "reason": reason})
    return rows


def check_e2(freq, datasets):
    """scVI cohesion falls with frequency on train AND test, and the test
    slope is at least as steep as the train slope."""
    rows = []
    for dataset in datasets:
        sub = freq[(freq["dataset"] == dataset) & (freq["baseline"] == INDUCTIVE_BASELINE)
                   & (freq["metric"] == "cohesion")].set_index("job")
        if not {"train", "test"} <= set(sub.index):
            rows.append({"dataset": dataset, "status": "N/A",
                         "detail": f"{INDUCTIVE_BASELINE} cohesion for train and test not both found"})
            continue
        tr, te = sub.loc["train"], sub.loc["test"]
        tr_status, _ = e1_status(tr["slope_per_log10_freq"], tr["spearman_corr"], tr["spearman_pvalue"])
        te_status, _ = e1_status(te["slope_per_log10_freq"], te["spearman_corr"], te["spearman_pvalue"])
        steeper = _is_num(te["slope_per_log10_freq"]) and _is_num(tr["slope_per_log10_freq"]) \
            and te["slope_per_log10_freq"] >= tr["slope_per_log10_freq"]
        status = "PASS" if (tr_status == "PASS" and te_status == "PASS" and steeper) else "FAIL"
        detail = (f"train: falls={tr_status}, slope={fmt(tr['slope_per_log10_freq'])}; "
                  f"test: falls={te_status}, slope={fmt(te['slope_per_log10_freq'])}; "
                  f"test at least as steep={'yes' if steeper else 'no'}")
        rows.append({"dataset": dataset, "status": status, "detail": detail})
    return rows


def check_vs_reference(headline, metrics, skip_jobs_without_reference=False):
    """For every non-reference baseline job, compare each (metric,
    population) against the SEACells job it's fairly comparable to.
    PASS = SEACells strictly higher."""
    rows = []
    if headline.empty:
        return rows
    combos = headline[["dataset", "baseline", "job"]].drop_duplicates().itertuples(index=False)
    for dataset, baseline, job in combos:
        if baseline == REFERENCE_BASELINE:
            continue
        ref_job = REFERENCE_JOB_FOR.get(job)
        for metric, population in metrics:
            base = {"dataset": dataset, "baseline": baseline, "job": job,
                    "metric": f"{metric} ({population})"}
            if ref_job is None:
                if not skip_jobs_without_reference:
                    rows.append({**base, "value": math.nan, "reference": math.nan, "status": "N/A",
                                 "reason": f"no SEACells equivalent for job '{job}'"})
                continue
            value = lookup(headline, dataset, baseline, job, metric, population)
            ref = lookup(headline, dataset, REFERENCE_BASELINE, ref_job, metric, population)
            if not _is_num(ref):
                status, reason = "N/A", f"SEACells '{ref_job}' metrics not found"
            elif not _is_num(value):
                status, reason = "N/A", "value undefined"
            elif value < ref:
                status, reason = "PASS", ""
            else:
                status, reason = "FAIL", "not below SEACells"
            rows.append({**base, "value": value, "reference": ref, "status": status, "reason": reason})
    return rows


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt(x, digits=3):
    if not _is_num(x):
        return "—"
    return f"{x:.{digits}f}"


def fmt_p(p):
    if not _is_num(p):
        return "—"
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def mean_std(headline, dataset, baseline, job, metric, population, with_coverage=False):
    hit = headline[(headline["dataset"] == dataset) & (headline["baseline"] == baseline)
                   & (headline["job"] == job) & (headline["metric"] == metric)
                   & (headline["population"] == population)]
    if hit.empty:
        return "—"
    r = hit.iloc[0]
    text = f"{fmt(r['mean'])} ± {fmt(r['std'])}"
    if with_coverage and _is_num(r["coverage"]):
        text += f" (cov {r['coverage']:.0%})"
    return text


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def count_line(rows):
    counts = {s: sum(r["status"] == s for r in rows) for s in ("PASS", "FAIL", "N/A")}
    return f"**{counts['PASS']} PASS, {counts['FAIL']} FAIL, {counts['N/A']} N/A** (of {len(rows)} checks)"


def render(task, headline, freq, bins, runs):
    lines = [
        f"# {task} — generated tables",
        "",
        "> **GENERATED FILE — do not edit.** Made by `baselines/build_results.py` from",
        f"> `runs/{task}/*/*/metrics/`. Re-run that script to update. The written",
        f"> verdict lives in `results/{task}.md`.",
        "",
        "Every value is mean ± std, higher = better. \"Rare\" = locally-rare (type, batch)",
        "pairs, as defined in the task file. `cov` = share of cells where the metric is",
        "defined (cross-batch agreement is undefined for a cell whose cluster has no",
        "other-batch cells).",
        "",
        "## Inputs",
        "",
    ]
    if not runs:
        lines += [f"No metrics found under `runs/{task}/`. Run `compute_metrics.py` first.", ""]
        return "\n".join(lines) + "\n"

    import platform, numpy, scipy  # noqa: E401 -- only for the environment line
    lines += [f"Environment (this script): python {platform.python_version()}, pandas {pd.__version__}, "
              f"numpy {numpy.__version__}, scipy {scipy.__version__}. Metric versions: see each "
              f"metrics folder's own config.", ""]
    lines.append(md_table(
        ["dataset", "baseline", "jobs", "rare pairs", "rare threshold (std)", "metrics folder"],
        [[r["dataset"], r["baseline"], ", ".join(r["jobs"]), r["config"].get("n_rare_pairs", "—"),
          r["config"].get("std_threshold", "—"), f"`{r['path']}`"] for r in runs],
    ))
    datasets = sorted({r["dataset"] for r in runs})
    for dataset in datasets:
        present = {r["baseline"] for r in runs if r["dataset"] == dataset}
        missing = [b for b in EXPECTED_BASELINES if b not in present]
        if missing:
            lines += ["", f"**{dataset}: not scored yet:** {', '.join(missing)}"]
    lines.append("")

    # --- headline tables, one pair per dataset -----------------------------
    for dataset in datasets:
        combos = headline[headline["dataset"] == dataset][["baseline", "job"]].drop_duplicates()
        combos = list(combos.itertuples(index=False))
        lines += [f"## {dataset}", "", "### Agreement (all cells / rare cells)", ""]
        header = ["baseline", "job"]
        for m in AGREEMENT_METRICS:
            short = m.replace("agreement_", "")
            header += [f"{short} — all", f"{short} — rare"]
        rows = []
        for baseline, job in combos:
            row = [baseline, job]
            for m in AGREEMENT_METRICS:
                cov = m == "agreement_cross"
                row += [mean_std(headline, dataset, baseline, job, m, "all", cov),
                        mean_std(headline, dataset, baseline, job, m, "rare", cov)]
            rows.append(row)
        lines += [md_table(header, rows), "", "### Purity, rare coverage, rare F1, cohesion", ""]
        rows = [[b, j,
                 mean_std(headline, dataset, b, j, "purity", "all"),
                 mean_std(headline, dataset, b, j, "coverage", "rare"),
                 mean_std(headline, dataset, b, j, "rare_f1", "rare"),
                 mean_std(headline, dataset, b, j, "cohesion", "all"),
                 mean_std(headline, dataset, b, j, "cohesion", "rare")] for b, j in combos]
        lines += [md_table(["baseline", "job", "purity", "rare coverage", "rare F1",
                            "cohesion — all", "cohesion — rare"], rows), ""]

    # --- expectation checks ------------------------------------------------
    e1 = check_e1(freq)
    e2 = check_e2(freq, datasets)
    e3 = check_vs_reference(headline, [("cohesion", "rare")])
    e4 = check_vs_reference(headline, E4_METRICS, skip_jobs_without_reference=True)
    e4 = [r for r in e4 if r["job"] == "full"]

    lines += [
        "## Expectation checks",
        "",
        "Rules are fixed in `build_results.py` (top of file) and mirror the task file's",
        "\"What I expect\" section. PASS/FAIL only says whether a saved number crossed a",
        "pre-set line; whether the gap is *real* is the handwritten verdict's job.",
        "",
        "### E1 — every metric drops as frequency falls",
        "",
        "PASS = `slope` > 0. `slope` = change in the metric per 10x change in frequency",
        "(positive = drops as frequency drops); it decides because it measures how big",
        "the drop is. `spearman` only measures order, so a tiny consistent difference can",
        f"outvote a big one; when it disagrees (not > 0 with p < {ALPHA}) the row is flagged —",
        "check that metric's binned table in the appendix. Caveat: with thousands of cells",
        "almost any p-value is tiny, and cells in one cluster aren't independent — judge",
        "the size of the slope and the binned tables, not the p-value.",
        "",
        count_line(e1),
        "",
        md_table(["dataset", "baseline", "job", "metric", "spearman", "p", "slope", "status"],
                 [[r["dataset"], r["baseline"], r["job"], r["metric"], fmt(r["spearman"]), fmt_p(r["p"]),
                   fmt(r["slope"]), r["status"] + (f" — {r['reason']}" if r["reason"] else "")] for r in e1]),
        "",
        "### E2 — scVI cohesion falls with frequency on train and test, test at least as steep",
        "",
        count_line(e2),
        "",
        md_table(["dataset", "status", "detail"], [[r["dataset"], r["status"], r["detail"]] for r in e2]),
        "",
        "### E3 — every baseline's rare cohesion is below SEACells'",
        "",
        "Compared like for like: train job vs SEACells train run, full job vs SEACells full",
        "run. scVI's test job has no SEACells equivalent (SEACells can't score unseen cells).",
        "",
        count_line(e3),
        "",
        md_table(["dataset", "baseline", "job", "rare cohesion", "SEACells", "status"],
                 [[r["dataset"], r["baseline"], r["job"], fmt(r["value"]), fmt(r["reference"]),
                   r["status"] + (f" — {r['reason']}" if r["reason"] else "")] for r in e3]),
        "",
        "### E4 — SEACells (full) beats every correction baseline (full) on assignment metrics",
        "",
        "From the task file's SEACells section. Only jobs named `full` are compared, so",
        "scVI (train/test only) is not in this check.",
        "",
        count_line(e4),
        "",
        md_table(["dataset", "baseline", "metric", "value", "SEACells", "status"],
                 [[r["dataset"], r["baseline"], r["metric"], fmt(r["value"]), fmt(r["reference"]),
                   r["status"] + (f" — {r['reason']}" if r["reason"] else "")] for r in e4]),
        "",
    ]

    # --- binned frequency tables (appendix) --------------------------------
    if not bins.empty:
        lines += [
            "## Appendix — metric by frequency bin",
            "",
            "Cells grouped into bins by their type's frequency within their batch (rarest",
            "first). A smooth fall across bins = gradual loss; a flat line then one drop at",
            "the low end = loss concentrated at the rare threshold. Both support the claim;",
            "flat everywhere does not.",
            "",
        ]
        for (dataset, baseline, job, metric), sub in bins.groupby(
                ["dataset", "baseline", "job", "metric"], sort=True):
            lines += [f"**{dataset} / {baseline} / {job} / {metric}**", "",
                      md_table(["frequency range", "mean", "std", "cells"],
                               [[f"{r.freq_low:.4f} – {r.freq_high:.4f}", fmt(r.mean), fmt(r.std), int(r.count)]
                                for r in sub.itertuples(index=False)]), ""]

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------

def run(task, runs_root="runs", results_dir="results"):
    headline, freq, bins, runs = load_all(runs_root, task)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    tables_path = results_dir / f"{task}_tables.md"
    tables_path.write_text(render(task, headline, freq, bins, runs), encoding="utf-8", newline="\n")

    summary_path = results_dir / f"{task}_summary.csv"
    headline.to_csv(summary_path, index=False, lineterminator="\n", float_format="%.6g")

    print(f"[results] {len(runs)} metrics folder(s) read")
    print(f"[results] wrote {tables_path}")
    print(f"[results] wrote {summary_path}")
    return tables_path, summary_path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, help="task name, e.g. rare_cell_preservation")
    p.add_argument("--runs-root", default="runs", help="(for tests) folder holding runs/<task>/")
    p.add_argument("--results-dir", default="results", help="(for tests) where output goes")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.task, args.runs_root, args.results_dir)
