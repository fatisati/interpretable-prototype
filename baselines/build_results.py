"""Stage 5 (results) for the rare-cell-preservation task.

Reads every baseline's saved metrics (stage 4's output, under
runs/<task>/<dataset>/<baseline>/metrics/) and writes:

  results/<task>_tables.md    comparison tables. GENERATED -- never
                              hand-edit; re-run this script instead.
  results/<task>_summary.csv  every headline number, one row each (tidy
                              format, for plots or spreadsheets later).

Mental model: a scoreboard. It (1) collects every saved metrics folder
and says which baselines are still missing, (2) lines the numbers up side
by side, (3) prints them. It never computes a number -- any new number (a
slope, an average across batches, ...) belongs in compute_metrics.py, so
there is exactly one place numbers come from -- and it never judges
them: no PASS/FAIL. Reading the tables against the task's expectations
and writing the verdict is done by hand, in results/<task>.md.

Output is deterministic: same metrics files in -> byte-identical files
out (sorted order, fixed number formatting, no timestamps), so a diff
of the generated file shows exactly what changed between runs.

Usage:
    python baselines/build_results.py --task rare_cell_preservation
"""
import argparse
import json
import math
import platform
from pathlib import Path

import numpy
import pandas as pd
import scipy

# Only used to list which baselines haven't been scored yet, per dataset.
EXPECTED_BASELINES = ["bbknn", "combat", "harmony", "scvi", "seacells"]

AGREEMENT_METRICS = ["agreement_global", "agreement_within", "agreement_cross"]


# ---------------------------------------------------------------------------
# 1. Collect
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

    def add(job, metric, population, stats):
        rows.append({
            "dataset": dataset, "baseline": baseline, "job": job,
            "metric": metric, "population": population,
            "mean": stats.get("mean", math.nan), "std": stats.get("std", math.nan),
            "coverage": stats.get("coverage", math.nan), "n": stats.get("n", math.nan),
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
    per-group frequency DataFrame, list of run info dicts)."""
    headline, freq, groups, runs = [], [], [], []
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

        groups_path = mdir / "frequency_groups.csv"
        if not groups_path.exists() or "n_groups" not in fd.columns:
            raise ValueError(
                f"{mdir}: no per-group frequency results -- made by an older compute_metrics.py. "
                f"Re-run compute_metrics.py for this baseline."
            )
        fg = pd.read_csv(groups_path)
        fg.insert(0, "baseline", baseline)
        fg.insert(0, "dataset", dataset)
        groups.append(fg)

        config_path = mdir / "metrics_config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        runs.append({"dataset": dataset, "baseline": baseline, "jobs": sorted(summary),
                     "path": mdir.as_posix(), "config": config})

    headline_df = pd.DataFrame(headline)
    freq_df = pd.concat(freq, ignore_index=True) if freq else pd.DataFrame(
        columns=["dataset", "baseline", "job", "metric", "spearman_corr", "n_groups", "spearman_pvalue", "slope_per_log10_freq"])
    groups_df = pd.concat(groups, ignore_index=True) if groups else pd.DataFrame()
    sort_cols = ["dataset", "baseline", "job", "metric"]
    if not headline_df.empty:
        headline_df = headline_df.sort_values(sort_cols + ["population"], kind="stable").reset_index(drop=True)
    freq_df = freq_df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    if not groups_df.empty:
        groups_df = groups_df.sort_values(sort_cols + ["freq", "batch", "true_type"], kind="stable").reset_index(drop=True)
    return headline_df, freq_df, groups_df, runs


# ---------------------------------------------------------------------------
# 2. Line up (formatting helpers)
# ---------------------------------------------------------------------------

def _is_num(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


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


def render(task, headline, freq, groups, runs, figures):
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

    lines += [f"Environment (this script): python {platform.python_version()}, pandas {pd.__version__}, "
              f"numpy {numpy.__version__}, scipy {scipy.__version__}. Versions used to compute the "
              f"metrics: each metrics folder's `metrics_config.json`.", ""]
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

    for dataset in datasets:
        combos = headline[headline["dataset"] == dataset][["baseline", "job"]].drop_duplicates()
        combos = list(combos.itertuples(index=False))

        # --- agreement -----------------------------------------------------
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
        lines += [md_table(header, rows), ""]

        # --- the rest of the headline metrics ------------------------------
        lines += ["### Purity, rare coverage, rare F1, cohesion", ""]
        rows = [[b, j,
                 mean_std(headline, dataset, b, j, "purity", "all"),
                 mean_std(headline, dataset, b, j, "coverage", "rare"),
                 mean_std(headline, dataset, b, j, "rare_f1", "rare"),
                 mean_std(headline, dataset, b, j, "cohesion", "all"),
                 mean_std(headline, dataset, b, j, "cohesion", "rare")] for b, j in combos]
        lines += [md_table(["baseline", "job", "purity", "rare coverage", "rare F1",
                            "cohesion — all", "cohesion — rare"], rows), ""]

        # --- frequency question: per group (cell type x batch) ------------
        sub = freq[freq["dataset"] == dataset]
        lines += [
            "### Does each metric fall as frequency falls?",
            "",
            "Asked across **groups** (one cell type in one batch), not cells: each group's",
            "average counts once, whatever its size, so big common groups can't outvote",
            "small rare ones. `spearman` > 0 = lower-frequency groups have lower averages",
            "(order only; p-value based on the number of groups). `slope` = how big the",
            "drop is: change in the group average per 10x change in frequency.",
            "",
            md_table(["baseline", "job", "metric", "groups", "spearman", "p", "slope"],
                     [[r.baseline, r.job, r.metric, int(r.n_groups), fmt(r.spearman_corr),
                       fmt_p(r.spearman_pvalue), fmt(r.slope_per_log10_freq)]
                      for r in sub.itertuples(index=False)]),
            "",
        ]
        for metric in sorted(groups.loc[groups["dataset"] == dataset, "metric"].unique()):
            if (dataset, metric) in figures:
                lines += [f"![{dataset} {metric} vs frequency]({figures[(dataset, metric)]})", ""]

    # --- appendix: every group behind the trend ---------------------------
    if not groups.empty:
        lines += [
            "## Appendix — every group",
            "",
            "One row per group (cell type in one batch), rarest first. `cells` = cells with a",
            "defined value. `std` = spread of the metric across that group's cells: high std",
            "with a middling mean means the group split (some cells kept together, some",
            "scattered); low std means all cells did about equally well or badly.",
            "",
        ]
        for (dataset, baseline, job, metric), sub in groups.groupby(
                ["dataset", "baseline", "job", "metric"], sort=True):
            lines += [f"<details><summary>{dataset} / {baseline} / {job} / {metric} — {len(sub)} groups</summary>",
                      "",
                      md_table(["batch", "cell type", "frequency", "cells", "mean", "std", "rare"],
                               [[r.batch, r.true_type, f"{r.freq:.4f}", int(r.n_cells), fmt(r.mean), fmt(r.std),
                                 "rare" if r.is_rare else ""] for r in sub.itertuples(index=False)]),
                      "", "</details>", ""]

    return "\n".join(lines).rstrip() + "\n"


# Colors: default categorical slots 1-2 of the dataviz reference palette,
# checked with its colorblind validator (protan worst-case dE 24.7, PASS).
COMMON_COLOR, RARE_COLOR = "#2a78d6", "#eb6834"


def plot_frequency(task, dataset, metric, groups, freq, fig_dir):
    """One figure per (dataset, metric): one panel per (baseline, job), each
    dot = one group (cell type in one batch), x = its frequency (log scale),
    y = its average, bar = +/- std across its cells. Shared y-axis so panels
    compare directly. Only plots saved numbers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    sub = groups[(groups["dataset"] == dataset) & (groups["metric"] == metric)]
    panels = list(sub[["baseline", "job"]].drop_duplicates().itertuples(index=False))
    ncols = min(3, len(panels))
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.6 * nrows), sharey=True, squeeze=False)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)

    for ax, (baseline, job) in zip(axes.flat, panels):
        g = sub[(sub["baseline"] == baseline) & (sub["job"] == job)].dropna(subset=["mean"])
        for is_rare, color, label in ((False, COMMON_COLOR, "common group"), (True, RARE_COLOR, "rare group")):
            gg = g[g["is_rare"] == is_rare]
            ax.errorbar(gg["freq"], gg["mean"], yerr=gg["std"].fillna(0), fmt="o", ms=5, color=color,
                        ecolor=color, elinewidth=1, alpha=0.85, capsize=0, label=label,
                        markeredgecolor="white", markeredgewidth=0.8)
        t = freq[(freq["dataset"] == dataset) & (freq["baseline"] == baseline)
                 & (freq["job"] == job) & (freq["metric"] == metric)]
        trend = ""
        if len(t):
            r = t.iloc[0]
            # Slope next to Spearman: Spearman is order-only, so a tiny but
            # consistent difference can read as -1.00; the slope shows its size.
            trend = (f"\nSpearman {fmt(r['spearman_corr'], 2)} (p {fmt_p(r['spearman_pvalue'])}), "
                     f"slope {fmt(r['slope_per_log10_freq'], 2)}, {int(r['n_groups'])} groups")
        ax.set_title(f"{baseline} / {job}{trend}", fontsize=9, color="#333333")
        ax.set_xscale("log")
        # Plain percent labels at 1-2-5 steps (1%, 2%, 5%, 10%) -- the
        # default log labels (2x10^-3, 4x10^-2, ...) overlap.
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:g}%"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, color="#e6e6e3", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#999999")
        ax.tick_params(colors="#555555", labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("frequency of the type within its batch (log)", fontsize=8, color="#555555")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"{metric} (group average)", fontsize=8, color="#555555")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle(f"{dataset}: {metric} vs frequency", fontsize=11, color="#222222")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False,
               fontsize=8, title="dot = one cell type in one batch;  bar = ± std across its cells",
               title_fontsize=8)
    fig.tight_layout()

    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"{task}_{dataset}_{metric}.png"
    fig.savefig(path, dpi=120, metadata={"Software": None}, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 3. Print
# ---------------------------------------------------------------------------

def run(task, runs_root="runs", results_dir="results"):
    headline, freq, groups, runs = load_all(runs_root, task)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    figures = {}
    if not groups.empty:
        for dataset, metric in groups[["dataset", "metric"]].drop_duplicates().itertuples(index=False):
            path = plot_frequency(task, dataset, metric, groups, freq, results_dir / "figures")
            figures[(dataset, metric)] = path.relative_to(results_dir).as_posix()

    tables_path = results_dir / f"{task}_tables.md"
    tables_path.write_text(render(task, headline, freq, groups, runs, figures), encoding="utf-8", newline="\n")

    summary_path = results_dir / f"{task}_summary.csv"
    headline.to_csv(summary_path, index=False, lineterminator="\n", float_format="%.6g")
    groups_path = results_dir / f"{task}_groups.csv"
    groups.to_csv(groups_path, index=False, lineterminator="\n", float_format="%.6g")

    print(f"[results] {len(runs)} metrics folder(s) read")
    for path in (tables_path, summary_path, groups_path):
        print(f"[results] wrote {path}")
    print(f"[results] wrote {len(figures)} figure(s) to {results_dir / 'figures'}")
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
