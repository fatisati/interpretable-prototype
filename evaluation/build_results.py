"""Stage 5 (results) for the rare-cell-preservation task.

Reads every baseline's saved metrics (stage 4's output, under
runs/<task>/<dataset>/<baseline>/metrics/) and writes:

  results/<task>/<version>/            one folder per version of the numbers:
                                       "official" (task file's rare rule) or
                                       "whatif_rare-<rule>" (another rare rule)
      tables.md                        comparison tables. GENERATED -- never
                                       hand-edit; re-run this script instead.
      figures/<dataset>_<metric>.png   frequency plots (linked from tables.md)
      data/summary.csv                 every headline number, one row each
      data/groups.csv                  every group (cell type in one batch)

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
    python evaluation/build_results.py --task rare_cell_preservation
    # what-if copy made with another rare rule (compute_metrics --rare-rule fixed):
    python evaluation/build_results.py --task rare_cell_preservation --metrics-subdir metrics__rare-fixed
"""
import argparse
import json
import math
import platform
from pathlib import Path

from run_record import record_run, uncommitted  # records the script's start time

import numpy
import pandas as pd
import scipy

# Only used to list which baselines haven't been scored yet, per dataset.
EXPECTED_BASELINES = ["bbknn", "combat", "harmony", "pca", "scvi", "seacells"]

AGREEMENT_METRICS = ["agreement_global", "agreement_within", "agreement_cross"]
KNN_METRICS = ["knn_same_type_global", "knn_same_type_within", "knn_same_type_cross", "knn_affinity_within"]


# ---------------------------------------------------------------------------
# 1. Collect
# ---------------------------------------------------------------------------

def find_metrics_dirs(runs_root, task, subdir="metrics"):
    """Every runs/<task>/<dataset>/<baseline>/<subdir>/ that has a summary,
    sorted so output order never depends on filesystem order."""
    task_dir = Path(runs_root) / task
    found = sorted(task_dir.glob(f"*/*/{subdir}/metrics_summary.json"))
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
        for metric in KNN_METRICS:  # only baselines with an embedding have these
            if metric in result:
                add(job, metric, "all", result[metric]["all_overall"])
                add(job, metric, "rare", result[metric]["rare_overall"])
        add(job, "cohesion", "all", result["cohesion"]["overall"])
        add(job, "cohesion", "rare", result["cohesion"]["rare"])
    return rows


def load_all(runs_root, task, subdir="metrics"):
    """Returns (headline DataFrame, frequency-dependence DataFrame,
    per-group frequency DataFrame, list of run info dicts, rare-decision
    DataFrame, per-rare-pair F1 DataFrame)."""
    headline, freq, groups, runs, rare, f1 = [], [], [], [], [], []
    for dataset, baseline, mdir in find_metrics_dirs(runs_root, task, subdir):
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
        if not {"min", "q25", "q75", "max"} <= set(fg.columns):
            raise ValueError(
                f"{mdir}: frequency_groups.csv has no min/q25/q75/max columns -- made by an older "
                f"compute_metrics.py. Re-run compute_metrics.py for this baseline."
            )
        fg.insert(0, "baseline", baseline)
        fg.insert(0, "dataset", dataset)
        groups.append(fg)

        for name, out in (("rare_groups.csv", rare), ("metrics_per_rare_type.csv", f1)):
            path = mdir / name
            if not path.exists():
                raise ValueError(f"{mdir}: no {name} -- made by an older compute_metrics.py. "
                                 f"Re-run compute_metrics.py for this baseline.")
            t = pd.read_csv(path)
            t.insert(0, "baseline", baseline)
            t.insert(0, "dataset", dataset)
            out.append(t)

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
    rare_df = pd.concat(rare, ignore_index=True) if rare else pd.DataFrame()
    f1_df = pd.concat(f1, ignore_index=True) if f1 else pd.DataFrame()
    return headline_df, freq_df, groups_df, runs, rare_df, f1_df


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


RARE_METRICS = ["agreement_global", "agreement_within", "agreement_cross", "cohesion",
                "knn_same_type_global", "knn_affinity_within"]


def _pct(x):
    return "—" if not _is_num(x) else f"{x:.2%}"


def render_rare_section(dataset, rare, groups, f1):
    """Which groups are rare, why, how many cells, and how each did.
    Only formats saved numbers (rare_groups.csv, frequency_groups.csv,
    metrics_per_rare_type.csv); "why" is the saved frequency vs the saved
    cutoff, written out."""
    r = rare[rare["dataset"] == dataset]
    if r.empty:
        return []
    # The rare decision is the same for every baseline (same cells, same
    # rule) -- show it once, from the first baseline that has it.
    r = r[r["baseline"] == sorted(r["baseline"].unique())[0]]
    rr = r[r["is_rare"]].sort_values(["freq", "batch", "true_type"], kind="stable")
    batches = sorted(r["batch"].unique())
    without = [b for b in batches if b not in set(rr["batch"])]
    rule = r["rule"].iloc[0]

    lines = [
        "### Rare groups",
        "",
        f"**{len(rr)} rare groups** (cell type in one batch) in **{rr['batch'].nunique()} of "
        f"{len(batches)} batches**, {int(rr['n_cells'].sum())} cells in total. Rule: `{rule}`.",
        "",
    ]
    if without:
        lines += [f"Batches with **no** rare group: {', '.join(without)}.", ""]
    if rr.empty:
        return lines

    # Cells per job (train / test), from the per-group table.
    coh = groups[(groups["dataset"] == dataset) & (groups["metric"] == "cohesion")]
    jobs = sorted(coh["job"].unique())

    def cells_in(job, batch, cell_type):
        hit = coh[(coh["job"] == job) & (coh["batch"] == batch) & (coh["true_type"] == cell_type)]
        return int(hit["n_cells"].iloc[0]) if len(hit) else 0

    header = ["batch", "cell type", "cells (all)"] + [f"cells ({j})" for j in jobs] + \
             ["frequency", "cutoff in this batch", "why rare"]
    rows = []
    for g in rr.itertuples(index=False):
        why = f"{_pct(g.freq)} < {_pct(g.threshold)}"
        if getattr(g, "fallback", False):
            why += " (lowest-quartile fallback)"
        rows.append([g.batch, g.true_type, int(g.n_cells)] + [cells_in(j, g.batch, g.true_type) for j in jobs]
                    + [_pct(g.freq), _pct(g.threshold), why])
    lines += ["**Which groups, and why** (rarest first):", "", md_table(header, rows), ""]

    # Near misses: the lowest-frequency groups that are NOT rare -- shows
    # where the cutoff bites (or doesn't).
    near = r[~r["is_rare"]].sort_values("freq", kind="stable").head(5)
    if len(near):
        lines += ["**Closest to rare, but not rare** (5 lowest-frequency common groups):", "",
                  md_table(["batch", "cell type", "cells (all)", "frequency", "cutoff in this batch"],
                           [[g.batch, g.true_type, int(g.n_cells), _pct(g.freq), _pct(g.threshold)]
                            for g in near.itertuples(index=False)]), ""]

    # How each rare group did, per baseline and job: mean ± std over its cells.
    sub = groups[(groups["dataset"] == dataset) & groups["metric"].isin(RARE_METRICS)]
    f1d = f1[f1["dataset"] == dataset]
    for (baseline, job), gj in sub.groupby(["baseline", "job"], sort=True):
        rows = []
        for g in rr.itertuples(index=False):
            row = [g.batch, g.true_type]
            hit = gj[(gj["batch"] == g.batch) & (gj["true_type"] == g.true_type)]
            n = int(hit["n_cells"].max()) if len(hit) else 0
            row.append(n)
            for m in RARE_METRICS:
                h = hit[hit["metric"] == m]
                row.append("—" if h.empty else f"{fmt(h['mean'].iloc[0])} ± {fmt(h['std'].iloc[0])}")
            fh = f1d[(f1d["baseline"] == baseline) & (f1d["job"] == job)
                     & (f1d["batch"] == g.batch) & (f1d["true_type"] == g.true_type)]
            if fh.empty:
                row.append("—")
            else:
                x = fh.iloc[0]
                row.append(f"{fmt(x['f1'], 2)} (found {int(x['tp'])} of {int(x['tp'] + x['fn'])})")
            rows.append(row)
        lines += [f"**How each rare group did — {baseline} / {job}** (mean ± std over its cells; "
                  "std is empty for a 1-cell group):", "",
                  md_table(["batch", "cell type", "cells", "agreement global", "agreement within",
                            "agreement cross", "cohesion", "neighbors same type", "neighbor affinity",
                            "F1 (cells in a cluster labeled its type)"], rows), ""]
    return lines


def render(task, headline, freq, groups, runs, figures, rare, f1, subdir="metrics"):
    title = f"# {task} — generated tables"
    if subdir != "metrics":
        title += f" — WHAT-IF: `{subdir}`"
    lines = [
        title,
        "",
        "> **GENERATED FILE — do not edit.** Made by `evaluation/build_results.py` from",
        f"> `runs/{task}/*/*/{subdir}/`. Re-run that script to update. The written",
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
        lines += [f"No metrics found under `runs/{task}/*/*/{subdir}/`. Run `compute_metrics.py` first.", ""]
        return "\n".join(lines) + "\n"

    lines += [f"Environment (this script): python {platform.python_version()}, pandas {pd.__version__}, "
              f"numpy {numpy.__version__}, scipy {scipy.__version__}. Versions used to compute the "
              f"metrics: each metrics folder's `metrics_config.json`.", ""]
    lines.append(md_table(
        ["dataset", "baseline", "jobs", "rare groups", "rare rule", "metrics folder"],
        [[r["dataset"], r["baseline"], ", ".join(r["jobs"]), r["config"].get("n_rare_pairs", "—"),
          r["config"].get("rare_rule_text", "—"), f"`{r['path']}`"] for r in runs],
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
        lines += [f"## {dataset}", ""]
        lines += render_rare_section(dataset, rare, groups, f1)
        lines += ["### Agreement (all cells / rare cells)", ""]
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

        # --- neighbor metrics (embedding only, no clusters) ----------------
        knn_combos = [(b, j) for b, j in combos if not headline[
            (headline["dataset"] == dataset) & (headline["baseline"] == b) & (headline["job"] == j)
            & (headline["metric"] == "knn_affinity_within")].empty]
        if knn_combos:
            lines += ["### Neighbor metrics (embedding only, no clusters; all cells / rare cells)", "",
                      "Each cell's 15 nearest neighbors in the embedding. `same type` = share of them",
                      "with the cell's own type (among all cells / own batch / other batches).",
                      "`affinity` = mean ground-truth affinity to the 15 nearest own-batch neighbors.", ""]
            header = ["baseline", "job"]
            for m in KNN_METRICS:
                short = m.replace("knn_same_type_", "same type ").replace("knn_affinity_within", "affinity")
                header += [f"{short} — all", f"{short} — rare"]
            rows = []
            for b, j in knn_combos:
                row = [b, j]
                for m in KNN_METRICS:
                    row += [mean_std(headline, dataset, b, j, m, "all"), mean_std(headline, dataset, b, j, m, "rare")]
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
BOX_MIN_CELLS = 5  # fewer cells -> no "middle half" bar, just min..max


def plot_frequency(task, dataset, metric, groups, freq, fig_dir):
    """One figure per (dataset, metric): one panel per (baseline, job), each
    dot = one group (cell type in one batch), x = its frequency (log scale),
    y = its average, thin line = min..max of its cells, thick line = middle
    half (25th-75th percentile). Shared y-axis so panels
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
            # Slim box plot per group. Not mean +/- std: for a bounded,
            # skewed metric that bar leaves the possible range (> 1).
            #   thin line  = min..max (real range of its cells)
            #   thick line = q25..q75 (middle half), only with >= BOX_MIN_CELLS
            #                cells -- a "middle half" of 2 cells means nothing
            #   dot        = group average (what the tables and Spearman use)
            ax.vlines(gg["freq"], gg["min"], gg["max"], color=color, linewidth=0.8, alpha=0.6)
            big = gg[gg["n_cells"] >= BOX_MIN_CELLS]
            ax.vlines(big["freq"], big["q25"], big["q75"], color=color, linewidth=3, alpha=0.6)
            ax.plot(gg["freq"], gg["mean"], "o", ms=5, color=color, alpha=0.9, label=label,
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
               fontsize=8, title=("dot = average of one cell type in one batch;  thin line = min to max "
                                  f"of its cells;\nthick line = middle half of its cells (25th-75th "
                                  f"percentile, groups with {BOX_MIN_CELLS}+ cells)"),
               title_fontsize=8)
    fig.tight_layout()

    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / f"{dataset}_{metric}.png"
    fig.savefig(path, dpi=120, metadata={"Software": None}, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 3. Print
# ---------------------------------------------------------------------------

def run(task, runs_root="runs", results_dir="results", subdir="metrics"):
    headline, freq, groups, runs, rare, f1 = load_all(runs_root, task, subdir)
    # One folder per version of the numbers, so versions never mix:
    # results/<task>/official/ or results/<task>/whatif_rare-<rule>/.
    version = "official" if subdir == "metrics" else subdir.replace("metrics__", "whatif_")
    out_dir = Path(results_dir) / task / version
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    figures = {}
    if not groups.empty:
        for dataset, metric in groups[["dataset", "metric"]].drop_duplicates().itertuples(index=False):
            path = plot_frequency(task, dataset, metric, groups, freq, out_dir / "figures")
            figures[(dataset, metric)] = path.relative_to(out_dir).as_posix()

    tables_path = out_dir / "tables.md"
    tables_path.write_text(render(task, headline, freq, groups, runs, figures, rare, f1, subdir),
                           encoding="utf-8", newline="\n")

    summary_path = data_dir / "summary.csv"
    headline.to_csv(summary_path, index=False, lineterminator="\n", float_format="%.6g")
    groups_path = data_dir / "groups.csv"
    groups.to_csv(groups_path, index=False, lineterminator="\n", float_format="%.6g")

    print(f"[results] {len(runs)} metrics folder(s) read")
    for path in (tables_path, summary_path, groups_path):
        print(f"[results] wrote {path}")
    print(f"[results] wrote {len(figures)} figure(s) to {out_dir / 'figures'}")
    return tables_path, summary_path


def unreproducible_runs(runs_root, task, subdir="metrics"):
    """Hard gate: every run record feeding a result must come from committed code.

    For each metrics folder, checks the latest run_history entry of the records along its
    chain -- the metrics, the baseline, and the dataset's shared graph / cluster count.
    Returns one "path: reason" line per problem (empty list = all clean). A record with no
    run_history predates run_record.py, so we can't tell what code made it: also a problem."""
    problems = []
    for _, _, mdir in find_metrics_dirs(runs_root, task, subdir):
        chain = [mdir / "metrics_config.json", mdir.parent / "config.json",
                 mdir.parent.parent / "affinity_config.json",
                 mdir.parent.parent / "cluster_config.json"]
        for path in chain:
            if not path.exists():
                continue
            history = json.loads(path.read_text()).get("run_history")
            if not history:
                problems.append(f"{path.as_posix()}: no run record (made before run_record.py)")
                continue
            last = history[-1]
            if last.get("git_dirty"):
                problems.append(f"{path.as_posix()}: made from uncommitted code "
                                f"(commit {str(last.get('git_commit'))[:7]} + edits)")
            elif last.get("script_committed") is False:
                problems.append(f"{path.as_posix()}: script {last.get('script')} was not in git")
    return sorted(set(problems))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, help="task name, e.g. rare_cell_preservation")
    p.add_argument("--runs-root", default="runs", help="(for tests) folder holding runs/<task>/")
    p.add_argument("--results-dir", default="results", help="(for tests) where output goes")
    p.add_argument("--metrics-subdir", default="metrics",
                   help="metrics folder name per run; metrics__rare-<rule> = a what-if rare rule")
    p.add_argument("--allow-dirty", action="store_true",
                   help="build anyway from runs made with uncommitted code (for checks only; "
                        "the flag is kept in run.json's command, so such tables stay traceable)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Hard gate (see CLAUDE.md): results only from committed code, this script included.
    problems = unreproducible_runs(args.runs_root, args.task, args.metrics_subdir)
    changed, script_untracked = uncommitted()
    if changed or script_untracked:
        problems.insert(0, f"build_results.py itself: uncommitted code ({', '.join(changed[:5]) or 'script not in git'})")
    if problems and not args.allow_dirty:
        raise SystemExit(
            "[results] REFUSED: these results would not be reproducible:\n  - "
            + "\n  - ".join(problems)
            + "\nCommit the code, re-run the listed stages (docs/runbook.md), then build again."
            "\n(For a throwaway check only: add --allow-dirty.)")
    if problems:
        print(f"[results] --allow-dirty: building despite {len(problems)} problem(s); "
              f"these tables are NOT reproducible")
    tables_path, _ = run(args.task, args.runs_root, args.results_dir, args.metrics_subdir)
    # Kept out of tables.md (which stays byte-identical across runs). Fresh file
    # each time: it describes the tables just written, not older builds.
    run_json = tables_path.parent / "run.json"
    run_json.unlink(missing_ok=True)
    record_run(run_json)
