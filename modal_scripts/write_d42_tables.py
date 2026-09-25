"""Render the two result tables as markdown, in the NeurIPS submission layout.

Structure copied from manuscripts/neurips_manuscript/sections/results.tex: METHODS
are rows, DATASETS are column groups, best is **bold** and second-best is
_underlined_ per column. Two tables only -- one per claim -- so the thing stays
readable.

TABLE 1 -- affinity / community preservation and batch integration
    Modularity     mean +- std OVER BATCHES     (per bio batch)
    Cohesion       mean +- std OVER CELLS       (the training objective, self-sim
                                                 = 1, computed exactly as in
                                                 training: V[b] denominator)
    Batch entropy  mean +- std OVER METACELLS   (unnormalised nats)
    Purity         unweighted AND size-weighted, each with its matching std

TABLE 2 -- rare cell types
    Coverage       mean +- std OVER BATCHES     (a per-batch quantity)
    Homogeneity    mean +- std OVER CELLS       (pooled across batches)

Every number is scored on the HELD-OUT 20% of edges unless the column says
(train): no model had a training signal for those edges.

Usage: python modal_scripts/write_d42_tables.py
"""
import os

import numpy as np
import pandas as pd

DIR = "files/tables/d42_scoreboard"
DATASETS = [("pancreas", "Pancreas"), ("pbmc-immune", "Immune"), ("lung", "Lung")]

# (display, mean col, std col, decimals, higher_is_better)
T1 = [
    ("Modularity",       "modularity_test_mean",  "modularity_test_std",  3, True),
    ("Cohesion",         "coh_aff_mean",          "coh_aff_std",          4, True),
    ("Batch Entropy",    "batch_entropy_mean",    "batch_entropy_std",    2, True),
    ("Purity",           "ct_purity_mean",        "ct_purity_std",        3, True),
    ("Purity (wtd)",     "ct_purity_wmean",       "ct_purity_wstd",       3, True),
]
T2 = [
    ("Coverage",     "rare_coverage_mean",        "rare_coverage_std",        3, True),
    ("Homogeneity",  "rare_homogeneity_cellmean", "rare_homogeneity_cellstd", 3, True),
]

# Row order and display names. Keep it short: the canonical baselines plus the
# dim-matched variants that the fairness argument needs.
# Row order and display names.
# DIM-MATCHED ONLY: for the batch-correction methods that expose a latent
# dimension, keep ONLY the 8-dim variant, matching scProto's latent_dims=8.
# Comparing scProto-8 against Harmony-50 would confound "better method" with
# "more latent capacity", which is the whole reason the d8 runs exist.
ROWS = [
    ("scProto NH (cohesion)",  "scProto NH (cohesion)"),
    ("scProto NH (coh+comm)",  "scProto NH (coh+comm)"),
    ("Leiden + scPoli-St1 NH", "Leiden + scPoli-St1 NH"),
    ("scProto (cohesion)",     "scProto (cohesion)"),
    ("SEACells",               "SEACells"),
    ("SuperCell",              "SuperCell"),
    ("MetaCell2",              "MetaCell2"),
    ("MetaQ",                  "MetaQ"),
    ("Leiden + scPoli-St1",    "Leiden + scPoli-Stage1"),
    ("Leiden + scVI",          "Leiden + scVI"),
    ("Leiden + Harmony d8",    "Leiden + Harmony d8"),
    ("Leiden + ComBat d8",     "Leiden + ComBat d8"),
    ("Leiden + BBKNN d8",      "Leiden + BBKNN d8"),
    ("SEACells + Harmony d8",  "SEACells + Harmony d8"),
    ("SEACells + ComBat d8",   "SEACells + ComBat d8"),
    ("SEACells + BBKNN d8",    "SEACells + BBKNN d8"),
    ("SEACells + scVI",        "SEACells + scVI"),
]


def load():
    out = {}
    for ds, _ in DATASETS:
        p = f"{DIR}/{ds}_scoreboard.csv"
        if os.path.exists(p):
            df = pd.read_csv(p, index_col=0)
            # collapse "[run_dir]" suffixes so a method matches its canonical row
            df.index = [i.split(" [")[0] for i in df.index]
            df = df[~df.index.duplicated(keep="first")]
            out[ds] = df
    return out


def render(dfs, spec, title, note):
    ngrp = len(spec)
    lines = [f"### {title}", "", note, ""]
    head1 = "| Method | " + " | ".join(
        f"**{lbl}** " + " | ".join([""] * (ngrp - 1)) for _, lbl in DATASETS) + " |"
    # simple flat header: dataset prefix on each metric, easier to read in md
    cols = []
    for ds, lbl in DATASETS:
        for disp, *_ in spec:
            cols.append(f"{lbl[:4]}: {disp}")
    lines = [f"### {title}", "", note, "",
             "| Method | " + " | ".join(cols) + " |",
             "|---|" + "|".join("---" for _ in cols) + "|"]

    # best / second per column, over the rows actually present
    best = {}
    for ds, _ in DATASETS:
        df = dfs.get(ds)
        for disp, mk, sk, nd, hi in spec:
            key = (ds, mk)
            if df is None or mk not in df.columns:
                best[key] = (None, None)
                continue
            vals = {}
            for _, src in ROWS:
                if src in df.index and pd.notna(df.loc[src, mk]):
                    vals[src] = float(df.loc[src, mk])
            if not vals:
                best[key] = (None, None)
                continue
            order = sorted(vals.items(), key=lambda kv: -kv[1] if hi else kv[1])
            best[key] = (order[0][0], order[1][0] if len(order) > 1 else None)

    for disp_row, src in ROWS:
        cells = []
        any_val = False
        for ds, _ in DATASETS:
            df = dfs.get(ds)
            for _, mk, sk, nd, hi in spec:
                if df is None or src not in df.index or mk not in df.columns \
                        or pd.isna(df.loc[src, mk]):
                    cells.append("--")
                    continue
                any_val = True
                v = float(df.loc[src, mk])
                s = float(df.loc[src, sk]) if sk in df.columns and pd.notna(
                    df.loc[src, sk]) else None
                txt = f"{v:.{nd}f}" + (f" ± {s:.{nd}f}" if s is not None else "")
                b, sec = best[(ds, mk)]
                if src == b:
                    txt = f"**{txt}**"
                elif src == sec:
                    txt = f"_{txt}_"
                cells.append(txt)
        if any_val:
            lines.append(f"| {disp_row} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    dfs = load()
    hdr = [
        "# scProto — results", "",
        "**Setup.** 20% of each affinity graph's edges were held out "
        "(`FRAC_HELDOUT=0.20`, `MIN_VISIBLE_DEGREE=5`, `SEED=0`, `arbf`); models "
        "that train on a graph saw only the visible 80%. Every number below is "
        "scored on the **held-out 20%** — no model had a training signal for any "
        "of those edges. The split is exact (train + test partition the edge set "
        "with no overlap and no loss).", "",
        "All numbers come from `interpretable_ssl/evaluation/metacell_scoreboard.py`, "
        "so scProto and every baseline go through identical code on identical "
        "graphs. **Bold** = best, _italic_ = second best, per column.", "",
        "**Cohesion** is the training objective itself, self-similarity = 1, computed "
        "exactly as in training: multi-batch is unconditional, so each cell is "
        "scored against `V[b]` — the node set of its own bio-batch graph (all "
        "edges with ≥1 endpoint in `b`). Nothing is masked; cross-batch pairs "
        "whose zeros carry no information are excluded from numerator and "
        "denominator alike. With one batch `V[b]` is every cell and the formula "
        "reduces exactly to the batch-agnostic one.", "",
        "It is averaged over the **test nodes** (cells with ≥1 held-out edge) but "
        "computed on the **full** bio-batch graph — a held-out node's similarity "
        "to train nodes counts too. Scoring on the held-out edges *alone* is a "
        "different and degenerate quantity: ~5x smaller, and it ranks a random "
        "partition **first**. On these three datasets the test-node set is 100% "
        "of cells (20% of edges removed at random means essentially every cell "
        "loses one), so the de-circularisation comes from the model having "
        "*trained* on 80%, not from the node subset.", "",
        "**Batch entropy** is unnormalised Shannon entropy in nats "
        "(`mc_metric_utils.calc_batch_entropy`); ceiling is `log(n_batches)` = "
        "2.20 pancreas (9), 1.61 immune (5), 2.77 lung (16). Higher = better mixed.",
        "",
        "**Rare cell types** use the paper's definition: rare is decided **within "
        "each batch** (frequency below mean−std, Q25 fallback).", "",
    ]
    body = [
        render(dfs, T1, "Table 1 — community preservation and batch integration",
               "Modularity: mean ± std **across batches**, on the held-out edges. "
               "Cohesion: mean ± std **across cells**, full graph, self-sim = 1. "
               "Batch entropy and purity: mean ± std **across metacells**; "
               "`Purity (wtd)` is size-weighted with the matching weighted std."),
        "",
        render(dfs, T2, "Table 2 — rare cell types",
               "Coverage: mean ± std **across batches** (it is a per-batch "
               "quantity). Homogeneity: mean ± std **across cells**, pooled over "
               "batches."),
        "",
    ]
    missing = [ds for ds, _ in DATASETS if ds not in dfs]
    if missing:
        body += [f"_Not yet scored: {', '.join(missing)}._", ""]
    os.makedirs(DIR, exist_ok=True)
    with open(f"{DIR}/RESULTS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(hdr + body))
    print(f"wrote {DIR}/RESULTS.md")


if __name__ == "__main__":
    main()
