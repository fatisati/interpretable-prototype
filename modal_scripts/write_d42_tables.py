"""Render the two requested tables as markdown from the scoreboard CSVs.

Section 1 -- "scProto preserves the similarities defined in the graph while also
              mixing batches"
    cohesion        mean +- std OVER CELLS, on the HELD-OUT edges (and on train)
    modularity      mean +- std OVER BATCHES, on the held-out edges
    batch entropy   mean +- std over metacells, AND size-weighted mean +- weighted std
    ct purity       the same two pairs

Section 2 -- "scProto preserves rare cell types better"
    coverage        mean +- std OVER BATCHES     (a per-batch quantity)
    homogeneity     mean +- std OVER BATCHES     (paper aggregation)
    f1              mean +- std OVER BATCHES
    plus cross_batch_homogeneity, the only one of these that cannot be gamed by
    giving each batch its own private cluster

Usage: python modal_scripts/write_d42_tables.py  > /dev/null
       (writes files/tables/d42_scoreboard/RESULTS.md)
"""
import os
import sys

import numpy as np
import pandas as pd

DIR = "files/tables/d42_scoreboard"
DATASETS = ["pancreas", "lung", "pbmc-immune"]

S1 = [
    ("cohesion aff (test edges)",  "coh_aff_test_mean",  "coh_aff_test_std",  4),
    ("cohesion gram (test edges)", "coh_gram_test_mean", "coh_gram_test_std", 4),
    ("cohesion aff (train)",       "coh_aff_train_mean", "coh_aff_train_std", 4),
    ("modularity (test edges)",    "modularity_test_mean", "modularity_test_std", 4),
    ("modularity (train)",         "modularity_train_mean", "modularity_train_std", 4),
    ("batch entropy",              "batch_entropy_mean", "batch_entropy_std", 4),
    ("batch entropy (size-wtd)",   "batch_entropy_wmean", "batch_entropy_wstd", 4),
    ("ct purity",                  "ct_purity_mean",     "ct_purity_std",     4),
    ("ct purity (size-wtd)",       "ct_purity_wmean",    "ct_purity_wstd",    4),
]
S2 = [
    ("rare coverage",              "rare_coverage_mean", "rare_coverage_std", 4),
    ("rare homogeneity",           "rare_homogeneity_mean", "rare_homogeneity_std", 4),
    ("rare F1",                    "rare_f1_mean",       "rare_f1_std",       4),
    ("rare cross-batch homog.",    "rare_cross_batch_homogeneity_mean",
                                   "rare_cross_batch_homogeneity_std", 4),
    ("rare recall",                "rare_recall_mean",   "rare_recall_std",   4),
    ("rare precision",             "rare_precision_mean", "rare_precision_std", 4),
]
EXTRA = [("K (metacells)", "n_mc", None, 0),
         ("median size", "median_size", None, 1)]


def fmt(v, s, nd):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return f"{v:.{nd}f}"
    return f"{v:.{nd}f} ± {s:.{nd}f}"


def table(df, spec, title, note=""):
    lines = [f"### {title}", ""]
    if note:
        lines += [note, ""]
    hdr = "| method | " + " | ".join(n for n, *_ in spec) + " |"
    sep = "|---|" + "|".join("---" for _ in spec) + "|"
    lines += [hdr, sep]
    for m in df.index:
        cells = []
        for _, mk, sk, nd in spec:
            v = df.loc[m, mk] if mk in df.columns else None
            s = df.loc[m, sk] if (sk and sk in df.columns) else None
            cells.append(fmt(v, s, nd))
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    out = ["# scProto results — held-out-edge evaluation", "",
           "Every number below is produced by "
           "`interpretable_ssl/evaluation/metacell_scoreboard.py`, so scProto and "
           "every baseline go through **identical code on identical graphs**.", "",
           "**Setup.** 20% of each affinity graph's edges were hidden "
           "(`FRAC_HELDOUT=0.20`, `MIN_VISIBLE_DEGREE=5`, `SEED=0`, `arbf`). "
           "Models that train on a graph saw only the visible 80%. The `test "
           "edges` columns are scored on the hidden 20% — **no model had a "
           "training signal for any of those edges**, which is what "
           "de-circularises the metric. The split is exact: train + test "
           "partition the edge set with no overlap and no loss.", "",
           "**Cohesion** is the training objective itself. Multi-batch is handled "
           "unconditionally: each cell is scored against `V[b]`, the node set of "
           "its own bio-batch graph (all edges with ≥1 endpoint in `b`). Nothing "
           "is masked — cross-batch similarities keep their real values; what is "
           "excluded are pairs whose zeros carry no information. With one batch "
           "`V[b]` is every cell, so the formula reduces exactly to the "
           "batch-agnostic one.", "",
           "**Batch entropy** is unnormalised Shannon entropy in nats "
           "(`mc_metric_utils.calc_batch_entropy`), so the ceiling is "
           "`log(n_batches)`. Higher = better mixed.", "",
           "**Rare metrics** are the paper's own definitions "
           "(`paper_figures._rare_table_one`): rare types are defined **per "
           "batch** (within-batch frequency below mean−std, Q25 fallback) and "
           "every quantity is aggregated mean ± std **across batches**.", ""]

    for ds in DATASETS:
        p = f"{DIR}/{ds}_scoreboard.csv"
        if not os.path.exists(p):
            out += [f"## {ds}", "", "_not scored yet_", ""]
            continue
        df = pd.read_csv(p, index_col=0)
        nb = df["modularity_test_nbatch"].dropna().iloc[0] if \
            "modularity_test_nbatch" in df.columns and \
            df["modularity_test_nbatch"].notna().any() else "?"
        out += [f"## {ds}", "",
                f"_{int(nb) if nb != '?' else nb} biological batches; "
                f"batch-entropy ceiling log({nb}) = "
                f"{np.log(int(nb)):.3f}_" if nb != "?" else f"## {ds}", ""]
        out += [table(df, S1 + EXTRA,
                      "Section 1 — preserves graph similarity while mixing batches",
                      "std over **cells** for cohesion; over **batches** for "
                      "modularity; over **metacells** for entropy/purity "
                      "(size-weighted rows use the weighted std)."), ""]
        out += [table(df, S2,
                      "Section 2 — rare cell types",
                      "std over **batches** throughout (the paper's aggregation). "
                      "`rare cross-batch homog.` is the only column here that "
                      "cannot be maximised by giving each batch its own private "
                      "cluster."), ""]

    os.makedirs(DIR, exist_ok=True)
    with open(f"{DIR}/RESULTS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"wrote {DIR}/RESULTS.md")


if __name__ == "__main__":
    main()
