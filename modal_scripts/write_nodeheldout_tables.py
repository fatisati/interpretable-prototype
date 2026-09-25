"""Render the NODE-HOLDOUT result tables as markdown (NeurIPS layout).

Same structure as write_d42_tables.py (methods = rows, datasets = column
groups, bold best / italic second), but every metric is computed on the 20%
HELD-OUT CELLS of the node-holdout split (the protocol decided 2026-09-24),
from files/tables/node_heldout_scoreboard/<ds>_scoreboard.csv
(produced by score_nodeheldout.py).

Reuses load/render/T1/T2/ROWS from write_d42_tables by pointing its DIR at the
node-holdout CSVs -- one rendering implementation, two protocols.

Usage: python modal_scripts/write_nodeheldout_tables.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import modal_scripts.write_d42_tables as W  # noqa: E402

DIR = "files/tables/node_heldout_scoreboard"


def main():
    W.DIR = DIR
    dfs = W.load()
    hdr = [
        "# scProto — results (NODE-HOLDOUT protocol)", "",
        "**Setup.** 20% of cells (stratified by batch; the rebuttal's cached "
        "`nodeheldout20_seed0` split) are the **test set**. Every metric below "
        "is computed **restricted to those held-out cells**, against the full "
        "affinity graph.", "",
        "**HONESTY CAVEAT — read before citing.** The baselines in these tables "
        "were fitted on **all** cells: they saw the test cells during "
        "clustering, an advantage in their favor. Only rows labelled "
        "`scProto (node-holdout)` are leak-free (trained on the 80% train "
        "graph; test cells placed by a frozen-encoder forward pass). "
        "`scProto (cohesion)` rows are the edge-holdout-trained models, also "
        "fitted with the test cells present — they are context, not the "
        "protocol's headline row.", "",
        "All numbers come from `metacell_scoreboard.py` + "
        "`score_nodeheldout.py`: identical code for every method. "
        "**Bold** = best, _italic_ = second best, per column.", "",
        "**Cohesion**: the training objective (self-sim = 1, V[b] multi-batch "
        "semantics), per-cell, averaged over test cells only. "
        "**Modularity**: per-batch Newman modularity over edges **touching a "
        "test cell** (the rebuttal's `test_node_edges` statistic), mean ± std "
        "across batches. **Batch entropy / purity**: per metacell over its "
        "test members only. **Rare metrics**: the paper's batch-local "
        "definitions computed on the test cells.", "",
    ]
    body = [
        W.render(dfs, W.T1,
                 "Table 1 — community preservation and batch integration "
                 "(held-out cells)",
                 "Modularity: mean ± std across batches (test-touching edges). "
                 "Cohesion: mean ± std across held-out cells, full graph. "
                 "Batch entropy and purity: mean ± std across metacells, test "
                 "members only; `Purity (wtd)` weighted by test-cell counts."),
        "",
        W.render(dfs, W.T2, "Table 2 — rare cell types (held-out cells)",
                 "Coverage: mean ± std across batches. Homogeneity: mean ± std "
                 "across cells, pooled over batches. Both on test cells only."),
        "",
    ]
    missing = [ds for ds, _ in W.DATASETS if ds not in dfs]
    if missing:
        body += [f"_Not yet scored: {', '.join(missing)}._", ""]
    os.makedirs(DIR, exist_ok=True)
    with open(f"{DIR}/RESULTS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(hdr + body))
    print(f"wrote {DIR}/RESULTS.md")


if __name__ == "__main__":
    main()
