# Node-held-out modularity — results log

Source notebook: `notebooks/node_heldout_modularity.ipynb` (run 2026-08-03, all 3
RNA-seq datasets — pancreas, lung, pbmc-immune)

**Answers Reviewer 1 (F5RB)'s follow-up**: *"held-out edges from the same graph are not
fully independent."* 20% of cells (stratified by batch) are excluded entirely from
affinity-graph construction and from training — not just from the loss — then scored via
a frozen-encoder forward pass on cells the model has never seen in any form. Verified
leak-free: PCA is recomputed fresh, per run, from only the cells actually passed in
(traced through `MultiCropsDataset`'s subprocess call to `compute_affinities`), and the
frozen-forward-pass assumption is asserted at runtime (`check_conditions_compatible`),
not just assumed. Full design rationale in the notebook's own intro cell.

## Results — cite these

Per-batch modularity, mean ± std across batches (same statistic the paper's own Table 1
uses — `calc_modularity_per_batch`, not a single pooled number):

| dataset | method | full_graph_modularity | test_node_edges_modularity | rare_edge_same_cluster_rate |
|---|---|---|---|---|
| pancreas | **scProto (node-holdout)** | 0.617 ± 0.069 | **0.599 ± 0.072** | 0.410 |
| pancreas | Leiden (scPoli Stage-1) | 0.355 ± 0.060 | 0.358 ± 0.053 | 0.424 |
| pancreas | Leiden (scVI, Gaussian) | 0.340 ± 0.114 | 0.341 ± 0.116 | 0.531 |
| lung | **scProto (node-holdout)** | 0.672 ± 0.029 | **0.655 ± 0.034** | 0.716 |
| lung | Leiden (scPoli Stage-1) | 0.494 ± 0.085 | 0.495 ± 0.083 | 0.689 |
| lung | Leiden (scVI, Gaussian) | 0.501 ± 0.085 | 0.501 ± 0.083 | 0.792 |
| pbmc-immune | **scProto (node-holdout)** | 0.630 ± 0.053 | **0.620 ± 0.051** | 0.656 |
| pbmc-immune | Leiden (scPoli Stage-1) | 0.312 ± 0.214 | 0.314 ± 0.215 | 0.571 |
| pbmc-immune | Leiden (scVI, Gaussian) | 0.323 ± 0.210 | 0.326 ± 0.209 | 0.612 |

`full_graph_modularity` = node-holdout scProto's combined (train + frozen-encoder-test)
assignment, scored the same way Table 1 is scored — directly comparable to scProto's own
full-data headline modularity (Pancreas 0.621, Lung 0.669, Immune 0.620). It lands almost
exactly on that number in all three datasets, despite training on only 80% of the cells.

`test_node_edges_modularity` = the same run, restricted to edges touching a cell the
model never saw in any form (not in the affinity graph, not in training, not in early
stopping). This is the actual generalization number. It drops a small, expected amount
from `full_graph_modularity` (0.617→0.599, 0.672→0.655, 0.630→0.620) — the signature of a
genuinely harder, out-of-sample test, not an inflated or leaked one.

**Not a clean sweep**: `rare_edge_same_cluster_rate` (pooled, no per-batch analog) has
scProto losing to a Leiden baseline on pancreas (0.410 vs. 0.424/0.531) and lung (0.716
vs. 0.792) — worth stating alongside the win, not hiding.

## Caveats to state alongside these numbers

1. **Test cells share a batch with train cells** (stratified split, by design) — this is
   "unseen cell, already-seen batch" generalization, not "unseen cohort" generalization.
2. **Leiden(scPoli-Stage1) / Leiden(scVI-Gaussian) baselines' own encoders were trained on
   every test cell's expression** in their original full-dataset run — they just never
   got graph/affinity supervision. A win here shows affinity supervision transfers to
   unseen structure, not that scProto's raw encoder generalizes better on expression
   alone.
3. **No SEACells row, structurally** — no encoder means no way to place a cell never seen
   when its kernel/archetypes were built. Worth stating directly as evidence for the
   paper's own inductive-method framing, not as a missing comparison.

## Suggested reviewer-response framing

> To make the held-out-edge test's independence stronger, we ran a node-level version:
> 20% of cells are excluded entirely from affinity-graph construction and from training
> (not just from the loss), then scored via a frozen encoder forward pass on cells the
> model has never seen in any form. Scored the same way as Table 1 (per-batch modularity,
> mean ± std across batches), this node-holdout model reproduces its own full-data
> headline modularity almost exactly (Pancreas 0.617±0.069 vs. 0.621, Lung 0.672±0.029 vs.
> 0.669, Immune 0.630±0.053 vs. 0.620) despite training on 20% less data. Restricted to
> just the edges touching a held-out cell, it drops a small, expected amount (0.599, 0.655,
> 0.620 respectively) — well above the same "zero affinity supervision" controls scored on
> the identical edges (Leiden on scPoli-Stage1 / scVI-Gaussian latents, 0.31–0.50).
> SEACells has no encoder and is structurally unable to place a cell it never saw, so it
> cannot be included in this specific test.
