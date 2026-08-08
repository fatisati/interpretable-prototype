# scVI at its own defaults as Stage 1 — results log

Source notebook: `notebooks/scvi_stage2_pancreas.ipynb` (run 2026-08-04)
Code: `interpretable_ssl/models/scvi_backbone.py`, `interpretable_ssl/trainers/scvi_proto.py`,
`interpretable_ssl/experiments/scvi_stage2.py`

**Status: all three datasets complete (04 Aug 2026).**

## What was run

One pretrained scVI model per dataset, at scvi-tools' own defaults (`n_latent=10`,
`gene_likelihood='zinb'`, raw counts, 50 epochs), then two continuations that differ in
exactly one step:

| arm | continuation |
|---|---|
| A1 | Leiden on the scVI latent |
| A2 | SEACells on the scVI latent |
| B  | scProto Stage 2 keeps training that same encoder |

Same weights loaded once in one process, same adaptive-RBF graph, same K as the paper
(220 / 300), same metric code, no scPoli anywhere. Stage 2 uses the paper's canonical
`LAMBDA_PROTO_UMAP_PRECON` configuration unchanged, early-stopped on modularity;
evaluation loads the best checkpoint, as `find_metacells` does.

Two configuration notes, both deliberate: scProto's prototype layer is built at scVI's
latent dimension (d=10) rather than the paper's d=8, so the comparison runs at the
baseline's setting rather than ours; and arm A clusters the raw scVI posterior mean,
which is what a user of scVI would cluster.

## 1. Modularity (per-batch mean ± std, canonical ARBF-on-PCA graph)

| Method | Pancreas (K=220) | Lung (K=300) | Immune (K=300) |
|---|---|---|---|
| **scProto Stage 2 (scVI)** | **0.616 ± 0.082** | **0.650 ± 0.039** | **0.618 ± 0.059** |
| Leiden (scVI) | 0.379 ± 0.122 | 0.522 ± 0.077 | 0.364 ± 0.183 |
| SEACells (scVI) | 0.296 ± 0.078 | 0.327 ± 0.053 | 0.240 ± 0.130 |
| scProto (scPoli Stage 1) — the paper's own run | 0.615 | 0.654 | 0.631 |

Two things to read off this table:

- Stage 2 on a scVI backbone reproduces the paper's own modularity almost exactly
  (0.616 vs 0.615; 0.650 vs 0.654; 0.618 vs 0.631) with scPoli removed from the pipeline
  entirely.
- Clustering that same corrected latent reaches 0.24–0.52.

## 2. The frozen-encoder control (whole-graph modularity — not the per-batch statistic above)

Prototypes waypoint-initialised on the **frozen** scVI latent, before any Stage-2
gradient step, versus the same encoder after Stage 2:

| | Pancreas | Lung | Immune |
|---|---|---|---|
| frozen scVI latent, zero Stage-2 training | 0.394 | 0.342 | 0.251 |
| after Stage 2 on that same encoder | 0.699 | 0.727 | 0.674 |

This is the cleanest isolation of the Stage-2 objective available: identical encoder,
identical graph, identical K, one variable. Reported here as whole-graph Newman
modularity (`t.modularity()`), which is **not** comparable to the per-batch numbers in
section 1 — quote the two separately.

## 3. Rare-cell metrics (mean ± std across batches, paired one-sided Wilcoxon vs scProto Stage 2, Bonferroni)

**Pancreas (n=8 batches)**

| Method | Rare F1 | Homogeneity | Cross-batch homog. | Coverage | Recall | Precision |
|---|---|---|---|---|---|---|
| scProto Stage 2 (scVI) | 0.614 ± 0.218 (ref) | 0.601 ± 0.173 (ref) | 0.205 ± 0.133 (ref) | 0.729 | 0.592 | **0.919** |
| SEACells (scVI) | 0.639 ± 0.196, ns (3/8) | 0.588 ± 0.180, ns (3/8) | 0.411 ± 0.123, ns (0/8) | 0.885 | 0.720 | 0.722 |
| Leiden (scVI) | 0.258 ± 0.276, **p=0.012\*** (8/8) | 0.317 ± 0.242, **p=0.012\*** (8/8) | 0.237 ± 0.146, ns (4/8) | 0.302 | 0.294 | 0.332 |
| scProto (scPoli Stage 1) | 0.535 ± 0.208, ns (6/8) | 0.565 ± 0.155, ns (5/8) | 0.323 ± 0.179, ns (2/8) | 0.635 | 0.537 | 0.768 |

**Lung (n=15 batches)**

| Method | Rare F1 | Homogeneity | Cross-batch homog. | Coverage | Recall | Precision |
|---|---|---|---|---|---|---|
| scProto Stage 2 (scVI) | 0.557 ± 0.255 (ref) | 0.560 ± 0.257 (ref) | 0.466 ± 0.242 (ref) | 0.706 | 0.555 | **0.853** |
| SEACells (scVI) | 0.682 ± 0.162, ns (1/15) | 0.658 ± 0.180, ns (2/15) | 0.598 ± 0.154, ns (3/15) | 0.856 | 0.712 | 0.810 |
| Leiden (scVI) | 0.689 ± 0.167, ns (4/15) | 0.635 ± 0.182, ns (4/15) | 0.597 ± 0.159, ns (3/15) | 0.856 | 0.718 | 0.812 |
| scProto (scPoli Stage 1) | 0.599 ± 0.176, ns (8/15) | 0.578 ± 0.221, ns (7/15) | 0.508 ± 0.197, ns (6/15) | 0.789 | 0.604 | 0.702 |

**Immune (n=5 batches)**

| Method | Rare F1 | Homogeneity | Cross-batch homog. |
|---|---|---|---|
| scProto Stage 2 (scVI) | **0.897 ± 0.030** (ref) | **0.859 ± 0.064** (ref) | 0.303 ± 0.195 (ref) |
| SEACells (scVI) | 0.892 ± 0.049, ns (3/5) | 0.819 ± 0.082, ns (4/5) | 0.335 ± 0.194, ns (2/5) |
| Leiden (scVI) | 0.855 ± 0.108, ns (4/5) | 0.729 ± 0.158, ns (5/5, p=.094) | 0.407 ± 0.208, ns (2/5) |
| scProto (scPoli Stage 1) | 0.854 ± 0.138, ns (2/5) | 0.827 ± 0.102, ns (4/5) | 0.349 ± 0.198, ns (3/5) |

## What these say

1. **On rare-cell metrics the two approaches come out level once scVI runs at its own
   configuration.** Every comparison is non-significant except Leiden on Pancreas. Across
   the three datasets: one in our favour (Immune, highest on both metrics), one level
   (Pancreas — highest homogeneity, F1 within 0.025), one against (Lung, 1/15 and 4/15). This is a real change from the
   d=8 + Gaussian setting, where scVI's rare-cell F1 was 0.18–0.35: the count likelihood
   at its native dimension is simply much better for sparse populations, which is
   mechanistically what one would expect.
2. **On modularity the separation is undiminished on all three** — roughly two-fold on
   Pancreas, 1.25–2× on Lung, 1.7–2.6× on Immune, the same size as with our own Stage 1.
3. **On Pancreas and Lung the rare-cell pattern is a precision/coverage trade.** scProto
   Stage 2 has the highest precision there (0.919 / 0.853) with the lowest coverage and
   recall: purer rare metacells, dedicated to fewer rare types. That single fact accounts
   for the F1 gaps on those two datasets.
4. **Cross-batch homogeneity is scProto Stage 2's weakest metric**, behind on all three
   (0.205 vs 0.411 Pancreas, losing every batch; 0.466 vs 0.598 Lung; 0.303 vs 0.407
   Immune). This is a scProto property rather than a consequence of the scVI swap — the
   paper's own scPoli-based run shows the same shape (0.323 / 0.508 / 0.349), and batch
   entropy agrees (0.399/0.538/0.229 vs 0.301/0.559/0.223).
5. **On Lung the two-step baselines also beat the paper's own scProto run** on rare F1
   (0.689 / 0.682 vs 0.599). The narrowing is a property of running scVI properly, not
   of this particular variant.

## Caveats worth knowing before citing

1. **Early stopping selects on modularity, and the objectives disagree.** Pancreas rare
   F1 was 0.683 at the final epoch and 0.614 at the best-modularity checkpoint. The
   number above is the best-modularity one, which is how every scProto run in the paper
   is scored, but part of the rare-cell deficit is model selection rather than
   capability. Untested: training the full 20-epoch budget and scoring both checkpoints.
   All three datasets stopped at epoch 9 of 20.
2. **d=10 vs the paper's d=8**, deliberately — scProto adapts to the baseline's setting
   here, so this arm is not identical to the paper's own configuration.
3. **The pancreas h5ad's counts are mixed**: 4 of 9 batches (celseq, celseq2, fluidigmc1,
   smarter) carry non-integer values. This is a property of the standard benchmark file
   and is the same matrix the existing scVI baselines were trained on, so all arms see
   identical input.
4. **The notebook's own "scProto (scPoli Stage 1)" row is not the canonical run** — it
   resolves to folders giving 0.601 (Pancreas) / 0.664 (Lung), because `extract_model_key`
   strips `_cvae_e50` and several sweep variants normalise to one key. The 0.615 / 0.654 /
   0.631 cited above are the paper's own reported numbers.
