# scProto vs SEACells — BANKSY lambda sweep, s28nsc / K=200

Measured 2026-09-23. Every number from `modal_scripts/eval_dge_all_seacells.py`
(one pass, so all rows are computed by identical code).

**The scoreboard metrics**, as decided:
`dge_r` (PEARSON, niche-program recovery within cell type, macro over
(cell type, niche) pairs) · `joint_purity` (per metacell, the largest fraction
of one (cell type, niche) pair, size-weighted over cells) · `ct_purity` (same,
cell type alone) · `n_covered` ((cell type, niche) states with at least one
niche-positive metacell) · `n_dead` (prototypes holding no cells).

`joint_purity`, `ct_purity`, `dge_r` and `n_covered` are computed from
assignments + labels and are INDEPENDENT of which affinity graph is loaded, so
they are directly comparable across every row. Only `coh_aff`/`coh_gram` depend
on the graph; those columns below are scored on bk02.

## SEACells across graphs

| graph        | dge_r  | joint_pur | ct_pur | n_cov | dead | median | max/med |
|--------------|--------|-----------|--------|-------|------|--------|---------|
| lambda 0.2   | 0.8487 | 0.5035    | 0.8227 | 19    | 0    | 258    | 3.6     |
| lambda 0.4   | 0.7223 | 0.4672    | 0.6974 | 21    | 0    | 242    | 3.8     |
| lambda 0.8   | 0.6648 | 0.3770    | 0.5127 | 16    | 0    | 208    | 9.3     |
| PCA (K200)   | 0.7470 | 0.4013    | 0.8200 | 16    | 0    | 248    | 4.5     |
| arbf (PCA)   | 0.7171 | 0.3433    | 0.4968 | 14    | 0    | 266    | 3.9     |

## scProto — cohesion "both" (gram + aff, 50/50 authority), community OFF

| graph        | dge_r  | joint_pur | ct_pur | n_cov | dead | median | max/med |
|--------------|--------|-----------|--------|-------|------|--------|---------|
| lambda 0.2   | 0.8105 | 0.4576    | 0.8681 | 16    | 0    | 250    | 3.8     |
| lambda 0.4   | 0.8404 | 0.4563    | 0.8491 | 13    | 0    | 252    | 4.8     |
| lambda 0.8   | pending|           |        |       |      |        |         |
| arbf         | pending|           |        |       |      |        |         |

## scProto ablations, all at lambda 0.2

| arm                  | dge_r  | joint_pur | ct_pur | n_cov | dead |
|----------------------|--------|-----------|--------|-------|------|
| both (gram+aff)      | 0.8105 | 0.4576    | 0.8681 | 16    | 0    |
| aff only             | 0.8006 | 0.4472    | 0.8603 | 16    | 0    |
| gram only            | 0.6538 | 0.4558    | 0.8710 | 18    | 5    |
| + community (UMAP)   | 0.3603 | 0.3951    | 0.8235 | 11    | 125  |
| free-logit ceiling   | 0.8518 | 0.5079    | 0.8233 | 14    | 0    |

## Readings

**1. SEACells degrades monotonically with BANKSY lambda; scProto does not.**
dge_r 0.8487 -> 0.7223 -> 0.6648 (a 22% loss) and ct_purity 0.8227 -> 0.6974 ->
0.5127 (38%). scProto's dge_r goes the other way, 0.8105 -> 0.8404.

Mechanism: SEACells' only input is the kernel built from the embedding
(`||M - M B A||^2`, SEACells/cpu.py:520). As lambda rises, BANKSY mixes more
NEIGHBOUR expression into each cell, diluting its own transcriptional identity,
and SEACells has nothing else to recover it from. scProto's encoder is a CVAE on
raw expression, so cell identity survives independently of the graph.

**2. scProto essentially matches SEACells' best.** Our 0.8404 (lambda 0.4) vs
their 0.8487 (lambda 0.2), within 1%. On the SAME graph at lambda 0.4 we lead by
12 points (0.8404 vs 0.7223).

**3. We win ct_purity at every lambda measured** (0.8681 / 0.8491 vs 0.8227 /
0.6974).

**4. They win joint_purity and n_covered** -- 0.5035 vs 0.4576, and 19-21 vs
13-16. This is the real remaining gap.

**5. The objective is not the limitation.** Free-logit optimisation of the same
cohesion objective, with no encoder at all, reaches dge_r 0.8518 / joint_purity
0.5079 -- matching SEACells. The trained model reaches 0.8105 / 0.4576, so the
shortfall is the encoder, not the loss. Consistent with EXPERIMENTS.md section
8b: expression-only features cap around int_frac 0.28-0.30 regardless of head or
dimension, while spatially-aware BANKSY features close 87% of the gap with a
trivial linear head.

**6. Combining gram+aff beats either alone** -- best dge_r, best joint_purity,
zero dead. Gram alone loses 5 prototypes and 54 groups fall below 50 cells;
adding aff cuts fragments to 19 and recovers all 200.

**7. The community/UMAP term is harmful here** -- same cohesion term and lambda,
adding it costs 125 of 200 prototypes and more than halves dge_r.

## Caveats

- The two PCA-based SEACells runs disagree sharply (`K200` ct_pur 0.8200 /
  dge_r 0.7470 vs `arbf` 0.4968 / 0.7171) despite both being expression-only.
  They differ in kernel construction: K200 uses SEACells' native path on X_pca,
  arbf was fed our precomputed adaptive-RBF affinity. Quote **K200** as the
  expression-only baseline, not arbf.
- ct_purity is NOT monotone in lambda once arbf is included (0.497 at arbf vs
  0.823 at lambda 0.2), which is why arbf should not be treated as the lambda=0
  endpoint.
- lambda 0.5 has an affinity (`banksy0.5`) but no SEACells baseline anywhere, so
  it cannot be compared and is omitted.
- `cohesion_diag=1.0` injects roughly 20% of the 1-hop affinity into the gram
  similarity (measured: neighbour-pair gram 0.2612 with delta=1 vs 0.2184 with
  delta=0), so "gram only" is not a pure 2-hop arm.
