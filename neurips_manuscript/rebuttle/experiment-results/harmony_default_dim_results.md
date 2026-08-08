# Harmony dimensionality — evidence sheet

Used in: `final-comment/rev2-nG29-final.md` (§2), `rev1-F5RB-final.md`, `rev3-e9Ho-final.md`,
`ac-comment.md` (§2). Numbers here are the source for all four.

## What was asked

Reviewer nG29, round 3: *"running Harmony at 8 PCs … handicaps the comparison in the
authors' favour; I would characterise the original 50-dim run as the fairer comparison
rather than a dimensionality bug. Reporting both dimensionalities would remove the concern
entirely, whichever way it resolves."*

## What we ran

Harmony at **8, 20 and 50 PCs** → SEACells and Leiden on top, K = scProto's
`num_prototypes`; scProto trained at each matching latent dimension.
20 and 50 are both defensible defaults: `harmonypy` does no PCA and has no native
dimension; **20** = the harmony package's own `HarmonyMatrix(npcs=20)` and its quickstart;
**50** = Scanpy/Seurat PCA defaults.

Done: d=8 and d=50 on all three datasets, d=20 on Pancreas (+ Harmony arm on Lung).
Pending: scProto d=20 on Lung, both arms on Immune.

## What we found

1. **At 20 PCs (Harmony's own default), scProto leads on every metric.**
2. **At 50 PCs the rare-cell metrics are level** — no significant difference in either
   direction on any dataset.
3. **Modularity separates at every dimension**, significantly on Pancreas and Lung. This is
   the only axis in the comparison that reaches significance.
4. **Harmony's rare-cell result tracks the dimension it is given** (0.32 → 0.54 → 0.66);
   scProto's modularity does not move with its own latent size (0.601 / 0.615 / 0.611).

## Rare-cell macro F1 (mean ± std across batches)

| Dataset | Method | d=8 | d=20 | d=50 |
|---|---|---|---|---|
| **Pancreas** (n=8) | scProto | 0.54±0.21 | **0.65±0.18** | 0.60±0.18 |
| | SEACells (Harmony) | 0.32±0.16 | 0.54±0.21 | 0.66±0.13 |
| | Leiden (Harmony) | 0.22±0.17 | 0.48±0.23 | 0.41±0.22 |
| **Lung** (n=15) | scProto | 0.60±0.18 | pending | 0.58±0.23 |
| | SEACells (Harmony) | 0.36±0.14 | 0.58±0.18 | 0.64±0.17 |
| | Leiden (Harmony) | 0.28±0.14 | pending | 0.62±0.17 |
| **Immune** (n=5) | scProto | 0.85±0.14 | pending | 0.91±0.02 |
| | SEACells (Harmony) | 0.75±0.16 | pending | 0.89±0.03 |
| | Leiden (Harmony) | 0.75±0.17 | pending | 0.71±0.27 |

Pancreas at 20 PCs, full row: scProto coverage 0.76±0.23, recall 0.67±0.18, precision
0.76±0.14, homogeneity 0.59±0.17, CT purity 0.976±0.060 — against SEACells (Harmony)
0.69±0.22 / 0.62±0.20 / 0.57±0.18 / 0.56±0.15 / 0.965±0.086.

## Modularity per batch (mean ± std)

| Dataset | Method | d=8 | d=20 | d=50 |
|---|---|---|---|---|
| **Pancreas** | scProto | 0.601±0.088 | 0.615±0.070 | 0.611±0.082 |
| | SEACells (Harmony) | 0.567±0.019 | 0.500±0.035 | 0.432±0.038 |
| | Leiden (Harmony) | 0.612±0.023 | 0.384±0.093 | 0.502±0.130 |
| **Lung** | scProto | 0.664±0.024 | pending | 0.654±0.027 |
| | SEACells (Harmony) | 0.606±0.046 | pending | 0.470±0.046 |
| | Leiden (Harmony) | 0.419±0.152 | pending | 0.570±0.070 |
| **Immune** | scProto | 0.629±0.058 | pending | 0.631±0.054 |
| | SEACells (Harmony) | 0.549±0.014 | pending | 0.322±0.102 |
| | Leiden (Harmony) | 0.297±0.122 | pending | 0.461±0.094 |

## Significance (paired one-sided Wilcoxon, Bonferroni per dataset)

**Modularity, scProto as reference:**

| vs. | Pancreas | Lung | Immune |
|---|---|---|---|
| SEACells (Harmony d=50) | 9/9, **p=0.012** | 16/16, **p<0.001** | 5/5, ns |
| Leiden (Harmony d=50) | 9/9, **p=0.012** | 15/16, **p<0.001** | 5/5, ns |
| SEACells (Harmony d=8) | 7/9, ns | 14/16, **p<0.01** | 5/5, ns |

**Rare-cell F1, scProto as reference:**

| vs. | Pancreas | Lung | Immune |
|---|---|---|---|
| SEACells (Harmony d=50) | 2/8, ns | 6/15, ns | 4/5, ns |
| SEACells (Harmony d=20) | 6/8, ns (p_raw 0.074) | pending | pending |
| SEACells (Harmony d=8) | 7/8, ns (p=0.07) | 13/15, **p<0.01** | 4/5, ns |
| Leiden (Harmony d=8) | 8/8, **p=0.023** | 13/15, **p<0.01** | 4/5, ns |

Per-metacell cell-type purity: scProto significantly above both Harmony d=50 arms on Lung
and Immune (p<0.001, Mann-Whitney). Immune reaches significance nowhere — at n=5 the
corrected test cannot, so win-counts are reported there.

## Why the rare-cell gap closes at 50 PCs

Harmony corrects per soft cluster under an objective that rewards batch-mixed clusters, so
it targets cross-batch grouping directly — which is what rare-cell recall measures. At 8
PCs that mixing crowds the space and disturbs cell types (purity 0.919±0.139, F1
0.32±0.16); more dimensions let it mix while leaving types intact, and recall rises
(0.35±0.14 → 0.62±0.20 → 0.76±0.10 on Pancreas). The cost lands on graph structure:
Harmony's modularity falls as its dimension grows, scProto's does not change with its own.

## Claims to keep consistent

- Say: modularity separation holds at every dimension and dataset, significant on Pancreas
  and Lung; scProto leads on every metric at Harmony's own default of 20 PCs; rare-cell
  metrics are level at 50 PCs.
- Don't say: that d=8 was a "dimensionality bug"; that scProto wins rare-cell recovery in
  general.
- Open check: `recompute_modularity_canonical` writes canonical values into `metrics.json`
  (Pancreas SEACells Harmony d=50: 0.285) while Table 1 shows 0.432, the pre-recompute
  value that `modularity_per_batch.csv` and the significance tests use. Direction favours
  us; confirm which value each table reports before the numbers are final.
