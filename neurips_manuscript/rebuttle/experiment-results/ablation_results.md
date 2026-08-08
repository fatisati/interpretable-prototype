# Ablation: full numbers, all 3 datasets

Pancreas numbers are recomputed straight from `models/pancreas/ablation_*/metrics.json` +
saved CSVs/`proto_vectors.npy` (the notebook's live pancreas cell was interrupted this
session, but the underlying run from disk is complete and matches the notebook's own
formulas exactly — cross-checked against the numbers already in
`reviewer2_nG29_full.md`, which match). Lung/pbmc-immune are read straight from the
notebook's own output cells. Rare-cell coverage/homogeneity for pancreas is taken from
that run's own printed log (the `rare_celltype_purity_table` computation), not
recomputed — the other columns are.

## Task 1 / Task 2 (point values)

| dataset | variant | purity | batch_entropy | modularity | coverage | dge_rbo | dge_kendall | dge_jaccard | scgraph_corr |
|---|---|---|---|---|---|---|---|---|---|
| pancreas | Full model | 0.908330 | 0.398255 | 0.687871 | 0.928571 | 0.175432 | 0.184560 | 0.228959 | 0.910262 |
| pancreas | – community loss | 0.955219 | 0.514148 | 0.499379 | 0.857143 | 0.201823 | 0.183045 | 0.233910 | 0.930914 |
| pancreas | – recon loss | 0.899229 | 0.389049 | 0.678304 | 0.928571 | 0.139816 | 0.156356 | 0.195371 | 0.846026 |
| pancreas | – nassoc | 0.885544 | 0.480759 | 0.724323 | 1.000000 | 0.153296 | 0.191668 | 0.199076 | 0.896755 |
| pancreas | – usage loss | 0.928503 | 0.705755 | 0.721936 | 0.857143 | 0.113520 | 0.139695 | 0.098701 | 0.901167 |
| pancreas | Fixed τ (no calibration) | 0.920349 | 0.345976 | 0.695411 | 0.928571 | 0.159015 | 0.153954 | 0.227656 | 0.873477 |
| pancreas | k-means init (vs waypoint) | 0.951218 | 0.405236 | 0.668646 | 0.857143 | 0.144454 | 0.145422 | 0.177689 | 0.914001 |
| pancreas | Stop-grad off (recon → encoder) | 0.888992 | 0.611555 | 0.618352 | 0.928571 | 0.211634 | 0.224831 | 0.224059 | 0.909445 |
| lung | Full model | 0.865135 | 0.551718 | 0.720402 | 1.000000 | 0.051369 | 0.042794 | 0.185458 | 0.854438 |
| lung | – community loss | 0.907488 | 0.632834 | 0.543126 | 1.000000 | 0.064398 | 0.071551 | 0.159265 | 0.839726 |
| lung | – recon loss | 0.893028 | 0.504655 | 0.725807 | 0.941176 | 0.018006 | 0.068504 | 0.111428 | 0.779597 |
| lung | – nassoc | 0.872539 | 0.447146 | 0.768751 | 0.882353 | 0.038334 | 0.062766 | 0.159509 | 0.838704 |
| lung | – usage loss | 0.843232 | 0.990548 | 0.757849 | 0.941176 | 0.050202 | 0.128469 | 0.077476 | 0.895605 |
| lung | Fixed τ (no calibration) | 0.880098 | 0.512151 | 0.737764 | 0.941176 | 0.046561 | 0.105200 | 0.170938 | 0.810429 |
| lung | k-means init (vs waypoint) | 0.870014 | 0.483152 | 0.730979 | 1.000000 | 0.038988 | 0.095574 | 0.149754 | 0.883070 |
| lung | Stop-grad off (recon → encoder) | 0.872491 | 0.536222 | 0.726190 | 1.000000 | 0.073115 | 0.137252 | 0.180384 | 0.869379 |
| pbmc-immune | Full model | 0.900238 | 0.275494 | 0.667064 | 1.000000 | 0.053054 | 0.002429 | 0.146891 | 0.840849 |
| pbmc-immune | – community loss | 0.882164 | 0.269695 | 0.397270 | 1.000000 | 0.061903 | 0.074800 | 0.135710 | 0.801553 |
| pbmc-immune | – recon loss | 0.894989 | 0.256759 | 0.680611 | 1.000000 | 0.026348 | 0.063078 | 0.094042 | 0.670425 |
| pbmc-immune | – nassoc | 0.909741 | 0.246544 | 0.691810 | 0.937500 | 0.045725 | 0.091837 | 0.119648 | 0.862753 |
| pbmc-immune | – usage loss | 0.902346 | 0.477765 | 0.680276 | 0.875000 | 0.021184 | 0.047423 | 0.078530 | 0.837851 |
| pbmc-immune | Fixed τ (no calibration) | 0.893932 | 0.234200 | 0.691351 | 1.000000 | 0.064893 | 0.061715 | 0.160715 | 0.855794 |
| pbmc-immune | k-means init (vs waypoint) | 0.861979 | 0.313209 | 0.681440 | 1.000000 | 0.048758 | -0.009523 | 0.116456 | 0.861766 |
| pbmc-immune | Stop-grad off (recon → encoder) | 0.894227 | 0.203397 | 0.675209 | 0.937500 | 0.082130 | 0.061447 | 0.164750 | 0.831603 |

(`niche_purity` is `None` everywhere — only populates for the spatial dataset, see bottom.)

## Per-batch / per-metacell variance (mean ± std)

| dataset | variant | purity | batch_entropy | modularity |
|---|---|---|---|---|
| pancreas | Full model | 0.971 ± 0.088 | 1.006 ± 0.596 | 0.615 ± 0.080 |
| pancreas | – community loss | 0.967 ± 0.080 | 0.467 ± 0.468 | 0.449 ± 0.043 |
| pancreas | – recon loss | 0.970 ± 0.072 | 1.160 ± 0.598 | 0.614 ± 0.078 |
| pancreas | – nassoc | 0.962 ± 0.086 | 1.324 ± 0.479 | 0.656 ± 0.080 |
| pancreas | – usage loss | 0.970 ± 0.078 | 1.401 ± 0.645 | 0.663 ± 0.083 |
| pancreas | Fixed τ (no calibration) | 0.979 ± 0.054 | 0.958 ± 0.642 | 0.623 ± 0.079 |
| pancreas | k-means init (vs waypoint) | 0.980 ± 0.043 | 0.832 ± 0.470 | 0.594 ± 0.068 |
| pancreas | Stop-grad off (recon → encoder) | 0.976 ± 0.069 | 1.147 ± 0.566 | 0.555 ± 0.070 |
| lung | Full model | 0.837 ± 0.199 | 1.307 ± 0.576 | 0.655 ± 0.030 |
| lung | – community loss | 0.916 ± 0.127 | 0.573 ± 0.510 | 0.483 ± 0.049 |
| lung | – recon loss | 0.846 ± 0.180 | 1.394 ± 0.585 | 0.666 ± 0.027 |
| lung | – nassoc | 0.804 ± 0.196 | 1.661 ± 0.499 | 0.708 ± 0.033 |
| lung | – usage loss | 0.824 ± 0.212 | 1.506 ± 0.573 | 0.698 ± 0.022 |
| lung | Fixed τ (no calibration) | 0.823 ± 0.190 | 1.380 ± 0.584 | 0.674 ± 0.026 |
| lung | k-means init (vs waypoint) | 0.866 ± 0.171 | 1.401 ± 0.590 | 0.674 ± 0.031 |
| lung | Stop-grad off (recon → encoder) | 0.855 ± 0.167 | 1.433 ± 0.574 | 0.667 ± 0.029 |
| pbmc-immune | Full model | 0.870 ± 0.118 | 1.002 ± 0.443 | 0.623 ± 0.060 |
| pbmc-immune | – community loss | 0.878 ± 0.145 | 0.244 ± 0.286 | 0.379 ± 0.091 |
| pbmc-immune | – recon loss | 0.840 ± 0.165 | 0.957 ± 0.457 | 0.630 ± 0.072 |
| pbmc-immune | – nassoc | 0.852 ± 0.121 | 1.084 ± 0.391 | 0.653 ± 0.056 |
| pbmc-immune | – usage loss | 0.859 ± 0.138 | 1.075 ± 0.439 | 0.647 ± 0.048 |
| pbmc-immune | Fixed τ (no calibration) | 0.852 ± 0.135 | 1.044 ± 0.463 | 0.645 ± 0.062 |
| pbmc-immune | k-means init (vs waypoint) | 0.866 ± 0.129 | 0.953 ± 0.436 | 0.631 ± 0.061 |
| pbmc-immune | Stop-grad off (recon → encoder) | 0.869 ± 0.117 | 0.913 ± 0.498 | 0.606 ± 0.093 |

## Prototype diagnostics (nassoc redundancy + usage collapse)

| dataset | variant | proto_cosine_sim_mean | proto_cosine_sim_max | n_active_prototypes | K | active_frac |
|---|---|---|---|---|---|---|
| pancreas | Full model | 0.064211 | 0.989161 | 219 | 220 | 0.995455 |
| pancreas | – community loss | 0.002867 | 0.954762 | 220 | 220 | 1.000000 |
| pancreas | – recon loss | 0.060544 | 0.984602 | 219 | 220 | 0.995455 |
| pancreas | – nassoc | 0.026590 | 0.986760 | 218 | 220 | 0.990909 |
| pancreas | – usage loss | 0.058627 | 0.990364 | 63 | 220 | 0.286364 |
| pancreas | Fixed τ (no calibration) | 0.054880 | 0.997472 | 219 | 220 | 0.995455 |
| pancreas | k-means init (vs waypoint) | 0.064685 | 0.967665 | 220 | 220 | 1.000000 |
| pancreas | Stop-grad off (recon → encoder) | 0.016039 | 0.975781 | 220 | 220 | 1.000000 |
| lung | Full model | 0.009140 | 0.995389 | 298 | 300 | 0.993333 |
| lung | – community loss | -0.000935 | 0.962259 | 300 | 300 | 1.000000 |
| lung | – recon loss | 0.018305 | 0.996159 | 298 | 300 | 0.993333 |
| lung | – nassoc | 0.007878 | 0.998138 | 290 | 300 | 0.966667 |
| lung | – usage loss | 0.033185 | 0.996567 | 77 | 300 | 0.256667 |
| lung | Fixed τ (no calibration) | 0.024797 | 0.997698 | 298 | 300 | 0.993333 |
| lung | k-means init (vs waypoint) | 0.011463 | 0.966660 | 298 | 300 | 0.993333 |
| lung | Stop-grad off (recon → encoder) | 0.018097 | 0.996852 | 296 | 300 | 0.986667 |
| pbmc-immune | Full model | 0.048151 | 0.995285 | 288 | 300 | 0.960000 |
| pbmc-immune | – community loss | 0.013254 | 0.966774 | 300 | 300 | 1.000000 |
| pbmc-immune | – recon loss | 0.060677 | 0.993824 | 288 | 300 | 0.960000 |
| pbmc-immune | – nassoc | 0.103079 | 0.998983 | 285 | 300 | 0.950000 |
| pbmc-immune | – usage loss | 0.070263 | 0.998253 | 48 | 300 | 0.160000 |
| pbmc-immune | Fixed τ (no calibration) | 0.063345 | 0.993434 | 285 | 300 | 0.950000 |
| pbmc-immune | k-means init (vs waypoint) | 0.042112 | 0.971966 | 292 | 300 | 0.973333 |
| pbmc-immune | Stop-grad off (recon → encoder) | 0.087943 | 0.998167 | 286 | 300 | 0.953333 |

## Rare-cell coverage / homogeneity (mean ± std)

| dataset | variant | batch_rare_coverage | batch_rare_homogeneity |
|---|---|---|---|
| pancreas | Full model | 0.69 ± 0.22 | 0.63 ± 0.16 |
| pancreas | – community loss | 0.50 ± 0.23 | 0.41 ± 0.23 |
| pancreas | – recon loss | 0.70 ± 0.29 | 0.60 ± 0.18 |
| pancreas | – nassoc | 0.60 ± 0.29 | 0.51 ± 0.19 |
| pancreas | – usage loss | 0.43 ± 0.27 | 0.40 ± 0.21 |
| pancreas | Fixed τ (no calibration) | 0.72 ± 0.21 | 0.56 ± 0.16 |
| pancreas | k-means init (vs waypoint) | 0.56 ± 0.28 | 0.48 ± 0.22 |
| pancreas | Stop-grad off (recon → encoder) | 0.67 ± 0.29 | 0.51 ± 0.16 |
| lung | Full model | 0.71 ± 0.27 | 0.56 ± 0.23 |
| lung | – community loss | 0.65 ± 0.25 | 0.54 ± 0.24 |
| lung | – recon loss | 0.72 ± 0.28 | 0.54 ± 0.25 |
| lung | – nassoc | 0.56 ± 0.22 | 0.48 ± 0.23 |
| lung | – usage loss | 0.65 ± 0.25 | 0.52 ± 0.24 |
| lung | Fixed τ (no calibration) | 0.73 ± 0.29 | 0.55 ± 0.23 |
| lung | k-means init (vs waypoint) | 0.82 ± 0.21 | 0.57 ± 0.22 |
| lung | Stop-grad off (recon → encoder) | 0.69 ± 0.25 | 0.56 ± 0.24 |
| pbmc-immune | Full model | 1.00 ± 0.00 | 0.90 ± 0.03 |
| pbmc-immune | – community loss | 1.00 ± 0.00 | 0.70 ± 0.12 |
| pbmc-immune | – recon loss | 1.00 ± 0.00 | 0.87 ± 0.06 |
| pbmc-immune | – nassoc | 0.80 ± 0.40 | 0.76 ± 0.17 |
| pbmc-immune | – usage loss | 1.00 ± 0.00 | 0.79 ± 0.15 |
| pbmc-immune | Fixed τ (no calibration) | 1.00 ± 0.00 | 0.85 ± 0.07 |
| pbmc-immune | k-means init (vs waypoint) | 0.93 ± 0.13 | 0.81 ± 0.13 |
| pbmc-immune | Stop-grad off (recon → encoder) | 1.00 ± 0.00 | 0.87 ± 0.06 |

## Macro-averaged (rare-type-fair) cell-type purity

Headline (cell-count-weighted) purity for the same rows is in Table 1 — not repeated here.

| dataset | variant | macro_celltype_purity |
|---|---|---|
| pancreas | Full model | 0.818088 |
| pancreas | – community loss | 0.752213 |
| pancreas | – recon loss | 0.792327 |
| pancreas | – nassoc | 0.768131 |
| pancreas | – usage loss | 0.745895 |
| pancreas | Fixed τ (no calibration) | 0.802129 |
| pancreas | k-means init (vs waypoint) | 0.766850 |
| pancreas | Stop-grad off (recon → encoder) | 0.794976 |
| lung | Full model | 0.750703 |
| lung | – community loss | 0.845893 |
| lung | – recon loss | 0.747779 |
| lung | – nassoc | 0.660070 |
| lung | – usage loss | 0.724798 |
| lung | Fixed τ (no calibration) | 0.728020 |
| lung | k-means init (vs waypoint) | 0.771513 |
| lung | Stop-grad off (recon → encoder) | 0.757140 |
| pbmc-immune | Full model | 0.772473 |
| pbmc-immune | – community loss | 0.775485 |
| pbmc-immune | – recon loss | 0.760946 |
| pbmc-immune | – nassoc | 0.684564 |
| pbmc-immune | – usage loss | 0.715562 |
| pbmc-immune | Fixed τ (no calibration) | 0.744547 |
| pbmc-immune | k-means init (vs waypoint) | 0.743088 |
| pbmc-immune | Stop-grad off (recon → encoder) | 0.777502 |

## Std not available (for any of the 3 datasets)

- `coverage`, `dge_rbo_avg`, `dge_kendall_avg`, `dge_jaccard_avg`, `scgraph_corr_avg`
  (Table 1) — the notebook never computes a std for these, only a single point value.
- `macro_celltype_purity` — its own function returns one aggregate number, no spread.
- `proto_cosine_sim_mean/max`, `active_frac` (Table 3) — each is already a single
  summary statistic per run (a similarity or a fraction), not an average over a
  distribution, so there's nothing to attach a std to.
- Everything else (Table 1's purity/batch_entropy/modularity, Table 2, Table 4) does
  have std, for all 3 datasets.

## Verdicts (which claims replicate — see table/row above for the actual numbers)

> **Updated 04 Aug 2026.** These verdicts were based on point values only. A paired
> per-batch significance test now exists for every arm on all 3 datasets — see
> `final-comment-evidence/ablation-variance-findings.md`. Two rows below are revised
> (marked ⚠); the rest stand.

| Component | Claim | Where to check | Replicates on all 3? |
|---|---|---|---|
| `no_community` | largest modularity drop | Table 1 | ✅ now also significant: 8/9 pancreas, 16/16 lung, 5/5 immune |
| `no_usage` | prototype collapse | Table 3, `active_frac` | ✅ |
| `no_recon` | purity/modularity flat | Table 1 | ✅ |
| `fixed_temp` | purity/modularity flat-or-up | Table 1 | ✅ |
| community/usage/nassoc | rare coverage/homogeneity drops | Table 4 | ⚠ direction holds on all 3, but the **lung** effects are near zero (community +0.015, usage +0.032, both ns) — significant only on pancreas |
| `no_nassoc` | largest purity drop | Table 1 | ❌ pancreas-only — largest is `no_usage` on lung, `kmeans_init` on pbmc. **Not testable at all** (purity's std is per metacell, cannot be paired) |
| `stopgrad_off` | modularity drop (F5RB Q1 evidence) | Table 1 | ⚠ pancreas-only, but now **significant there**: 9/9 batches, p=0.0039, q=0.0091. Reverses significantly on lung (4/16, q=0.026) |
| `kmeans_init` | modularity drop | Table 1 | ❌ reverses direction on lung/pbmc |
| `no_community` | headline-up / macro-down purity paradox | Table 1 vs. Table 5 | ❌ pancreas-only framing |

## Before writing the reviewer response

- **Keep general:** community (modularity) and usage (active_frac) — confirmed on all 3,
  and community is now significant on all 3.
- **Now defensible with the dataset named:** the stopgrad-modularity claim on pancreas —
  9/9 batches, p=0.0039 (the exact test floor for n=9). Scope it to pancreas.
- **Drop entirely:** the nassoc-purity claim. Purity is per metacell, so no paired test is
  possible, and the gap (0.009) is a tenth of the spread (0.088).
- **Re-anchor nassoc to rare homogeneity, but as direction only** — +0.121 / +0.070 /
  +0.143 (7/8, 11/15, 5/5). Raw one-sided p < 0.05 on all three, but none survives BH
  correction, so do not write "replicates on all 3".
- **Drop:** the headline/macro purity "paradox" framing for `no_community` — pancreas-only.
