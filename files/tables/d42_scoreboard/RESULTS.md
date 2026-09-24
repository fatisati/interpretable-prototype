# scProto — results

**Setup.** 20% of each affinity graph's edges were held out (`FRAC_HELDOUT=0.20`, `MIN_VISIBLE_DEGREE=5`, `SEED=0`, `arbf`); models that train on a graph saw only the visible 80%. Every number below is scored on the **held-out 20%** — no model had a training signal for any of those edges. The split is exact (train + test partition the edge set with no overlap and no loss).

All numbers come from `interpretable_ssl/evaluation/metacell_scoreboard.py`, so scProto and every baseline go through identical code on identical graphs. **Bold** = best, _italic_ = second best, per column.

**Cohesion** is the training objective itself, with self-similarity = 1, computed exactly as in training: multi-batch is unconditional, so each cell is scored against `V[b]` — the node set of its own bio-batch graph (all edges with ≥1 endpoint in `b`). Nothing is masked; cross-batch pairs whose zeros carry no information are excluded from numerator and denominator alike. With one batch `V[b]` is every cell and the formula reduces exactly to the batch-agnostic one.

**Batch entropy** is unnormalised Shannon entropy in nats (`mc_metric_utils.calc_batch_entropy`); ceiling is `log(n_batches)` = 2.20 pancreas (9), 1.61 immune (5), 2.77 lung (16). Higher = better mixed.

**Rare cell types** use the paper's definition: rare is decided **within each batch** (frequency below mean−std, Q25 fallback).

### Table 1 — community preservation and batch integration

Modularity: mean ± std **across batches**. Cohesion: mean ± std **across cells**. Batch entropy and purity: mean ± std **across metacells**; `Purity (wtd)` is weighted by metacell size with the matching weighted std.

| Method | Panc: Modularity | Panc: Cohesion | Panc: Batch Entropy | Panc: Purity | Panc: Purity (wtd) | Immu: Modularity | Immu: Cohesion | Immu: Batch Entropy | Immu: Purity | Immu: Purity (wtd) | Lung: Modularity | Lung: Cohesion | Lung: Batch Entropy | Lung: Purity | Lung: Purity (wtd) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SEACells | 0.494 ± 0.060 | 0.0474 ± 0.0356 | 0.18 ± 0.31 | 0.956 ± 0.091 | 0.967 ± 0.076 | 0.398 ± 0.025 | 0.0288 ± 0.0370 | 0.14 ± 0.22 | 0.894 ± 0.140 | 0.881 ± 0.141 | 0.528 ± 0.053 | 0.0386 ± 0.0400 | 0.44 ± 0.49 | _0.905 ± 0.148_ | 0.900 ± 0.154 |
| SuperCell | 0.394 ± 0.038 | 0.0413 ± 0.0470 | 0.28 ± 0.38 | 0.903 ± 0.157 | 0.917 ± 0.149 | 0.277 ± 0.026 | 0.0225 ± 0.0377 | 0.14 ± 0.23 | 0.888 ± 0.161 | 0.857 ± 0.147 | 0.374 ± 0.035 | 0.0335 ± 0.0398 | 0.68 ± 0.51 | 0.899 ± 0.147 | 0.894 ± 0.153 |
| MetaCell2 | 0.217 ± 0.106 | 0.0546 ± 0.1277 | 0.29 ± 0.48 | 0.946 ± 0.144 | 0.860 ± 0.222 | 0.192 ± 0.076 | 0.0298 ± 0.0982 | 0.20 ± 0.35 | **0.924 ± 0.159** | 0.801 ± 0.183 | 0.231 ± 0.057 | 0.0370 ± 0.0528 | 0.71 ± 0.56 | 0.900 ± 0.139 | _0.901 ± 0.139_ |
| MetaQ | 0.325 ± 0.079 | 0.0379 ± 0.0409 | 0.40 ± 0.42 | 0.933 ± 0.134 | 0.955 ± 0.106 | 0.259 ± 0.100 | 0.0209 ± 0.0507 | 0.48 ± 0.37 | 0.798 ± 0.164 | 0.807 ± 0.146 | 0.356 ± 0.123 | 0.0303 ± 0.0391 | 0.54 ± 0.54 | **0.908 ± 0.135** | **0.914 ± 0.132** |
| Leiden + scPoli-St1 | 0.342 ± 0.056 | _0.0826 ± 0.1154_ | 1.25 ± 0.41 | 0.963 ± 0.104 | **0.970 ± 0.080** | 0.312 ± 0.191 | _0.0380 ± 0.0491_ | _0.84 ± 0.37_ | 0.853 ± 0.146 | 0.868 ± 0.145 | 0.483 ± 0.084 | 0.0500 ± 0.0877 | 1.21 ± 0.46 | 0.850 ± 0.201 | 0.833 ± 0.191 |
| Leiden + scVI | 0.335 ± 0.112 | 0.0515 ± 0.0701 | 0.73 ± 0.43 | 0.959 ± 0.095 | 0.961 ± 0.088 | 0.324 ± 0.185 | 0.0347 ± 0.0448 | 0.77 ± 0.37 | 0.878 ± 0.146 | **0.893 ± 0.137** | 0.493 ± 0.087 | 0.0474 ± 0.0808 | 1.14 ± 0.47 | 0.877 ± 0.168 | 0.869 ± 0.180 |
| Leiden + Harmony | 0.511 ± 0.090 | 0.0641 ± 0.1059 | 1.01 ± 0.44 | _0.974 ± 0.089_ | 0.958 ± 0.084 | 0.523 ± 0.101 | 0.0179 ± 0.0383 | 0.61 ± 0.58 | 0.887 ± 0.149 | 0.863 ± 0.139 | _0.597 ± 0.090_ | 0.0441 ± 0.1140 | 1.04 ± 0.57 | 0.846 ± 0.164 | 0.791 ± 0.185 |
| Leiden + Harmony d8 | 0.253 ± 0.056 | 0.0727 ± 0.0889 | _1.29 ± 0.32_ | 0.923 ± 0.151 | 0.903 ± 0.165 | 0.296 ± 0.109 | 0.0366 ± 0.0445 | **0.95 ± 0.38** | 0.849 ± 0.133 | 0.841 ± 0.145 | 0.414 ± 0.149 | 0.0483 ± 0.0830 | _1.34 ± 0.31_ | 0.774 ± 0.205 | 0.723 ± 0.231 |
| Leiden + ComBat | **0.527 ± 0.146** | 0.0617 ± 0.1186 | 0.72 ± 0.51 | 0.969 ± 0.090 | 0.968 ± 0.070 | **0.576 ± 0.147** | 0.0239 ± 0.0659 | 0.65 ± 0.33 | 0.802 ± 0.145 | 0.841 ± 0.144 | 0.558 ± 0.079 | 0.0347 ± 0.0859 | 0.69 ± 0.51 | 0.875 ± 0.186 | 0.810 ± 0.199 |
| Leiden + ComBat d8 | 0.300 ± 0.073 | 0.0745 ± 0.1039 | 0.94 ± 0.46 | 0.944 ± 0.123 | 0.938 ± 0.119 | 0.514 ± 0.155 | 0.0276 ± 0.0586 | 0.54 ± 0.33 | 0.825 ± 0.163 | 0.834 ± 0.154 | 0.468 ± 0.083 | 0.0440 ± 0.0833 | 0.96 ± 0.49 | 0.775 ± 0.198 | 0.817 ± 0.190 |
| Leiden + BBKNN | _0.527 ± 0.063_ | 0.0502 ± 0.0814 | 0.60 ± 0.43 | **0.977 ± 0.079** | 0.958 ± 0.081 | _0.523 ± 0.089_ | 0.0265 ± 0.0513 | 0.37 ± 0.29 | 0.829 ± 0.144 | 0.850 ± 0.130 | **0.617 ± 0.050** | 0.0259 ± 0.0601 | 1.01 ± 0.67 | 0.851 ± 0.172 | 0.856 ± 0.147 |
| Leiden + BBKNN d8 | 0.368 ± 0.070 | 0.0486 ± 0.0659 | 0.83 ± 0.43 | 0.899 ± 0.161 | 0.921 ± 0.125 | 0.427 ± 0.079 | 0.0203 ± 0.0339 | 0.25 ± 0.29 | 0.828 ± 0.152 | 0.828 ± 0.148 | 0.466 ± 0.090 | 0.0344 ± 0.0528 | 1.07 ± 0.44 | 0.777 ± 0.198 | 0.793 ± 0.202 |
| SEACells + Harmony | 0.234 ± 0.069 | **0.0944 ± 0.0957** | **1.50 ± 0.46** | 0.941 ± 0.112 | 0.964 ± 0.083 | 0.290 ± 0.059 | **0.0403 ± 0.0595** | 0.61 ± 0.51 | 0.884 ± 0.152 | 0.859 ± 0.146 | 0.282 ± 0.072 | **0.0638 ± 0.0769** | **1.56 ± 0.37** | 0.826 ± 0.184 | 0.823 ± 0.186 |
| SEACells + ComBat | 0.385 ± 0.087 | 0.0733 ± 0.1060 | 0.74 ± 0.53 | 0.953 ± 0.091 | _0.969 ± 0.069_ | 0.308 ± 0.039 | 0.0312 ± 0.0513 | 0.30 ± 0.35 | _0.894 ± 0.145_ | 0.877 ± 0.141 | 0.428 ± 0.039 | 0.0457 ± 0.0666 | 0.71 ± 0.56 | 0.895 ± 0.158 | 0.885 ± 0.157 |
| SEACells + BBKNN | 0.438 ± 0.063 | 0.0574 ± 0.1254 | 0.97 ± 0.63 | 0.958 ± 0.109 | 0.964 ± 0.081 | 0.296 ± 0.095 | 0.0333 ± 0.0591 | 0.59 ± 0.30 | 0.838 ± 0.174 | 0.830 ± 0.165 | 0.423 ± 0.067 | 0.0423 ± 0.0929 | 1.08 ± 0.75 | 0.838 ± 0.204 | 0.819 ± 0.240 |
| SEACells + scVI | 0.275 ± 0.065 | 0.0492 ± 0.0650 | 0.65 ± 0.48 | 0.940 ± 0.112 | 0.961 ± 0.088 | 0.193 ± 0.025 | 0.0317 ± 0.0484 | 0.50 ± 0.47 | 0.890 ± 0.142 | _0.893 ± 0.134_ | 0.325 ± 0.066 | _0.0504 ± 0.0734_ | 1.08 ± 0.54 | 0.874 ± 0.167 | 0.884 ± 0.160 |


### Table 2 — rare cell types

Coverage: mean ± std **across batches** (it is a per-batch quantity). Homogeneity: mean ± std **across cells**, pooled over batches.

| Method | Panc: Coverage | Panc: Homogeneity | Immu: Coverage | Immu: Homogeneity | Lung: Coverage | Lung: Homogeneity |
|---|---|---|---|---|---|---|
| SEACells | 0.427 ± 0.268 | 0.558 ± 0.400 | **1.000 ± 0.000** | 0.815 ± 0.282 | 0.522 ± 0.227 | 0.540 ± 0.430 |
| SuperCell | 0.125 ± 0.177 | 0.173 ± 0.149 | _1.000 ± 0.000_ | 0.846 ± 0.268 | 0.650 ± 0.220 | 0.462 ± 0.380 |
| MetaCell2 | 0.260 ± 0.332 | 0.332 ± 0.411 | 0.783 ± 0.194 | 0.521 ± 0.410 | 0.850 ± 0.202 | 0.585 ± 0.409 |
| MetaQ | 0.250 ± 0.331 | 0.295 ± 0.290 | 0.950 ± 0.100 | 0.545 ± 0.324 | 0.700 ± 0.245 | 0.657 ± 0.413 |
| Leiden + scPoli-St1 | **0.771 ± 0.199** | 0.602 ± 0.349 | 1.000 ± 0.000 | 0.842 ± 0.236 | 0.733 ± 0.242 | 0.625 ± 0.397 |
| Leiden + scVI | 0.198 ± 0.328 | 0.381 ± 0.391 | 1.000 ± 0.000 | 0.831 ± 0.256 | _0.856 ± 0.181_ | _0.715 ± 0.374_ |
| Leiden + Harmony | 0.531 ± 0.212 | 0.559 ± 0.398 | 1.000 ± 0.000 | 0.766 ± 0.271 | 0.706 ± 0.217 | 0.576 ± 0.371 |
| Leiden + Harmony d8 | 0.396 ± 0.242 | 0.257 ± 0.209 | 0.933 ± 0.133 | 0.710 ± 0.282 | 0.433 ± 0.257 | 0.314 ± 0.242 |
| Leiden + ComBat | _0.771 ± 0.199_ | **0.694 ± 0.312** | 1.000 ± 0.000 | 0.826 ± 0.210 | 0.667 ± 0.316 | 0.691 ± 0.373 |
| Leiden + ComBat d8 | 0.333 ± 0.270 | 0.287 ± 0.236 | 0.933 ± 0.133 | 0.831 ± 0.235 | 0.533 ± 0.260 | 0.443 ± 0.336 |
| Leiden + BBKNN | 0.333 ± 0.270 | 0.502 ± 0.425 | 1.000 ± 0.000 | 0.806 ± 0.208 | 0.772 ± 0.244 | 0.645 ± 0.398 |
| Leiden + BBKNN d8 | 0.167 ± 0.132 | 0.144 ± 0.145 | 0.700 ± 0.267 | 0.656 ± 0.296 | 0.450 ± 0.239 | 0.323 ± 0.235 |
| SEACells + Harmony | 0.698 ± 0.232 | _0.657 ± 0.375_ | 1.000 ± 0.000 | _0.870 ± 0.232_ | 0.833 ± 0.183 | 0.710 ± 0.384 |
| SEACells + ComBat | 0.635 ± 0.243 | 0.592 ± 0.392 | 1.000 ± 0.000 | **0.892 ± 0.199** | 0.689 ± 0.303 | 0.695 ± 0.382 |
| SEACells + BBKNN | 0.396 ± 0.242 | 0.567 ± 0.378 | 1.000 ± 0.000 | 0.758 ± 0.287 | 0.789 ± 0.301 | 0.549 ± 0.418 |
| SEACells + scVI | 0.260 ± 0.307 | 0.478 ± 0.382 | 1.000 ± 0.000 | 0.868 ± 0.229 | **0.894 ± 0.151** | **0.726 ± 0.353** |

