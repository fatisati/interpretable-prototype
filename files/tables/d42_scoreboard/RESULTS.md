# scProto — results

**Setup.** 20% of each affinity graph's edges were held out (`FRAC_HELDOUT=0.20`, `MIN_VISIBLE_DEGREE=5`, `SEED=0`, `arbf`); models that train on a graph saw only the visible 80%. Every number below is scored on the **held-out 20%** — no model had a training signal for any of those edges. The split is exact (train + test partition the edge set with no overlap and no loss).

All numbers come from `interpretable_ssl/evaluation/metacell_scoreboard.py`, so scProto and every baseline go through identical code on identical graphs. **Bold** = best, _italic_ = second best, per column.

**Cohesion** is the training objective itself, self-similarity = 1, computed exactly as in training: multi-batch is unconditional, so each cell is scored against `V[b]` — the node set of its own bio-batch graph (all edges with ≥1 endpoint in `b`). Nothing is masked; cross-batch pairs whose zeros carry no information are excluded from numerator and denominator alike. With one batch `V[b]` is every cell and the formula reduces exactly to the batch-agnostic one.

It is averaged over the **test nodes** (cells with ≥1 held-out edge) but computed on the **full** bio-batch graph — a held-out node's similarity to train nodes counts too. Scoring on the held-out edges *alone* is a different and degenerate quantity: ~5x smaller, and it ranks a random partition **first**. On these three datasets the test-node set is 100% of cells (20% of edges removed at random means essentially every cell loses one), so the de-circularisation comes from the model having *trained* on 80%, not from the node subset.

**Batch entropy** is unnormalised Shannon entropy in nats (`mc_metric_utils.calc_batch_entropy`); ceiling is `log(n_batches)` = 2.20 pancreas (9), 1.61 immune (5), 2.77 lung (16). Higher = better mixed.

**Rare cell types** use the paper's definition: rare is decided **within each batch** (frequency below mean−std, Q25 fallback).

### Table 1 — community preservation and batch integration

Modularity: mean ± std **across batches**, on the held-out edges. Cohesion: mean ± std **across cells**, full graph, self-sim = 1. Batch entropy and purity: mean ± std **across metacells**; `Purity (wtd)` is size-weighted with the matching weighted std.

| Method | Panc: Modularity | Panc: Cohesion | Panc: Batch Entropy | Panc: Purity | Panc: Purity (wtd) | Immu: Modularity | Immu: Cohesion | Immu: Batch Entropy | Immu: Purity | Immu: Purity (wtd) | Lung: Modularity | Lung: Cohesion | Lung: Batch Entropy | Lung: Purity | Lung: Purity (wtd) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| scProto (cohesion) | 0.240 ± 0.088 | **0.2569 ± 0.1342** | **1.36 ± 0.32** | 0.873 ± 0.154 | 0.911 ± 0.130 | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| SEACells | **0.476 ± 0.056** | 0.1752 ± 0.1087 | 0.18 ± 0.31 | 0.956 ± 0.091 | _0.967 ± 0.076_ | 0.393 ± 0.028 | **0.1014 ± 0.1005** | 0.14 ± 0.22 | _0.894 ± 0.140_ | 0.881 ± 0.141 | **0.505 ± 0.057** | **0.1371 ± 0.0944** | 0.44 ± 0.49 | _0.905 ± 0.148_ | 0.900 ± 0.154 |
| SuperCell | _0.389 ± 0.034_ | 0.1356 ± 0.1180 | 0.28 ± 0.38 | 0.903 ± 0.157 | 0.917 ± 0.149 | 0.275 ± 0.027 | 0.0696 ± 0.0994 | 0.14 ± 0.23 | 0.888 ± 0.161 | 0.857 ± 0.147 | 0.372 ± 0.034 | 0.0979 ± 0.0881 | 0.68 ± 0.51 | 0.899 ± 0.147 | 0.894 ± 0.153 |
| MetaCell2 | 0.217 ± 0.105 | 0.1022 ± 0.1300 | 0.29 ± 0.48 | 0.946 ± 0.144 | 0.860 ± 0.222 | 0.188 ± 0.076 | 0.0530 ± 0.1065 | 0.20 ± 0.35 | **0.924 ± 0.159** | 0.801 ± 0.183 | 0.226 ± 0.056 | 0.0905 ± 0.0670 | 0.71 ± 0.56 | 0.900 ± 0.139 | _0.901 ± 0.139_ |
| MetaQ | 0.323 ± 0.077 | 0.1128 ± 0.0731 | 0.40 ± 0.42 | 0.933 ± 0.134 | 0.955 ± 0.106 | 0.257 ± 0.102 | 0.0378 ± 0.0526 | 0.48 ± 0.37 | 0.798 ± 0.164 | 0.807 ± 0.146 | 0.348 ± 0.114 | 0.0916 ± 0.0701 | 0.54 ± 0.54 | **0.908 ± 0.135** | **0.914 ± 0.132** |
| Leiden + scPoli-St1 | 0.356 ± 0.054 | 0.1694 ± 0.1286 | 1.25 ± 0.41 | **0.963 ± 0.104** | **0.970 ± 0.080** | 0.312 ± 0.190 | 0.0869 ± 0.0689 | _0.84 ± 0.37_ | 0.853 ± 0.146 | 0.868 ± 0.145 | 0.493 ± 0.082 | 0.1042 ± 0.1000 | 1.21 ± 0.46 | 0.850 ± 0.201 | 0.833 ± 0.191 |
| Leiden + scVI | 0.341 ± 0.108 | 0.1254 ± 0.0851 | 0.73 ± 0.43 | _0.959 ± 0.095_ | 0.961 ± 0.088 | 0.323 ± 0.186 | 0.0778 ± 0.0662 | 0.77 ± 0.37 | 0.878 ± 0.146 | **0.893 ± 0.137** | _0.503 ± 0.083_ | 0.1039 ± 0.0990 | 1.14 ± 0.47 | 0.877 ± 0.168 | 0.869 ± 0.180 |
| Leiden + Harmony d8 | 0.262 ± 0.054 | 0.1453 ± 0.1069 | 1.29 ± 0.32 | 0.923 ± 0.151 | 0.903 ± 0.165 | 0.297 ± 0.109 | 0.0768 ± 0.0651 | **0.95 ± 0.38** | 0.849 ± 0.133 | 0.841 ± 0.145 | 0.420 ± 0.147 | 0.0851 ± 0.0938 | _1.34 ± 0.31_ | 0.774 ± 0.205 | 0.723 ± 0.231 |
| Leiden + ComBat d8 | 0.294 ± 0.065 | 0.1690 ± 0.1208 | 0.94 ± 0.46 | 0.944 ± 0.123 | 0.938 ± 0.119 | **0.510 ± 0.158** | 0.0627 ± 0.0812 | 0.54 ± 0.33 | 0.825 ± 0.163 | 0.834 ± 0.154 | 0.453 ± 0.085 | 0.0969 ± 0.1069 | 0.96 ± 0.49 | 0.775 ± 0.198 | 0.817 ± 0.190 |
| Leiden + BBKNN d8 | 0.376 ± 0.064 | 0.1130 ± 0.0942 | 0.83 ± 0.43 | 0.899 ± 0.161 | 0.921 ± 0.125 | _0.427 ± 0.079_ | 0.0506 ± 0.0556 | 0.25 ± 0.29 | 0.828 ± 0.152 | 0.828 ± 0.148 | 0.472 ± 0.089 | 0.0734 ± 0.0764 | 1.07 ± 0.44 | 0.777 ± 0.198 | 0.793 ± 0.202 |
| SEACells + Harmony d8 | 0.188 ± 0.042 | 0.1518 ± 0.1094 | _1.31 ± 0.34_ | 0.884 ± 0.163 | 0.919 ± 0.139 | 0.183 ± 0.024 | 0.0857 ± 0.0912 | 0.64 ± 0.51 | 0.843 ± 0.165 | 0.847 ± 0.142 | 0.203 ± 0.039 | 0.0971 ± 0.0751 | **1.52 ± 0.27** | 0.778 ± 0.202 | 0.757 ± 0.214 |
| SEACells + ComBat d8 | 0.242 ± 0.036 | _0.1773 ± 0.1250_ | 0.97 ± 0.47 | 0.911 ± 0.146 | 0.942 ± 0.116 | 0.279 ± 0.032 | _0.0908 ± 0.0990_ | 0.35 ± 0.36 | 0.875 ± 0.155 | 0.858 ± 0.149 | 0.295 ± 0.039 | 0.1160 ± 0.0994 | 0.76 ± 0.55 | 0.865 ± 0.170 | 0.837 ± 0.182 |
| SEACells + BBKNN d8 | 0.307 ± 0.089 | 0.1187 ± 0.0900 | 0.88 ± 0.42 | 0.881 ± 0.175 | 0.904 ± 0.168 | 0.258 ± 0.181 | 0.0665 ± 0.0738 | 0.54 ± 0.32 | 0.780 ± 0.201 | 0.764 ± 0.207 | 0.290 ± 0.055 | 0.0903 ± 0.0655 | 1.17 ± 0.40 | 0.847 ± 0.188 | 0.824 ± 0.200 |
| SEACells + scVI | 0.274 ± 0.062 | 0.1299 ± 0.1071 | 0.65 ± 0.48 | 0.940 ± 0.112 | 0.961 ± 0.088 | 0.190 ± 0.026 | 0.0811 ± 0.0904 | 0.50 ± 0.47 | 0.890 ± 0.142 | _0.893 ± 0.134_ | 0.326 ± 0.068 | _0.1207 ± 0.0973_ | 1.08 ± 0.54 | 0.874 ± 0.167 | 0.884 ± 0.160 |


### Table 2 — rare cell types

Coverage: mean ± std **across batches** (it is a per-batch quantity). Homogeneity: mean ± std **across cells**, pooled over batches.

| Method | Panc: Coverage | Panc: Homogeneity | Immu: Coverage | Immu: Homogeneity | Lung: Coverage | Lung: Homogeneity |
|---|---|---|---|---|---|---|
| scProto (cohesion) | _0.740 ± 0.218_ | _0.589 ± 0.356_ | -- | -- | -- | -- |
| SEACells | 0.427 ± 0.268 | 0.558 ± 0.400 | **1.000 ± 0.000** | 0.815 ± 0.282 | 0.522 ± 0.227 | 0.540 ± 0.430 |
| SuperCell | 0.125 ± 0.177 | 0.173 ± 0.149 | _1.000 ± 0.000_ | _0.846 ± 0.268_ | 0.650 ± 0.220 | 0.462 ± 0.380 |
| MetaCell2 | 0.260 ± 0.332 | 0.332 ± 0.411 | 0.783 ± 0.194 | 0.521 ± 0.410 | 0.850 ± 0.202 | 0.585 ± 0.409 |
| MetaQ | 0.250 ± 0.331 | 0.295 ± 0.290 | 0.950 ± 0.100 | 0.545 ± 0.324 | 0.700 ± 0.245 | 0.657 ± 0.413 |
| Leiden + scPoli-St1 | **0.771 ± 0.199** | **0.602 ± 0.349** | 1.000 ± 0.000 | 0.842 ± 0.236 | 0.733 ± 0.242 | 0.625 ± 0.397 |
| Leiden + scVI | 0.198 ± 0.328 | 0.381 ± 0.391 | 1.000 ± 0.000 | 0.831 ± 0.256 | _0.856 ± 0.181_ | _0.715 ± 0.374_ |
| Leiden + Harmony d8 | 0.396 ± 0.242 | 0.257 ± 0.209 | 0.933 ± 0.133 | 0.710 ± 0.282 | 0.433 ± 0.257 | 0.314 ± 0.242 |
| Leiden + ComBat d8 | 0.333 ± 0.270 | 0.287 ± 0.236 | 0.933 ± 0.133 | 0.831 ± 0.235 | 0.533 ± 0.260 | 0.443 ± 0.336 |
| Leiden + BBKNN d8 | 0.167 ± 0.132 | 0.144 ± 0.145 | 0.700 ± 0.267 | 0.656 ± 0.296 | 0.450 ± 0.239 | 0.323 ± 0.235 |
| SEACells + Harmony d8 | 0.562 ± 0.285 | 0.342 ± 0.311 | 0.933 ± 0.133 | 0.788 ± 0.311 | 0.544 ± 0.187 | 0.362 ± 0.293 |
| SEACells + ComBat d8 | 0.562 ± 0.285 | 0.335 ± 0.238 | 1.000 ± 0.000 | 0.823 ± 0.255 | 0.456 ± 0.307 | 0.508 ± 0.404 |
| SEACells + BBKNN d8 | 0.250 ± 0.306 | 0.228 ± 0.172 | 0.900 ± 0.200 | 0.643 ± 0.369 | 0.539 ± 0.267 | 0.310 ± 0.282 |
| SEACells + scVI | 0.260 ± 0.307 | 0.478 ± 0.382 | 1.000 ± 0.000 | **0.868 ± 0.229** | **0.894 ± 0.151** | **0.726 ± 0.353** |

