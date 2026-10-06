# scProto — results (NODE-HOLDOUT protocol)

**Setup.** 20% of cells (stratified by batch; the rebuttal's cached `nodeheldout20_seed0` split) are the **test set**. Every metric below is computed **restricted to those held-out cells**, against the full affinity graph.

**HONESTY CAVEAT — read before citing.** The baselines in these tables were fitted on **all** cells: they saw the test cells during clustering, an advantage in their favor. Only rows labelled `scProto (node-holdout)` are leak-free (trained on the 80% train graph; test cells placed by a frozen-encoder forward pass). `scProto (cohesion)` rows are the edge-holdout-trained models, also fitted with the test cells present — they are context, not the protocol's headline row.

All numbers come from `metacell_scoreboard.py` + `score_nodeheldout.py`: identical code for every method. **Bold** = best, _italic_ = second best, per column.

**Cohesion**: the training objective (self-sim = 1, V[b] multi-batch semantics), per-cell, averaged over test cells only. **Modularity**: per-batch Newman modularity over edges **touching a test cell** (the rebuttal's `test_node_edges` statistic), mean ± std across batches. **Batch entropy / purity**: per metacell over its test members only. **Rare metrics**: the paper's batch-local definitions computed on the test cells.

### Table 1 — community preservation and batch integration (held-out cells)

Modularity: mean ± std across batches (test-touching edges). Cohesion: mean ± std across held-out cells, full graph. Batch entropy and purity: mean ± std across metacells, test members only; `Purity (wtd)` weighted by test-cell counts.

| Method | Panc: Modularity | Panc: Cohesion | Panc: Batch Entropy | Panc: Purity | Panc: Purity (wtd) | Immu: Modularity | Immu: Cohesion | Immu: Batch Entropy | Immu: Purity | Immu: Purity (wtd) | Lung: Modularity | Lung: Cohesion | Lung: Batch Entropy | Lung: Purity | Lung: Purity (wtd) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| scProto NH (cohesion) | 0.127 ± 0.059 | 0.1391 ± 0.1210 | **1.38 ± 0.41** | 0.709 ± 0.202 | 0.760 ± 0.182 | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| scProto NH (coh+comm) | 0.176 ± 0.074 | 0.1591 ± 0.1207 | _1.19 ± 0.43_ | 0.809 ± 0.190 | 0.864 ± 0.156 | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| scProto NH (4-term share25) | 0.278 ± 0.086 | _0.1890 ± 0.1209_ | 0.98 ± 0.47 | 0.908 ± 0.153 | 0.944 ± 0.106 | 0.222 ± 0.095 | 0.0886 ± 0.0960 | 0.53 ± 0.37 | 0.815 ± 0.180 | 0.821 ± 0.162 | 0.340 ± 0.059 | **0.1450 ± 0.1032** | 1.06 ± 0.50 | 0.727 ± 0.201 | 0.712 ± 0.194 |
| Leiden + scPoli-St1 NH | 0.307 ± 0.068 | 0.1608 ± 0.1178 | 1.01 ± 0.49 | **0.983 ± 0.064** | 0.967 ± 0.081 | _0.504 ± 0.125_ | 0.0578 ± 0.0642 | 0.71 ± 0.42 | 0.821 ± 0.165 | 0.870 ± 0.142 | _0.505 ± 0.102_ | 0.0984 ± 0.0978 | 0.92 ± 0.53 | 0.871 ± 0.180 | 0.839 ± 0.196 |
| scProto (cohesion) | 0.275 ± 0.088 | **0.2608 ± 0.1360** | 1.09 ± 0.44 | 0.891 ± 0.164 | 0.916 ± 0.142 | 0.224 ± 0.067 | **0.1308 ± 0.1120** | **0.95 ± 0.32** | 0.736 ± 0.201 | 0.755 ± 0.199 | -- | -- | -- | -- | -- |
| SEACells | **0.498 ± 0.058** | 0.1780 ± 0.1099 | 0.14 ± 0.28 | 0.962 ± 0.097 | _0.970 ± 0.079_ | 0.397 ± 0.022 | _0.1001 ± 0.1000_ | 0.12 ± 0.23 | 0.899 ± 0.154 | 0.880 ± 0.151 | **0.528 ± 0.052** | _0.1367 ± 0.0954_ | 0.36 ± 0.43 | _0.909 ± 0.148_ | 0.901 ± 0.152 |
| SuperCell | _0.391 ± 0.042_ | 0.1411 ± 0.1246 | 0.24 ± 0.37 | 0.919 ± 0.152 | 0.927 ± 0.140 | 0.273 ± 0.023 | 0.0688 ± 0.0988 | 0.11 ± 0.23 | _0.901 ± 0.162_ | 0.860 ± 0.146 | 0.373 ± 0.033 | 0.0967 ± 0.0861 | 0.59 ± 0.49 | 0.907 ± 0.144 | 0.897 ± 0.149 |
| MetaCell2 | 0.221 ± 0.106 | 0.1043 ± 0.1315 | 0.41 ± 0.50 | 0.919 ± 0.169 | 0.875 ± 0.206 | 0.190 ± 0.078 | 0.0514 ± 0.1007 | 0.29 ± 0.38 | 0.872 ± 0.195 | 0.801 ± 0.188 | 0.233 ± 0.056 | 0.0905 ± 0.0671 | 0.64 ± 0.53 | 0.902 ± 0.147 | _0.902 ± 0.146_ |
| MetaQ | 0.326 ± 0.080 | 0.1144 ± 0.0730 | 0.32 ± 0.41 | 0.941 ± 0.139 | 0.956 ± 0.114 | 0.260 ± 0.107 | 0.0366 ± 0.0449 | 0.40 ± 0.36 | 0.824 ± 0.178 | 0.814 ± 0.157 | 0.351 ± 0.116 | 0.0908 ± 0.0708 | 0.47 ± 0.49 | **0.909 ± 0.129** | **0.915 ± 0.129** |
| Leiden + scPoli-St1 | 0.346 ± 0.053 | 0.1708 ± 0.1276 | 1.02 ± 0.43 | _0.971 ± 0.099_ | **0.972 ± 0.086** | 0.314 ± 0.192 | 0.0863 ± 0.0687 | 0.76 ± 0.39 | 0.859 ± 0.151 | 0.873 ± 0.145 | 0.488 ± 0.083 | 0.1026 ± 0.0954 | 0.95 ± 0.48 | 0.862 ± 0.203 | 0.831 ± 0.196 |
| Leiden + scVI | 0.336 ± 0.113 | 0.1261 ± 0.0870 | 0.60 ± 0.44 | 0.961 ± 0.106 | 0.962 ± 0.100 | 0.327 ± 0.185 | 0.0761 ± 0.0638 | 0.70 ± 0.40 | 0.885 ± 0.153 | **0.899 ± 0.142** | 0.493 ± 0.085 | 0.1030 ± 0.0978 | 0.92 ± 0.52 | 0.882 ± 0.169 | 0.867 ± 0.184 |
| Leiden + Harmony d8 | 0.259 ± 0.056 | 0.1465 ± 0.1112 | 1.10 ± 0.38 | 0.939 ± 0.136 | 0.910 ± 0.159 | 0.302 ± 0.117 | 0.0754 ± 0.0637 | _0.86 ± 0.40_ | 0.850 ± 0.148 | 0.843 ± 0.154 | 0.417 ± 0.146 | 0.0841 ± 0.0922 | _1.11 ± 0.40_ | 0.793 ± 0.201 | 0.723 ± 0.233 |
| Leiden + ComBat d8 | 0.303 ± 0.072 | 0.1685 ± 0.1157 | 0.77 ± 0.44 | 0.947 ± 0.130 | 0.938 ± 0.126 | **0.517 ± 0.156** | 0.0620 ± 0.0799 | 0.44 ± 0.37 | 0.835 ± 0.177 | 0.838 ± 0.155 | 0.462 ± 0.083 | 0.0975 ± 0.1071 | 0.77 ± 0.49 | 0.794 ± 0.198 | 0.823 ± 0.186 |
| Leiden + BBKNN d8 | 0.367 ± 0.069 | 0.1130 ± 0.0896 | 0.68 ± 0.44 | 0.917 ± 0.149 | 0.925 ± 0.124 | 0.428 ± 0.082 | 0.0508 ± 0.0570 | 0.21 ± 0.29 | 0.834 ± 0.167 | 0.834 ± 0.154 | 0.461 ± 0.091 | 0.0739 ± 0.0759 | 0.90 ± 0.47 | 0.791 ± 0.209 | 0.796 ± 0.203 |
| SEACells + Harmony d8 | 0.190 ± 0.041 | 0.1552 ± 0.1153 | 1.12 ± 0.38 | 0.895 ± 0.162 | 0.929 ± 0.136 | 0.184 ± 0.023 | 0.0849 ± 0.0903 | 0.57 ± 0.51 | 0.856 ± 0.164 | 0.852 ± 0.145 | 0.205 ± 0.038 | 0.0979 ± 0.0761 | **1.31 ± 0.36** | 0.779 ± 0.205 | 0.758 ± 0.213 |
| SEACells + ComBat d8 | 0.256 ± 0.043 | 0.1774 ± 0.1209 | 0.80 ± 0.49 | 0.921 ± 0.146 | 0.947 ± 0.116 | 0.285 ± 0.027 | 0.0900 ± 0.0961 | 0.29 ± 0.34 | 0.883 ± 0.153 | 0.861 ± 0.153 | 0.316 ± 0.040 | 0.1148 ± 0.0957 | 0.67 ± 0.55 | 0.867 ± 0.177 | 0.837 ± 0.186 |
| SEACells + BBKNN d8 | 0.299 ± 0.093 | 0.1194 ± 0.0908 | 0.73 ± 0.44 | 0.899 ± 0.171 | 0.905 ± 0.170 | 0.252 ± 0.172 | 0.0648 ± 0.0746 | 0.47 ± 0.36 | 0.799 ± 0.201 | 0.767 ± 0.211 | 0.285 ± 0.061 | 0.0910 ± 0.0672 | 1.00 ± 0.43 | 0.852 ± 0.192 | 0.823 ± 0.200 |
| SEACells + scVI | 0.277 ± 0.063 | 0.1320 ± 0.1077 | 0.54 ± 0.47 | 0.949 ± 0.117 | 0.966 ± 0.093 | 0.193 ± 0.024 | 0.0805 ± 0.0899 | 0.45 ± 0.46 | **0.909 ± 0.142** | _0.897 ± 0.143_ | 0.327 ± 0.068 | 0.1196 ± 0.0963 | 0.90 ± 0.57 | 0.880 ± 0.171 | 0.884 ± 0.165 |


### Table 2 — rare cell types (held-out cells)

Coverage: mean ± std across batches. Homogeneity: mean ± std across cells, pooled over batches. Both on test cells only.

| Method | Panc: Coverage | Panc: Homogeneity | Immu: Coverage | Immu: Homogeneity | Lung: Coverage | Lung: Homogeneity |
|---|---|---|---|---|---|---|
| scProto NH (cohesion) | 0.552 ± 0.331 | 0.326 ± 0.260 | -- | -- | -- | -- |
| scProto NH (coh+comm) | 0.625 ± 0.298 | 0.494 ± 0.305 | -- | -- | -- | -- |
| scProto NH (4-term share25) | 0.656 ± 0.340 | 0.608 ± 0.386 | 0.867 ± 0.267 | 0.784 ± 0.272 | 0.590 ± 0.368 | 0.500 ± 0.373 |
| Leiden + scPoli-St1 NH | _0.750 ± 0.204_ | _0.682 ± 0.357_ | **1.000 ± 0.000** | 0.816 ± 0.256 | _0.731 ± 0.324_ | 0.646 ± 0.376 |
| scProto (cohesion) | 0.729 ± 0.219 | 0.648 ± 0.301 | 0.933 ± 0.133 | 0.699 ± 0.299 | -- | -- |
| SEACells | 0.521 ± 0.269 | 0.642 ± 0.363 | _1.000 ± 0.000_ | 0.840 ± 0.288 | 0.628 ± 0.293 | 0.536 ± 0.403 |
| SuperCell | 0.427 ± 0.327 | 0.465 ± 0.367 | 1.000 ± 0.000 | 0.871 ± 0.270 | 0.538 ± 0.218 | 0.508 ± 0.384 |
| MetaCell2 | 0.333 ± 0.363 | 0.292 ± 0.346 | 0.483 ± 0.351 | 0.538 ± 0.409 | 0.731 ± 0.355 | 0.588 ± 0.409 |
| MetaQ | 0.458 ± 0.406 | 0.468 ± 0.313 | 0.883 ± 0.145 | 0.581 ± 0.335 | 0.692 ± 0.375 | **0.726 ± 0.374** |
| Leiden + scPoli-St1 | **0.823 ± 0.188** | **0.707 ± 0.360** | 1.000 ± 0.000 | 0.860 ± 0.229 | 0.564 ± 0.367 | 0.539 ± 0.398 |
| Leiden + scVI | 0.417 ± 0.400 | 0.419 ± 0.331 | 0.833 ± 0.211 | _0.903 ± 0.236_ | 0.731 ± 0.324 | 0.680 ± 0.383 |
| Leiden + Harmony d8 | 0.656 ± 0.237 | 0.399 ± 0.319 | 0.883 ± 0.145 | 0.706 ± 0.308 | 0.423 ± 0.250 | 0.309 ± 0.273 |
| Leiden + ComBat d8 | 0.490 ± 0.284 | 0.409 ± 0.294 | 0.933 ± 0.133 | 0.844 ± 0.254 | 0.654 ± 0.317 | 0.459 ± 0.291 |
| Leiden + BBKNN d8 | 0.448 ± 0.279 | 0.343 ± 0.335 | 0.667 ± 0.422 | 0.550 ± 0.340 | 0.500 ± 0.226 | 0.381 ± 0.257 |
| SEACells + Harmony d8 | 0.583 ± 0.312 | 0.494 ± 0.331 | 0.933 ± 0.133 | 0.808 ± 0.314 | 0.513 ± 0.265 | 0.381 ± 0.302 |
| SEACells + ComBat d8 | 0.562 ± 0.263 | 0.489 ± 0.352 | 1.000 ± 0.000 | 0.834 ± 0.262 | 0.449 ± 0.323 | 0.522 ± 0.385 |
| SEACells + BBKNN d8 | 0.406 ± 0.316 | 0.368 ± 0.317 | 0.767 ± 0.200 | 0.642 ± 0.369 | 0.397 ± 0.274 | 0.335 ± 0.297 |
| SEACells + scVI | 0.552 ± 0.331 | 0.637 ± 0.375 | 1.000 ± 0.000 | **0.921 ± 0.220** | **0.756 ± 0.362** | _0.696 ± 0.370_ |

