# scProto results — held-out-edge evaluation

Every number below is produced by `interpretable_ssl/evaluation/metacell_scoreboard.py`, so scProto and every baseline go through **identical code on identical graphs**.

**Setup.** 20% of each affinity graph's edges were hidden (`FRAC_HELDOUT=0.20`, `MIN_VISIBLE_DEGREE=5`, `SEED=0`, `arbf`). Models that train on a graph saw only the visible 80%. The `test edges` columns are scored on the hidden 20% — **no model had a training signal for any of those edges**, which is what de-circularises the metric. The split is exact: train + test partition the edge set with no overlap and no loss.

**Cohesion** is the training objective itself. Multi-batch is handled unconditionally: each cell is scored against `V[b]`, the node set of its own bio-batch graph (all edges with ≥1 endpoint in `b`). Nothing is masked — cross-batch similarities keep their real values; what is excluded are pairs whose zeros carry no information. With one batch `V[b]` is every cell, so the formula reduces exactly to the batch-agnostic one.

**Batch entropy** is unnormalised Shannon entropy in nats (`mc_metric_utils.calc_batch_entropy`), so the ceiling is `log(n_batches)`. Higher = better mixed.

**Rare metrics** are the paper's own definitions (`paper_figures._rare_table_one`): rare types are defined **per batch** (within-batch frequency below mean−std, Q25 fallback) and every quantity is aggregated mean ± std **across batches**.

## pancreas

_9 biological batches; batch-entropy ceiling log(9) = 2.197_

### Section 1 — preserves graph similarity while mixing batches

std over **cells** for cohesion; over **batches** for modularity; over **metacells** for entropy/purity (size-weighted rows use the weighted std).

| method | cohesion aff (test edges) | cohesion gram (test edges) | cohesion aff (train) | modularity (test edges) | modularity (train) | batch entropy | batch entropy (size-wtd) | ct purity | ct purity (size-wtd) | K (metacells) | median size |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SEACells | 0.0474 ± 0.0356 | 0.0796 ± 0.0480 | 0.1433 ± 0.0901 | 0.4943 ± 0.0598 | 0.4953 ± 0.0616 | 0.1792 ± 0.3089 | 0.1909 ± 0.3007 | 0.9564 ± 0.0910 | 0.9667 ± 0.0762 | 220 | 64.5 |
| SuperCell | 0.0413 ± 0.0470 | 0.0658 ± 0.0574 | 0.1117 ± 0.0987 | 0.3935 ± 0.0377 | 0.3901 ± 0.0380 | 0.2834 ± 0.3798 | 0.3938 ± 0.4297 | 0.9027 ± 0.1571 | 0.9166 ± 0.1487 | 220 | 45.0 |
| MetaCell2 | 0.0546 ± 0.1277 | 0.0705 ± 0.1272 | 0.0898 ± 0.1273 | 0.2170 ± 0.1057 | 0.2171 ± 0.1028 | 0.2882 ± 0.4763 | 0.6857 ± 0.6113 | 0.9463 ± 0.1444 | 0.8598 ± 0.2216 | 426 | 13.0 |
| MetaQ | 0.0379 ± 0.0409 | 0.0602 ± 0.0460 | 0.0938 ± 0.0626 | 0.3253 ± 0.0795 | 0.3253 ± 0.0800 | 0.3992 ± 0.4179 | 0.3753 ± 0.4135 | 0.9331 ± 0.1345 | 0.9549 ± 0.1059 | 217 | 83.0 |
| Leiden + scPoli-Stage1 | 0.0826 ± 0.1154 | 0.1074 ± 0.1152 | 0.1467 ± 0.1213 | 0.3422 ± 0.0563 | 0.3404 ± 0.0588 | 1.2471 ± 0.4104 | 1.2980 ± 0.4254 | 0.9629 ± 0.1036 | 0.9703 ± 0.0796 | 220 | 54.0 |
| Leiden + scVI | 0.0515 ± 0.0701 | 0.0750 ± 0.0718 | 0.1063 ± 0.0769 | 0.3348 ± 0.1118 | 0.3332 ± 0.1111 | 0.7262 ± 0.4303 | 0.7149 ± 0.4390 | 0.9592 ± 0.0949 | 0.9611 ± 0.0885 | 220 | 68.0 |
| Leiden + scVI (gauss) | 0.0471 ± 0.0679 | 0.0664 ± 0.0709 | 0.0893 ± 0.0814 | 0.3480 ± 0.1061 | 0.3493 ± 0.1016 | 0.8143 ± 0.4836 | 0.9612 ± 0.5025 | 0.9192 ± 0.1636 | 0.9415 ± 0.1314 | 220 | 42.0 |
| Leiden + Harmony | 0.0641 ± 0.1059 | 0.0851 ± 0.1065 | 0.1190 ± 0.1217 | 0.5114 ± 0.0903 | 0.5103 ± 0.0915 | 1.0118 ± 0.4362 | 1.3883 ± 0.4825 | 0.9739 ± 0.0893 | 0.9579 ± 0.0836 | 220 | 32.5 |
| Leiden + Harmony d8 | 0.0727 ± 0.0889 | 0.0950 ± 0.0903 | 0.1269 ± 0.0994 | 0.2528 ± 0.0563 | 0.2539 ± 0.0569 | 1.2932 ± 0.3214 | 1.3467 ± 0.3313 | 0.9234 ± 0.1507 | 0.9026 ± 0.1647 | 220 | 61.0 |
| Leiden + Harmony d20 | 0.0768 ± 0.1023 | 0.1010 ± 0.1031 | 0.1395 ± 0.1151 | 0.3693 ± 0.0861 | 0.3679 ± 0.0898 | 1.2111 ± 0.3597 | 1.3834 ± 0.4205 | 0.9650 ± 0.0962 | 0.9578 ± 0.0940 | 220 | 53.0 |
| Leiden + Harmony d50 | 0.0681 ± 0.1074 | 0.0908 ± 0.1093 | 0.1298 ± 0.1289 | 0.4847 ± 0.1209 | 0.4850 ± 0.1217 | 1.0193 ± 0.4345 | 1.3712 ± 0.4912 | 0.9685 ± 0.1024 | 0.9586 ± 0.0862 | 220 | 38.0 |
| Leiden + ComBat | 0.0617 ± 0.1186 | 0.0846 ± 0.1183 | 0.1230 ± 0.1312 | 0.5268 ± 0.1460 | 0.5258 ± 0.1484 | 0.7224 ± 0.5053 | 1.1004 ± 0.6651 | 0.9693 ± 0.0896 | 0.9680 ± 0.0695 | 220 | 32.0 |
| Leiden + ComBat d8 | 0.0745 ± 0.1039 | 0.0998 ± 0.1046 | 0.1448 ± 0.1120 | 0.3001 ± 0.0734 | 0.2994 ± 0.0763 | 0.9420 ± 0.4591 | 0.9439 ± 0.4802 | 0.9439 ± 0.1233 | 0.9377 ± 0.1186 | 220 | 66.0 |
| Leiden + BBKNN | 0.0502 ± 0.0814 | 0.0710 ± 0.0847 | 0.1046 ± 0.1069 | 0.5268 ± 0.0635 | 0.5268 ± 0.0632 | 0.5955 ± 0.4294 | 1.2526 ± 0.6436 | 0.9774 ± 0.0788 | 0.9580 ± 0.0809 | 220 | 30.0 |
| Leiden + BBKNN d8 | 0.0486 ± 0.0659 | 0.0685 ± 0.0694 | 0.0963 ± 0.0836 | 0.3681 ± 0.0698 | 0.3677 ± 0.0700 | 0.8282 ± 0.4302 | 0.9041 ± 0.4124 | 0.8989 ± 0.1613 | 0.9208 ± 0.1253 | 220 | 43.0 |
| SEACells + scVI | 0.0492 ± 0.0650 | 0.0733 ± 0.0711 | 0.1095 ± 0.0928 | 0.2748 ± 0.0652 | 0.2749 ± 0.0641 | 0.6465 ± 0.4806 | 0.7258 ± 0.4635 | 0.9398 ± 0.1122 | 0.9610 ± 0.0876 | 220 | 53.0 |
| SEACells + Harmony | 0.0944 ± 0.0957 | 0.1231 ± 0.0967 | 0.1753 ± 0.1095 | 0.2342 ± 0.0695 | 0.2335 ± 0.0679 | 1.5004 ± 0.4590 | 1.5972 ± 0.4135 | 0.9414 ± 0.1116 | 0.9642 ± 0.0831 | 220 | 68.5 |
| SEACells + Harmony d8 | 0.0735 ± 0.0879 | 0.0968 ± 0.0905 | 0.1318 ± 0.1004 | 0.1871 ± 0.0435 | 0.1881 ± 0.0443 | 1.3053 ± 0.3410 | 1.3356 ± 0.3194 | 0.8840 ± 0.1633 | 0.9188 ± 0.1386 | 220 | 66.0 |
| SEACells + ComBat | 0.0733 ± 0.1060 | 0.1038 ± 0.1072 | 0.1604 ± 0.1208 | 0.3854 ± 0.0870 | 0.3870 ± 0.0869 | 0.7405 ± 0.5301 | 0.7837 ± 0.5089 | 0.9527 ± 0.0909 | 0.9689 ± 0.0692 | 220 | 64.0 |
| SEACells + BBKNN | 0.0574 ± 0.1254 | 0.0793 ± 0.1259 | 0.1033 ± 0.1248 | 0.4376 ± 0.0630 | 0.4337 ± 0.0650 | 0.9682 ± 0.6323 | 0.9026 ± 0.5282 | 0.9582 ± 0.1089 | 0.9644 ± 0.0812 | 220 | 10.5 |


### Section 2 — rare cell types

std over **batches** throughout (the paper's aggregation). `rare cross-batch homog.` is the only column here that cannot be maximised by giving each batch its own private cluster.

| method | rare coverage | rare homogeneity | rare F1 | rare cross-batch homog. | rare recall | rare precision |
|---|---|---|---|---|---|---|
| SEACells | 0.4271 ± 0.2682 | 0.4206 ± 0.1951 | 0.3706 ± 0.2509 | 0.2218 ± 0.1006 | 0.4074 ± 0.2448 | 0.5409 ± 0.2209 |
| SuperCell | 0.1250 ± 0.1768 | 0.1438 ± 0.0623 | 0.0744 ± 0.1112 | 0.0493 ± 0.0504 | 0.1122 ± 0.1634 | 0.4086 ± 0.1550 |
| MetaCell2 | 0.2604 ± 0.3319 | 0.2019 ± 0.2267 | 0.1970 ± 0.2609 | 0.0903 ± 0.1518 | 0.1726 ± 0.2378 | 0.7533 ± 0.1357 |
| MetaQ | 0.2500 ± 0.3307 | 0.2007 ± 0.1162 | 0.1095 ± 0.1362 | 0.1129 ± 0.0582 | 0.0808 ± 0.0940 | 0.4017 ± 0.2657 |
| Leiden + scPoli-Stage1 | 0.7708 ± 0.1987 | 0.5047 ± 0.1753 | 0.5497 ± 0.1763 | 0.4386 ± 0.1553 | 0.6693 ± 0.1720 | 0.5599 ± 0.1851 |
| Leiden + scVI | 0.1979 ± 0.3279 | 0.2306 ± 0.2102 | 0.1636 ± 0.2812 | 0.1679 ± 0.1277 | 0.1620 ± 0.2652 | 0.2474 ± 0.3127 |
| Leiden + scVI (gauss) | 0.1562 ± 0.3292 | 0.2145 ± 0.2846 | 0.1370 ± 0.2968 | 0.1412 ± 0.2033 | 0.1433 ± 0.3208 | 0.2682 ± 0.2284 |
| Leiden + Harmony | 0.5312 ± 0.2122 | 0.3995 ± 0.2060 | 0.3900 ± 0.2308 | 0.3403 ± 0.1770 | 0.4158 ± 0.2201 | 0.4823 ± 0.2376 |
| Leiden + Harmony d8 | 0.3958 ± 0.2421 | 0.2184 ± 0.0854 | 0.2236 ± 0.1712 | 0.1883 ± 0.0849 | 0.2839 ± 0.1903 | 0.2422 ± 0.1799 |
| Leiden + Harmony d20 | 0.6042 ± 0.1654 | 0.5085 ± 0.2122 | 0.4807 ± 0.2333 | 0.4470 ± 0.2001 | 0.4888 ± 0.2370 | 0.5120 ± 0.2324 |
| Leiden + Harmony d50 | 0.5625 ± 0.1852 | 0.4365 ± 0.1862 | 0.4148 ± 0.2181 | 0.3767 ± 0.1614 | 0.4314 ± 0.2121 | 0.4738 ± 0.2367 |
| Leiden + ComBat | 0.7708 ± 0.1987 | 0.6039 ± 0.1509 | 0.6402 ± 0.1847 | 0.5331 ± 0.1647 | 0.7099 ± 0.1574 | 0.6811 ± 0.1807 |
| Leiden + ComBat d8 | 0.3333 ± 0.2700 | 0.2316 ± 0.1012 | 0.1459 ± 0.0666 | 0.1700 ± 0.0596 | 0.1952 ± 0.1049 | 0.2502 ± 0.2501 |
| Leiden + BBKNN | 0.3333 ± 0.2700 | 0.3665 ± 0.2141 | 0.2878 ± 0.2262 | 0.3346 ± 0.2077 | 0.3136 ± 0.2392 | 0.3337 ± 0.2056 |
| Leiden + BBKNN d8 | 0.1667 ± 0.1318 | 0.1644 ± 0.0597 | 0.1002 ± 0.0839 | 0.1270 ± 0.0616 | 0.1375 ± 0.1241 | 0.2868 ± 0.2187 |
| SEACells + scVI | 0.2604 ± 0.3074 | 0.3319 ± 0.2085 | 0.2116 ± 0.2651 | 0.1854 ± 0.1252 | 0.2280 ± 0.2456 | 0.3526 ± 0.2619 |
| SEACells + Harmony | 0.6979 ± 0.2318 | 0.5983 ± 0.1884 | 0.5813 ± 0.2189 | 0.5099 ± 0.1897 | 0.6432 ± 0.2206 | 0.5945 ± 0.1765 |
| SEACells + Harmony d8 | 0.5625 ± 0.2849 | 0.2854 ± 0.1150 | 0.3165 ± 0.1594 | 0.2278 ± 0.0981 | 0.3472 ± 0.1361 | 0.3642 ± 0.1782 |
| SEACells + ComBat | 0.6354 ± 0.2427 | 0.5501 ± 0.1692 | 0.5468 ± 0.2261 | 0.4624 ± 0.1437 | 0.5847 ± 0.2018 | 0.5505 ± 0.2257 |
| SEACells + BBKNN | 0.3958 ± 0.2421 | 0.4363 ± 0.2022 | 0.3350 ± 0.2476 | 0.3392 ± 0.0959 | 0.3630 ± 0.2279 | 0.3476 ± 0.2754 |


## lung

_not scored yet_

## pbmc-immune

_not scored yet_
