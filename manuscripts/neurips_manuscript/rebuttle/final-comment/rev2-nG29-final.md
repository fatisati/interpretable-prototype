Thank you for the follow-up and for raising your score. Both points are answered below.

**1. Variance in the ablations**

Every effect we claim is now tested the same way — paired per batch, Wilcoxon, BH-corrected, on every arm and all three datasets — with win-counts and p-values. The ± you were reading is across-batch spread, not the uncertainty of the effect: the same batches appear in the full and ablated runs, so they pair, and a 0.06 drop repeated in every batch is consistent even where the spread is 0.08.

| Arm removed | Metric | Pancreas | Lung | Immune |
|---|---|---|---|---|
| Community loss | Modularity | 0.615→0.449, 8/9, **p=.008** | 0.655→0.483, 16/16, **p=3.1e-5** | 0.623→0.379, 5/5 |
| Usage loss | Active prototypes | 220→63 | 300→77 | 300→48 |
| Reconstruction | Modularity | flat, 5/9, ns | flat, 5/16, ns | flat, 3/5, ns |
| Stop-gradient | Modularity | 0.615→0.555, **9/9, p=.0039** | no drop | no drop |

Three conclusions hold on all three datasets: the community loss is the largest single effect in the study; the usage loss is what prevents prototype collapse; and removing reconstruction leaves modularity unchanged, as the stop-gradient predicts. On Immune, with 5 batches, we report win-counts rather than p-values.

*Lung and Immune with variance:*

| Dataset | Variant | Purity | Modularity |
|---|---|---|---|
| Lung | Full model | 0.837±0.199 | 0.655±0.030 |
| Lung | – community loss | 0.916±0.127 | 0.483±0.049 |
| Lung | – nassoc | 0.804±0.196 | 0.708±0.033 |
| Lung | – usage loss | 0.824±0.212 | 0.698±0.022 |
| Lung | Stop-grad off | 0.855±0.167 | 0.667±0.029 |
| Immune | Full model | 0.870±0.118 | 0.623±0.060 |
| Immune | – community loss | 0.878±0.145 | 0.379±0.091 |
| Immune | – nassoc | 0.852±0.121 | 0.653±0.056 |
| Immune | – usage loss | 0.859±0.138 | 0.647±0.048 |
| Immune | Stop-grad off | 0.869±0.117 | 0.606±0.093 |

*The two cases you raised.*

- **Stop-gradient on modularity (0.615→0.555).** It holds: the drop occurs in 9 of 9 Pancreas batches, p=0.0039. The overlapping deviations come from batch difficulty, identical in both runs. The effect is specific to Pancreas, where reconstruction has most to conflict over, and we scope the claim there.
- **nassoc.** Its effect is consistent on rare-cell homogeneity, in the same direction on all three datasets: 0.63→0.51 (7/8 batches), 0.56→0.48 (11/15), 0.90→0.76 (5/5) — so we anchor the component there, as a direction. On purity (0.972→0.962) you are right that the claim does not hold: purity cannot be paired, its ± being per metacell and two runs producing different metacells.

*Seeds.* Agreed, a separate axis; seed repeats will be in the camera-ready.

**2. Harmony at both dimensionalities**

At Harmony's own package default of 20 PCs, scProto leads on both metric families on Pancreas — community structure and rare cells. At the 50 PCs you named, on all three datasets, the modularity separation is significant and the rare-cell metrics come out level. `harmonypy` performs no PCA and has no native dimensionality — its entry point is `HarmonyMatrix(npcs=20)`, while 50 comes from Scanpy's and Seurat's defaults — so we report both, with scProto trained at each matching latent dimension.

*20 PCs, Pancreas:*

| Method | Modularity | Rare F1 | Rare precision | CT purity |
|---|---|---|---|---|
| scProto | **0.615±0.070** | **0.65±0.18** | **0.76±0.14** | **0.976±0.060** |
| SEACells (Harmony) | 0.500±0.035 | 0.54±0.21 | 0.57±0.18 | 0.965±0.086 |
| Leiden (Harmony) | 0.384±0.093 | 0.48±0.23 | 0.51±0.23 | 0.958±0.094 |

scProto wins 6 of 8 batches on rare F1 against each Harmony arm, with coverage 0.76±0.23 against 0.69±0.22 and homogeneity 0.59±0.17 against 0.56±0.15.

*50 PCs, all three datasets* — modularity per batch, paired one-sided Wilcoxon:

| Dataset | scProto (d=50) | SEACells (Harmony) | Leiden (Harmony) |
|---|---|---|---|
| Pancreas | **0.611±0.082** | 0.432±0.038, 9/9, **p=0.012** | 0.502±0.130, 9/9, **p=0.012** |
| Lung | **0.654±0.027** | 0.470±0.046, 16/16, **p<0.001** | 0.570±0.070, 15/16, **p<0.001** |
| Immune | **0.631±0.054** | 0.322±0.102, 5/5 | 0.461±0.094, 5/5 |

scProto keeps the higher rare-cell precision here (0.75±0.15 against 0.60±0.16 on Pancreas) and significantly higher per-metacell purity on Lung and Immune (p<0.001), with rare-cell F1 level throughout: 0.60±0.18 against 0.66±0.13 (Pancreas), 0.58±0.23 against 0.64±0.17 (Lung), 0.91±0.02 against 0.89±0.03 (Immune).

On Pancreas, scProto's modularity is unchanged by its own latent size (0.601/0.615/0.611) while Harmony's falls as its dimension grows (0.567→0.500→0.432). Harmony's rare-cell gains with dimension are on recall (0.35±0.14 → 0.62±0.20 → 0.76±0.10 on Pancreas), which is what its correction targets: correcting per soft cluster under an objective rewarding batch-mixed clusters groups a rare state across batches directly, and more dimensions let it do that without disturbing cell types.
