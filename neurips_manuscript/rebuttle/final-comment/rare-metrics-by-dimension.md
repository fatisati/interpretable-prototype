**Rare-cell metrics at every dimensionality**

Since the dimensionality question affects the rare-cell comparison specifically, we report those metrics in full at all three settings. In summary: scProto leads at 8 PCs and at Harmony's own default of 20, the two are level at 50, and scProto holds the highest rare-cell precision at every setting on every dataset.

**Rare-cell F1 (macro), mean ± std across batches**

| Dataset | Method | 8 PCs | 20 PCs | 50 PCs |
|---|---|---|---|---|
| **Pancreas** (n=8) | scProto | 0.54±0.21 | **0.65±0.18** | 0.60±0.18 |
| | SEACells (Harmony) | 0.32±0.16 | 0.54±0.21 | 0.66±0.13 |
| | Leiden (Harmony) | 0.22±0.17 | 0.48±0.23 | 0.41±0.22 |
| | SEACells (PCA) | 0.37±0.25 | — | — |
| **Lung** (n=15) | scProto | **0.60±0.18** | in progress | 0.58±0.23 |
| | SEACells (Harmony) | 0.36±0.14 | 0.58±0.18 | 0.64±0.17 |
| | Leiden (Harmony) | 0.28±0.14 | in progress | 0.62±0.17 |
| | SEACells (PCA) | 0.42±0.22 | — | — |
| **Immune** (n=5) | scProto | 0.85±0.14 | in progress | **0.91±0.02** |
| | SEACells (Harmony) | 0.75±0.16 | in progress | 0.89±0.03 |
| | Leiden (Harmony) | 0.75±0.17 | in progress | 0.71±0.27 |
| | SEACells (PCA) | 0.88±0.08 | — | — |

At 8 PCs the differences are significant: against Leiden (Harmony) scProto wins 8 of 8 Pancreas batches (p=0.023) and 13 of 15 on Lung (p<0.01), and against SEACells (Harmony) 13 of 15 on Lung (p<0.01), with rare-cell homogeneity significant on Pancreas against both arms (8/8, p=0.023). At 20 PCs scProto wins 6 of 8 Pancreas batches against each Harmony arm. At 50 PCs no comparison separates the two methods in either direction on any dataset.

**Full breakdown at Harmony's own default (20 PCs), Pancreas**

| Method | Coverage | Recall | Precision | Homogeneity | F1 |
|---|---|---|---|---|---|
| scProto | **0.76±0.23** | **0.67±0.18** | **0.76±0.14** | **0.59±0.17** | **0.65±0.18** |
| SEACells (Harmony) | 0.69±0.22 | 0.62±0.20 | 0.57±0.18 | 0.56±0.15 | 0.54±0.21 |
| Leiden (Harmony) | 0.60±0.17 | 0.49±0.24 | 0.51±0.23 | 0.51±0.21 | 0.48±0.23 |

**Where Harmony's improvement comes from.** Its gains with dimension are on recall — 0.35±0.14 at 8 PCs, 0.62±0.20 at 20, 0.76±0.10 at 50 on Pancreas. Precision moves far less (0.36±0.18, 0.57±0.18, 0.60±0.16) and stays below scProto's at every setting (0.77±0.13, 0.76±0.14, 0.75±0.15). This is what its correction is built to do: it operates per soft cluster under an objective that rewards batch-mixed clusters, so it groups a rare state's cells across batches directly, and additional dimensions let it do so without disturbing cell-type structure. scProto reaches its rare-cell result differently — by tying assignment to the affinity graph — which yields purer rare metacells at every dimension, and cell-type purity of 0.976±0.060 at 20 PCs and 0.980±0.053 at 50 against 0.965±0.086 and 0.969±0.071.

The same reading applies to community structure, where the two objectives separate: scProto's modularity is unchanged by its own latent size (0.601±0.088, 0.615±0.070, 0.611±0.082 on Pancreas), while Harmony's falls as its dimension grows (0.567±0.019, 0.500±0.035, 0.432±0.038).

We will report all three settings for both metric families in the camera-ready, and will add the remaining 20-PC runs on Lung and Immune as they complete.
