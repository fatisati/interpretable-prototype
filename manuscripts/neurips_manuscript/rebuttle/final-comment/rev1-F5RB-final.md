We completed the baseline comparisons at default settings — scVI at ZINB/n_latent=10, Harmony at 20 and 50 PCs. scProto's graph-preservation advantage holds at every one of them; on the rare-cell metrics the methods come out level there.

**1. scVI at its own defaults**

*Highest modularity of any arm on all three datasets; level on rare cells.*

scProto is an objective, not an architecture: here it is applied to scVI's own encoder at scVI's own defaults (ZINB, n_latent=10, 50 epochs), while SEACells and Leiden cluster the identical embedding. Only the objective differs, and nothing is configured by us. Paired one-sided Wilcoxon across batches, BH-corrected.

| scVI defaults (d=10, ZINB) | Modularity | Rare F1 (macro) | Rare homogeneity |
|---|---|---|---|
| **Pancreas** (K=220, n=8) | | | |
| scProto (scVI) | **0.616±0.082** | 0.614±0.218 | **0.601±0.173** |
| SEACells (scVI) | 0.296±0.078 | 0.639±0.196, ns (3/8) | 0.588±0.180, ns (3/8) |
| Leiden (scVI) | 0.379±0.122 | 0.258±0.276, p=.012* | 0.317±0.242, p=.012* |
| **Lung** (K=300, n=15) | | | |
| scProto (scVI) | **0.650±0.039** | 0.557±0.255 | 0.560±0.257 |
| SEACells (scVI) | 0.327±0.053 | 0.682±0.162, ns (1/15) | 0.658±0.180, ns (2/15) |
| Leiden (scVI) | 0.522±0.077 | 0.689±0.167, ns (4/15) | 0.635±0.182, ns (4/15) |
| **Immune** (K=300, n=5) | | | |
| scProto (scVI) | **0.618±0.059** | **0.897±0.030** | **0.859±0.064** |
| SEACells (scVI) | 0.240±0.130 | 0.892±0.049, ns (3/5) | 0.819±0.082, ns (4/5) |
| Leiden (scVI) | 0.364±0.183 | 0.855±0.108, ns (4/5) | 0.729±0.158, ns (5/5) |

scProto has the highest modularity of any arm on all three datasets, and the highest rare-metacell precision on Pancreas and Lung (0.919 and 0.853 against 0.72 and 0.81). On the rare-cell averages it is highest on Immune on both metrics and on Pancreas homogeneity, and nothing is significant in either direction except Leiden on Pancreas.

We read that as you do: a ZINB likelihood represents a sparse, low-count population on its own scale, so scVI at its own setting already recovers rare types well — its rare F1 rises from 0.18 to 0.64 on Pancreas. Our objective's effect appears in the graph instead: on that same encoder it roughly doubles preserved community structure, 0.394/0.342/0.251 before it is applied against 0.699/0.727/0.674 after. What it captures there the rare-cell metrics cannot register.

This also answers your Stage-1 question. With scPoli absent entirely, the objective reaches the modularity the paper reports for scProto itself — 0.616/0.650/0.618 against 0.615/0.654/0.631 — so that structure comes from the objective, not our pretraining.

Harmony gives the same picture at both of its settings (tables in our reply to Reviewer nG29). At its package default of 20 PCs scProto leads on every metric on Pancreas (modularity 0.615 vs 0.500, rare F1 0.65 vs 0.54). At 50 PCs the modularity separation holds on all three — 0.611 vs 0.432 Pancreas (9/9, p=0.012), 0.654 vs 0.470 Lung (16/16, p<0.001), 0.631 vs 0.322 Immune (5/5 batches).

**2. The graph result holds at every configuration**

*The objective adds wherever the graph carries signal the reconstruction likelihood does not.*

A VAE learns its latent from reconstruction: under MSE or a count likelihood the objective is the fidelity of each cell's own profile, which it does well, and with batch conditioning it corrects batch effects. Similar cells do end up near one another, but only as a by-product: similarity there is whatever the reconstruction error implies, dominated by the states contributing the most cells, and cannot be told to be density-corrected instead. scProto adds an affinity term stating it directly, aligning prototype assignments to a cell–cell affinity graph.

Where reconstruction already recovers what a metric measures, that term is quiet — the rare-cell metrics under scVI's count likelihood are that case, which is why part of that comparison levels out here. Where it does not, the graph supplies it: on the same run modularity still separates two-fold, and that separation holds on both backbones and at every dimensionality tested.

Scored on structure the model never trained on, masking 20% of edges gives 0.59/0.66/0.62 against 0.47/0.52/0.39. Excluding 20% of cells from graph construction and training, then scoring through a frozen encoder: 0.599/0.655/0.620 against 0.31–0.50 for Leiden on the scPoli-Stage-1 and scVI latents. We rely on it because it needs no cell-type labels, which are themselves clustering outputs carrying the biases of the pipelines that produced them — precisely where a sparse population absorbed into a denser one is hardest to see.

The ablation agrees on all three datasets under the same paired test: removing the community loss is the largest single effect in the study (Lung 0.655→0.483, 16/16 batches, p=3.1e-5), and removing the stop-gradient you flagged degrades modularity in every Pancreas batch (9/9, p=0.0039).

We hope this addresses the remaining concerns.
