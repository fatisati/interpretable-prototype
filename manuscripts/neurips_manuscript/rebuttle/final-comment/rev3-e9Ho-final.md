We thank the reviewer for the close reading throughout. One update first: the second spatial benchmark is complete, on a different tissue and platform, and the Fig. 3 result reproduces on it (§3).

**1. Modularity against SEACells**

scProto matches SEACells' community structure while mixing batches three to six times more, and that joint result is the paper's claim. SEACells is the one baseline that removes no batch effect at all: it builds metacells within a single batch, on the same PCA affinity structure the metric is computed on, and is never asked to place cells from different batches together — batch entropy 0.16–0.41 against scProto's 0.94–1.39. Against every method that does mix batches, scProto's modularity is significantly higher — across the board on Lung, against Parametric UMAP and MetaQ on Pancreas, and against Harmony at its own 50 PCs (9 of 9 batches, p=0.012). On the full graph the two are level, and the paper will not claim otherwise.

This holds only where both methods are scored on the graph they were fitted to. On structure neither was fitted to, scProto leads in all three settings we tested:

- *Held-out edges.* 20% masked before training, scProto and SEACells fit on the identical visible 80%, scored only on the hidden 20%: 0.59/0.66/0.62 against 0.47/0.52/0.39.
- *Held-out nodes.* 20% of cells excluded entirely from graph construction and training, then scored through a frozen encoder: 0.599/0.655/0.620, against 0.31–0.50 for Leiden on the scPoli-Stage-1 and scVI latents. SEACells cannot enter this test: its archetypal fit is defined only over the cells present when it is built.
- *A graph scProto never trains on.* On the second spatial dataset (§3), scored on the adaptive-RBF-on-PCA graph the transcriptomic baseline is itself fitted to, scProto reaches 0.510 against 0.363 and 0.103.

**2. Matched dimension**

Every comparison in Tables 1–2 and all significance testing is already dimension-matched, so we also ran the baselines at their own settings; full tables in our reply to Reviewer nG29. The community-structure separation does not move with dimension. At Harmony's package default of 20 PCs scProto leads on Pancreas on every metric (modularity 0.615±0.070 vs 0.500±0.035, rare F1 0.65±0.18 vs 0.54±0.21). At 50 PCs, with scProto also trained at d=50: 0.611±0.082 vs 0.432±0.038 on Pancreas (9/9, p=0.012), 0.654±0.027 vs 0.470±0.046 on Lung (16/16, p<0.001), 0.631±0.054 vs 0.322±0.102 on Immune (5/5), with significantly higher per-metacell purity on Lung and Immune (p<0.001). With scVI at its own defaults (ZINB, n_latent=10) as the backbone: 0.603±0.088 vs 0.296±0.078 and 0.379±0.122. The baselines' rare-cell metrics improve at native settings and come out level there, which follows from what Harmony optimizes — its per-soft-cluster objective targets cross-batch grouping directly.

**3. Spatial: a second tissue and a second platform**

The claim no longer rests on one slide. The second benchmark is a colorectal-cancer Xenium cohort — different tissue and platform from the NSCLC CosMx slide. Three whole sections from three patients are kept intact rather than subsampled: 50,130 cells, 13 cell types, 5 niches. Niche labels are the source study's own NicheCompass annotations, used only for evaluation; no niche label enters training, and the graph is adaptive-RBF on BANKSY embeddings (λ=0.8, BANKSY's recommended value). All methods run at the same number of metacells. Computed exactly as Fig. 3 (per-metacell majority labels, unweighted, one-sided Mann-Whitney U vs scProto):

| | niche purity | cell-type purity |
|---|---|---|
| **scProto** | **0.915** | 0.781 |
| SEACells (transcriptomic) | 0.885 | 0.788 |
| SEACells (spatially-aware) | 0.930 | 0.604 |

The trade-off holds in the same direction as on NSCLC: scProto is above the transcriptomics-only baseline on niche purity while matching it on cell-type purity (0.781 vs 0.788), and 0.18 above the spatial-affinity baseline on cell-type purity, which again reaches niche purity by giving up cell-type identity.

The per-cell-type tests carry this rather than the averages. scProto's niche purity is significantly higher than SEACells (transcriptomic) on 6 of the 9 cell types testable for all three — cancer cells, endothelium, macrophages, neutrophils, smooth muscle, T cells — and its cell-type purity is significantly higher than SEACells (spatially-aware) on 7 of 9. Both halves hold simultaneously and significantly on five of these. The §1 modularity figures are this dataset's per-section means (0.441±0.074, 0.352±0.010, 0.084±0.030), reported descriptively — three sections is too few for the paired test used elsewhere.

We will state the spatial claim at that level — niche-correlated structure captured while cell-type identity is preserved, on two tissues and two platforms — and keep the per-pair recovery result at pair level.
