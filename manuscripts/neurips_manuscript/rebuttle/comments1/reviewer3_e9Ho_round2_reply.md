We thank the reviewer for the follow-up and for confirming our first response resolved the concerns it addressed. Below we extend each remaining point from one dataset/example to full coverage, and add kBET/iLISI as requested.

**1. Ablation now covers all three datasets**

The same ablation (previously Pancreas-only) now runs on Lung and Immune too; full numbers in our reply to Reviewer nG29. On all three datasets: removing the community loss is the largest modularity drop of any arm (Pancreas 0.615→0.449, Lung 0.655→0.483, Immune 0.623→0.379); removing nassoc or the usage loss both drop rare-cell homogeneity (nassoc: Pancreas 0.63→0.51, Immune 0.90→0.76; usage: Pancreas 0.63→0.40). Removing reconstruction leaves modularity flat on all three (stop-gradient blocks it from the encoder) but still drops rare-cell homogeneity (Immune 0.90→0.87). The stop-gradient's own modularity effect and waypoint-vs-k-means-init are Pancreas-specific, flat on Lung/Immune — Pancreas's 9 sequencing chemistries give reconstruction a non-biological axis to conflict over that the others don't.

**2. Significance tests for Tables 1-2**

Paired one-sided Wilcoxon (modularity, rare F1/homogeneity), Bonferroni-corrected per dataset. The cited example, Immune modularity (scProto 0.62±0.08 vs. SEACells (PCA) 0.55±0.04): p_adj=0.25, ns.

| Dataset | Metric | scProto | vs. SEACells (PCA) |
|---|---|---|---|
| Pancreas | Modularity | 0.621 | 0.658, ns |
| Pancreas | Rare F1 | 0.438 | 0.373, ns |
| Pancreas | Rare Homog. | 0.522 | 0.369, p=.035\* |
| Lung | Modularity | 0.669 | 0.671, ns |
| Lung | Rare F1 | 0.611 | 0.498, p<.05\* |
| Lung | Rare Homog. | 0.586 | 0.495, p<.05\* |
| Immune | Modularity | 0.620 | 0.569, ns |
| Immune | Rare F1/Homog. | 0.897/0.856 | ns |

Modularity ties SEACells (PCA) on all three — the trade-off already central to the abstract — while scProto significantly wins rare-cell recovery on Lung, numerically ahead elsewhere: SEACells builds metacells within a single batch, so a rare state without enough cells in any one batch gets absorbed into a denser neighbor; scProto's batch correction pools that state's cells across batches instead. Against the batch-correction-only baselines (Parametric UMAP, MetaQ, scVI, Harmony), scProto's modularity wins are significant across the board on Lung, and on Pancreas against Parametric UMAP/MetaQ specifically.

Purity (near-ceiling: 0.98/0.86/0.87): significantly ahead of most baselines on Lung/Immune, leads SEACells (Harmony) on Pancreas (p=0.0003\*\*\*), ties the rest there; trails SEACells/MetaQ on Lung/Immune — same mixing trade-off. Immune's small size (5 batches) keeps most individual comparisons at ns after Bonferroni correction.

**3. Baselines**

scVI, dimension- and loss-matched (n_latent=8, Gaussian likelihood), SEACells/Leiden on top:

| Dataset | Metric | scProto | SEACells (scVI) | Leiden (scVI) | SuperCell | Metacell-2† |
|---|---|---|---|---|---|---|
| Pancreas | F1 | 0.54±.21 | 0.18±.31, p=.039\* | 0.14±.30, p=.020\* | 0.07±.11, p=.012\* | 0.20±.26 |
| Pancreas | Homog. | 0.56±.16 | 0.19±.30, p=.039\* | 0.21±.28, p=.039\* | 0.14±.06, p=.012\* | 0.20±.23 |
| Lung | F1 | 0.60±.18 | 0.35±.20, p<.001\*\*\* | 0.34±.16, p<.001\*\*\* | 0.47±.16, p=.045\* | 0.64±.16 |
| Lung | Homog. | 0.58±.22 | 0.39±.20, p<.001\*\*\* | 0.38±.20, p<.001\*\*\* | 0.45±.16, p=.032\* | 0.56±.20 |
| Immune | F1/Homog. | 0.85±.14/0.83±.10 | ns | ns | 0.88±.05/0.84±.04, ns | 0.52±.24/0.44±.28 |

† Metacell-2's K (own outlier policy) is too far from scProto's for significance testing; shown for reference.

SEACells (PCA) — the paper's own baseline — follows the same direction (significant on Lung, ns on Pancreas F1, mixed on Immune; full numbers in §2). Dimension-matched Harmony and ComBat show the identical pattern (nG29 reply, §3–§4) — scVI, Harmony, and ComBat all lose the rare-cell test at matched dimension. SuperCell matches that on Pancreas/Lung (significant); Immune is a tie, ns. Metacell-2 (K mismatch, footnote) trails scProto on 5/6 cells, ahead only on Lung F1. Second spatial dataset in progress; otherwise we'll narrow the claim to the single NSCLC slide.

**4. kBET / iLISI**

Standard `scib-metrics` Benchmarker on scProto's own latent embedding, vs. uncorrected PCA and the scPoli Stage-1 embedding scProto is initialized from, all three datasets. iLISI: 0.000–0.002 (PCA), 0.088–0.174 (scPoli), 0.046–0.109 (scProto). KBET: 0.160–0.217 (PCA), 0.233–0.318 (scPoli), 0.229–0.286 (scProto). scProto is above uncorrected PCA on both metrics everywhere — Stage 2 doesn't undo Stage 1's alignment. Against scPoli's own embedding, scProto is close on KBET (largest gap 0.03, on Lung; ahead on Pancreas) and behind on iLISI — consistent with the modularity/purity-vs-mixing trade-off already central to this response.
