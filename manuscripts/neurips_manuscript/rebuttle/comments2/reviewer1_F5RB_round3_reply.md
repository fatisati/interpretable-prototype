We thank the reviewer for the constructive feedback.

To address the independence gap raised by the reviewer for the edge held out experiment results, in addition to edge level cross validation (edge heldout) we also performed a cell level cross validation by traininhg the model on 80% of cells and testing on the remaining 20%. 
We observed that consistent to the edge level cross validation results, the differnce between cross validation and full-data headline modularity were minimal (0.617 / 0.672 /
0.630 vs. 0.621 / 0.669 / 0.620 for Pancreas / Lung / Immune) suggesting that these metrics reflect honest results and are not skewed by leakage or overfitting. This analysis also showed that scProto outperforms Leiden on scPoli-Stage1 / scVI latents scored the same way. 
It is of note that this test was applied on the scProto method as it is not applicable to non-encoder based methods such as SEACells.
Full per-dataset breakdown is given below.

**Node-held-out modularity** (per-batch mean±std, modularity on edges touching only
held-out cells)

| Method | Pancreas | Lung | Immune |
|---|---|---|---|
| scProto (node-holdout) | **0.599±.072** | **0.655±.034** | **0.620±.051** |
| Leiden (scPoli-Stage1) | 0.358±.053 | 0.495±.083 | 0.314±.215 |
| Leiden (scVI) | 0.341±.116 | 0.501±.083 | 0.326±.209 |




** we want the respinse to be clear and objective. explain necessary details in suscinct and objective way without being too verbose . we want it to be on positive note and be honest and professional and accurate but remove any unnecessary details that does not help with the overall response flow. 

** regarding the niche-correlated transcriptional programs, we want to appreciate the reviewer's concern and provide partial results of the analysis we have done and show few annectodes and acknowledge that a more comrehensive analysis expanding to additional datasets is needed and that will be included in the camera ready version of the manuscript.  (give a brief description of the analysis that was done and provide some results (such as  Vascular endothelium x T cell;  and  Macrophages x Alveolar spaces) and the corresponding marker genes and biological relevance and reference papers) keep this section shorter as it is a work in progress and we are only showing viniettes . keep it sussinct, professional, honest and clear and  accurate  but remove any unnecessary details that does not help with the overall response flow so keep it on positive.  


critically review the entire response and polish/ rewrite as needed. we want to keep it sussinct, professional, honest and clear and accurate . we want to make it as simple as possible for the reviewrs to read and follow and get their answers. we want it to be of positive note while honest and technichally accurate and sound. remove any unnecessary details or talkitive sentences that does not help with the overall flow to make it more polished and professional and easy to read and follow.  

**4. Spatial niche recovery**

We tested whether scProto's metacells recover niche-correlated
transcriptional programs: per (cell type, niche) pair, we computed
ground-truth DE from real single cells, then checked whether scProto's
metacells recover the same genes (Pearson r / Kendall's tau on log
fold-change), against SEACells on the identical spatial graph and on a
PCA-only graph. On the circularity concern specifically: niche labels are
defined only by local *cell-type composition*, never by gene expression --
the genes we recover were never part of that definition, so their recovery
is independent evidence, not a restatement of the label.

scProto recovers programs in 6 pairs neither SEACells baseline can test at
all (4 strongly, r=0.78-0.94), and beats both baselines head-to-head on 2
more: Macrophages/"macrophage islands" (r=0.76 vs. 0.53/0.65) and
Fibroblasts/"desmoplastic stroma" (r=0.88 vs. 0.80/0.81).

The genes behind these two are coherent, independently-recognized programs,
not markers selected after the fact. "Macrophage islands": *SPP1*, *GPNMB*,
*MARCO*, *CD68* -- a tissue-repair macrophage phenotype; [1] identifies the
same two genes in this tumor, and both recur together independently in [2,3].
"Desmoplastic stroma": *COL14A1*, *IGF1*, *CXCL12* -- an
inflammatory/matrix-remodeling CAF signature (distinct from the
myofibroblastic COL11A1/COL8A1 program); [1] independently reports the same
three genes here, each separately established as a CAF marker elsewhere
[4,5].

We think this reflects scProto's design: the encoder is pretrained on real
reconstruction before any spatial-graph signal is introduced, so its latent
space starts grounded in expression, not graph structure, and reconstruction
keeps anchoring it through Stage 2. SEACells' archetypal method has no such
grounding -- it fits directly on the graph, with no way to separate a real
niche edge from a same-cell-type edge that's merely spatially adjacent
(known noise in BANKSY graphs). We don't read this as scProto being broadly
better: of the 9 pairs where a direct comparison against both SEACells
variants is possible, scProto wins clearly on 2, is mixed on 1, and trails
on the remaining 6. How much the mechanism matters in practice is itself
open, since the reconstruction term is weighted well below the
community/association terms in our current configuration.

[1] Pentimalli et al., "Combining spatial transcriptomics and ECM imaging in
3D for mapping cellular interactions in the tumor microenvironment," Cell
Systems 16(5), 2025.

[2] King et al., "Gpnmb and Spp1 mark a conserved macrophage injury response
masking fibrosis-specific programming in the lung," JCI Insight, 2024.

[3] Matsubara et al., "Significance of SPP1 in Lung Cancers...," Cancers 15,
2023.

[4] Thorlacius-Ussing et al., "Collagen landscape in cancer...," J. Pathology
262, 2024.

[5] Ma et al., "CXCL12 alone is enough to Reprogram Normal Fibroblasts into
Cancer-Associated Fibroblasts," Cell Death Discovery, 2025.
