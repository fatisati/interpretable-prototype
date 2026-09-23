We thank the reviewer for the feedback.

**1. Held-out-edge independence**

To strengthen the edge-level cross-validation already presented, we also ran
a cell-level cross-validation: train on 80% of cells, test on the
remaining 20%. The gap between cross-validation and full-data modularity
was minimal (0.617/0.672/0.630 vs. 0.621/0.669/0.620 for Pancreas/Lung/
Immune), so the reported metrics are not inflated by leakage or
overfitting. scProto also outperforms Leiden on scPoli-Stage1/scVI latents
scored the same way. This test applies to scProto specifically, since it
requires an encoder -- SEACells has none.

**Node-held-out modularity** (per-batch mean±std, modularity on edges
touching only held-out cells)

| Method | Pancreas | Lung | Immune |
|---|---|---|---|
| scProto (node-holdout) | **0.599±.072** | **0.655±.034** | **0.620±.051** |
| Leiden (scPoli-Stage1) | 0.358±.053 | 0.495±.083 | 0.314±.215 |
| Leiden (scVI) | 0.341±.116 | 0.501±.083 | 0.326±.209 |

**2. Baseline configuration (d=8, Gaussian likelihood)**

Reasonable point. We set every baseline to d=8 with a Gaussian likelihood so
all methods share one configuration. Other configurations are testable:
we can adapt scProto to each baseline's native setting -- higher
latent dimension, NB reconstruction loss -- and compare at baselines'
standard configuration (d=50, ZINB). We will include this in the
camera-ready version.

**3. Wording corrections**

Both noted. "No global step": Stage 1 (scPoli pretraining) *is* global; the
claim is that Stage 2 -- scProto's rare-cell mechanism -- adds no
*additional* global correction, unlike Harmony/ComBat/scVI. "Never reverse":
ComBat's mean F1 (0.86) is marginally, non-significantly higher than
scProto's (0.85) on Immune, though scProto wins more batches (3/5).
Win-counts favor scProto everywhere except this one non-significant case.

**4. Spatial niche recovery**

To test whether scProto's metacells discover niche-correlated
transcriptional programs rather than just preserve predefined composition,
we ran MetaQ's recovery protocol [1]: for each (cell type, niche) pair
with enough real single cells, we derived ground-truth differential
expression, then checked whether scProto's metacells recover the same
genes (log fold-change correlation).
scProto trains on a spatial affinity graph built with adaptive-RBF on
BANKSY [2] embeddings (spatial + transcriptional information); we compare
against SEACells on this same graph and on SEACells' original
adaptive-RBF-on-PCA graph (no spatial information). Niche labels come only
from local cell-type composition, never gene expression, so recovering the
correct genes is independent evidence.

**Why neither baseline can test these.** Testing a pair requires isolating
cells from that niche into their own metacells, so there is an expression
profile to compare against ground truth. For 6 pairs, scProto does this
but neither baseline can: the PCA baseline has no spatial information, so
it cannot separate niches within a cell type when the difference is
subtle -- scProto's graph carries that information. The BANKSY-graph
baseline trains on the identical graph as scProto but is purely
archetypal, unable to tell a real niche edge from an adjacent
same-cell-type edge; scProto's encoder is pretrained on reconstruction
before the graph signal is added and keeps training throughout, so it
starts from real expression. This is a head start, not a guarantee -- how
strong it is in practice is still open.

**Biological validation.** Genes behind two of the six describe coherent,
independently-recognized programs, not isolated markers picked out after
the fact:

- **Vascular endothelium x "T cell aggregates"** (r=0.94): *ACKR1* and MHC
  class II genes (*HLA-DRA*, *CD74*, *HLA-DPA1*) rise while *KDR*/*FLT1*
  fall -- consistent with high endothelial venules specialized for
  lymphocyte recruitment, described in tertiary lymphoid structures in
  tumors [3].
- **Macrophages x "alveolar spaces"** (r=0.92): *CCL18*, a well-established
  marker of tissue-resident alveolar macrophages [4], recovered as
  upregulated -- consistent with normal resident-macrophage identity in
  this non-tumor lung compartment.

Outside these six, neither method consistently outperforms the other --
not a claim scProto is broadly better at spatial recovery. We plan to
extend this to more datasets for camera-ready.

[1] Li, Y., Li, H., Lin, Y., et al., "MetaQ: fast, scalable and accurate
metacell inference via single-cell quantization," Nature Communications,
2025.

[2] Singhal, V., Chou, N., Lee, J., et al., "BANKSY unifies cell typing and
tissue domain segmentation for scalable spatial omics data analysis," Nature
Genetics 56, 2024: 431-441.

[3] Vella, G., Guelfi, S., Bergers, G., "High Endothelial Venules: A
Vascular Perspective on Tertiary Lymphoid Structures in Cancer," Frontiers
in Immunology, 2021.

[4] Chenivesse, C., Tsicopoulos, A., "CCL18 -- beyond chemotaxis," Cytokine
109, 2018: 52-56.
