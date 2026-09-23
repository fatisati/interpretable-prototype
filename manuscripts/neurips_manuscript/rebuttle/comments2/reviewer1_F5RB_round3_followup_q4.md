**Update: spatial niche recovery results (Question 4)**

We tested whether scProto's metacells recover niche-correlated transcriptional
programs, as requested. For each (cell type, niche) pair with enough real
single cells to define ground truth, we computed niche-vs-rest differential
expression at the single-cell level, then tested whether scProto's metacells
recover the same genes -- Pearson correlation and Kendall's tau between
metacell-level and single-cell-level log fold-change (MetaQ's recovery
protocol), against SEACells trained on the identical spatial graph and on a
PCA-only (non-spatial) graph.

We find scProto recovers real niche-specific transcriptional programs in cell
type / niche combinations neither SEACells baseline can produce a testable
result for at all -- 6 such programs, 4 of them strongly: Vascular endothelium
in "T cell aggregates" (r=0.94), Vascular endothelium in "vascular stroma"
(r=0.92), Macrophages in "alveolar spaces" (r=0.92), Macrophages in
"desmoplastic stroma" (r=0.80). On two further programs where all methods can
be compared directly, scProto also outperforms both SEACells variants:
Macrophages in "macrophage islands" (r=0.76 vs. 0.53/0.65) and Fibroblasts in
"desmoplastic stroma" (r=0.88 vs. 0.80/0.81).

**Biological validation.** The genes behind these two head-to-head recoveries
describe coherent, independently-recognized programs, not isolated markers we
picked out after the fact:

In "macrophage islands," scProto's top recovered genes are *SPP1* (osteopontin),
*GPNMB*, *MARCO*, and *CD68* -- together describing a lipid-laden,
matrix-remodeling, tissue-repair macrophage phenotype. *SPP1*+ macrophages are
specifically linked to hypoxic tumor niches and poor prognosis in lung
adenocarcinoma [2]; *GPNMB* independently marks the same lipid-associated state
in lung macrophage injury responses [3]. The source dataset's own published
analysis [1] identifies these same two genes as marking pro-tumoral,
tissue-repair macrophages in this tumor -- and the pairing recurs in lung
biology unrelated to this dataset.

In "desmoplastic stroma," scProto's top recovered genes are *COL14A1*, *IGF1*,
and *CXCL12* -- a cancer-associated fibroblast (CAF) signature specific to the
inflammatory/matrix-remodeling CAF subtype (distinct from the myofibroblastic
*COL11A1*/*COL8A1* program reported elsewhere in the CAF literature) [4].
*CXCL12* is a chemokine axis through which CAFs recruit and reprogram
surrounding stroma and immune cells across multiple cancer types [5]. [1]
independently reports the same three genes as enriched fibroblast/ECM markers
in this exact compartment.

**Why we think this happens.** scProto's encoder is pretrained with a real
reconstruction objective before any spatial-graph signal is introduced, so its
latent space starts grounded in real gene expression rather than in graph
structure alone; the spatial community objective is then layered on top, with
reconstruction continuing to anchor the encoder throughout (it is not frozen
during this stage). SEACells' archetypal method has no analogous grounding at
any stage -- it fits directly on the graph, with no way to distinguish a
genuine niche edge from a same-cell-type edge that happens to be spatially
adjacent, a known source of noise in BANKSY-derived graphs. We do not think
this makes scProto broadly better at spatial recovery: most (cell type, niche)
pairs are a wash between the methods, and how strongly this mechanism
manifests in practice is itself an open question, since the reconstruction
term is weighted well below the community/association terms in our current
configuration.

[1] Pentimalli, T.M., Schallenberg, S., León-Periñán, D., Legnini, I.,
Theurillat, I., Thomas, G., Boltengagen, A., Fritzsche, S., Nimo, J., Ruff, L.,
et al. "Combining spatial transcriptomics and ECM imaging in 3D for mapping
cellular interactions in the tumor microenvironment." Cell Systems 16(5) (2025).

[2] Matsubara, E., Yano, H., Pan, C., Komohara, Y., Fujiwara, Y., Zhao, S.,
Shinchi, Y., Kurotaki, D., Suzuki, M. "The Significance of SPP1 in Lung Cancers
and Its Impact as a Marker for Protumor Tumor-Associated Macrophages." Cancers
15 (2023): 2250.

[3] King, E.M., Zhao, Y., Moore, C.M., Steinhart, B., Anderson, K.C., Vestal,
B., Moore, P.K., McManus, S.A., Evans, C.M., Mould, K.J., Redente, E.F.,
McCubbrey, A.L., Janssen, W.J. "Gpnmb and Spp1 mark a conserved macrophage
injury response masking fibrosis-specific programming in the lung." JCI
Insight (2024).

[4] Thorlacius-Ussing, J., Jensen, C., Nissen, N.I., Cox, T.R., Kalluri, R.,
Karsdal, M., Willumsen, N. "The collagen landscape in cancer: profiling
collagens in tumors and in circulation reveals novel markers of
cancer-associated fibroblast subtypes." Journal of Pathology 262 (2024): 22–36.

[5] Ma, Z., Yu, D., Tan, S., Li, H., Zhou, F., Qiu, L., Xie, X., Wu, X.
"CXCL12 alone is enough to Reprogram Normal Fibroblasts into Cancer-Associated
Fibroblasts." Cell Death Discovery (2025).
