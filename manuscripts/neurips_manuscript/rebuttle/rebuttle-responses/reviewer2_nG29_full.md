# Reviewer 2 (nG29) — Full Official Review

**Submission:** 15985 — scProto
**Reviewer:** nG29
**Posted:** 24 Jun 2026, 14:57 (modified 23 Jul 2026, 18:27)

**Rating:** 2 — *Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility and incompletely addressed ethical considerations.*
**Confidence:** 4 — *You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.*

| Quality | Clarity | Significance | Originality |
|---|---|---|---|
| 2: not good | 2: not good | 3: good | 2: not good |

---

## Summary

This manuscript introduces scProto, a deep clustering method to construct metacells from single-cell transcriptomics data that is robust to batch effects. The method starts by constructing a batch-specific affinity graph, with affinities defined either by gene expression alone, or by a combination of gene expression and spatial information. The method then learns a set of prototypes that represent the metacells, and soft-assigns cells to these prototypes in latent space. Cell representations are augmented by concatenated batch embeddings, which are also learnable vectors. A decoder (also exposed to batch information) is then used to reconstruct the original gene expression for each prototype, and individual cells are then reconstructed as a weighted combination of the prototypes they originally mapped to. The provided loss mainly includes a community term, which encourages cells that are connected in the initial affinity graph to be assigned to the same prototype, and a reconstruction term, which encourages the prototypes to be representative of the cells assigned to them. Other auxiliary loss terms are also included, such as a prototype usage penalization that discourages prototype collapse. scProto is evaluated on several single-cell transcriptomics datasets, including a spatial one, and is benchmarked against a plethora of baselines across a relevant set of metrics, including modularity, cell-type purity of the prototypes, and batch entropy.

**Contribution Type:** General: Most submissions will fall into this type.

---

## Strengths and Weaknesses

### Strengths

1. The problem this manuscript attempts to address is significant, well-framed, and well-motivated. Batch effects are one of the largest sources of technical variation in single-cell experiments, and the proposed method aims to distill real biological variation in the presence of batched contexts (which most datasets have).
2. The inclusion of a spatial dataset is a nice addition, and opens the door for expanding the definition of affinity beyond gene expression.
3. The presented results are competitive when it comes to modularity and batch entropy, and the trade-off between these two metrics is well-discussed.
4. The presented results on coverage and homogeneity for rare cell types are strong.

### Weaknesses

1. **The evaluation is incomplete — no ablation studies for the loss function terms.** The authors include four different terms (community, reconstruction, batch-aware normalization, and prototype usage penalization, which they indeed properly discuss), and neither ablations nor hyperparameter sweeps are provided to show how each term affects performance. The lack of ablation studies makes it difficult to assess the contribution of each component of the method, and limits the reproducibility of the work.

2. **No tuning or sensitivity analysis for K.** The number of prototypes is described in the discussion as a key hyperparameter that can affect performance, but there is no presented tuning or sensitivity analysis to justify this claim. In all these cases a set of fixed hyperparameters are reported in appendix D, with the only justification based on prior literature.

3. **The text mildly overstates performance, especially on Pancreas (line 229).** Here scProto is second best in terms of modularity, and second-to-last in batch entropy, which contradicts the claim that it "matches or exceeds SEACells on modularity across all three datasets while substantially outperforming it on batch entropy." I would suggest to tone down these claims, and provide a more balanced discussion of the results.

4. **Originality is incremental; missing comparison to related batch-aware latent methods.** The method is a combination of existing ideas, and the novelty seems incremental. The main original contribution is how they deal with batch effects, but performance in this regard is still debatable (as mentioned for pancreas), and there exist other methods which also incorporate batch information in the latent space ([Nature Communications 2025](https://www.nature.com/articles/s41467-025-63915-z), [Nature Methods 2018](https://www.nature.com/articles/s41592-018-0229-2)). A comparison to these methods (or a justification on why the authors may think they are not relevant) would be highly appreciated.

### Presentation Issues (minor, reviewer's own list — "none critical and all hopefully fixable")

- Most figure labels and legends are too small and difficult to read.
- The PBMC dataset is often referred to as "Immune" (this is a minor thing, but easily addressable).
- Spatial niches are not defined, and the term might be confusing to the non-expert reader.
- In line 194, the pancreas dataset is described as having 9 "technologies." Are these equivalent to batches? If so, replace the term; if not, please explain.
- The cell-cell similarity matrix is referred to as a "similarity graph" even before thresholding, which is confusing. While this is not technically incorrect (as we could think of it as a fully connected graph), it would be clearer to refer to it as a similarity matrix, and only call it a graph after thresholding. This is just a suggestion.
- Equations are not numbered.

---

## Questions

1. **How much would the performance of scProto change if the loss function was ablated to remove one or more of the terms?** For example, how much would performance change if the prototype usage penalization term was removed? Ablation studies on the loss function would shed light on the contribution of each term to overall performance. My score would increase if these ablations were included.

2. **How optimal is the choice of K (the number of prototypes)?** How sensitive is performance to this hyperparameter? A sensitivity analysis on K would shed light on the robustness of the method. My score would increase if this analysis was included. NOTE: K should of course be tuned on all relevant baselines too, not just scProto, to ensure a fair comparison.

3. **How would batch-correction preprocessing affect the comparison to other baselines?** It would be interesting to see how the performance of scProto would compare to other baselines if batch effects were removed a priori (for example with Harmony or ComBat). My score would increase if this analysis was included.

---

## Limitations

Yes. Limitations are discussed in the discussion section, and in my opinion the main issues with the presented method are addressed. The discussion on potential negative societal impact is missing, but I think this is a relatively low-risk area.

---

## Other Fields

- **Ethical Concerns:** NO or VERY MINOR ethics concerns only
- **Paper Formatting Concerns:** none
- **Code Of Conduct Acknowledgement:** Yes
- **Responsible Reviewing Acknowledgement:** Yes

---

## Our Response

We thank the reviewer for recognizing the problem is significant and well-motivated,
and address each point below.

### 1. Ablation of loss terms (Weakness 1; Question 1)

Pancreas only so far, extending to Lung/Immune:

| Removed / changed | Intended role | Ablation result |
|---|---|---|
| $\mathcal{L}_{\text{community}}$ | Pulls affinity-graph neighbors toward shared assignments | modularity 0.615±.08→**0.449±.04** (largest drop); rare coverage 0.69±.22→0.50±.23; rare homogeneity 0.63±.16→0.41±.23 |
| $\mathcal{L}_{\text{nassoc}}$ | Keeps prototypes tight and non-redundant | purity 0.972±.09→**0.962±.09** (largest purity drop, though modest); rare homogeneity 0.63±.16→0.51±.19 |
| $\mathcal{L}_{\text{usage}}$ | Anti-collapse penalty, preserves capacity for rare states | rare coverage 0.69±.22→**0.43±.27**; rare homogeneity 0.63±.16→0.40±.21 |
| $\mathcal{L}_{\text{rec}}$ (removed) | Grounds prototype content in real, denoised expression | Pancreas: purity/modularity flat (0.972→0.970, 0.615→0.614) — expected, see below; decoded-prototype fidelity drops instead (gene-expression-space scGraph [1] variant, 0.91→0.85). Spatial (NSCLC): purity 0.50→0.49 and niche purity 0.91→0.87 both drop |
| Stop-gradient removed | Prevents recon's dense-favoring bias from fighting community/nassoc over the shared encoder | modularity 0.615±.08→0.555±.07, rare homogeneity 0.63±.16→0.51±.16 both drop; purity flat |
| Waypoint init → random | Seeds prototypes to cover sparse/rare regions from the start | rare coverage 0.69±.22→0.56±.28; rare homogeneity 0.63±.16→0.48±.22; modularity 0.615±.08→0.594±.07 |
| Fixed $\tau$ (no calibration) | Calibrates assignment sharpness | rare homogeneity 0.63±.16→**0.56±.16** drops; purity, modularity, rare coverage flat or improve |

Every term earns its place: community/nassoc/usage each protect a different metric;
the stop-gradient ablation shows the training-signal separation from reconstruction
is load-bearing, not incidental. $\mathcal{L}_{\text{rec}}$'s null effect on
purity/modularity is expected (the affinity graph is itself expression-derived, so
community/nassoc stay anchored without it) — its real effect shows up in the
gene-expression-space scGraph [1] variant instead, and on the spatial dataset, where
that anchor disappears.

### 2. K sensitivity (Weakness 2; Question 2)

We swept K over {0.25×, 0.5×, 1×, 2×, 4×} the paper's own K (Pancreas:
55/110/220/440/880), tuned identically across scProto and the three baselines the
reviewer names, not just scProto — SEACells on raw PCA (our own Table 1 baseline),
SEACells on the scPoli Stage-1 embedding, and Leiden on that same embedding.
Cells are Modularity / Batch Entropy, mean ± std across batches (Tables 1–2's
convention) — the two metrics this reviewer's own Strength 3 singles out:

| Method | K=55 | K=110 | K=220 (paper's K) | K=440 | K=880 |
|---|---|---|---|---|---|
| scProto | .663±.06 / 1.224±.61 | .618±.07 / 0.880±.61 | .617±.09 / 1.249±.64 | .620±.09 / 1.237±.61 | .644±.08 / 1.269±.50 |
| SEACells (PCA) | .704±.06 / 0.580±.52 | .725±.05 / 0.319±.41 | .673±.05 / 0.202±.31 | .593±.05 / 0.144±.26 | .487±.04 / 0.111±.23 |
| SEACells (Stage-1) | .547±.06 / 1.525±.39 | .461±.07 / 1.356±.40 | .368±.07 / 1.249±.42 | .285±.07 / 1.129±.43 | .216±.06 / 1.023±.46 |
| Leiden (Stage-1) | .592±.08 / 1.482±.40 | .515±.08 / 1.400±.41 | .445±.07 / 1.298±.43 | .392±.08 / 1.223±.52 | .382±.08 / 1.192±.60 |

Both metrics are flat for scProto across a 16-fold range of K; every baseline
degrades, SEACells (PCA) most sharply on batch entropy (0.58→0.11). Cell-type
purity converges across every method and K (0.86-0.98, all four ≥0.94 by
K≥110), so not tabled separately. Rare-cell coverage/homogeneity/F1 improve
with K for every method, but rank inconsistently, so we draw no K-robustness
conclusion from it. Pancreas only so far; extending to Lung/Immune.

### 3. Overstated claim on Pancreas (Weakness 3)

Fair — on Pancreas SEACells leads modularity (0.67 vs. 0.61), so "matches or exceeds
SEACells on modularity across all three datasets" is wrong as written; we'll fix the
wording for camera-ready. But the claim behind it — that scProto gets community structure without sacrificing
batch mixing — still holds: every existing method picks one side of that trade-off.
SEACells preserves community structure but barely mixes batches; scPoli-cVAE and
UMAP mix batches but flatten community structure. scProto's modularity is near
SEACells (Pancreas 0.61 vs. 0.67, Immune 0.62 vs. 0.55, Lung 0.67 vs. 0.67) and
2-2.6x higher than scPoli-cVAE/UMAP everywhere (Pancreas 0.61 vs. 0.23/0.26); its
batch mixing is far higher than SEACells (Pancreas 1.04 vs. 0.19) and comparable to
scPoli-cVAE/UMAP everywhere (Pancreas 1.04 vs. 1.38/1.51, Immune 0.94 vs. 0.81/0.80,
Lung 1.39 vs. 1.33/1.32).

This isn't a clustering-quality gap either. To avoid the same circularity concern
Reviewer 1 (F5RB) raised, we score on held-out edges only (20% masked before
training; full protocol in our response to Reviewer 1 (F5RB) §2). On Pancreas (our
own K=220): SEACells and Leiden on the scPoli-Stage1 embedding score 0.25±.07 and
0.36±.06 on held-out edges — both well below scProto's own 0.59±.09 — with batch
entropy 1.24±.43 and 1.30±.43. Once Stage-1 has distorted the affinity structure,
no downstream clustering method recovers it (same mechanism, Reviewer 1 (F5RB) §1).

### 4. Originality (Weakness 4)

scProto solves something no prior objective does: cross-batch metacells whose
assignments respect real, affinity-defined community structure — not just
batch-mixing — using a signal from any modality, transcriptomic or spatial,
with zero change to the objective. That structure is exactly what's easiest to
lose, and often the most biologically important: rare cell states and spatial
niches.

No prior method protects it. scVI/SURE/MetaQ reconstruct without ever
referencing an affinity graph, so a rare state numerically thin in every batch
has nothing shielding it from being swamped by dominant states. SEACells/MetaCell have no
cross-batch mechanism at all: SEACells' kernel is density-corrected against
exactly this swamping, but stays within-batch; MetaCell partitions by topology
without using expression directly. The two batch-aware latent methods the
reviewer names lack a piece too: **Harmony** [2] corrects per
soft-cluster, and its clustering step rewards clusters that mix batches — a
rare population too thin to form its own cluster gets pooled under, and
corrected toward, whatever larger cluster it's nearest to: the same swamping
risk as scVI/SURE/MetaQ, by a different mechanism. **SpatialMETA** (Nat.
Commun. 2025) shares scProto's CVAE-style backbone, but nothing in its
training objective ties assignment to an affinity/community-preserving signal
the way scProto's does — the same gap as scVI/SURE, for a different problem
(modality fusion, not metacell learning).

This is a real gap, not an asserted one: no existing embedding,
downstream-clustered by a strong graph-aware method (§5), closes it on its own.
The stop-gradient ablation (§1) is direct proof the pieces don't compose for
free: without it, reconstruction's dense-favoring bias fights the
affinity-driven terms over the shared encoder. Generality shows the
combination is load-bearing, not arbitrary glue: the identical loss accepts
spatial affinity in place of transcriptomic affinity with zero redesign,
producing niche-correlated metacells while keeping cell-type purity (Fig. 3) —
SEACells' own archetypal analysis can take that swap but trades away purity for
niche purity without a reconstruction term anchoring it to real expression.
scProto is the only method that keeps both.

### 5. Batch-correction preprocessing comparison (Question 3)

First, scPoli's own Stage-1 pretraining (reconstruction + KL only, no
affinity-graph training) as the correction step, with two strong graph-aware
clusterers (SEACells, Leiden) applied to the resulting embedding — replacing the
paper's original K-means-only comparison:

| Method (Pancreas / Lung / Immune) | F1 (macro) | Rare-cell homogeneity |
|---|---|---|
| scProto | **0.56±.22** / **0.57±.22** / **0.91±.02** | **0.54±.18** / **0.60±.23** / **0.86±.04** |
| SEACells on Stage-1 embedding | 0.40±.23 / 0.55±.25 / 0.89±.06 | 0.40±.20 / 0.53±.21 / 0.83±.09 |
| Leiden on Stage-1 embedding | 0.55±.18 / 0.54±.20 / 0.78±.15 | 0.50±.18 / 0.52±.18 / 0.70±.11 |

Neither baseline closes the gap on this identical, already-corrected embedding —
scProto leads on both metrics, all three datasets.

Second, Harmony — the reviewer's suggested example. Our mechanistic account
(Stage 1's reconstruction-dominated-by-common-states bias) is specific to
VAE-style correction (scPoli, scVI); Harmony instead corrects PCA embeddings via
iterative soft-clustering and per-cluster linear regression, not reconstruction,
so it doesn't predict Harmony's rare-state behavior. We lack a first-principles
account for what we see: SEACells-on-Harmony's F1 is mixed — ahead of scProto
on some datasets, behind on others. The claim we rest on is the
architecture-level generality argued in §4, which Harmony has no path to.

### 6. Presentation (minor issues)

All noted presentation issues (figure legibility, naming consistency, undefined
terms, the l.194 ambiguity, matrix terminology, equation numbering) will be fixed
for camera-ready; no new experiments needed.

### Ref

[1] Wang, H., Leskovec, J., & Regev, A. "Limitations of cell embedding metrics assessed
using drifting islands." *Nature Biotechnology* 44 (2026): 574–577.

[2] Korsunsky, I., Millard, N., Fan, J., et al. "Fast, sensitive and accurate integration
of single-cell data with Harmony." *Nature Methods* 16 (2019): 1289–1296.
