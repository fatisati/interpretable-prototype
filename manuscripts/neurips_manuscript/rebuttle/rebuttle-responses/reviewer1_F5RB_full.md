# Reviewer 1 (F5RB) — Full Official Review

**Submission:** 15985 — scProto
**Reviewer:** F5RB
**Posted:** 26 Jun 2026, 15:06 (modified 23 Jul 2026, 18:27)

**Rating:** 3 — Borderline reject: *Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.*
**Confidence:** 3 — *You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.*

| Quality | Clarity | Significance | Originality |
|---|---|---|---|
| 3: good | 3: good | 3: good | 3: good |

---

## Summary

The paper addresses a real gap between metacell construction and batch correction. Existing metacell methods often preserve local transcriptomic neighborhoods, but the resulting metacells are still largely batch-specific. Batch-correction methods can mix batches, but they may also weaken local communities or rare cell states. I find this motivation convincing.

The prototype-based design is also reasonable. It gives a clear connection between cells, prototypes, and metacells. The affinity-guided objective is easy to understand, and the framework can in principle use different affinity graphs, including transcriptomic and spatial graphs. These are useful design choices.

My main concern is that the paper sometimes gives a stronger mechanistic story than the current method and experiments can support. In particular, the cross-batch alignment seems to come mainly from the Stage 1 scPoli-style pretraining. In Stage 2, the reconstruction loss stops the gradient through the prototype assignment, so it does not directly update the encoder through this path. This makes the method look more like affinity-guided prototype refinement on a pretrained batch-corrected latent space, rather than a fully joint model for reconstruction, batch alignment, and community preservation.

The evaluation also needs stronger controls. Modularity is closely related to the training affinity graph. It is also used for early stopping and then reported as a main result. This makes the modularity evidence less independent. The paper is promising, but stronger baselines, clearer ablations, and less circular validation are needed.

**Contribution Type:** General: Most submissions will fall into this type.

---

## Strengths and Weaknesses

### Strengths

1. Existing metacell methods can preserve local neighborhoods, but they often build metacells within each batch. This is not ideal when related cell states are split across batches. Batch-correction methods can improve mixing, but they may also blur local communities or rare states. The paper identifies this tension clearly and studies a useful problem.
2. Each cell is softly assigned to prototypes, and the prototypes are used as metacells. This design is easy to follow. The affinity loss also has a clear meaning: cells that are close in the input graph should have similar prototype assignments. This makes the method relatively interpretable.
3. The experiments do not rely on only one metric. The paper reports community structure, batch mixing, cell-type purity, rare-cell coverage, rare-cell homogeneity, and spatial niche purity. These metrics cover several important aspects of metacell quality. This makes the empirical study more informative than a simple clustering comparison.

### Weaknesses

1. **The training design does not fully support the mechanism claimed in the paper.** The manuscript suggests that reconstruction helps cross-batch alignment during prototype learning. However, the reconstruction loss stops the gradient through the assignment $s_{ik}$. Therefore, this loss does not update the encoder through the prototype assignment in Stage 2. The encoder is mainly shaped by the affinity/community objective after pretraining. The cross-batch correction seems to be largely inherited from the Stage 1 scPoli-style model. This point should be stated more clearly.

2. **The modularity result is not fully independent from the training objective.** The community loss directly encourages prototype assignment similarity to match the input affinity graph. Later, modularity is computed using the per-batch affinity subgraph. Modularity is also used for early stopping. Because of this, high modularity is partly expected. It is still a useful diagnostic, but it is not strong enough as independent evidence of biological metacell quality.

3. **The most important baseline is missing.** A natural comparison would be: first apply a batch-correction method, such as scPoli, scVI, Harmony, or BBKNN, and then run an existing graph-based metacell method such as SEACells on the corrected latent space. The current "scPoli + K-means" baseline is weaker. K-means is not designed to preserve graph communities. Without a corrected-latent-space plus graph-metacell baseline, it is hard to know whether scProto improves over a simple combination of existing tools.

4. **The spatial claim is stronger than the current evidence.** The paper says that scProto captures niche-correlated transcriptional programs. The presented results mainly show niche purity and cell-type purity. This is useful, but it does not yet prove discovery of transcriptional programs. Also, the niche labels are defined from local cell-type composition, and the spatial affinity graph is built from neighborhood-averaged PCA features. Therefore, the evaluation is not fully independent from the training signal.

5. **The ablation study does not isolate the main components well enough.** The method contains several important parts, including the community loss, reconstruction loss, normalized association regularization, usage loss, temperature calibration, and waypoint initialization. These components are not separately tested in a complete way. As a result, it is unclear which component is responsible for the main improvement.

---

## Questions

1. **Where does the cross-batch alignment mainly come from?** Since $L_{\mathrm{rec}}$ does not update the encoder through the prototype assignment in Stage 2, is the alignment mostly inherited from Stage 1 scPoli-style pretraining? Please compare scProto with stronger two-step baselines, such as scPoli latent + SEACells, scVI latent + SEACells, Harmony latent + SEACells, and BBKNN graph + SEACells.

2. **Can the authors provide less circular evidence for community preservation?** Modularity is related to the training graph and is also used for early stopping. It would be helpful to report held-out-edge graph preservation, evaluation on an independently constructed affinity graph, or results with an early-stopping rule that does not use modularity.

3. **Which component gives the main gain?** Please include a fuller ablation study. Important variants include removing $L_{\mathrm{community}}$, removing $L_{\mathrm{rec}}$, allowing reconstruction gradients to update the encoder, removing $L_{\mathrm{nassoc}}$, removing $L_{\mathrm{usage}}$, using random prototype initialization, and using fixed instead of calibrated temperature.

4. **What is the biological evidence for spatial programs?** The niche-purity result is interesting, but it does not by itself prove that scProto discovers niche-correlated transcriptional programs. Can the authors show niche-specific marker genes, pathway enrichment, ligand–receptor analysis, or validation on additional slides?

---

## Limitations

Yes.

---

## Other Fields

- **Ethical Concerns:** NO or VERY MINOR ethics concerns only
- **Paper Formatting Concerns:** None
- **Code Of Conduct Acknowledgement:** Yes
- **Responsible Reviewing Acknowledgement:** Yes

---

## Our Response

We thank the reviewer for a precise, constructive review. Below: where cross-batch
alignment comes from, why Stage 2 is a distinct mechanism rather than a re-derivation of
Stage 1, and controlled experiments backing both — including where current phrasing
understates the modularity evidence.

### 1. Where cross-batch alignment comes from (Weaknesses 1 & 3; Question 1)

Alignment comes from Stage 1, not Stage 2. Stage 1 (scPoli-style pretraining) conditions
the decoder on batch, so the encoder discards batch-specific technical variation it
doesn't need for reconstruction — giving scProto a good initial, batch-corrected latent
space, the real source of cross-batch alignment. But that objective favors dense,
high-abundance states: its reconstruction gradient is summed over cells, so common states
dominate, and a low-density population can get absorbed into a denser neighbor as a side
effect, though often biologically important — the well-documented
over-correction/over-integration phenomenon [1, 2].

This is why batch-correct-then-cluster isn't a fix: once Stage 1 scatters a rare state
into a denser neighbor, any downstream step — K-means, Leiden, even SEACells' archetypal
analysis — only partitions a geometry that already lost the signal. Stage 2 avoids this by
continuing to train the encoder and prototypes so assignment similarity matches a
cell-cell affinity graph built separately within each batch, where technical noise
cancels out and only real biology remains. Prototypes are shared across batches by
construction, so pulling each batch's cells toward them using only that batch's own
affinity signal pools structure across batches without comparing raw expression directly —
how a rare state too sparse in one batch to resist the dense bias can still accumulate
into a shared prototype (specifics in Section 3). We agree the manuscript (l.150–152)
currently credits reconstruction with alignment, contradicting the stop-gradient at
l.156–157; we will revise this for camera-ready to state plainly that alignment comes
from Stage 1, not reconstruction.

We tested this with the requested two-step baselines on scProto's Stage-1 embedding (held
fixed, only the downstream step varies), scored on two metrics restricted to the paper's
locally-rare cell types (Appendix F), each capturing a different aspect of the failure
mode above. Common types are excluded, not down-weighted, so a method can't compensate
elsewhere. Homogeneity is per-cell, graded: how much of a rare cell's metacell, wherever
it landed, shares its type — no dedicated prototype need exist. F1 is stricter, per-type —
does a dedicated prototype for that type exist at all: `recall_c` catches a type that
never clears a 50% majority anywhere; `precision_c` catches one whose dedicated
prototype, once formed, is impure. Let `M_c` be the metacells whose majority label is
`c`: `precision_c = (1/|M_c|) * sum over m in M_c of purity(m)`,
`recall_c = |{i : y_i = c, m(i) in M_c}| / |{i : y_i = c}|`,
`F1_c` their harmonic mean; "macro" averages per-type scores equally, not by cell
count, across a batch then across batches:

| Method (Pancreas / Lung / Immune) | F1 (macro) | Rare-cell homogeneity |
|---|---|---|
| scProto | **0.56±.22** / **0.57±.22** / **0.91±.02** | **0.54±.18** / **0.60±.23** / **0.86±.04** |
| SEACells on Stage-1 embedding | 0.40±.23 / 0.55±.25 / 0.89±.06 | 0.40±.20 / 0.53±.21 / 0.83±.09 |
| Leiden on Stage-1 embedding | 0.55±.18 / 0.54±.20 / 0.78±.15 | 0.50±.18 / 0.52±.18 / 0.70±.11 |

SEACells and Leiden on scProto's Stage-1 embedding cannot close the gap — scProto leads
both on both metrics, all three datasets. We deliberately
omit a raw modularity column here: scProto's modularity is scored against the exact graph
`L_community` trains it to match, so a full-graph comparison here would
reopen Weakness 2's circularity concern before we've earned the right to use it as
evidence — that requires the held-out-edge protocol in §2, scoring scProto only on edges
it never trained on. Harmony/scVI/BBKNN+SEACells (embedding fixed across downstream methods) are still
running; only scVI is encoder-based, so Stage 2 can continue training it like scPoli
above — Harmony/BBKNN lack an encoder, so those two are necessarily full-pipeline
comparisons, not an isolated Stage-2 test.

### 2. Less circular evidence for community preservation (Weakness 2; Question 2)

We think scProto's affinity guidance does more than memorize trained edges: the affinity
losses act on each cell's expression-derived embedding, not edge identity, so the encoder
learns a mapping from expression to assignment — not a per-edge lookup — that should
transfer to edges it never saw. We tested this with a held-out-edge protocol: on each
RNA-seq dataset we masked a random 20% of affinity-graph edges before training (symmetric,
per-cell floor so no neighborhood is starved), trained scProto and fit SEACells on the
identical masked 80%, and scored modularity **only** on the hidden 20%:

| Held-out-edge modularity (per-batch mean±std) | Pancreas | Lung | Immune |
|---|---|---|---|
| scProto (trained on masked graph) | **0.59±.09** | **0.66±.02** | **0.62±.06** |
| SEACells (identical masked graph) | 0.47±.08 | 0.52±.07 | 0.39±.02 |
| SEACells on scPoli-Stage1 embedding | 0.25±.07 | 0.31±.05 | 0.21±.03 |
| Leiden on scPoli-Stage1 embedding | 0.36±.06 | 0.49±.08 | 0.45±.18 |

scProto's modularity on unseen edges stays close to its headline number and ahead of
SEACells on the identical masked graph, SEACells on the scPoli Stage-1 embedding, and
Leiden on that embedding — not an artifact of scoring against its own training objective.
The separation is clean (non-overlapping per-batch ranges) on Pancreas and Lung; on
Immune, Leiden's per-batch variance (±.18) edges into scProto's range, though scProto's
mean leads.

### 3. Fuller ablation (Weakness 5; Question 3)

Pancreas only so far; extending to Lung/Immune. Purity is cell-count weighted (Appendix
F); rare coverage/homogeneity carry the rare-state signal instead.

| Removed / changed | Intended role | What changed, and why |
|---|---|---|
| `L_community` | Matches assignments to the affinity graph | **Modularity collapses** (0.615±.08→0.449±.04, largest drop of any arm) — this term drives community structure directly. Rare coverage (0.69±.22→0.50±.23) and homogeneity (0.63±.16→0.41±.23) drop too. |
| `L_nassoc` | Keeps prototypes tight and non-redundant | **Rare homogeneity drops** (0.63±.16→0.51±.19) — prototypes become looser/more redundant without this term. Purity drops too, most of any arm, though modest (0.972±.09→0.962±.09). |
| `L_usage` | Anti-collapse penalty for rare states | **Rare coverage and homogeneity drop sharply** (0.69±.22→0.43±.27; 0.63±.16→0.40±.21) — exactly the rare-state capacity this term exists to protect. |
| `L_rec` (removed entirely) | Grounds prototype content in real expression (claimed under spatial-affinity guidance) | **Decoded-prototype fidelity drops most of any arm** (gene-expression-space scGraph [3] variant, 0.91→0.85 on Pancreas) — since `L_rec`'s gradient is stopped before the encoder, it shapes decoder content only, so purity/modularity stay flat (0.972±.09→0.970±.07 / 0.615±.08→0.614±.08). On the spatial dataset (NSCLC), the one setting this loss should also protect cell-type identity, purity (0.50±.21→0.49±.20, weighted) and niche purity (0.91±.19→0.87±.22, unweighted) drop modestly too (single run). |
| Stop-gradient (removed, recon gradient reaches encoder) | Prevents recon's dense-favoring bias fighting community/nassoc over the shared encoder | **Modularity and homogeneity drop** (0.615±.08→0.555±.07; 0.63±.16→0.51±.16) once reconstruction reaches the encoder — it fights the affinity objectives, as designed to prevent. Ranges overlap; purity doesn't move (0.972±.09→0.976±.07, within noise). |
| Waypoint init (→ random init) | SEACells-style informed init — seeds prototypes to cover sparse/rare regions | **Rare coverage and homogeneity drop** with random init (0.69±.22→0.56±.28; 0.63±.16→0.48±.22) — losing the head start on sparse regions. Modularity drops too (0.615±.08→0.594±.07). |
| Fixed $\tau$ (no calibration) | Calibrates assignment temperature (avoids too-diffuse/sharp assignments) | **Only rare homogeneity drops** (0.63±.16→0.56±.16) — specific to homogeneity. Purity (0.972±.09→0.979±.05), modularity (0.615±.08→0.623±.08), and rare coverage (0.69±.22→0.72±.21) stay flat or improve. |

`L_rec`'s non-effect on assignment is expected either way: the affinity
graph is itself built from expression, so community/nassoc stay anchored to it regardless
of `L_rec`. This is a gene-expression-space variant of scGraph [3]: PCA of decoded-prototype
expression, cell-type-pairwise distances correlated against scGraph's own per-batch
real-expression consensus graph.

### 4. Biological evidence for spatial programs (Weakness 4; Question 4)

Marker-gene, pathway-enrichment, and ligand–receptor analyses are a natural next step,
building on the niche/purity results already shown. We will try to run these and share
results in the discussion comments.

### Ref

[1] Luecken, M. D., et al. "Benchmarking atlas-level data integration in single-cell
genomics." *Nature Methods* 19 (2022): 41–50.

[2] Büttner, M., et al. "A test metric for assessing single-cell RNA-seq batch
correction." *Nature Methods* 16 (2019): 43–49.

[3] Wang, H., Leskovec, J., & Regev, A. "Limitations of cell embedding metrics assessed
using drifting islands." *Nature Biotechnology* 44 (2026): 574–577.
