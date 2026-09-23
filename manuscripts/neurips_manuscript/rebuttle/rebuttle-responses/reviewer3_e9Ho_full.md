# Reviewer 3 (e9Ho) — Full Official Review

**Submission:** 15985 — scProto
**Reviewer:** e9Ho
**Posted:** 23 Jun 2026, 05:53 (modified 23 Jul 2026, 18:27)

**Rating:** 3 — Borderline reject: *Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.*
**Confidence:** 4 — *You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.*

| Quality | Clarity | Significance | Originality |
|---|---|---|---|
| 2: not good | 3: good | 2: not good | 2: not good |

---

## Summary

scProto learns cross-batch prototypes (metacells) with an adapted batch-conditioned cVAE (scPoli): a community-preserving objective that matches pairwise prototype-assignment similarity ($s_{ik}$) to an input cell-cell affinity graph (transcriptomic or spatial), a reconstruction objective, and several other regularization terms. On three scIB integration datasets and one NSCLC spatial dataset it reports competitive modularity, high batch entropy, leading rare-cell homogeneity, and a favorable niche-purity vs cell-type-purity trade-off. The method is lightweight (under an hour per dataset) and the hyperparameters are documented.

**Contribution Type:** General: Most submissions will fall into this type.

---

## Strengths and Weaknesses

**Reviewer's own framing paragraph:** The problem is real and the formulation is reasonable, but the core claim is never tested. The community-structure win is confounded by the baseline clusterer and by scoring on the model's own training objective, the standard integration metrics are missing, and the spatial claim is built on a single dataset.

### Strengths

1. **Sensible, swappable formulation.** Matching prototype (metacell) assignments to an affinity graph that accepts either transcriptomic or spatial input is a reasonable and flexible design (l.114-119, l.142-152).
2. **Reproducible configuration.** Appendix D/E document the hyperparameters (l.424-484), so the missing ablations are not a compute problem.

### Weaknesses

1. **No ablation, so the core claim is untested (vital).** No experiment removes any introduced loss terms, sweeps $K$, or toggles the encoder stop-gradient (§4, App A-G), so the thesis that the affinity-community objective drives the gains is never separated from the scPoli backbone.

2. **K-means handicaps the baselines.** Embedding baselines are clustered with plain K-means (l.213-215) while scProto uses graph-aware assignment and is scored on modularity, so the community-structure win is confounded with the clusterer, not the embedding (Leiden/Louvain is the field default).

3. **No standard scIB metrics on scIB data.** The datasets come from scIB [28] (App A l.383-384), but none of its metrics (ARI/NMI/ASW/kBET/iLISI) appear (l.220-223), and the headline metric, modularity, is also the training loss and early-stop criterion (App D l.434-436), so the comparison is neither independent nor fair.

4. **The headline metric is under-specified.** Modularity lacks a formula, resolution, or null model (App F l.486-488), so the numbers are neither reproducible nor comparable across the different-sized batches Table 1 averages.

5. **The cross-batch mechanism is inconsistent.** The text credits cross-batch alignment to reconstruction (l.150-152), but the stop-gradient leaves the encoder no reconstruction gradient (l.156-157), so the alignment actually comes from the inherited Stage-1 cVAE.

6. **Abstract claims exceed the tables.** Purity trails SEACells on Immune (0.87 vs 0.88) and Lung (0.86 vs 0.90), coverage trails on Lung (0.72 vs 0.74), and the "modularity across all three datasets" claim (l.229) fails on Pancreas (0.61 vs 0.67) (l.13-14, Tables 1-2).

7. **Thin significance and baselines.** Tables 1-2 report no significance test (only the spatial Fig.3 does), and some "wins" overlap in std (e.g. Immune 0.62±0.08 vs 0.55±0.04). Only SEACells and MetaQ are true metacell baselines; SuperCell [11], Metacell-2 [4], and scVI [15] are not run.

8. **The spatial claim rests on one dataset.** The spatial results come from a single subsampled NSCLC slide (App A l.385-386), which cannot show robustness to tissue or platform. Either narrow the claim or add a second benchmark dataset.

---

## Questions

1. **Leave-one-term-out ablation.** Remove each loss term, toggle the stop-gradient, and sweep $K$ on one dataset, with the backbone fixed.

2. **What drives cross-batch alignment?** l.152 credits reconstruction, but the stop-gradient leaves the encoder no reconstruction gradient (l.156-157).

3. **Report the standard scIB metrics.** Will you add standard metrics like ARI, NMI, ASW, kBET, iLISI, and graph-connectivity on the datasets?

4. **Make the modularity comparison fair.** Select on a held-out criterion and cluster the baselines with Leiden/Louvain, then report modularity on the held-out data.

5. **Baselines, prior art, scope.** Will you add more baselines like scVI [15] / SuperCell [11], state the delta over [26], narrow the spatial claim, and soften the abstract to match the tables?

---

## Limitations

The two stated limitations (K set in advance; affinity-graph quality, l.283-285) are genuine but narrow, and omit the gaps that bear on the conclusions: no ablation, no standard integration metrics, no significance tests on Tables 1-2, the single spatial dataset, and every metric where scProto trails.

---

## Other Fields

- **Ethical Concerns:** NO or VERY MINOR ethics concerns only
- **Paper Formatting Concerns:** None
- **Code Of Conduct Acknowledgement:** Yes
- **Responsible Reviewing Acknowledgement:** Yes

---

## Our Response

We thank the reviewer for the detailed, evidence-based review. Full
tables for the two-step Stage-1-embedding baseline, held-out-edge modularity,
and leave-one-term-out ablation are in our response to Reviewer F5RB (§1–§3); we
reference those numbers here and focus below on what's specific to this review.

### 1. No ablation, so the core claim is untested (Weakness 1; Question 1)

Full 7-arm leave-one-term-out ablation (every loss term, stop-gradient, waypoint
init, temperature calibration) is in our response to Reviewer F5RB, §3. The two
arms that most directly test this weakness: removing $\mathcal{L}_{\text{community}}$
collapses modularity (0.61→0.45, largest drop of any arm, though not all the way
to the scPoli-cVAE+K-means backbone's own 0.23); removing the stop-gradient
(letting reconstruction reach the encoder) drops modularity and homogeneity
(0.61→0.56; 0.63→0.51). Read with §2 below: the gains aren't simply inherited
from the scPoli backbone.

### 2. K-means handicaps the baselines (Weakness 2)

scProto is not only a clustering step — Stage 2's affinity loss keeps updating
the encoder, reshaping the latent geometry itself rather than partitioning a
fixed embedding. Batch-correction-only training (reconstruction + KL) is
dominated by common states, so a rare state can get absorbed into a denser
neighbor; any affinity graph built on that embedding is already corrupted, and
no downstream clusterer can recover a separation the embedding no longer
contains.

We tested this directly (F5RB §1): SEACells and Leiden applied to scProto's own
frozen Stage-1 embedding, not K-means. scProto leads both F1 (macro) and
rare-cell homogeneity on all three datasets — not a clusterer
artifact, since Leiden is exactly the graph-aware alternative requested and
still doesn't close the gap.

### 3. No standard scIB metrics on scIB data (Weakness 3)

We did not originally include the standard scIB battery because we don't think
it measures what scProto is designed for. It evaluates general-purpose
batch-integration embeddings — is the corrected space well-mixed, does
clustering it agree with cell-type labels — while scProto pools already-corrected
cells into groups that preserve fine-grained cell *states*, not just coarse
annotated *types*, a different target these benchmarks don't test. This metric
family can be actively misleading here: scProto uses
220–300 prototypes, deliberately finer than annotated cell type (9–17 here);
KMeans-based NMI/ARI forces $k$=#cell-types, so a method with no fine substructure
to lose can score *higher* than one whose finer, deliberately-preserved structure
gets merged by a coarse re-clustering — rewarding exactly the collapse scProto is
built to avoid. NMI/ARI are also cell-mass weighted, so a method can score well
by handling only common types while failing rare states entirely. Most directly,
Wang, Leskovec & Regev [1] show a model trained to
*directly* fit cell-type labels tops all 12 standard scIB metrics while
measurably distorting biological structure — a high score on this battery does
not certify a good embedding, since it largely rewards agreement with the same
coarse label used to define "conservation." Their proposed remedy, scGraph [1],
checks local neighborhood structure instead; we already apply it to scProto's
decoded prototypes in the ablation (F5RB §3) and will extend it to baselines as a
follow-up.

The evidence we rely on for scProto's actual claim is Table 2's rare-cell
coverage/homogeneity and the frozen-embedding comparison in §2, where scProto
beats SEACells and Leiden on the identical Stage-1 embedding on all three
datasets.

On modularity doubling as training loss and early-stopping criterion: addressed
in §4 with a held-out-edge protocol.

### 4. The headline metric is under-specified (Weakness 4; Question 4)

**Formula.** Standard weighted Newman modularity,
$$Q = \frac{1}{2m}\sum_k \left[e_k - \frac{d_k^2}{2m}\right],$$
computed on the continuous RBF-kernel affinity weights (no thresholding), with
the standard configuration-model null and a fixed resolution $\gamma=1$ (not
swept). We will add this to Appendix F.

**Circularity.** We address this with a held-out-edge protocol (F5RB §2): 20% of
each dataset's affinity-graph edges masked before training, scProto and SEACells
fit on the identical masked 80%, modularity scored only on the hidden 20%.
scProto's modularity on unseen edges stays close to its headline number and
ahead of every baseline, including Leiden — not an artifact of scoring against
its own training objective.

**Batch-size comparability.** We report modularity per batch, not pooled, so
structure preservation is visible batch-by-batch rather than dominated by the
largest batch. Every method is scored against identical per-batch subgraphs, so
batch-size variation is a shared covariate affecting every row equally. All
claims are within-dataset, cross-method comparisons, never magnitude comparisons
across datasets — we will state this explicitly.

### 5. The cross-batch mechanism is inconsistent (Weakness 5; Question 2)

The stop-gradient is deliberate, not an oversight — it stops reconstruction's
dense-favoring bias
from fighting the affinity/community objective over the shared encoder in
Stage 2. Our ablation (§1) shows why it matters: removing it drops modularity
and homogeneity (0.61→0.56; 0.63→0.51), purity flat.

We agree l.150–152 currently credits reconstruction with alignment, which the
stop-gradient contradicts. We will correct this for camera-ready: alignment is
inherited from Stage 1; Stage 2 preserves and pools community structure over
that already-aligned space, not re-aligning it.

### 6. Abstract claims exceed the tables (Weakness 6)

Agreed the wording at l.229 overclaims; we will fix it for camera-ready. The
trade-off the reviewer's numbers surface is in fact the result we want to make,
stated precisely instead of oversold.

Against every method that actually mixes batches (scPoli-cVAE, UMAP, MetaQ),
scProto's modularity is highest on all three datasets (Pancreas 0.61 vs
0.23/0.26/0.41; Immune 0.62 vs 0.26/0.26/0.28; Lung 0.67 vs 0.32/0.32/0.42).
Against SEACells, which matches or beats scProto's modularity on Pancreas/Lung,
the reason is visible in the same table: SEACells' batch entropy is 0.16–0.41
throughout — barely mixing at all — while scProto reaches 0.94–1.39. High
community structure and real mixing are not usually compatible; scProto is the
only method here that gets both. Rare-cell homogeneity drops the trade-off
entirely: best on all three datasets with no exceptions (Pancreas 0.54, Immune
0.86, Lung 0.60).

Where scProto trails, it trails methods that don't pay the mixing cost: Lung
purity (0.86) ties the other real mixers (scPoli-cVAE, UMAP: 0.86); SEACells
(0.90) and MetaQ (0.91) both mix far less (batch entropy 0.41, 0.53 vs 1.39).
Immune's purity gap to SEACells is 1pp (0.87 vs 0.88) against a ~6x mixing
difference. Pancreas purity is in fact scProto's best of all five methods
(0.98). Coverage follows the same pattern: best/tied-best on Pancreas/Immune,
second on Lung (0.72 vs 0.74), within one std of the top score. We will revise
the abstract and l.229 to state this as a trade-off (modularity/purity vs.
mixing) rather than an unqualified win, and state the Lung/Immune purity and
Lung coverage gaps explicitly.

### 7. Thin significance and baselines (Weakness 7)

On the Immune modularity comparison (0.62±0.08 for scProto vs. 0.55±0.04 for
SEACells): the paper's claim isn't that scProto's modularity strictly exceeds
SEACells', but that it's comparable while substantially outperforming on batch
mixing — the modularity numbers are the "matches" half of that joint claim, not
the evidence for an advantage. Batch entropy on the same dataset is 0.94±0.51 vs
0.16±0.24 — non-overlapping even accounting for spread. We will state this
framing explicitly.

We agree every Tables 1–2 comparison would be stronger with a significance
test, not only the ones large enough to flag by eye. We are adding paired tests
across batches/metacells for both tables, using the same Mann-Whitney approach
as Fig. 3, and will report results as a follow-up during the discussion period.

On baseline completeness: Tables 1–2 include SEACells and MetaQ as true
metacell baselines, with scPoli (cVAE)/UMAP as batch-correction-only controls.
SuperCell, Metacell-2, and scVI are discussed in Related Work as conceptually
limited, but each has a close empirical analog already in our tables:
SuperCell/Metacell-2 are restricted to within-batch aggregation — the same
limitation SEACells already shows (batch entropy 0.16–0.41, barely mixing);
scVI optimizes reconstruction without affinity guidance — the same property
scPoli (cVAE) already isolates. We can run all three directly during the
discussion period if needed.

### 8. The spatial claim rests on one dataset (Weakness 8; Question 5)

We will try to add a second spatial benchmark during the discussion period. If
we cannot complete it in time, we will narrow the spatial claim (§6 above) to
reflect evidence from a single NSCLC slide, deferring additional
tissues/platforms to future work.

### Ref

[1] Wang, H., Leskovec, J., & Regev, A. "Limitations of cell embedding metrics
assessed using drifting islands." *Nature Biotechnology* 44 (2026): 574–577.
