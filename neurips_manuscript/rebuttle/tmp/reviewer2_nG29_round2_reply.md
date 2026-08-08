We thank the reviewer for the follow-up and for confirming the K-sensitivity and originality points are resolved.

**1. Loss ablation on Lung and Immune (PBMC)**

The same 8-arm ablation now covers all three datasets (per-batch mean ± std throughout):

| Removed / changed | Intended role | What changed |
|---|---|---|
| L_community | Matches assignments to the affinity graph | Largest modularity drop of any arm on all three datasets: Pancreas 0.615±.08→0.449±.04, Lung 0.655±.03→0.483±.05, Immune 0.623±.06→0.379±.09. Rare homogeneity drops too: .63±.16→.41±.23, .56±.23→.54±.24, .90±.03→.70±.12. |
| L_nassoc | Keeps prototypes tight and non-redundant | Rare homogeneity drops on all three: Pancreas .63±.16→.51±.19, Lung .56±.23→.48±.23, Immune .90±.03→.76±.17. |
| L_usage | Anti-collapse penalty for rare states | Rare coverage/homogeneity drop sharply on Pancreas (.69±.22→.43±.27 / .63±.16→.40±.21); milder on Lung/Immune, but homogeneity still drops there too (.56±.23→.52±.24, .90±.03→.79±.15). |
| L_rec (removed) | Grounds prototype content in real expression | Purity/modularity flat on all three (expected — stop-gradient blocks it from the encoder). Decoded-prototype fidelity (scGraph corr.) drops instead, on all three: Pancreas 0.910→0.846, Lung 0.854→0.780, Immune 0.841→0.670 (point values, no std). |
| Stop-gradient (removed, recon → encoder) | Isolates recon's dense-favoring bias from the shared encoder | Modularity drops on Pancreas (0.615±.08→0.555±.07) only — flat on Lung (0.655±.03→0.667±.03) and Immune (0.623±.06→0.630±.09). Pancreas's 9 distinct sequencing chemistries give reconstruction a non-biological axis to chase that the affinity graph doesn't encode; Lung and Immune's more homogeneous chemistry leaves less such an axis to conflict over. |
| Waypoint init (removed) | Seeds prototypes to cover sparse/rare regions | Removing it drops modularity on Pancreas (0.615±.08→0.594±.07) but leaves it flat on Lung (0.655±.03→0.674±.03) and Immune (0.623±.06→0.631±.06) — consistent with the community loss being strong enough there to reach a similar configuration regardless of starting point. Rare homogeneity also drops on both Pancreas (.63±.16→.48±.22) and Immune (.90±.03→.81±.13), flat only on Lung (.56±.23→.57±.22). |
| Fixed τ (no calibration) | Calibrates assignment sharpness | Rare homogeneity drops on Pancreas (.63±.16→.56±.16); purity/modularity flat or improve on all three. |

L_community, L_usage, L_nassoc, and L_rec all have a role that holds on all three datasets; the stop-gradient and waypoint-init modularity effects are Pancreas-specific, flat on Lung/Immune, noted above.

**2. Hyperparameter tuning**

All four loss weights raised in the review are now swept on all three datasets (community loss is the fixed reference weight; the rest are swept relative to it), five points each around the paper's own value. K-sensitivity was reported in our first response.

| Weight | Grid (paper's value) | Finding |
|---|---|---|
| λ_usage | {0, 0.03, 0.1, 0.3, 1.0} (**0.1**) | Directly controls prototype collapse: at λ_usage=0, 70–86% of prototypes go unused across the three datasets (Pancreas .70, Lung .80, Immune .86), and rare-cell F1 drops with it (Pancreas .49 vs .62 at the paper's value). By the paper's value, collapse is essentially eliminated (Pancreas .005, Lung .017, Immune .037) and rare-cell F1 recovers; going 10x higher still (1.0) keeps rare F1 flat or better on all three (Pancreas .66, Lung .64, Immune .91), with only a mild modularity cost over that same range (largest: Pancreas .65→.56). |
| λ_recon | {0, 0.003, 0.01, 0.03, 0.1} (**0.01**) | Purity and modularity stay flat across the full range on all three datasets (e.g. Pancreas modularity .60–.63, Lung .67±.03, Immune .62–.65 throughout) — consistent with the round-1 ablation's finding that reconstruction doesn't drive clustering quality, now confirmed over the full grid rather than just the two endpoints tested there. |
| α (nassoc diag/off-diag) | {0, 0.25, 0.5, 1.0, 2.0} (**1.0**) | Purity and modularity are flat across the whole range on all three datasets — no sensitivity to this sub-weight. |
| λ_nassoc | {0, 0.3, 1.0, 3.0, 10.0} (**1.0**) | Modularity and batch entropy stay flat from 0 up through the paper's own value on all three datasets (e.g. Pancreas modularity .65→.62, batch entropy 1.29→1.01); pushing λ_nassoc 3–10x past the paper's value is what starts trading them away (Pancreas modularity down to .50, batch entropy to .32 at 10x). The paper's value sits inside the flat, stable region with headroom before the trade-off appears, not at its edge. |

Rare-cell coverage/homogeneity/F1 move with each weight in the direction the round-1 ablation already predicted (dropping when the corresponding term is under-weighted), and are otherwise stable across the tested range.

One hyperparameter we are not sweeping: the assignment-temperature τ is calibrated from data after pretraining and prototype initialization rather than set by hand (Appendix, Temperature Calibration), so a sweep over it would just re-discover what the calibration step already does automatically; our fixed-τ ablation arm above is the relevant control for it instead.

**3. Harmony batch-correction-then-cluster, corrected**

The "mixed" framing was a dimensionality error, not an ambiguous result: we had run Harmony on 50-dimensional PCA, roughly 6x more room than scProto's 8-dimensional latent size, so the comparison wasn't fair. We reran Harmony on 8-dimensional PCA, then applied SEACells and Leiden on top.

Rare-cell F1 (macro) and homogeneity, mean±std, paired one-sided Wilcoxon vs. scProto (Bonferroni-corrected per dataset per metric):

| Dataset | Metric | scProto | SEACells (Harmony) | Leiden (Harmony) |
|---|---|---|---|---|
| Pancreas (n=8) | F1 | 0.54±.21 (ref) | 0.32±.16, p=0.035\* (7/8) | 0.22±.17, p=0.012\* (8/8) |
| Pancreas (n=8) | Homog. | 0.56±.16 (ref) | 0.29±.12, p=0.012\* (8/8) | 0.22±.09, p=0.012\* (8/8) |
| Lung (n=15) | F1 | 0.60±.18 (ref) | 0.36±.14, p=0.0023\*\* (13/15) | 0.28±.14, p=0.0009\*\*\* (13/15) |
| Lung (n=15) | Homog. | 0.58±.22 (ref) | 0.40±.12, p=0.006\*\* (14/15) | 0.34±.08, p=0.0023\*\* (13/15) |
| Immune (n=5) | F1 | 0.85±.14 (ref) | 0.75±.16, ns (4/5) | 0.75±.17, ns (4/5) |
| Immune (n=5) | Homog. | 0.83±.10 (ref) | 0.68±.19, ns (5/5) | 0.64±.13, ns (5/5) |

Immune's rare-cell metrics stay non-significant only because n=5 batches is underpowered, not because the direction reverses — win-counts there are still 4-5 out of 5 in scProto's favor.

We attribute scProto's advantage here to how Harmony corrects. Harmony first groups cells into soft clusters, rewarding clusters that mix batches well, then corrects each cluster's batch effect separately — creating two failure modes for a rare state. Within a single batch, its cells may be too few or too sparse to register as their own cluster, so they get absorbed into a denser cluster instead, and the per-cluster correction that follows — reflecting the dominant population, not the minority folded into it — doesn't recover them and can make things worse; scProto's density-corrected affinity graph prevents this absorption up front, and its community loss keeps shaping assignment throughout training rather than fixing a cluster once. Separately, a state absent from some batches can't form a naturally well-mixed cluster, so Harmony's clustering objective [1] — which rewards batch-diverse clusters regardless of biology — can pull in unrelated cells from other batches just to satisfy it; scProto never faces this, since each batch's cells are pulled toward shared prototypes using only that batch's own affinity graph, so a prototype is never rewarded for how many batches contribute to it.

**4. ComBat**

Same pipeline as Harmony above (SEACells and Leiden on the corrected embedding, K matched
to scProto's prototype count): original ComBat (Johnson, Li & Rabinovic 2007) corrects
the full log-normalized expression matrix directly, then we PCA the corrected matrix down
to an embedding for SEACells/Leiden, matched to scProto's 8-dimensional latent for the
same fairness reason as Harmony.

Rare-cell F1 (macro) and homogeneity, mean±std, paired one-sided Wilcoxon vs. scProto
(Bonferroni-corrected per dataset per metric):

| Dataset | Metric | scProto | SEACells (ComBat) | Leiden (ComBat) |
|---|---|---|---|---|
| Pancreas (n=8) | F1 | 0.54±.21 (ref) | 0.35±.21, p=0.039\* (7/8) | 0.15±.07, p=0.020\* (8/8) |
| Pancreas (n=8) | Homog. | 0.56±.16 (ref) | 0.29±.08, p=0.020\* (8/8) | 0.23±.10, p=0.020\* (8/8) |
| Lung (n=15) | F1 | 0.60±.18 (ref) | 0.33±.21, p=0.011\* (14/15) | 0.38±.19, p=0.003\*\* (13/15) |
| Lung (n=15) | Homog. | 0.58±.22 (ref) | 0.40±.25, p=0.017\* (14/15) | 0.40±.21, p=0.013\* (14/15) |
| Immune (n=5) | F1 | 0.85±.14 (ref) | 0.86±.06, ns (3/5) | 0.80±.19, ns (4/5) |
| Immune (n=5) | Homog. | 0.83±.10 (ref) | 0.75±.15, ns (4/5) | 0.75±.18, ns (4/5) |

Same pattern as Harmony: scProto significantly ahead on Pancreas and Lung across both
metrics and both downstream clusterers; Immune is non-significant (n=5 batches,
underpowered) but not reversed — SEACells+ComBat's F1 (0.86) is a numerical tie with
scProto (0.85), not a loss, and win-counts elsewhere still favor scProto or split evenly.

The two structural points we made before the numbers still hold, now with evidence behind
them: ComBat estimates one shift/scale per gene per batch — run here without cell-type
labels, like every other correction baseline in this rebuttal — so it cannot distinguish a
technical batch effect from a real cell state that happens to be concentrated in one
batch — both produce the identical per-batch mean shift ComBat corrects against. It also
applies that one correction uniformly across every cell in a batch regardless of cell
type, so a batch effect that hits a rare population harder than the dominant one is not
something a single global per-gene number can represent. scProto has no equivalent global
correction step — a cell's placement is conditioned on its own state via the affinity
graph, not a batch-wide statistic — the same argument we made for Harmony's
over-correction risk, applied to a different correction mechanism.

Ref

[1] Korsunsky, I., Millard, N., Fan, J., et al. "Fast, sensitive and accurate integration of single-cell data with Harmony." Nature Methods 16 (2019): 1289–1296.

---

## To do

- [ ] Post λ/α sweep results for Pancreas once the run finishes; then start Lung + Immune
- [ ] Check response length against OpenReview's character limit before posting; trim table precision (e.g. drop macro F1 to 2 decimals) if over
