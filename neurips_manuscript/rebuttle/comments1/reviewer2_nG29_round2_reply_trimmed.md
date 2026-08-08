Thanks for the follow-up, and for confirming the K-sensitivity and originality points are resolved.

**1. Loss ablation on Lung and Immune (PBMC)**

Same 8-arm ablation, all three datasets (per-batch mean±std). L_community removal is the largest modularity drop everywhere (Pancreas .615→.449, Lung .655→.483, Immune .623→.379) and drops rare homogeneity too (.63→.41, .56→.54, .90→.70). L_nassoc removal drops homogeneity on all three (.63→.51, .56→.48, .90→.76). L_usage removal drops coverage/homogeneity sharply on Pancreas (.69→.43, .63→.40), milder on Lung/Immune homogeneity (.56→.52, .90→.79). L_rec removal leaves purity/modularity flat everywhere (expected — stop-gradient blocks it) but drops decoded-prototype fidelity (scGraph corr. .91→.85, .85→.78, .84→.67). Two arms are Pancreas-only: removing the stop-gradient drops modularity only on Pancreas (.615→.555, flat on Lung/Immune); removing waypoint init drops Pancreas modularity (.615→.594, flat elsewhere) and homogeneity on Pancreas+Immune (.63→.48, .90→.81, flat on Lung). Fixed τ only drops Pancreas homogeneity (.63→.56), flat/better elsewhere.

**2. Hyperparameter tuning**

All four weights swept, 5 points each, all three datasets (community loss is the fixed reference weight). λ_usage∈{0,.03,.1,.3,1}(.1): controls collapse directly — 70-86% unused at 0 (Pancreas .70, Lung .80, Immune .86), rare F1 drops with it (Pancreas .49 vs .62 at paper's value); by .1, collapse is gone (.005/.017/.037) and F1 recovers; 1.0 keeps F1 flat/better (.66/.64/.91) with only a mild modularity cost (largest: Pancreas .65→.56). λ_recon∈{0,.003,.01,.03,.1}(.01): purity/modularity flat across the whole range, all three datasets. α∈{0,.25,.5,1,2}(1): purity/modularity flat everywhere, no sensitivity to this sub-weight. λ_nassoc∈{0,.3,1,3,10}(1): modularity/entropy flat from 0 through the paper's value (Pancreas modularity .65→.62), only trading off once pushed 3-10x past it (down to .50). τ isn't swept — it's calibrated from data after pretraining/init, not hand-set, so a sweep would just re-discover that calibration; the fixed-τ ablation arm above is the control for it.

**3. Harmony, corrected**

The "mixed" framing was a dimensionality bug: Harmony ran at 50-dim PCA vs. scProto's 8-dim latent. Rerun at matched d=8, SEACells/Leiden on top:

| Dataset | Metric | scProto | SEACells (Harmony) | Leiden (Harmony) |
|---|---|---|---|---|
| Pancreas (n=8) | F1 | .54±.21 (ref) | .32±.16, p=.035\* (7/8) | .22±.17, p=.012\* (8/8) |
| Pancreas (n=8) | Homog. | .56±.16 (ref) | .29±.12, p=.012\* (8/8) | .22±.09, p=.012\* (8/8) |
| Lung (n=15) | F1 | .60±.18 (ref) | .36±.14, p=.002\*\* (13/15) | .28±.14, p<.001\*\*\* (13/15) |
| Lung (n=15) | Homog. | .58±.22 (ref) | .40±.12, p=.006\*\* (14/15) | .34±.08, p=.002\*\* (13/15) |
| Immune (n=5) | F1 | .85±.14 (ref) | .75±.16, ns (4/5) | .75±.17, ns (4/5) |
| Immune (n=5) | Homog. | .83±.10 (ref) | .68±.19, ns (5/5) | .64±.13, ns (5/5) |

Immune's non-significance is underpowered (n=5), not reversed (4-5/5 wins). Two reasons: Harmony's clustering step can absorb a sparse rare state into a denser cluster before correction sees it, which scProto's density-corrected affinity graph avoids; and its diversity-rewarding objective [1] can force a batch-exclusive state into a mixed cluster with unrelated cells, which scProto never faces, since each batch's cells reach shared prototypes using only that batch's own affinity graph.

**4. ComBat**

ComBat corrects gene expression directly (not an embedding, unlike Harmony), then we PCA the corrected matrix to d=8:

| Dataset | Metric | scProto | SEACells (ComBat) | Leiden (ComBat) |
|---|---|---|---|---|
| Pancreas (n=8) | F1 | .54±.21 (ref) | .35±.21, p=.039\* (7/8) | .15±.07, p=.020\* (8/8) |
| Pancreas (n=8) | Homog. | .56±.16 (ref) | .29±.08, p=.020\* (8/8) | .23±.10, p=.020\* (8/8) |
| Lung (n=15) | F1 | .60±.18 (ref) | .33±.21, p=.011\* (14/15) | .38±.19, p=.003\*\* (13/15) |
| Lung (n=15) | Homog. | .58±.22 (ref) | .40±.25, p=.017\* (14/15) | .40±.21, p=.013\* (14/15) |
| Immune (n=5) | F1 | .85±.14 (ref) | .86±.06, ns (3/5) | .80±.19, ns (4/5) |
| Immune (n=5) | Homog. | .83±.10 (ref) | .75±.15, ns (4/5) | .75±.18, ns (4/5) |

Same pattern: significant on Pancreas/Lung; Immune underpowered, not reversed (SEACells+ComBat ties scProto on F1, .86 vs .85). ComBat fits one shift/scale per gene per batch, without cell-type labels like every baseline here, so it can't tell a technical effect from a real state concentrated in one batch, and applies that shift uniformly regardless of cell type. scProto has no equivalent global step — a cell's placement comes from the affinity graph, not a batch-wide statistic.

Ref

[1] Korsunsky, I., Millard, N., Fan, J., et al. "Fast, sensitive and accurate integration of single-cell data with Harmony." Nature Methods 16 (2019): 1289–1296.
