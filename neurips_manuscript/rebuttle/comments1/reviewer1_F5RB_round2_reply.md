We thank the reviewer for the follow-up and for confirming the Stage 1/Stage 2 clarification resolves the ambiguity. Below: the scVI/Harmony/BBKNN comparisons now complete, why the held-out-edge test is already independent, and where the spatial analysis stands.

**1. scVI, Harmony, and BBKNN baselines, now complete**

All three now run at scProto's own latent dimension (d=8; Harmony/BBKNN previously used d=50, ~6x more room), with SEACells on top — the two-step baseline requested in Question 1 — plus ComBat, as Reviewer nG29 asked. scVI is also loss-matched (Gaussian likelihood, to mirror scProto's MSE reconstruction), not run with its ZINB default. Rare-cell F1 (macro) and homogeneity, mean±std, paired one-sided Wilcoxon vs. scProto, Bonferroni-corrected per dataset per metric (wins in parentheses):

| Dataset | Metric | scProto | +Harmony | +scVI | +BBKNN | +ComBat |
|---|---|---|---|---|---|---|
| Pancreas (n=8) | F1 | 0.54±.21 (ref) | 0.32±.16, p=.035\* (7/8) | 0.18±.31, p=.039\* (7/8) | 0.18±.22, p=.0195\* (8/8) | 0.35±.21, p=.039\* (7/8) |
| Pancreas (n=8) | Homog. | 0.56±.16 (ref) | 0.29±.12, p=.012\* (8/8) | 0.19±.30, p=.039\* (7/8) | 0.20±.07, p=.0195\* (8/8) | 0.29±.08, p=.020\* (8/8) |
| Lung (n=15) | F1 | 0.60±.18 (ref) | 0.36±.14, p=.0023\*\* (13/15) | 0.35±.20, p<.001\*\*\* (15/15) | 0.39±.22, p<.01\*\* (13/15) | 0.33±.21, p=.011\* (14/15) |
| Lung (n=15) | Homog. | 0.58±.22 (ref) | 0.40±.12, p=.006\*\* (14/15) | 0.39±.20, p<.001\*\*\* (14/15) | 0.40±.16, p<.01\*\* (13/15) | 0.40±.25, p=.017\* (14/15) |
| Immune (n=5) | F1 | 0.85±.14 (ref) | 0.75±.16, ns (4/5) | 0.62±.18, ns (5/5) | 0.61±.21, ns (4/5) | 0.86±.06, ns (3/5) |
| Immune (n=5) | Homog. | 0.83±.10 (ref) | 0.68±.19, ns (5/5) | 0.57±.13, ns (5/5) | 0.55±.22, ns (5/5) | 0.75±.15, ns (4/5) |

All four independent correction mechanisms show the same pattern at matched dimension: scProto's advantage is significant on Pancreas and Lung, both metrics, against all four. Immune is non-significant only because n=5 batches is underpowered for Bonferroni; win-counts (3-5/5) still favor scProto, never reverse.

The reason is consistent across all four: each leans on a batch- or cluster-wide correction statistic that can't distinguish a real sparse population from a technical outlier the way a rare state's own cell-cell affinities can (mechanism specifics for Harmony and ComBat in our reply to Reviewer nG29, §3–§4). scProto has no such global step — a cell's placement is conditioned on its own local, per-batch affinity graph, so a rare state's signal isn't averaged away before scProto sees it.

**2. Held-out-edge independence**

Fair point. scProto and SEACells are both fit on the visible 80% of each affinity graph's edges; reported modularity comes only from the hidden 20%, which enters neither method's loss nor its stopping criterion. scProto's encoder maps each cell to a latent position from its own gene expression alone, with no access to the affinity graph at inference, so a held-out edge is scored from two embeddings computed independently of which edges were masked, not looked up through neighboring edges. scProto's held-out modularity stays close to its headline number and ahead of every baseline, including Leiden. We mask edges rather than holding out whole cells or batches because SEACells has no encoder — its kernel and archetypal fit are defined only over the cells present when it is built, with no way to place a cell or batch it never saw — so edge-masking within a shared, fully-observed node set is the version of this test both methods can actually be scored on. We're exploring whether an independent-graph version of this test is feasible, and will share results in this thread if it comes together during the discussion period.

**3. Spatial claim**

We're actively running a clearer experiment to test discovery of niche-correlated transcriptional programs, and will share results in this thread if it completes during the discussion period.

