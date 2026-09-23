# Is "scProto should capture niche from expression alone" a valid expectation?

Date: 2026-09-16. Dataset: `s28nsc` (58,423 cells x 960-gene panel, `celltypes` x `niches_2D`).
Script: `scripts/niche_probe_auc.py` -> `tables/s28nsc_niche_probe_auc.csv` (99 pairs, 259 s, CPU only).

## Method

Standard **linear probing / linear evaluation protocol**. Freeze the features, train a
simple classifier on top, report held-out score. Measures information that is *usable by a
bounded decoder*, not Shannon MI (which is invariant to invertible transforms and so says
nothing about whether a model can extract it).

    Alain & Bengio 2016, arXiv:1610.01644   linear classifier probes
    Hewitt & Liang 2019, EMNLP              control tasks; report selectivity
    Voita & Titov 2020, EMNLP               MDL probing (probe-capacity confound)
    Xu et al. 2020, ICLR                    V-information / usable information

    for each cell_type:
      for each niche in that cell_type (excluding 'Excluded'):
          y           = (niche_of_cell == this_niche)   one-vs-rest, WITHIN cell type
          auc         = 5-fold CV ROC-AUC, L2 logistic regression, class_weight=balanced
          auc_ctrl    = same with y permuted           (Hewitt & Liang control task)
          selectivity = auc - auc_ctrl

Within cell type, not joint `niche x celltype`: a joint probe scores high by predicting
cell type (the dominant axis) and would say nothing about niches.

Feature sets: `expr960` (full panel, exactly what the encoder receives -- this dataset has
no HVG subsetting, `var` has no `highly_variable` column), `pca50` (`X_pca[:, :50]`), and
`pca50` + HistGradientBoosting as a nonlinear capacity rung.

## Probe validity

Control AUC on permuted labels sits at chance, so the probes are not fitting noise:

    auc_expr960_ctrl      mean 0.494   sd 0.050
    auc_pca50_ctrl        mean 0.489   sd 0.043
    auc_pca50_gbm_ctrl    mean 0.490   sd 0.037

sd ~0.04-0.05 => roughly AUC > 0.60 is the meaningful-signal threshold here. The control is
a single permutation draw per pair, so per-pair control values are noisy; the mean is the
reliable quantity.

## Result 1 -- the information IS in gene expression

    auc_expr960     mean 0.656   sd 0.093
    auc_pca50       mean 0.715   sd 0.101
    auc_pca50_gbm   mean 0.695   sd 0.098
    best-of-probes  mean 0.727   median 0.727

    best AUC >= 0.60 :  91/99 pairs  (92%)
    best AUC >= 0.65 :  72/99  (73%)
    best AUC >= 0.70 :  61/99  (62%)
    best AUC >= 0.80 :  20/99  (20%)

**The expectation is valid.** For 92% of (cell type, niche) pairs the niche is decodable
from expression alone, within cell type. Asking the encoder to find these is not asking for
something that isn't there.

Note `pca50` BEATS `expr960` (0.715 vs 0.656). 960 correlated features with L2 and a few
hundred positives overfits; PCA denoises. So "probe the raw matrix" is a necessary check but
is not automatically the stronger probe -- report best-of-probes.

## Result 2 -- DGE is CONSERVATIVE here, not liberal (this reverses the prior expectation)

Rank agreement between DGE and decodability is substantial:

    spearman(n_dge_genes,   best AUC) = 0.674
    spearman(max_abs_logfc, best AUC) = 0.720
    pearson (n_dge_genes,   best AUC) = 0.379     (monotone but not linear)

2x2, `n_dge_genes >= 20` vs `best AUC >= 0.65`:

                      not decodable   decodable
    no/few DE genes        26            33
    many DE genes           1            39

- **Discordant A (many DE genes, not decodable) = 1 of 99.** The "significance inflates
  with sqrt(n_cells) while AUC does not" failure mode is real but RARE in this data -- and
  it appears exactly where predicted, at the largest group: `Tumor cells / Tumor surface`,
  n_pos = 2855, 270 DE genes, AUC 0.624.
- **Discordant B (few/no DE genes, yet decodable) = 33 of 99**, including 9 pairs with
  ZERO significant DE genes that still reach AUC 0.67-0.77.

Cause: per-gene Wilcoxon + BH at n_pos ~30-200 is underpowered, while the multivariate
probe pools weak evidence across all 960 genes. Pooling gain over independent genes:

    effect_size_pooled = sqrt( sum over genes of effect_size[g]^2 )

So on this dataset the DGE gene count UNDERCOUNTS how many niches are actually recoverable.
Using "has DE genes" as the feasibility screen is safe but leaves ~1/3 of recoverable pairs
off the list.

## Result 3 -- the signal survives the encoder and dies in the quantization

From the existing probes (`sep2.py`, `sep3.py`), mean balanced accuracy, chance 0.162:

    expression_PCA50      lift +0.155
    spatial_xy            lift +0.138    <- expression carries MORE niche info than xy
    scProto_z_8d          lift +0.141    ~91% of PCA50, in 8 dims; beats dim-matched PCA8
                                            in 10/15 cell types
    soft_assign_200d      lift +0.084    retains 60% of z
    hard_metacell_onehot  lift +0.088    retains 62% of z

The encoder is not the lossy step. `z -> prototype assignment` discards ~40% of the niche
signal that `z` holds. That points at K / `epsilon` / `nassoc_gamma`, not at the encoder's
loss weighting.

## What this licenses, and what it does not

Licensed: "niche identity is decodable from expression alone in 92% of pairs, so an
expression-only encoder is being asked for something that exists."

NOT licensed by this table alone: that a given unrecovered niche is the loss's fault. A
probe is supervised and sees the niche labels; scProto is unsupervised and sees only the
affinity. Probe AUC is an UPPER BOUND on what scProto could do, not a target it must hit.
The remaining condition is whether the affinity marks the niche -- testable by whether
SEACells on that same affinity recovers it.

Caveats not addressed here: segmentation spillover / ambient RNA (a cell's counts contain
its neighbours', so decodability can be real without any cell-state change) and nuisance
proxies for position (library size, density). Both threaten the biological INTERPRETATION of
a success; neither affects the attribution of a FAILURE.
