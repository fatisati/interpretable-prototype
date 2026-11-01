---
id:             D65
title:          Cross-batch signature alignment -- the first loss term that evaluates a cross-batch pair
status:         proposed
date_opened:    2026-11-01
date_closed:
supersedes:
superseded_by:
tags:           [cross-batch, chimeras, cider, ider, de-signature, contrastive, ema-memory]
datasets:       [pbmc-immune]
protocol:       NH-heldout, (cell type x batch)-stratified split arbf_ctb20_seed0, seed 31
comparable_to:  d60_leid_noc (same recipe without this term), d60_epoch0_leid
                (its own stage-1 init), d59_leiden_ctb (Leiden on the stage-1 latent).
                NOT comparable to anything at a different K.
runs:           [d65_diag-mse, d65_both-mse, d65_diag-nb, d65_both-nb]
lit:            [hu2021cider (read from source), xie2018mstn (verify),
                 jiang2020implicit (verify), kang2019can (verify)]
paper:          the cross-batch limitation section; promoted to a method
                contribution only if the primary bars pass
K:              300
code:           0c54d95 (term) + this ADR
cost:           1 probe per likelihood (~$0.5 each, T4) + 4 arms (~$1.5 each, T4)
---

# D65 — Cross-batch signature alignment

## Summary

> In the context of **stage-2 training measurably destroying the cross-batch
> alignment its own initialization had** (label-free gene-space coherence
> r_cross 0.702 -> 0.603, chimeric prototypes 0.0% -> 9.9%, batch-vote AGREE%
> 83.1 -> 66.0), facing that **no term in the loss has ever evaluated a
> cross-batch pair**, we decide for **a CIDER-style DE-signature alignment
> term with an EMA memory, prototypes detached, added at a 1/8 gradient share**
> and against **proto_recon in any form (closed by D61+D62+D63), explicit
> batch-mixing terms (forbidden: single-batch rare types), and a push-to-zero
> off-diagonal (would fight real biology)**, to achieve **r_cross restored to
> ~0.69 and chimeras near 0% with held-out cohesion not regressing**,
> accepting **one dataset, one seed per arm, a term that is blind to
> consistent merging, and a measure that becomes the training signal and so
> cannot be the headline evidence**.

## Context and Problem Statement

Measured 2026-10-31/11-01. Every arm against its OWN stage-1 init
(`d60_epoch0_leid` = stage-1 CVAE latent + the same Leiden prototype init,
argmax, zero stage-2 training) -- the comparison the earlier ADRs never made:

```
                        stage-1   protorecon  protorec   protorec   floor
                         only        OFF      +CVAE      no CVAE
  coh_aff_test_all      0.0784     0.0853     0.0773     0.0784    0.0012
  coh_aff_test_rare     0.1262     0.1376     0.1319     0.1217    0.0101
  coh_aff_TRAIN_all     0.0806     0.1310     0.1305     0.1289    0.0016
  AGREE% (>=10 cells)    83.1%      66.0%      61.8%      65.2%    ~3 pt
  r_cross (gene space)   0.702      0.603      0.581      0.578      --
  chimeras (r_cross<0.3)  0.0%       9.9%      13.4%       9.4%      --
```

Three readings, all of them load-bearing:

1. **Training buys TRAIN cohesion and loses cross-batch.** coh_aff_train rises
   0.0806 -> 0.1310 (31x the noise floor) while r_cross falls 0.702 -> 0.603
   and chimeras go 0% -> 9.9%. Held-out cohesion barely moves (0.0784 ->
   0.0853, 5.8x floor for the best arm).
2. **The base recipe does this, not proto_recon.** `d60_leid_noc` has no
   proto_recon at all and still loses 0.099 of r_cross and creates 9.9%
   chimeras. proto_recon only made it slightly worse.
3. **Nothing in the loss defends against it.** Cohesion is scored inside each
   bio-batch's own node set `V[b]` BY DESIGN (CLAUDE.md 9d: a cross-batch zero
   is mostly batch effect, not measured dissimilarity, so it is excluded from
   both numerator and denominator). GM-KL is batch-agnostic. CVAE-recon is
   per-cell. **No quantity in the objective changes value when a prototype
   becomes a chimera.**

### Why proto_recon could never have fixed it

Structural, not empirical: `|| x[i] - decode(prototype_k) ||^2` never mentions
batch. It asks "does k reconstruct you?", never "does k mean the same thing in
batch 1 and batch 2?". The measured confirmation, across three ADRs:

```
scores LIVE      CAN move assignments -- but [D62] measured its target is the
                 group MEAN EXPRESSION, and those targets correlate 0.72-0.75
                 BETWEEN DIFFERENT prototypes (0.11-0.16 after background
                 subtraction). The gradient is dominated by a shared component,
                 so it merges rather than aligns.
scores DETACHED  gradient reaches prototypes + decoder ONLY. Assignments are
                 exactly what a chimera IS. [D63]'s P3 failure was guaranteed
                 in advance by the fix that solved its P1.
```

Tested at 5 doses spanning 100x, both forms (`agg`/`mix3`), both routings, and
recon on/off. `lambda_proto_recon = 0` is closed.

### Why NB would not have fixed it either

NB changes the LIKELIHOOD, not the comparison structure -- still per-cell
reconstruction, still never a cross-batch pair. Its stated motivation is also
refuted on this dataset ([D62] TEST 2): rare types average 1,389 counts / **505
genes detected**, common types 25,685 counts / **412 genes**. Rare types have
MORE genes detected; count depth tracks protocol, not rarity. The mse/nb axis
is kept in this ADR because it changes the latent space the assignments live
in, not because it is expected to supply a cross-batch force.

## Decision

### The measure

CIDER / IDER (Hu, Ahmed & Yau, *Genome Biology* 22:337, 2021). A group is
characterised by **how it differs from its own batch**, which cancels the batch
effect -- their Fig 1B model is `expression = cluster + batch + patient` with
batch constant within a batch.

```
sig[k][b] = (soft-assignment-weighted mean lognorm of k's batch-b members)
            - (mean lognorm over ALL batch-b cells)
r(k,b1 ; j,b2) = pearson over genes of sig[k][b1], sig[j][b2]
```

Measured on pbmc-immune, this separates cleanly:

```
signature   contrast                n pairs  median r  10th pct  75th pct
plain       SAME type/diff batch        74     0.842     0.388     0.913
plain       diff type/diff batch       928    -0.019    -0.416     0.121
            --> separation                     0.860
```

**Read from the R source** (`zhiyuan-hu-lab/CIDER`, `R/getIDER.R`), not the
prose: it normalizes with edgeR `logCPM(prior.count=3)`, fits per-gene
ORDINARY least squares via `.lm.fit` on `~ 0 + group + batch + detrate`, and
correlates the two contrast coefficients. **There is no empirical Bayes and no
variance moderation anywhere in it.** With no covariates an OLS coefficient IS
the difference of means, so our batch-mean subtraction is the same estimator;
CIDER's batch covariate performs exactly the per-batch centering we do
directly. CIDER uses IDER to merge clusters and to evaluate integration --
**never as a training loss.**

### The objective

```
DIAGONAL (pull), MAXIMIZE:
    reward = sum over k, over batch pairs b1<b2 of
                 min(size[k][b1], size[k][b2]) * r(k,b1 ; k,b2)
    loss_diag = - reward / n_cells            <- CONSTANT denominator

OFF-DIAGONAL (push), MINIMIZE:
    loss_off = mean over k != j, b1 != b2 of  max(0, r(k,b1 ; j,b2) - margin)
```

### Every design choice, with the measurement behind it

| choice | why, measured |
|---|---|
| **reward, not penalty** | `sum(1 - r)` charges per batch pair, so abandoning a batch LOWERS the cost. Measured: dropping one batch costs only **17.8%** of a prototype's cells at a >=10-cell threshold -- the escape hatch is cheap and the model would go batch-pure, the opposite of integration. A reward only ever ADDS positive terms for spanning. |
| **min(size,size) weight** | a pair is only as reliable as its weaker side. Measured split-half reliability of `sig` (same cells, same batch, same prototype, disjoint halves): **0.34 at 3 cells, 0.47 at 5, 0.65 at 10, 0.81 at 20, 0.93 at 40**. This is the differentiable stand-in for CIDER's own fix, `downsampling.size = 40`. |
| **weight DETACHED** | live, the size path contributes `d(reward)/d(size) = r ~ 0.7` as a CONSTANT pull to join any coherent spanning prototype regardless of fit, while the r path -- the part that checks whether the cell belongs -- decays as `1/size`. For large groups it degenerates into a pure capacity charge, structurally the defect CLAUDE.md 9b identified in nassoc/CPM ("a function of the assignment COLUMN SUMS only, so it cannot move a cell for SIMILARITY reasons"). |
| **constant denominator `n_cells`** | dividing by `sum(weights)` makes it a weighted AVERAGE, which can be raised by deleting a below-average member -- that reopens the escape hatch. |
| **EMA memory, load-bearing** | a `(prototype, batch)` group gets a **MEDIAN OF 2 CELLS** in one minibatch (measured; only 194 of 1,384 groups reach 10 cells), where reliability is below 0.34. The per-step estimate alone is noise. Half-life **DERIVED** from the group revisit interval `33,506 / 6,700 ~ 5 steps` (CLAUDE.md 9d: never a constant; the old `coh_mem_hlife=500` was a measured no-op). Memory half is detached -- gradient flows only through the current step's cells. Mechanism and justification are MSTN's (Xie et al., ICML 2018), introduced there to "compensate for the insufficient categorical information within each mini-batch". |
| **hinge off-diagonal, never push-to-zero** | with K=300 and ~16 cell types, roughly **19 prototypes share each cell type**, so most "different prototype" negatives are the SAME cell type and genuinely correlate. Measured: different-type median `r = -0.019` but 75th pct `+0.12`, and the paper reports CD4/CD8 T and the monocyte subsets sharing high pairwise similarity. [D62] already concluded that a term forcing a wrong MERGE is bad (Oetjen's Mk-progs); the mirror risk is a term forcing a wrong SPLIT. |
| **prototypes DETACHED** (`xbatch_proto_sg=1`) | [D63]'s lesson: proto_recon's lambda was set from the ENCODER gradient, which is structurally zero for a detached-score term, so its PROTOTYPE-path pull ran 5.7-18.6x GM-KL's and collapsed the prototypes (median nearest-prototype cosine 0.81 -> 0.9997). Detaching means ONE budget to calibrate. Asserted in the probe via `gnorm_proto == 0`. |
| **lognorm, never counts** | asserted at setup. Under `recon_loss='nb'` the dataset swaps `adata.X` to raw counts, and a signature on counts is dominated by library size -- the size factor is not biology, the relative gene ratios are. scPoli's own NB head agrees: it is `Linear -> Softmax OVER GENES`, a composition summing to 1, with the size factor applied afterwards. Consequence: **this term is bit-identical under mse and nb.** |

### Measured and REJECTED (none of these are in the code)

```
CDR residualization (CIDER's detection-rate covariate)
    variance explained by CDR: 10X 6.89%, Sun 2.66%, Oetjen 1.86%,
                               Freytag 1.79%, Villani 1.29%
    ceiling 0.842 -> 0.820.  Rejected: costs signal, explains almost nothing.

Variance-trend gene weighting (limma-trend's idea, applied to the correlation)
    ceiling full size      0.842 -> 0.726
    ceiling at m=10        0.554 -> 0.259
    stage-1 prototypes     0.694 -> 0.303, chimeras 0.0% -> 48.8%
    Rejected: weighting by 1/variance downweights the HIGH-variance genes where
    the DE signal lives. In limma that ratio builds a t-statistic for
    SIGNIFICANCE TESTING; for correlating two signatures it strips the markers.

CIDER's gene filter (drop MT-, ERCC-, genes detected in <=5 cells)
    the 4,000-gene HVG set contains 0 MT- and 0 ERCC- genes. Nothing to remove.

edgeR logCPM(prior.count=3) instead of scanpy log1p
    NOT TESTED -- deliberately deferred. prior.count is a low-count damper and
    the HVG selection already removed the low-count genes. Shelved: revisit
    only if the arms show the ceiling is noise-limited rather than
    objective-limited.
```

## Arms

```
                  MSE              NB
diag only      d65_diag-mse     d65_diag-nb
diag + off     d65_both-mse     d65_both-nb
```

All: `d60_leid_noc` recipe + the new term; leiden init; cohesion gram+aff with
prototypes detached; GM-KL; community OFF; proto_recon OFF; K=300; 10 epochs;
`freeze_batch_embedding=True`; `umap_steps_per_epoch=None`; `early_stop=False`;
`batch_size=1024`; seed 31; same split.

**Shares** (user decision, 2026-11-01): cross-batch takes 1/8; the other four
keep their 1:1:2:2 ratio, renormalized to 1.

```
arm DIAG                      arm BOTH
  cohesion-gram  0.1458         cohesion-gram  0.1296
  cohesion-aff   0.1458         cohesion-aff   0.1296
  CVAE-recon     0.2917         CVAE-recon     0.2593
  GM-KL          0.2917         GM-KL          0.2593
  xbatch-diag    0.1250         xbatch-diag    0.1111
                                xbatch-off     0.1111
```

Each `lambda = share / measured gnorm_enc` from one probe pass at temporary
lambda = 1 for EVERY term including cohesion (9d-1's trap: old probe files
measured cohesion at its deployed lambda).

**KNOWN CONFOUND, stated not hidden.** Arm BOTH holds the diagonal share
constant and ADDS the off-diagonal, so DIAG vs BOTH isolates the off-diagonal's
effect -- but BOTH then carries more total new force and dilutes the other four
slightly more. This is exactly the dilution confound [D62] was built to
separate. The alternative (splitting 1/8 in half) would change the diagonal's
strength between arms, which is worse for the question being asked.

## Pre-registered expectations

```
PRIMARY -- what the term is FOR
  P1  r_cross >= 0.69                        (d60_leid_noc 0.603; ceiling ~0.70)
  P2  chimeras (r_cross < 0.3) <= 3%         (d60_leid_noc 9.9%; stage-1 0.0%)
  P3  AGREE% (>=10 cells) > 69.0             (d60_leid_noc 66.0; floor ~3 pts)

HARD BARS -- a regression here kills the term, as it killed D61/D62/D63
  H1  coh_aff_test_all >= 0.0853             (floor 0.0012)
  H2  pooled_test_all  >= 0.8160             (floor 0.0058)
  H3  n_dead = 0,  median_nn_cos >= 0.75,  frac_pairs>0.99 = 0%

ARM-BOTH-SPECIFIC
  B1  if median_nn_cos or n_dead degrade vs arm DIAG, the off-diagonal is
      doing harm -> drop it, keep DIAG.

RARE metrics are reported against the 6a noise floor (rare_f1 0.082,
within_test_rare 0.061, pooled_test_rare 0.056). NO rare ranking below those.
```

**Honest expectation: RECOVERY, not a new win.** The size-matched cell-type
ceiling at prototype group sizes is **r = 0.704** (cell types subsampled to 30
cells, median r; 0.554 at 10 cells), and the untrained stage-1 init already
sits at **0.694**. At the group sizes we actually have, an untrained prototype
is already as cross-batch-coherent as a real cell type. So the realistic job of
this term is to recover the ~0.09 that stage-2 destroys, not to exceed stage 1.

**What the term CANNOT do, asserted in the unit tests:** it rewards
CONSISTENCY of composition, not PURITY. A prototype holding the same 50/50
mixture of two cell types in every batch scores `r = 1`, exactly like a pure
one. Nothing in `r` measures within-prototype homogeneity. Cohesion -- scored
per bio-batch, so it sees the fusion in every batch -- is the counterweight,
and it is being diluted from 16.7% to 14.6% of the budget to fund this. H2
(`pooled_test_all`) is the direct detector and is a hard bar for that reason.

## Evaluation honesty

`r_cross` **becomes the training signal for these arms**, so it cannot be the
headline evidence -- a reviewer would be right to reject that. Therefore:

- lead the claim with **AGREE%** and **cross_test_all**, which are label/purity
  based and are NOT what the loss optimizes;
- report `r_cross` explicitly as a training diagnostic;
- the in-run `r_cross` printed by `train_d65_xbatch.py` is labelled as such.

A cleaner option exists and is NOT yet implemented: train the loss on a random
half of the 4,000 genes and evaluate `r_cross` on the held-out half. One line;
worth adding if the arms pass and the result goes in the paper.

## Verification built into the run (not left to discipline)

```
stale-code guard   inspect.getsource(scproto) must contain the D65 marker,
                   asserted BEFORE anything expensive. [D61] lost 6 arms
                   (~$6 of GPU) to a volume upload that silently went nowhere.
model-name guard   "lxb" (and "lxbof" on BOTH, "rlss-nb" on NB) must appear in
                   get_model_name(). CLAUDE.md 9 step 3 as an assert, not a
                   printout -- [D61]'s probe printed the name and nobody read it.
routing guard      the probe FAILS if cohesion/xbatch put nonzero gradient on
                   the prototypes.
9b range check     |r| <= 1 and |loss_diag| <= (n_batches-1)/2 asserted on
                   step 1, with all term values printed side by side.
zero guard         validate() asserts nassoc/modularity/usage/swav/kl/umap/
                   proto_recon are all 0 -- LAMBDA_PROTO_UMAP_PRECON turns
                   several of them ON.
unit tests         interpretable_ssl/tests/test_xbatch_signature.py, 9 tests:
                   signature vs brute force, correlation vs numpy, the
                   background-inclusion rescale, the bound, chimera detection,
                   consistency-not-purity, gradient reaches assignments,
                   weight-detach changes the gradient, and -- the design check
                   -- abandoning a batch LOSES reward while the penalty form
                   would have gained.
```

## Consequences

**Good.** The first term that can act on the measured defect. Label-free, so it
is immune to the annotation disagreement [D62] found (Oetjen's Megakaryocyte
progenitors are r = 0.18-0.21 against the other batches' and resemble HSPCs --
it would push to SPLIT them, which [D62] established is correct). Bounded in
`[-1, 1]`, so no vanishing-denominator blowup of the `nassoc` kind.

**Bad.** Blind to consistent merging. Costs cohesion 2.1 percentage points of
budget. Contaminates `r_cross` as an evaluation metric. Adds ~5 matmuls of
(K x n_batch) @ (n_batch x n_genes) per step. The off-diagonal risks fighting
genuine biology among same-type prototypes, and its margin (0.5) is chosen from
the cell-type null, NOT from a measured PROTOTYPE-pair null -- that measurement
is still outstanding and should be run before trusting arm BOTH.

**Neglected.** The free-logit ceiling (optimize assignments directly, no
encoder, to separate "is the objective right" from "can the encoder reach it")
-- designed, not run. A contrastive/InfoNCE formulation of the same matrix,
which would be self-normalizing and need no margin, rejected for now because
~19 prototypes per cell type makes the false-negative rate severe. Multi-seed
replication, which 6a says is required before ranking on any rare metric.

## Verdict

NOT YET RUN.
