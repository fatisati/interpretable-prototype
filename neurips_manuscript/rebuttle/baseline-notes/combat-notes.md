# ComBat limitations — where scProto's mechanism should win

Context: Reviewer nG29 (Question 3) asked for a ComBat batch-correction-then-cluster
baseline (`sc.pp.combat` -> {SEACells, Leiden}), analogous to the Harmony/scPoli-Stage1
baselines already run for Reviewer F5RB. Beyond just running the comparison, two
structural limitations of ComBat are worth calling out explicitly in the rebuttal/paper,
since they point at *why* scProto's affinity-guided mechanism should behave differently
on this data, not just report a number that happens to be better.

## 1. ComBat can't tell "batch artifact" from "real biology confined to one batch"

ComBat estimates each batch's effect on a gene as a shift/rescale of that gene's
per-batch mean and variance. It has no way to distinguish two situations that produce
the identical statistical signature:

- a technical artifact (this batch's protocol reads this gene high), vs.
- a genuinely rare cell state that happens to be sampled mostly/only in this one batch.

Both look like "batch i has an unusual mean for this gene" from ComBat's point of view,
and correcting for the former necessarily also flattens the latter. This is the same
"confounding" failure mode noted in the ComBat-seq paper's own limitations section
(confounded designs are hard for any per-gene linear correction), and it's a known,
documented risk in the batch-integration literature generally (over-correction /
over-integration, Luecken et al. 2022; Büttner et al. 2019 — already cited in our
rebuttal to F5RB).

**Why scProto's mechanism is structurally different here:** scProto's community loss
builds its affinity graph *within each batch separately* first, and only pools cells
across batches through shared prototypes driven by that within-batch affinity signal —
it never needs to compare raw cross-batch statistics to decide what's "batch effect" vs.
"real and rare." A rare population's cells still have each other as within-batch
neighbors (that's a within-batch geometric fact, not a cross-batch statistical
comparison), so nothing forces scProto to first decide "is this difference batch or
biology" the way ComBat's mean/variance correction structurally must.

**How to make this concrete for the rebuttal (not yet run):** identify a rare cell type
that is present in only one (or very few) batches per dataset, and show ComBat-corrected
+ SEACells/Leiden either merges it into a larger population or distorts its expression
profile, vs. scProto's rare coverage/homogeneity numbers on that exact subset. This would
be a sharper, mechanism-level point rather than an aggregate-metric comparison.

## 2. ComBat applies one shift per gene per whole batch — not per cell type/state

ComBat's correction is a single `(shift, spread)` pair per gene per batch, applied
uniformly to every cell in that batch regardless of what cell type or state it's in. But
batch effects in single-cell data are well documented to be *heterogeneous across cell
types* — e.g. a batch's technical shift can hit an immune population much harder than an
epithelial one, or affect low-expressing genes in rare states more than housekeeping
genes in abundant ones. A single global per-gene-per-batch number is too blunt to capture
that: it will over-correct cell types where the true batch effect is smaller than average,
and under-correct ones where it's larger.

**Why scProto's mechanism is structurally different here:** scProto has no single global
per-gene correction step at all. Cell-level assignment comes from a learned encoder plus
the affinity/community objective, so whatever "correction" happens is implicitly
cell/state-specific — a rare, strongly-affected state and a common, mildly-affected one
are not forced through the same scalar adjustment.

**How to make this concrete for the rebuttal (not yet run):** compute, per cell type,
how much ComBat's fitted per-gene shift actually "should" have differed (e.g. compare
per-cell-type batch means before/after correction, per gene) to show real heterogeneity
in batch-effect magnitude across cell types exists in these datasets — this is the
empirical evidence that a single global shift is mis-specified, independent of whatever
the final SEACells/Leiden-on-ComBat numbers turn out to be.

## Status

Both points are analysis/framing ideas, not yet run as experiments. The ComBat ->
{SEACells, Leiden} baseline itself is being run via
`notebooks/combat_then_cluster_baselines.ipynb`. These two limitations are candidates for
the rebuttal text explaining *why* scProto might come out ahead, not just reporting that
it does.
