# Task: Batch correction + clustering does not preserve rare cell types

## Claim

The standard pipeline for integrating multi-batch single-cell data —
*batch correction* (removing technical differences between
experiments/datasets so cells can be compared across them) followed by
*clustering* (grouping cells into discrete groups via Leiden community
detection — run directly on BBKNN's corrected graph, or on a neighbor
graph built from the corrected embedding for methods that output an
embedding instead of a graph, like Harmony, ComBat, and scVI; Leiden doesn't take a target cluster
count directly the way k-means would, and its resolution parameter
doesn't map cleanly onto cluster count — the count can stay flat across a
range of resolution values, then jump — so rather than solving for an
exact count, we sweep a small range of resolution values and pick
whichever gets closest to the desired number of clusters) — does not
preserve rare cell types. A rare cell state gets mixed into a denser,
more common one during this process. For example, an early-cancerous
population marked by one specific gene can get absorbed into a healthy,
abundant population, because the correction step never had a reason to
treat that marker as signal rather than noise.

A second, related claim: **SEACells** (an existing *metacell* method — it
groups similar cells into small aggregates called metacells to reduce
noise) works well with a **single batch**, but in a **multi-batch** setting
it fails to form a metacell for a rare cell type, because no single batch
has enough cells of that type to form one on its own.

The point of scProto is to give the model explicit *guidance* (the
affinity graph) so it keeps finding a rare pattern even when the
reconstruction loss alone would never surface it. This task is the
evidence step: show the gap is real and measurable in existing methods
before comparing scProto against it.

## Why this should happen, per baseline

Each method fails for a different mechanical reason. Listing them
separately means each baseline's result tests its own specific
sub-claim, instead of one vague claim about "batch correction" in general.

- **ComBat** — a linear, additive correction: it estimates one shift/scale
  factor per batch (via empirical Bayes — a statistical method that borrows
  strength across genes to get a stable estimate even with limited data)
  and applies it to every cell in that batch. With only a handful of
  rare-type cells per batch, there isn't enough data to estimate a
  correction specific to them, so they silently inherit the factor
  estimated from the abundant types around them.

- **Harmony** — iterative soft-clustering combined with linear correction:
  at each iteration it clusters cells, then explicitly rewards clusters
  that mix batches evenly (a "diversity" objective), then corrects within
  each cluster. Because the clustering step is density-based, a handful of
  rare cells can get pulled into a bigger, denser cluster just to satisfy
  the diversity objective, before any rare-type-specific correction is
  computed.

- **BBKNN** — doesn't correct expression values at all; it only rebuilds
  the cell-to-cell neighbor graph so each cell's neighbors are drawn evenly
  from every batch. If a batch has fewer rare-type cells than the neighbor
  count it's asking for, the "equal per batch" rule forces in neighbors
  that aren't actually the same type — corrupting the graph exactly at the
  rare cell's position, before clustering even runs.

- **scVI (cVAE-based)** — trains a shared latent space using only a
  reconstruction loss (plus a KL term, a regularizer pulling every cell's
  code toward a shared, simple distribution), averaged over *all* cells.
  Since abundant types dominate that average, the gradient signal shaping
  the latent space is mostly theirs; a rare marker gene has little
  influence on what structure gets encoded, so the model has no reason to
  carve out separate latent structure for it, and the KL pull further
  shrinks any rare representation toward the bulk. This is the specific
  weakness scProto's affinity guidance is meant to fix.

- **SEACells (multi-batch)** — a different failure mode, not a correction
  method: it builds metacells from local structure *within* a single
  batch. If a rare type is rare in every batch individually, no single
  batch ever has enough of its cells to form a metacell there — the
  problem isn't SEACells' method, it's that cross-batch pooling (which it
  doesn't do) is what the rare type actually needs.

Root cause shared across all of them: a rare cell type needs enough cells
*pooled across batches* before its structure becomes statistically
visible, and none of these methods do that pooling before making their
correction/clustering decisions.

## Rare cell type definition

Defined **per batch**, not globally. For a given batch, compute each cell
type's frequency *within that batch only* (its share of the batch's cells).
The same cell type can be rare in one batch and common in another — rarity
is a property of (cell type, batch), not of the cell type alone.

**Default rule — "log_std":** a cell type is rare in a batch if
log10(its frequency) is more than one standard deviation below the mean
log10 frequency of all cell types in that batch. The log scale stops one or
two huge types from stretching the spread so much that nothing counts as
rare. Fallback, if a batch has fewer than 4 cell types (too few for
"standard deviation" to be meaningful): the lowest quartile of frequencies.

Two other rules are kept only as what-if checks, never the headline result:
"std" (the same test without the log — finds almost nothing on uneven
batches: 12 of 14 heart batches had no rare type) and "fixed" (frequency
below 1% of the batch). Why log_std, with the numbers:
`docs/decisions/0002-rare-cell-type-rule.md`.

## Metrics

Two kinds of signal, because they catch different failures:
**assignment metrics** (did the final cluster label come out right) and
**cohesion** (did the method's own notion of similarity still agree with
the ground-truth affinity graph, independent of where the clustering
boundary ended up). A method can score fine on one and fail the other.

**Assignment metrics:**

- **Cell type purity** — global, no batch split. For cluster *m*, the
  fraction belonging to its majority type, counting every cell in it
  regardless of batch; averaged across all clusters (one mean ± std,
  dataset-wide). The standard, type-agnostic reference point.

      purity(m) = (1/|m|) * max_c |{i in m : type(i) = c}|

- **Agreement** — per cell *i*, the fraction of a chosen comparison
  group *within i's own cluster* that shares *i*'s type:

      agree(i, group) = (1/|group|) * |{j in group : type(j) = type(i)}|

  Three choices of `group`, all drawn from i's cluster:
  - **global** — every cell in the cluster (includes *i*).
  - **within-batch** — only cells from i's *same* batch (still includes *i*).
  - **cross-batch** — only cells from *other* batches (excludes *i* by
    construction; undefined if there are none — such cells are dropped
    from every cross-batch average below, with the dropped fraction
    reported as its own coverage number: low = clusters rarely mix
    batches).

  Each of the three (global / within-batch / cross-batch) is reported
  4 ways — which population of cells, crossed with how the result is
  summarized:

  | | all cells | rare cells only |
  |---|---|---|
  | **one number, whole dataset** | mean ± std | mean ± std |
  | **one number per batch** | B pairs of mean ± std | B pairs of mean ± std |

  (B = number of batches.) The within/cross split tells apart two
  failure modes global agreement can't: high within-batch + low
  cross-batch means batches each cluster fine alone but don't actually
  agree with each other (an un-integrated batch effect); both high means
  real integration.

- **Coverage** — per batch: fraction of that batch's locally-rare types
  with at least one cell in a cluster whose majority label is that type.
  A binary per-type check (present or not), not a ratio — catches a type
  dropped entirely, which agreement (always some number, never literally
  "absent") can't flag as cleanly.

- **Rare precision / recall / F1** — the fragmentation question (is
  this type scattered across many clusters) that nothing above measures.

  Step 1, done once, globally: every cluster gets exactly one label —
  its own majority type, counting every cell in it, any batch. Fixed,
  reused for every rare type and every batch below; never recomputed
  per batch.

  Step 2, for each (rare type *T*, batch *b*) pair in the rare list —
  using only batch *b*'s own cells, checked against the fixed cluster
  labels from step 1:
  - **TP** — batch *b*'s type-*T* cells sitting in a cluster labeled *T*.
  - **FN** — batch *b*'s type-*T* cells sitting in a cluster labeled
    something else (scattered, missed).
  - **FP** — batch *b*'s non-type-*T* cells sitting in a cluster labeled *T*
    (other types showing up where *T* should be).
  - precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = harmonic mean of
    the two.

  Report mean ± std of F1 over the full list of (rare type, batch)
  pairs. Each pair is its own independent measurement — not a global,
  per-type value reused across whichever batches happen to share that
  type — so this mean ± std is meaningful, not an artifact of overlap
  between batches' rare-type lists.

**Cohesion metrics** — how well a method's clusters agree with the
original RBF affinity graph's notion of "these cells belong together":

    cohesion(i) = average of sim(i, j) over every j in i's cluster
                  (with sim(i, i) := 1)

Tells us whether the affinity-defined similarity survives clustering, how
well it's preserved, and — via the frequency-dependence check below —
whether that preservation gets worse as frequency drops.

- **Cohesion** — the metric above, averaged over all cells.
- **Rare cell type cohesion** — the same, averaged only over locally-rare
  cells (same population used for the rare agreement metrics above).

**Frequency-dependence check.** In addition to the binary rare/common
split above, we check whether every metric above moves *continuously*
with a cell type's frequency within its batch, instead of only comparing
two buckets. A real gap should show scores declining smoothly as
frequency drops; a method that stays flat across the whole frequency
range is not actually showing the failure this task is testing for.

## Evaluation protocol, per baseline

Methods split into two groups, because of a structural difference:

- **Inductive methods** (scVI) have a trained encoder function that can
  embed a cell it has never seen, so we can hold cells out of training and
  still get a real, honest embedding for them afterward.
- **Transductive methods** (Harmony, BBKNN, ComBat) only ever produce a
  fixed output for the exact set of cells given to them — there's no
  encoder to run on a new cell without redoing the whole fit on the
  combined set. So a train/test split at the *model* level isn't possible
  for them; they must be fit on the full dataset.

This means only the inductive methods can tell us whether the failure
generalizes to cells the model never saw — that asymmetry should be
stated plainly in the results, not glossed over.

For every baseline, save its embedding (or corrected representation)
alongside its cluster assignment, keyed by cell ID, split (train/test),
and run type (train-only vs. full-embedding, for scVI) — one compact file
per baseline run, not one file per metric. This way, any new metric we
think of later can be computed straight from the saved embeddings without
rerunning the baseline.

### scVI (inductive)

1. Split cells into train/test, stratified jointly on (cell type, batch) —
   keeping each (cell type, batch) group's proportion the same on both
   sides, so rare groups aren't lost to chance in a plain random split.
2. Train scVI on the train cells only.
3. Embed both train cells (seen during training) and test cells (passed
   through the trained encoder, never seen during training).
4. Cluster twice, not once: (a) train embeddings only, giving train cluster
   labels; (b) the full embedding — train and test cells together — giving
   test cluster labels, keeping only the test cells' labels from this run.
   Clustering the full embedding (rather than assigning each test cell to
   whichever train cluster is nearest) is the part that actually tests the
   claim: a forced-nearest-cluster rule guarantees every test cell lands in
   an existing cluster regardless of where the encoder put it, so it can
   never show a rare cell sitting correctly on its own. Letting the
   clustering run freely on the full embedding means a rare test cell only
   gets folded into a dense cluster if that's genuinely where the encoder
   placed it — which is the actual thing we're trying to observe. This
   mirrors the train-graph/full-graph split already used for cohesion
   below.
5. Compute all assignment metrics on the train labels (from run a) and on
   the test cells' labels (from run b); check whether they're worse on
   test, and whether that gap widens as frequency drops.
6. Build two affinity graphs: one from train cells only, one from all
   cells (train + test).
7. Cohesion on train cells: computed against the train-only affinity
   graph.
8. Cohesion on test cells: computed against the full affinity graph,
   averaged only over the test cells.
9. For both train and test cohesion, run the frequency-dependence check:
   does cohesion for rare cell types drop as frequency decreases? On
   train, this asks whether scVI preserves affinity-graph similarity for
   cells it was trained on, and whether that holds up as a type gets
   rarer. On test, it asks the same question for cells scVI never saw.
   Compare train cohesion against SEACells run on the same train-split
   cells (a fair, same-data comparison — see SEACells section below); the
   test number has no SEACells equivalent, since SEACells can't be
   evaluated on unseen cells at all.

### Harmony, BBKNN, ComBat (transductive)

- Fit once on the full dataset — no train/test split at the model level.
- Still run the frequency-dependence check on the full output: does each
  metric drop as frequency decreases, same as for scVI.
- Cohesion is computed once, on the full affinity graph, with no
  train/test distinction.


### SEACells (metacell reference)

SEACells is transductive like Harmony/BBKNN/ComBat — no encoder, so it
can't be tested on unseen cells the way scVI can. That means it can't
produce a true "test" number at all. Run it twice instead, each run
answering a different question:

1. **Full dataset** (train + test pooled) — the practical number: how
   well SEACells does the way it's actually used. Also compute the same
   assignment metrics as the other baselines (purity, within/cross-batch
   agreement, rare homogeneity, rare F1) here, not just cohesion. Expectation: SEACells
   scores better than the correction-method baselines on all of them,
   especially the rare cell type versions — even forced into the
   multi-batch setting it isn't designed for, its within-batch archetypal
   structure should still capture local affinity better than a blunt
   linear/graph/generative correction does.
2. **Train-split cells only** (the same cells scVI trains on) — a fair,
   confound-free comparison against **scVI's train cohesion** specifically:
   same data given to both, so any gap reflects the method, not a
   difference in how much data each one saw. This is the number to use
   for "does scVI preserve affinity structure as well as a method built
   specifically to do that," not the full-dataset run.

Both runs act as the reference ceiling for cohesion: SEACells is built
specifically to maximize it, so run 1 tells us the best any baseline
should be expected to do in practice, and run 2 tells us the best
achievable on the exact data scVI had access to — and whether even that
ceiling drops for rare types once SEACells is pooled across batches,
a setting it isn't designed for.

## Baselines left out for now (future work)

Not run in this round, but worth adding later if the gap needs stronger
evidence or a reviewer expects them:

- **Seurat (CCA/RPCA)** — anchor/mutual-nearest-neighbor based correction,
  the most widely-used method missing from the current set. Same
  rare-type risk as the others, for its own reason: anchors require a
  reciprocal nearest-neighbor match between batches, and a handful of
  rare cells per batch are unlikely to produce one, so they inherit
  whatever correction nearby abundant-type anchors get instead.
- **trVAE** — a cVAE-based method like scVI, but adds an MMD (maximum
  mean discrepancy — a statistical distance between two distributions)
  penalty that explicitly forces each batch's distribution of embeddings
  to match the others. That extra push toward making batches
  indistinguishable is a plausible second way a rare, batch-specific
  state could get smoothed away, on top of the reconstruction-dominance
  issue already described for scVI.

## What I expect

- Every assignment metric and cohesion score should drop as frequency
  falls. A smooth, continuous decline across the whole frequency range
  would be the clearest version of this; a sharper drop concentrated
  right at the rare/common threshold still supports the claim. The only
  outcome that would weaken it is scores staying flat regardless of
  frequency.
- scVI's rare cell type cohesion drops as frequency decreases on both
  train and test, with the test-side drop at least as steep — showing the
  encoder doesn't generalize the rare pattern to cells it never saw — the
  specific failure scProto's affinity guidance is meant to fix.
- Every baseline's rare cell type cohesion sits below SEACells' — showing
  that correction methods lose the original neighborhood structure for
  rare cells regardless of which clustering method is applied downstream.
  
