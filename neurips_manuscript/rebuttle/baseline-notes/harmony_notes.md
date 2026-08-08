# Harmony vs scProto — rare cross-batch cell matching (notes)

## scProto claim

- scProto gets a batch-removed latent space, and corrects for over-correction
  (e.g. rare cells getting absorbed into dense/common states) by modifying the
  latent space directly.
- The encoder learns to recognize a rare state's pattern and place the cell
  correctly in latent space, regardless of batch.
- Evaluation approach: use rare cells in each batch to show this.
- Metrics: F1 (how well rare cells are correctly classified / given a
  dedicated correct group), homogeneity (continuous measure of how well rare
  cells are grouped together).

## Harmony — how it works (my understanding)

- Does k-means on PCA.
- Cells of the same state are assumed to have the same batch-effect vector.
  *(confirmed correct)*
- Assigns each cell to a cluster in a way that maximizes batch mixing per
  cluster — may not hold if a cell state doesn't exist in a batch, or more
  generally could be problematic.
- Then moves cell embeddings so the batch effect is removed.

## Reasoning / open questions

- If we compute ARBF on the Harmony embedding: since the correction is
  linear, rare cells within the same batch should stay close to each other —
  similar to running ARBF on raw PCA — so rare states should still be
  recoverable within each batch. Risk: cells from *other* batches getting
  pulled close to them could contaminate this.
  - Caveat: a cell's correction isn't one rigid shift for its whole batch —
    it's a blend across clusters, weighted by soft cluster membership
    (`sum_k cluster_probability[cell,k] * offset[k,batch]`). Nearby cells
    with similar membership profiles get nearly the same correction, so
    local structure holds for typical cells. Rare cells tend to sit in
    mixed-membership zones — they don't have enough local mass to firmly
    claim one cluster's identity — so their membership is more likely split
    across 2+ clusters with different offsets, making their own correction
    less smooth than a generic same-batch cell's. So within-batch rare-cell
    cohesion is the fragile case, not the safe case the "just a linear
    shift" intuition suggests.
- For evaluation, running Leiden on ARBF of the corrected embedding is the
  best option available right now — not sure if it's ideal, but Leiden on a
  good affinity graph can do very well if the affinity itself is good.
- Agree with the "no co-clustering" failure mode: rare cells in batch A land
  closest to one common population, rare cells of the same type in batch B
  land closest to a *different* common population (batch-specific noise) →
  they end up in different clusters → Harmony's per-(cluster,batch)
  correction never relates them → they can end up far apart post-correction
  not because correction failed, but because they were never compared.

## Proposed experiment to log

- **Cross-batch rare-cell kNN matching**: for each rare-type cell in batch A,
  find its k nearest neighbors restricted to cells from *other batches only*,
  in the corrected embedding (`X_harmony`, `X_stage1z`, `X_scvi`, scProto's
  latent). Measure the fraction of those neighbors sharing its label.
  Tests directly whether correction placed the rare cell near its true
  cross-batch counterpart, independent of SEACells/Leiden's clustering choices.
