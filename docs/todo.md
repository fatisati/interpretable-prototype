# To-do (Task 1, after Mahmoud's review)

Source: Mahmoud's review in PR #5 (`feedback/clean-v2-critical-review.html`) and our reply there.
Order matters: earlier steps decide whether later ones are worth doing.

## Decide
- [ ] Choose the framing: A = batch-aware metacells, B = rare states defined by niche, or both.
- [ ] Freeze the dataset criteria list (no edits after this).

## Fix what we have
- [x] Cohesion: only count cluster members in the cell's batch or linked to it, so mixing batches isn't punished (bd81600).
- [x] Held-out-node rule for scoring scProto on its own graph, written in the task file (bd81600).
- [ ] Regenerate `results/rare_cell_preservation/official/` so heart uses the new cohesion.
- [ ] Seeds: rerun PCA and scVI with several seeds on heart, then write the pass rule (e.g. a gap counts only if it's bigger than 2× the seed spread).
- [ ] Write `results/rare_cell_preservation.md` for heart: no gap, and why.

## Add
- [ ] Baselines: Harmony, BBKNN, ComBat (+ Leiden); scANVI; scPoli (scProto's backbone); MetaQ.
- [ ] SEACells variants: SEACells then Harmony on the metacells (the paper's own route); SEACells on a corrected embedding (scVI, Harmony).
- [ ] Metrics: isolated-label F1, silhouette, HLCA-style rare recall/precision, scGraph; a second, cell-type-level cluster count.
- [ ] Citations for each baseline's failure mechanism, or reword them as hypotheses.

## Test whether the gap exists
- [ ] **Planted rare state.** In a common type of a raw-count, many-batch dataset (e.g. NeurIPS bone marrow, 12 batches), boost a small gene set in about 5 cells per batch. Sweep the boost size and the number of genes. Does each method keep the planted cells together, compared with uncorrected PCA? The answer is known, so no labels are needed.
  - If every method matches PCA at every setting, the rare-cell claim is false. Don't look for another dataset.
- [ ] **Cheap checks per new dataset, then go / no-go.** Adams 2020, Habermann 2020, HLCA core without the cap; later Vannan Xenium and SEA-AD. For each rare state: cells per batch, markers kept after gene filtering, visible in plain PCA within one batch, how strong the batch effect is. Drop the dataset if PCA can't see the state.
- [ ] Full baselines + seeds on the datasets that pass. One verdict each: gap / no gap / inconclusive (with the reason).

## Then
- [ ] scProto, only where a gap was shown, against scPoli + the same clustering. Before porting, write down the old branch's lessons (D59–D64).
