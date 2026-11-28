# 0003 — Rare-cell metrics depend on one random clustering: what to report instead

Date: 2026-11-23, revised 2026-11-28 (steps 2 and 5 corrected, steps 6–7 added). Status:
**proposed**, open for discussion (Fatemeh, Mahmoud). Nothing in the code has changed yet.

## Context

PR #5 found that the Windows and Mac machines give different rare-cell numbers for the
same pipeline (PCA train rare F1: 0.696 Windows, 0.846 Mac), while all-cell metrics agree.
The Mac pack (`repro/`) listed four possible causes: H1 gene list, H2 split, H3 platform
numerics, H4 file format only.

We tested this directly on Windows: we forced the Mac's exact 2,000 genes into stage 1
(`repro/subset_to_gene_list.py`), reran stage 2, the PCA baseline and the metrics, and
compared against the Mac pack with `evaluation/compare_runs.py`.

## Evidence

**Step 1 — with the original Windows gene list.** 1,998 of 2,000 genes are shared (Windows
only: AP005329.3, LINC00603; Mac only: GIPR, LILRB4; caused by scanpy 1.12.4 vs 1.11.5).
The split is identical (18,641 of 18,641 cells on the same side). The affinity graph shares
99.71% of its edges with the Mac's.

**Step 2 — with the Mac's gene list forced on Windows.**

| Stage | Windows (Mac genes) vs Mac |
|---|---|
| Genes, order, split, k_train / k_full | identical (198 / 248) |
| `affinity_full` | 415,121 shared edges of 415,133 / 415,131; weight diff max 2.7e-05 |
| PCA embedding | max coordinate diff 0.002; 99.98% of cells have identical 15 nearest neighbours |
| Leiden clusters, train run | 199 vs 198 clusters; adjusted Rand index (ARI: 1 = same grouping, 0 = chance) **0.77** |
| Leiden clusters, full run | 248 vs 247 clusters at the **identical** resolution 22.478; ARI **0.77** |
| Train rare F1 | **0.696** vs **0.846** |

The 0.696 is the same as the original Windows run (0.696 to 3 decimals). *Corrected
2026-11-28:* we first read this as "the 2-gene difference has no effect". That was one seed
landing on the same value by chance. Step 6 shows the gene list does change rare F1, by
changing how often one rare type wins a cluster vote. The clustering is still where the
jump happens.

**Step 3 — exchange test** (repro README step 4): the Mac's own embedding and clusters,
scored by the Windows metric code, reproduce the Mac's metrics exactly (max diff 0). The
metric code is the same on both machines.

**Step 4 — which metrics are stable.** On the same embeddings as step 2 (Windows vs Mac):

| Rare-cell metric | Uses clusters? | Windows | Mac |
|---|---|---|---|
| `knn_same_type_global` (rare cells' 15 neighbours, share of same type), train | no | 0.8035 | 0.8035 |
| `knn_same_type_cross` (same, neighbours from other donors only), train | no | 0.7950 | 0.7950 |
| `knn_same_type_global`, test | no | 0.7985 | 0.7985 |
| `rare_f1`, train | yes | 0.696 | 0.846 |
| `rare_f1`, test | yes | 0.781 | 0.850 |
| `coverage`, train | yes | 0.691 | 0.893 |
| `agreement_global`, donor D2, train | yes | 0.19 | 0.58 |

Every neighbour-based rare metric matches to 4 decimals, per donor as well. Every
cluster-based one moves.

## Why this happens

1. **Leiden is random and has many near-equal answers.** It visits cells in random order
   and merges groups while modularity (how tightly connected the groups are) rises. With
   ~250 clusters on 18.6k cells there are many groupings of almost equal modularity; the
   seed and tiny input differences decide which one comes out (Good, de Montjoye &
   Clauset 2010, Phys. Rev. E — the "degeneracy" of modularity). Changing machine acts
   like changing the seed.
2. **The rare metrics turn small boundary moves into jumps.** `rare_f1` and `coverage`
   count a rare type as kept only if it is the *majority* of some cluster. A rare
   (type, donor) pair has a handful of cells inside a ~75-cell cluster, so moving a few
   cells across a boundary flips that pair between ~1 and ~0. With 27 rare pairs, a few
   flips shift the mean by 0.1–0.15. This matches the 0.08–0.15 rerun spread in
   Mahmoud's review. In heart, most of the jump is one type, Mesothelial (step 6).

**Step 5 — seed test on one machine** (`repro/leiden_seed_test.py`). Same PCA embedding,
same neighbour graph, same resolution; only Leiden's `random_state` changes. *Corrected
2026-11-28:* the embedding here is the one built on Windows **with the Mac gene list**
(step 2), not the official Windows run. Step 6 repeats this with both gene lists.
Seed 0 reproduces the official clusters exactly (ARI 1.000). Seeds 1–3 agree with seed 0
at ARI 0.77–0.80, the same as Windows vs Mac (0.77).

| Rare-cell metric | Seed 0 | 1 | 2 | 3 | Range | Mac |
|---|---|---|---|---|---|---|
| `rare_f1`, train | 0.696 | 0.846 | 0.768 | 0.852 | **0.156** | 0.846 |
| `rare_f1`, test | 0.781 | 0.859 | 0.824 | 0.859 | 0.078 | 0.850 |
| `coverage`, train | 0.690 | 0.893 | 0.786 | 0.893 | **0.202** | 0.893 |
| `agreement_global`, train (mean over rare cells) | 0.824 | 0.838 | 0.838 | 0.855 | 0.032 | 0.838 |
| `agreement`, mean over rare (type, donor) pairs | 0.706 | 0.768 | 0.734 | 0.777 | 0.071 | — |
| `knn_same_type_global`, train | 0.803 | 0.803 | 0.803 | 0.803 | 0.000 | 0.803 |
| `knn_same_type_cross`, train | 0.795 | 0.795 | 0.795 | 0.795 | 0.000 | 0.795 |

Caveat: at a fixed resolution, seeds 1–3 gave 201–206 clusters instead of 199; no trend
with F1 is visible.

**Step 6 — 8 seeds, both gene lists, both machines** (Windows: `repro/evidence/windows/leiden_seed_test/`;
Mac: `repro/evidence/mac-studio/leiden_seed_test/`, PR #7). Train job, 27 rare (type, donor)
pairs, seeds 0–7.

| Rare-cell metric | Frozen list, Windows | Frozen list, Mac | Mac list, Windows | Mac list, Mac |
|---|---|---|---|---|
| `rare_f1` | 0.692–**0.851**, sd 0.055 | 0.692–0.702, sd 0.004 | **0.696**–0.853, sd 0.059 | 0.766–0.851, sd 0.029 |
| `coverage` | 0.690 (7 seeds), 0.893 (1) | 0.690 (all 8) | 0.690–0.893 | 0.786–0.893 |
| `agreement_global`, rare cells | 0.802–0.850, sd 0.015 | 0.802–0.841, sd 0.014 | 0.824–0.856, sd 0.011 | 0.827–0.856, sd 0.012 |
| `cohesion`, rare | sd 0.002 | sd 0.010 | sd 0.009 | sd 0.011 |
| `knn_same_type_global`, rare | 0.805, all seeds | 0.805 | 0.803 | 0.803 |

- `rare_f1` and `coverage` sit on two levels, ~0.70 and ~0.85. Mostly one event decides
  which: whether the Mesothelial cells (17 in train, 6 of the 27 rare pairs) are the
  majority of some cluster. On Windows (frozen list, seed 0) their main cluster holds
  7 Mesothelial and 8 Endothelial cells, so they lose by one cell. Mesothelial F1 per seed:
  frozen list on Windows 0 in 7 of 8 seeds, Mac list on Windows 0 in 1 of 8.
- The gene list shifts the odds of that vote; it does not change the biology. The 4 swapped
  genes are expressed in 0–1 of the 17 Mesothelial cells, and their 15 nearest neighbours
  are the same with either list.
- Neither machine alone shows the full picture. On the Mac the two lists' ranges don't
  overlap; on Windows they do (frozen list seed 7 = 0.851, Mac list seed 0 = 0.696).
- Averaging seeds fixes the noise for agreement (sd ~0.015, so 10 seeds give a mean
  accurate to ~0.005). It does not fix `rare_f1`/`coverage`: their mean only reports how
  often a coin flip lands one way.

**Step 7 — same embedding, same seed, different machine.** The Mac's committed PCA
embedding, clustered on Windows at the Mac's resolutions with Leiden seed 0: 204 vs 198
clusters (train), ARI 0.78. On the same machine the same seed always gives identical
clusters (ARI 1.000). So "same seed" only reproduces on one machine. Two candidate causes:
- the neighbour graph: `sc.pp.neighbors` uses an approximate search (pynndescent, already
  seeded with 0). On Windows only 86% of cells get exactly their true 15 nearest
  neighbours, and pynndescent is 0.6.0 on Windows vs 0.7.0 on the Mac;
- Leiden itself: unlikely, since leidenalg 0.12.0 and igraph 1.0.0 are the same on both.

Test in progress: `repro/graph_leiden_check.py`. Windows saved its graph and seed-0
clusters (`repro/evidence/windows/graph_leiden_check/`); the Mac compares its own graph
with ours and runs Leiden on ours. If the graph is the cause, the fix is an exact neighbour
search (no shortcuts, so the same on every machine) in `evaluation/common.py`.

## Options considered

| Option | Fixes the randomness? | Notes |
|---|---|---|
| A. Another algorithm (Louvain, k-means, hierarchical) | No | Louvain has the same degeneracy. K-means depends on random starting points. Hierarchical (Ward) has no random step but still moves boundaries on tiny input changes, and the majority-vote jump stays. |
| B. One fixed resolution for all methods (count taken from PCA) | No, and less fair | The same resolution gives different cluster counts on different embeddings, so a method with more clusters looks better on rare types. Step 2 shows identical resolution on near-identical input still gives 248 vs 247 clusters and ARI 0.77. |
| C. Keep fixed K, average over seeds (and over 2–3 K values) | Measures it | Standard in recent benchmarks: report mean ± std over seeds. Repeating at e.g. 30 / 75 / 150 cells per cluster shows the claim doesn't hang on the SEACells-derived 75. |
| D. Make neighbour-based metrics the headline | Yes (deterministic given the embedding) | Clustering-free kNN metrics are what scIB (Luecken et al. 2022, Nat. Methods) and LISI (Korsunsky et al. 2019, Nat. Methods) rely on for the same reason. Already computed by `compute_metrics.py`. Matches the claim's own wording in `tasks/rare_cell_preservation.md`: "fewer of the rare cells' nearest neighbors are rare cells of the same kind." |
| E. Choose resolution by stability (chooseR, Patterson-Cross et al. 2021; consensus clustering, SC3, Kiselev et al. 2017) | Partly | Picks K by how stable the clusters are under subsampling. Changes K per method, so it makes cross-method comparison harder. Useful later, not needed for the claim. |
| F. Optimise resolution per method and keep the best (scIB's NMI/ARI and isolated-label F1) | No | Standard in scIB, but "best of N" rewards lucky seeds and still has single-run noise. |

## Published metrics checked (2026-11-23), and why none were adopted

Each was checked against its source code, and tested on the heart data where possible.

| Metric | Source | Verdict | Reason |
|---|---|---|---|
| Isolated-label F1 | scIB (`scib/metrics/isolated_labels.py`) | Not adopted | For one label: the single best cluster's F1, then the best resolution of 10 between 0.2 and 2.0. "Isolated" means present in the fewest batches, not rare in every batch as in our claim. With our rare types and fixed K, across 4 seeds it still moves 0.09 (pooled) to 0.10 (per type and donor). Keeping the best resolution makes methods fair to each other but doesn't remove randomness. |
| Balanced ARI / AMI / homogeneity / completeness | Maan et al. 2024, Nat. Biotechnol. (`hsmaan/balanced-clustering`) | Optional, as an overall score | Rescales each cell type's row of the type × cluster table to the average type size, so small types count as much as large ones. Gives one dataset-wide number. The reweighting uses global type size, so types that are rare in each donor but common overall get little extra weight. |
| Jaccard index | scIB-E, Yi et al. 2025, Genome Biol. (`Chenxin-Yi/scIB-E`, `calculate_mean_ja`) | Not adopted | Per batch, the share of 15-NN graph edges kept between a per-batch PCA and the method's embedding. Calibration on heart: the same per-donor PCA with 40 instead of 50 PCs scores 0.62; our PCA control scores 0.35; scVI 0.20. It counts exact neighbour identities, so it mixes "different coordinate system" with "lost biology". Swapping one same-type neighbour for another costs as much as losing a rare cell to its parent type. |
| PCR comparison | `scib-metrics` (batch-correction group), reused by scIB-E | Possible later | (batch variance before − after) ÷ before. Shows that correction removed batch effect. It doesn't test rare cells. |

## Decision so far (2026-11-23, Fatemeh)

Keep the metrics `compute_metrics.py` already produces. Stop using `coverage` and
`rare_f1` as evidence; they are still computed. Neighbour same-type metrics are the main
evidence for embedding methods. Agreement, purity and cohesion are the cluster-based
metrics, and the only ones available for SEACells and BBKNN; report them as mean ± spread
over Leiden seeds. Recorded in `tasks/rare_cell_preservation.md`, section "Metrics".
Status stays **proposed** until Mahmoud agrees.

## Proposal

1. **Measure the noise first** (~10 min per method): PCA, 10 Leiden seeds × K at 30 / 75 /
   150 cells per cluster. Find the resolution once per K, then rerun only Leiden with other
   seeds. Report the std of each rare metric. That std is the threshold every later
   comparison must beat.
2. **Headline rare-cell metrics = neighbour-based**: `knn_same_type_global` and
   `knn_same_type_cross` on rare (type, donor) pairs. Not `knn_affinity_within` as the
   headline, because the affinity graph mostly links cells from the same donor (Mahmoud's
   review).
3. **Cluster-based metrics (`rare_f1`, `coverage`, `agreement_*`) stay, as secondary**,
   reported as mean ± std over the seeds of step 1, at fixed K (same K for every method).
4. **A method beats another only if the gap exceeds the seed spread.** For trained methods
   (scVI), seeds of the model itself add to this (Phase D of `docs/reproducibility-plan.md`).
5. **Keep freezing the gene list and split** (plan 4.5). The gene list shifts the odds of
   the Mesothelial vote (step 6), so freezing it matters for comparing runs.
6. **Make clusters reproducible across machines** (step 7): freeze each official run's
   clusters (`evaluation/frozen_run.py`, PR #7), and, if the graph test confirms the cause,
   switch to an exact neighbour search. That changes every official cluster once.

## Open questions for discussion

- Is a neighbour metric enough for the claim, or does the claim need clusters, since
  SEACells' output *is* metacells and "lost in clustering" is part of the claim?
  (SEACells has no per-cell embedding, so it only gets cluster metrics either way.)
- Which K values: 30 / 75 / 150 cells per cluster, or tie them to the size of the rare
  groups?
- 10 seeds enough? Decide after seeing the step-1 spread.

## Reproduce

Worktree on branch `clean-v2-local-mac` (commit `538d255`), raw heart file already cached:
```
python repro/subset_to_gene_list.py <raw_dir> repro/evidence/mac-studio/heart_cell_atlas/hvg_genes.txt datasets/processed/heart_cell_atlas_hvg.h5ad
python evaluation/prepare_eval.py --h5ad datasets/processed/heart_cell_atlas_hvg.h5ad --batch-key donor --label-key cell_type --layer counts --out-dir runs/rare_cell_preservation/heart_cell_atlas
python baselines/pca_baseline.py --h5ad datasets/processed/heart_cell_atlas_hvg.h5ad --batch-key donor --label-key cell_type --layer counts --out-dir runs/rare_cell_preservation/heart_cell_atlas/pca
python evaluation/compute_metrics.py --baseline pca --embedding runs/rare_cell_preservation/heart_cell_atlas/pca/embedding.csv --affinity-dir runs/rare_cell_preservation/heart_cell_atlas --label-key cell_type --batch-key donor
python evaluation/compare_runs.py --a runs/rare_cell_preservation/heart_cell_atlas --b repro/evidence/mac-studio/heart_cell_atlas
```
Times on Windows: stage 2 ~16 s, PCA + metrics ~6 min.

## Environment (Windows run, 2026-11-23)

```
Windows 11, python 3.13.11
scanpy 1.12.4   anndata 0.13.2   numpy 2.4.6   scipy 1.18.0   pandas 3.0.5
scikit-learn 1.9.0   leidenalg 0.12.0   igraph 1.0.0   scvi-tools 1.5.0.post1
pynndescent 0.6.0   numba 0.67.0   umap-learn 0.5.12
```
Full list: `repro/evidence/windows/pip_freeze.txt`.
Mac versions: `repro/README.md`, section "Environment on the Mac". Note igraph and
leidenalg are the **same** versions on both machines, so the cluster difference is not a
Leiden version difference.
