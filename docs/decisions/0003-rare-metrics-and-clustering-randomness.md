# 0003 — Rare-cell metrics depend on one random clustering: what to report instead

Date: 2026-11-23. Status: **proposed** — open for discussion (Fatemeh, Mahmoud). Nothing in
the code has changed yet.

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

The 0.696 is the same as the original Windows run (0.696 to 3 decimals). **The 2-gene
difference had almost no effect. The gap comes from the clustering.**

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
   Mahmoud's review.

Within-machine seed spread has **not** been measured yet (see Proposal step 1). The
cross-machine gap is our only estimate so far.

## Options considered

| Option | Fixes the randomness? | Notes |
|---|---|---|
| A. Another algorithm (Louvain, k-means, hierarchical) | No | Louvain has the same degeneracy. K-means depends on random starting points. Hierarchical (Ward) has no random step but still moves boundaries on tiny input changes, and the majority-vote jump stays. |
| B. One fixed resolution for all methods (count taken from PCA) | No, and less fair | The same resolution gives different cluster counts on different embeddings, so a method with more clusters looks better on rare types. Step 2 shows identical resolution on near-identical input still gives 248 vs 247 clusters and ARI 0.77. |
| C. Keep fixed K, average over seeds (and over 2–3 K values) | Measures it | Standard in recent benchmarks: report mean ± std over seeds. Repeating at e.g. 30 / 75 / 150 cells per cluster shows the claim doesn't hang on the SEACells-derived 75. |
| D. Make neighbour-based metrics the headline | Yes (deterministic given the embedding) | Clustering-free kNN metrics are what scIB (Luecken et al. 2022, Nat. Methods) and LISI (Korsunsky et al. 2019, Nat. Methods) rely on for the same reason. Already computed by `compute_metrics.py`. Matches the claim's own wording in `tasks/rare_cell_preservation.md`: "fewer of the rare cells' nearest neighbors are rare cells of the same kind." |
| E. Choose resolution by stability (chooseR, Patterson-Cross et al. 2021; consensus clustering, SC3, Kiselev et al. 2017) | Partly | Picks K by how stable the clusters are under subsampling. Changes K per method, so it makes cross-method comparison harder. Useful later, not needed for the claim. |
| F. Optimise resolution per method and keep the best (scIB's NMI/ARI and isolated-label F1) | No | Standard in scIB, but "best of N" rewards lucky seeds and still has single-run noise. |

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
5. **Keep freezing the gene list and split** (plan 4.5). Not the cause here, but it removes
   one variable for free.

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
```
Mac versions: `repro/README.md`, section "Environment on the Mac". Note igraph and
leidenalg are the **same** versions on both machines, so the cluster difference is not a
Leiden version difference.
