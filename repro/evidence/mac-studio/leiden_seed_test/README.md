# Leiden seed test on the Mac, both gene lists (2026-11-24)

Question: is the Windows–Mac gap in PCA rare F1 (0.696 vs 0.846) ordinary Leiden seed
spread, as ADR 0003 concludes, or does the 2-gene difference in the gene list cause it?

Test: on one machine (Mac), PCA baseline on heart, 8 Leiden seeds (0–7) for each gene list,
using `repro/leiden_seed_test.py` (same embedding, same neighbour graph, resolution fixed at
the official run's value; only Leiden's `random_state` changes).

## Result (train job, 27 rare (type, donor) pairs)

| Gene list | Rare F1, 8 seeds | Coverage, 8 seeds | Neighbour same-type (global) |
|---|---|---|---|
| Frozen list (`datasets/frozen/heart_cell_atlas/genes.txt`, the Windows one) | 0.692–0.702, mean 0.697, sd 0.004 | 0.690 in all 8 | 0.805 in all 8 |
| Mac list (`repro/evidence/mac-studio/heart_cell_atlas/hvg_genes.txt`) | 0.766–0.851, mean 0.837, sd 0.029 | 0.893 in 7, 0.786 in 1 | 0.803 in all 8 |

- The two rare F1 ranges do not overlap. On this machine the gene list moves rare F1 by 0.14;
  the Leiden seed moves it by 0.01 (frozen list) to 0.085 (Mac list).
- The Mac with the frozen list gives 0.696 at seed 0, the same as the official Windows run.
- One cell type explains the gap: Mesothelial (6 of the 27 rare pairs). Its F1 is 0 in every
  seed with the frozen list and averages 0.64 with the Mac list. The lists differ by 2 genes
  (frozen only: AP005329.3, LINC00603; Mac only: GIPR, LILRB4).
- Neighbour metrics change by 0.002 between gene lists and by 0.000 across seeds.
- Test job: rare F1 0.759–0.816 (frozen list) and 0.824–0.859 (Mac list).

Limits: one machine, PCA only, fixed resolution (cluster count drifts 197–206 for a target of
198). The ADR's own seed test on Windows used the Mac-list embedding (its neighbour metric is
0.803); its seed-0 value of 0.696 is lower than anything seen here with the Mac list and was
not reproduced.

## Files

- `metrics_by_seed.csv`: one row per gene list, seed and job (train / test).
- `per_rare_pair_by_seed.csv`: `metrics_per_rare_type.csv` of every run, stacked.

## Reproduce

Code: `origin/clean-v2` (`6fc1612`) merged with `origin/adr-0003-leiden-randomness` (`1c0748e`).

Frozen list: stages 1–4 as in `docs/runbook.md` (stage 1 picks up the committed gene list), then
```
python repro/leiden_seed_test.py runs/rare_cell_preservation/heart_cell_atlas/pca runs/seed_test/heart_cell_atlas 0 1 2 3 4 5 6 7
python evaluation/compute_metrics.py --baseline pca --embedding runs/seed_test/heart_cell_atlas/pca_seed<N>/embedding.csv --affinity-dir runs/rare_cell_preservation/heart_cell_atlas --label-key cell_type --batch-key donor
```
Mac list: the same two commands on the run folder behind `repro/evidence/mac-studio/heart_cell_atlas/`
(built at commit `538d255`, before the freeze).

## Environment

```
macOS (Darwin 25.5.0), Apple M3 Ultra, python 3.11.18
scanpy 1.11.5   anndata 0.12.19   numpy 2.2.6   scipy 1.13.1   pandas 2.3.3
scikit-learn 1.9.1   leidenalg 0.12.0   igraph 1.0.0
```
