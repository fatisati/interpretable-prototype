# Leiden seed test on Windows, both gene lists (2026-11-28)

The Windows twin of `repro/evidence/mac-studio/leiden_seed_test/` (PR #7): same test, same
file format, so the two folders can be compared row by row.

Test: PCA baseline on heart, 8 Leiden seeds (0–7) per gene list, with
`repro/leiden_seed_test.py` (same embedding, same neighbour graph, resolution fixed at the
official run's value; only Leiden's `random_state` changes). Seed 0 reproduces each official
run exactly (ARI 1.000).

## Result (train job, 27 rare (type, donor) pairs)

| Gene list | Rare F1, 8 seeds | Coverage, 8 seeds | Agreement (rare cells) | Neighbour same-type (global) |
|---|---|---|---|---|
| Frozen list (Windows) | 0.692–0.851, mean 0.715, sd 0.055 | 0.690 in 7, 0.893 in 1 | 0.802–0.850, sd 0.015 | 0.805 in all 8 |
| Mac list | 0.696–0.853, mean 0.809, sd 0.059 | 0.690–0.893 | 0.824–0.856, sd 0.011 | 0.803 in all 8 |

Per seed, train rare F1:

| Seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| Frozen list | 0.696 | 0.697 | 0.692 | 0.695 | 0.695 | 0.698 | 0.692 | **0.851** |
| Mac list | **0.696** | 0.846 | 0.768 | 0.852 | 0.848 | 0.853 | 0.766 | 0.846 |

- Unlike on the Mac, the two lists' ranges **overlap**: the frozen list reaches 0.851
  (seed 7), the Mac list drops to 0.696 (seed 0). Both lists can land on either level; the
  gene list shifts how often.
- The level is set mostly by Mesothelial (6 of the 27 rare pairs). Mesothelial train F1 per
  seed: frozen list 0, 0, 0, 0, 0, 0, 0, 0.69; Mac list 0, 0.69, 0.32, 0.69, 0.69, 0.69, 0.32, 0.70.
- Neighbour metrics: no change across seeds, 0.002 between lists.
- Test job: rare F1 0.759–0.850 (frozen list), 0.781–0.859 (Mac list).

## Files

- `metrics_by_seed.csv`: one row per gene list, seed and job (train / test). Same columns as the Mac file.
- `per_rare_pair_by_seed.csv`: `metrics_per_rare_type.csv` of every run, stacked.

## Reproduce

Frozen list: the official Windows PCA run (built at `e11debf`; its 2,000 genes equal `datasets/frozen/heart_cell_atlas/genes.txt`), then
```
python repro/leiden_seed_test.py runs/rare_cell_preservation/heart_cell_atlas/pca runs/seed_test8/heart_cell_atlas 0 1 2 3 4 5 6 7
python evaluation/compute_metrics.py --baseline pca --embedding runs/seed_test8/heart_cell_atlas/pca_seed<N>/embedding.csv --affinity-dir runs/rare_cell_preservation/heart_cell_atlas --label-key cell_type --batch-key donor
```
Mac list: the same two commands on the run built in ADR 0003 step 2 (`repro/subset_to_gene_list.py`
with `repro/evidence/mac-studio/heart_cell_atlas/hvg_genes.txt`, commit `538d255`).

## Environment

```
Windows 11, Intel x86-64, python 3.13.11
scanpy 1.12.4   anndata 0.13.2   numpy 2.4.6   scipy 1.18.0   pandas 3.0.5
scikit-learn 1.9.0   leidenalg 0.12.0   igraph 1.0.0   pynndescent 0.6.0   numba 0.67.0
```
Full list: `../pip_freeze.txt`.
