# NeurIPS 2021 bone marrow (BMMC), CITE-seq part

**Source:** Luecken et al. 2021, NeurIPS Datasets & Benchmarks track, GEO GSE194122.
**Citation required:** cite Luecken et al. 2021 when using this data.

## Get it

Download command: stage 1 of [`docs/runbook.md`](../docs/runbook.md). Then run
`python datasets/scripts/prepare_neurips_bmmc.py`.

| | raw file | processed (`neurips_bmmc_hvg.h5ad`) |
|---|---|---|
| sha256 | `a0d193bb30f01a280c219c3ac40fd4b0a46b7c3bcb89223a1053f0f9a4cef434` | `6fd79801794aa24e9e5f56e7a70ee58b772842ed9e9f7bafed213cd4f95aac6f` |

Gene list and split are frozen in `datasets/frozen/neurips_bmmc/` (see `evaluation/frozen.py`).

## What the prepare script does

- **Drops 134 protein (ADT) features.** The raw file mixes 13,953 RNA genes
  (`feature_types == "GEX"`) with 134 antibody features (`"ADT"`). Protein counts are
  not gene counts, so they must not reach scVI.
- **Uses `layers['counts']`** (raw UMI counts, 0% non-integer, checked on every run).
  `X` in the raw file is already normalized and is not used.
- **Keeps all 90,261 cells.** No subsampling, so rare groups keep their real frequency.
- **2,000 genes**, `seurat_v3` on raw counts, batch-aware (`batch`), same recipe as heart.

## Shape

- 90,261 cells × 2,000 genes (processed)
- batch key: `batch`, 12 values (site × donor; 4 sites, 9 donors)
- label key: `cell_type`, 45 types
- split: 72,209 train / 18,052 test
- cluster count (75 cells per cluster): k_train = 962, k_full = 1,203

Rare groups: see the bone marrow section in `docs/dataset-options.md`.

## Environment (stage 1 and 2, 2026-11-28)

python 3.13.11, scanpy 1.12.4, scikit-misc 0.5.3, anndata 0.13.2, numpy 2.4.6,
scipy 1.18.0, pandas 3.0.5. Baseline versions are recorded in each run's `config.json`.

## Run times (CPU, 8 cores, 16 GB RAM)

- stage 1 (prepare): ~2 min
- stage 2 (`prepare_eval.py`): **105 min**, vs a few minutes on heart (5× fewer cells)
