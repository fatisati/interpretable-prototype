# Baselines: flag values and method notes

**Commands to run each stage are in [`docs/runbook.md`](../docs/runbook.md).** This file keeps
only (1) the flag values for each dataset and (2) notes on each method.

Every script takes the dataset's h5ad path, batch key, and label key as flags. Nothing
dataset-specific is hardcoded, so the same command works on any dataset; swap in the values below.

## Flag values per dataset

| dataset | `--h5ad` | `--batch-key` | `--label-key` | `--layer` |
|---|---|---|---|---|
| pancreas | `datasets/processed/pancreas_hvg.h5ad` | `tech` | `celltype` | *(omit — no true counts, see `datasets/pancreas.md`)* |
| lung | `datasets/processed/lung_hvg.h5ad` | `batch` | `cell_type` | *(omit — no true counts, see `datasets/lung.md`)* |
| pbmc_immune | `datasets/processed/pbmc_immune_hvg.h5ad` | `study` | `final_annotation` | *(omit — no true counts, see `datasets/pbmc_immune.md`)* |
| heart_cell_atlas | `datasets/processed/heart_cell_atlas_hvg.h5ad` | `donor` | `cell_type` (broad, 11 types — use this one for metrics; `cell_states` is the 65-type fine-grained alternative) | `counts` (true raw counts, confirmed) |

If a processed file is missing (`ls datasets/processed/`), regenerate it with stage 1 of the runbook.

## scVI (`scvi_baseline.py`)

Training runs for at most 100 epochs by default (`--max-epochs`). scvi-tools' own
automatic choice is 400 for a dataset this size, which was never a deliberate pick; 100
is enough for ~19k cells, and early stopping can still end it sooner. Each run's
`config.json` records `max_epochs`, `epochs_trained` (the actual count), and exact
package versions.

To change clustering only (no retraining), use `recluster.py` (runbook stage 3), then
re-run metrics.

## SEACells (`seacells_baseline.py`)

Runs twice per dataset (train-only, for a fair comparison against scVI's train
cohesion; and full dataset, the practical "how well does this actually do"
number) -- see the SEACells section of `tasks/rare_cell_preservation.md`.
Reuses the affinity graph `build_affinity.py` builds (building it first if it
doesn't exist yet) as SEACells' own kernel, via the library's
`add_precomputed_kernel_matrix`, instead of letting SEACells rebuild it itself.

`--n-pca-comps`/`--k`/`--graph-construction` must match whatever the dataset's
affinity graph was (or will be) built with -- defaults (50/15/union) match
`build_affinity.py`'s own defaults. The number of metacells is derived once
per dataset (`--cells-per-seacell`, default 75) and cached in
`seacells_config.json` next to `split.csv` -- delete that file to pick a
different ratio.

## PCA + Leiden, unintegrated control (`pca_baseline.py`)

Same pipeline as scVI with no batch correction: PCA (50 components, fit on train
cells, test cells projected), then the same shared clustering. Separates "integration
lost the rare type" from "clustering lost it" -- see the task file. Most of its ~5 min
runtime is the Leiden search for K.

## Harmony, BBKNN, ComBat

Not written yet.
