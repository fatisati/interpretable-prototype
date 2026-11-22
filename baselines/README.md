# Baselines — how to run each one, on each dataset

One script per baseline method (see repo root `README.md`). Every script takes the
dataset's h5ad path, batch key, and label key as command-line flags — nothing about a
specific dataset is hardcoded, so the same script runs on any dataset you add later. This
file is the lookup table: copy-paste the exact command for the baseline/dataset pair you
want, instead of re-deriving the flag values from each dataset's doc every time.

## scVI (`scvi_baseline.py`)

Local (CPU or your own GPU, whichever PyTorch finds):

```
python baselines/scvi_baseline.py \
  --h5ad datasets/processed/<file> \
  --batch-key <batch col> --label-key <label col> \
  --out-dir runs/rare_cell_preservation/<dataset>/scvi \
  [--layer counts]   # only for datasets with true raw counts, see table below
```

Training runs for at most 100 epochs by default (`--max-epochs`). scvi-tools' own
automatic choice is 400 for a dataset this size, which was never a deliberate pick; 100
is enough for ~19k cells, and early stopping can still end it sooner. Each run's
`config.json` records `max_epochs`, `epochs_trained` (the actual count), and exact
package versions.

To change clustering only (no retraining), e.g. after changing the cluster-count search:
`python baselines/recluster.py --run-dir runs/rare_cell_preservation/<dataset>/scvi`, then
re-run `compute_metrics.py`. Takes ~9 min on the heart atlas (CPU).

Same command on Modal GPU instead: see `baselines/scvi_modal.py`'s docstring for the
one-time volume setup, then `modal run baselines/scvi_modal.py --h5ad-path ... (same flags)`.

| dataset | `--h5ad` | `--batch-key` | `--label-key` | `--layer` |
|---|---|---|---|---|
| pancreas | `datasets/processed/pancreas_hvg.h5ad` | `tech` | `celltype` | *(omit — no true counts, see `datasets/pancreas.md`)* |
| lung | `datasets/processed/lung_hvg.h5ad` | `batch` | `cell_type` | *(omit — no true counts, see `datasets/lung.md`)* |
| pbmc_immune | `datasets/processed/pbmc_immune_hvg.h5ad` | `study` | `final_annotation` | *(omit — no true counts, see `datasets/pbmc_immune.md`)* |
| heart_cell_atlas | `datasets/processed/heart_cell_atlas_hvg.h5ad` | `donor` | `cell_type` (broad, 11 types — use this one for metrics; `cell_states` is the 65-type fine-grained alternative) | `counts` (true raw counts, confirmed) |

Worked example (heart atlas):

```
python baselines/scvi_baseline.py \
  --h5ad datasets/processed/heart_cell_atlas_hvg.h5ad \
  --batch-key donor --label-key cell_type --layer counts \
  --out-dir runs/rare_cell_preservation/heart_cell_atlas/scvi
```

**Before running on pancreas/lung/pbmc_immune:** confirm the processed file still exists
at the path above (`ls datasets/processed/`) — if missing, regenerate it with
`python datasets/scripts/prepare.py --dataset <name>` (see that dataset's own `.md` for
the raw-file download step first, if needed).

## SEACells (`seacells_baseline.py`)

Runs twice per dataset (train-only, for a fair comparison against scVI's train
cohesion; and full dataset, the practical "how well does this actually do"
number) -- see the SEACells section of `tasks/rare_cell_preservation.md`.
Reuses the affinity graph `build_affinity.py` builds (building it first if it
doesn't exist yet) as SEACells' own kernel, via the library's
`add_precomputed_kernel_matrix`, instead of letting SEACells rebuild it itself.

```
python baselines/seacells_baseline.py \
  --h5ad datasets/processed/<file> \
  --batch-key <batch col> --label-key <label col> \
  --out-dir runs/rare_cell_preservation/<dataset>/seacells \
  [--layer counts]   # only for datasets with true raw counts, see table above
```

`--n-pca-comps`/`--k`/`--graph-construction` must match whatever the dataset's
affinity graph was (or will be) built with -- defaults (50/15/union) match
`build_affinity.py`'s own defaults. The number of metacells is derived once
per dataset (`--cells-per-seacell`, default 75) and cached in
`seacells_config.json` next to `split.csv` -- delete that file to pick a
different ratio.

Worked example (heart atlas):

```
python baselines/seacells_baseline.py \
  --h5ad datasets/processed/heart_cell_atlas_hvg.h5ad \
  --batch-key donor --label-key cell_type --layer counts \
  --out-dir runs/rare_cell_preservation/heart_cell_atlas/seacells
```

## PCA + Leiden, unintegrated control (`pca_baseline.py`)

Same pipeline as scVI with no batch correction: PCA (50 components, fit on train
cells, test cells projected), then the same shared clustering. Separates "integration
lost the rare type" from "clustering lost it" -- see the task file. Most of its ~5 min
runtime is the Leiden search for K.

```
python baselines/pca_baseline.py \
  --h5ad datasets/processed/heart_cell_atlas_hvg.h5ad \
  --batch-key donor --label-key cell_type --layer counts \
  --out-dir runs/rare_cell_preservation/heart_cell_atlas/pca
```

## Metrics and results (every baseline)

```
# official numbers (task file's rare rule) -> <run>/metrics/
python baselines/compute_metrics.py --baseline scvi \
  --embedding runs/rare_cell_preservation/heart_cell_atlas/scvi/embedding.csv \
  --affinity-dir runs/rare_cell_preservation/heart_cell_atlas \
  --label-key cell_type --batch-key donor
# what-if copy with another rare rule -> <run>/metrics__rare-<rule>/
#   add: --rare-rule std   (or fixed)

# tables + plots -> results/rare_cell_preservation/official/
python baselines/build_results.py --task rare_cell_preservation
#   what-if: --metrics-subdir metrics__rare-std  -> results/.../whatif_rare-std/
```

## Harmony, BBKNN, ComBat

Not written yet.
