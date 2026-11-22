# Pancreas

**Source:** figshare article 12420968 — "Benchmarking atlas-level data integration in
single-cell genomics - integration task datasets" (Luecken, Buttner, Danese, Interlandi,
Muller, Strobl, Zappia, Dugas, Colome-Tatche, Theis, Chaichoompu; posted 2022-01-24).
https://figshare.com/articles/dataset/Benchmarking_atlas-level_data_integration_in_single-cell_genomics_-_integration_task_datasets/12420968

**Citation required by the dataset license:** cite the paper above (Luecken et al. 2022,
*Nature Methods*) and the original papers each pancreas sub-dataset was aggregated from.
Preprocessing details: https://github.com/theislab/scib

## Get it

1. Download `human_pancreas_norm_complexBatch.h5ad` by hand from the figshare link above
   (automated download is blocked on some networks/clients — don't rely on a script for
   this step).
2. Save it at `datasets/raw/human_pancreas_norm_complexBatch.h5ad`.
3. Run stage 1 of [`docs/runbook.md`](../docs/runbook.md) for this dataset.
   Writes `datasets/processed/pancreas_hvg.h5ad`.

| | raw file | processed (`pancreas_hvg.h5ad`) |
|---|---|---|
| sha256 | `97e6dfd65553e4d10aa3ef5d904362970a75c677c31d70fabc9234191a09db8c` | `a648c4f0ff27bc9d898de694dd0ae53dd3460d6eed8449fa7f84b404d557f187` |

(both measured directly from the downloaded/processed file on 2026-11-16 — if your
checksum differs, you don't have the same bytes; re-download or re-run.)

## Stats (measured directly, 2026-11-16)

- raw: 16,382 cells × 19,093 genes
- processed: 16,382 cells × 4,000 genes (HVG-subset)
- batch key: `tech` — 9 sequencing technologies (celseq, celseq2, fluidigmc1, smartseq2,
  inDrop1–4, smarter)
- label key: `celltype` — 14 cell types

## Preprocessing applied (`datasets/scripts/prepare.py`)

1. `X` ships already normalized by the depositors (scran pooling + log1p, per the
   figshare description) — confirmed directly: `X.max() = 13.0`, consistent with
   log-space data. The script's own `normalize_total`/`log1p` call is a safety net that
   does not fire for this file.
2. `layers['counts']` is **dropped**. Measured directly: 42.8% of its nonzero entries
   are non-integer — not true raw counts, almost certainly scaled during the
   depositors' own batch correction (the file name says `complexBatch`). Carrying it
   forward unlabeled risks someone later using it for an NB/ZINB likelihood or a
   size-factor recomputation that assumes real raw counts.
3. HVG: top 4,000 genes, `sc.pp.highly_variable_genes(..., batch_key='tech')` —
   batch-aware so no technology's variance dominates gene selection.
4. Subset to the HVG set, write to `datasets/processed/pancreas_hvg.h5ad`.

## Known caveat

**No true raw counts are available for this dataset from this source.** Any method here
must work on log-normalized `X` (e.g. `recon_loss='mse'`); NB/ZINB-likelihood methods
that need real raw counts cannot use this file as-is.
