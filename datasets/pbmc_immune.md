# PBMC / Immune (human)

**Source:** figshare article 12420968 — "Benchmarking atlas-level data integration in
single-cell genomics - integration task datasets" (Luecken, Buttner, Danese, Interlandi,
Muller, Strobl, Zappia, Dugas, Colome-Tatche, Theis, Chaichoompu; posted 2022-01-24).
https://figshare.com/articles/dataset/Benchmarking_atlas-level_data_integration_in_single-cell_genomics_-_integration_task_datasets/12420968

**Citation required by the dataset license:** cite the paper above (Luecken et al. 2022,
*Nature Methods*) and the original papers each immune sub-dataset was aggregated from.
Preprocessing details: https://github.com/theislab/scib

## Get it

1. Download `Immune_ALL_human.h5ad` by hand from the figshare link above (automated
   download is blocked on some networks/clients — don't rely on a script for this step).
2. Save it at `datasets/raw/Immune_ALL_human.h5ad`.
3. Run stage 1 of [`docs/runbook.md`](../docs/runbook.md) for this dataset.
   Writes `datasets/processed/pbmc_immune_hvg.h5ad`.

| | raw file | processed (`pbmc_immune_hvg.h5ad`) |
|---|---|---|
| sha256 | `96709b5fa677d5eafc734793a7873a2c763453ba8abddad24c0fba39e0d7f2c8` | `b76bf0b6f4a192f44c03cb489f89d900509128040820e19aa6cee6ad5a124572` |

(both measured directly from the downloaded/processed file on 2026-11-16 — if your
checksum differs, you don't have the same bytes; re-download or re-run.)

## Stats (measured directly, 2026-11-16)

- raw: 33,506 cells × 12,303 genes
- processed: 33,506 cells × 4,000 genes (HVG-subset)
- batch key: `study` — 5 distinct studies, including held-out-worthy studies `Freytag`,
  `Villani` (used as a test split in prior work)
- label key: `final_annotation` — 16 cell types

## Preprocessing applied (`datasets/scripts/prepare.py`)

1. `X` ships already normalized by the depositors (scran pooling + log1p, per the
   figshare description) — confirmed directly: `X.max()` is well below the 30 threshold,
   consistent with log-space data. The script's own `normalize_total`/`log1p` call is a
   safety net that does not fire for this file.
2. `layers['counts']` is **dropped**. Measured directly: 7.9% of its nonzero entries are
   non-integer — lower than pancreas (42.8%) or lung (91.6%), but still well above
   rounding noise, so it isn't trustworthy raw counts either. No published explanation
   for this specific dataset's cause was found; unlike lung (SoupX) and pancreas
   (mixed technologies), this one is unexplained.
3. HVG: top 4,000 genes, `sc.pp.highly_variable_genes(..., batch_key='study')` —
   batch-aware so no study's variance dominates gene selection.
4. Subset to the HVG set, write to `datasets/processed/pbmc_immune_hvg.h5ad`.

## Known caveat

**No true raw counts are available for this dataset from this source.** Any method here
must work on log-normalized `X` (e.g. `recon_loss='mse'`); NB/ZINB-likelihood methods
that need real raw counts cannot use this file as-is.
