# Lung

**Source:** figshare article 12420968 — "Benchmarking atlas-level data integration in
single-cell genomics - integration task datasets" (Luecken, Buttner, Danese, Interlandi,
Muller, Strobl, Zappia, Dugas, Colome-Tatche, Theis, Chaichoompu; posted 2022-01-24).
https://figshare.com/articles/dataset/Benchmarking_atlas-level_data_integration_in_single-cell_genomics_-_integration_task_datasets/12420968

**Citation required by the dataset license:** cite the paper above (Luecken et al. 2022,
*Nature Methods*), the original lung atlas paper (Braga et al.), and the papers each
sub-dataset was aggregated from. Preprocessing details:
https://github.com/theislab/scib

## Get it

1. Download `Lung_atlas_public.h5ad` by hand from the figshare link above (automated
   download is blocked on some networks/clients — don't rely on a script for this step).
2. Save it at `datasets/raw/Lung_atlas_public.h5ad`.
3. Run stage 1 of [`docs/runbook.md`](../docs/runbook.md) for this dataset.
   Writes `datasets/processed/lung_hvg.h5ad`.

| | raw file | processed (`lung_hvg.h5ad`) |
|---|---|---|
| sha256 | `623f84435efdce9af31fc2be393ae7d2dfcf427414567b2fd994ea7422f5feeb` | `2e7f8e80b9355f0ec0a7a99e8c201a2c7fa69c5b5d172f8eebd3c55fcd8b040a` |

(both measured directly from the downloaded/processed file on 2026-11-16 — if your
checksum differs, you don't have the same bytes; re-download or re-run.)

## Stats (measured directly, 2026-11-16)

- raw: 32,472 cells × 15,148 genes
- processed: 32,472 cells × 4,000 genes (HVG-subset)
- batch key: `batch` — 16 distinct batches (e.g. `A1`–`A6`, `B1`–`B4`)
- label key: `cell_type` — 17 cell types

## Preprocessing applied (`datasets/scripts/prepare.py`)

1. `X` ships already normalized by the depositors (scran pooling + log1p, per the
   figshare description) — confirmed directly: `X.max() = 7.3`, consistent with
   log-space data. The script's own `normalize_total`/`log1p` call is a safety net that
   does not fire for this file.
2. `layers['counts']` is **dropped**. Measured directly: 91.6% of its nonzero entries
   are non-integer. Cause (confirmed in the scIB paper and the scvi-tools lung-atlas
   tutorial): this atlas combines Drop-seq data (true raw UMI counts) with 10x Chromium
   data that arrived already **SoupX-corrected** (ambient-RNA subtraction, which produces
   fractional values) from the original lab, before the raw count matrix was ever
   available to the scIB team.
3. HVG: top 4,000 genes, `sc.pp.highly_variable_genes(..., batch_key='batch')` —
   batch-aware so no sample's variance dominates gene selection.
4. Subset to the HVG set, write to `datasets/processed/lung_hvg.h5ad`.

## Known caveat

**No true raw counts are available for this dataset from this source, as shipped.** Any
method here must work on log-normalized `X` (e.g. `recon_loss='mse'`). A fix exists in
principle — SoupX's own stochastic-rounding step (round each fractional value up or down
with probability equal to its fractional part, preserving the correction's expected value
while producing real integers) — but has not been applied here; `layers['counts']` is
dropped rather than used as-is.
