# HLCA — Human Lung Cell Atlas (core, subsampled)

**Source:** Sikkema et al. 2023, *Nature Medicine* — "An integrated cell atlas of the
human lung in health and disease." Core atlas (healthy lung, 14 datasets), via
cellxgene's public collection — direct download, no figshare-style blocking observed.
https://cellxgene.cziscience.com/collections/6f6d381a-7701-4781-935c-db10d30de293

**Citation required:** cite Sikkema et al. 2023 when using this data.

## Get it

1. Download the **core** dataset's h5ad directly (not the "full" one — that's 2.28M
   cells, 21.9GB):
   ```
   https://datasets.cellxgene.cziscience.com/688185ad-11c2-4172-a53a-f4f1f4076860.h5ad
   ```
   (This exact URL was obtained via cellxgene's public curation API — collection
   assets can change their object key over time; if this link 404s, re-resolve it from
   `https://api.cellxgene.cziscience.com/curation/v1/collections/6f6d381a-7701-4781-935c-db10d30de293`
   and find the dataset titled "...(core)".)
2. Save it at `datasets/raw/hlca_core.h5ad` (5.87 GB).
3. Run stage 1 of [`docs/runbook.md`](../docs/runbook.md) for this dataset.
   Writes `datasets/processed/hlca_core_hvg.h5ad`.

| | raw file | processed (`hlca_core_hvg.h5ad`) |
|---|---|---|
| sha256 | `1cbdde1e513a31bdb6a3f70296e2d0612760d2af4032e239c94596ea87b44007` | `5159504e1ad989027322a45f83a01321f10b1493a22994631fe75ce57500280d` |

Re-ran the whole pipeline twice on a clean `processed/` output — both runs produced the
**identical** checksum above (seeded subsampling, `seed=0`), confirming the pipeline is
deterministic.

## Environment (exact versions used to produce the checksums above, 2026-11-16)

```
python    3.13.11
scanpy    1.12.4
anndata   0.13.2
numpy     2.4.6
pandas    3.0.5
```

## Why this isn't the full atlas

The core atlas is 584,944 cells, 27,402 genes — too large to load into memory at once on
a 16GB machine (loading `X` alone needs >8GB for one sparse-matrix component). Read in
anndata's **backed mode** (metadata only, no full matrix load) to decide which cells to
keep, then materialized only the kept subset into memory.

## Why NOT uniform random subsampling

The entire point of using HLCA is 3 literature-documented rare cell types: **Ionocyte**
(561 cells, 0.096% of the full core), **Tuft** (165, 0.028%), **Neuroendocrine** (159,
0.027%) — the paper's own authors built a dedicated recall/precision metric because
Harmony and Seurat RPCA failed to keep these in their own clusters during integration.
A uniform random subsample to a manageable size (e.g. 30-50k cells, ~5-9% retention)
would shrink Tuft/Neuroendocrine to ~8-15 cells each — too few for any reliable per-type
metric, destroying the exact thing this dataset is for (the same failure mode found and
documented for Heart Cell Atlas's rarest neuronal sub-states).

**Fix applied:** cap every cell type (`ann_finest_level`, 61 types) at 1,000 cells
(randomly subsampled within type, `seed=0`); types already below that cap are kept in
full. Confirmed directly after subsampling — all three target rare types are
**completely unchanged**: 561→561, 165→165, 159→159.

## Stats (measured directly, 2026-11-16)

- raw (core): 584,944 cells × 27,402 genes, 14 datasets (batch), 61 fine cell types
- subsampled: 50,819 cells (8.7% of the core) × 27,402 genes, before HVG
- processed: 50,819 cells × 4,000 genes (HVG-subset)
- batch key: `dataset` — all 14 datasets still represented after subsampling, sizes
  range 208 (`Lafyatis_Rojas_2019_10Xv1`) to 10,666 (`Banovich_Kropski_2020`) cells
- label key: `ann_finest_level` — 61 fine-grained cell types (the finest annotation
  level shipped; a coarser `cell_type`/`ann_level_1-5` hierarchy also exists in the raw
  file if a broader label is ever wanted)

## Preprocessing applied (`datasets/scripts/prepare_hlca.py`)

1. Per-type-capped subsample (see above) — the one step unique to this dataset.
2. **Confirmed `.raw.X` is true raw counts directly**: 0.0000% of nonzero entries are
   non-integer (vs. 42.8%/91.6%/7.9% for pancreas/lung/pbmc_immune). `X` itself ships
   log-normalized (cellxgene's schema convention) — `.raw.X` is copied into
   `layers['counts']`; `X` is left as shipped, not renormalized.
3. HVG: top 4,000 genes, `sc.pp.highly_variable_genes(..., batch_key='dataset')` —
   batch-aware so no dataset's variance dominates gene selection. (4,000 rather than the
   Heart Atlas's 2,000, since this dataset spans far more distinct cell types — 61 vs 11
   — and needs more genes to tell them apart.)
4. Subset to the HVG set, write to `datasets/processed/hlca_core_hvg.h5ad`.

## Known caveat

The per-type cap means the processed file's cell-type balance is **not** the atlas's
natural composition — abundant types (e.g. Alveolar macrophages, originally 68,487
cells) are capped to 1,000 just like everything else above the cap, while rare types
keep their true, small counts. This is deliberate (see above), but means this file
should not be used to estimate natural prevalence/proportions of the common cell types —
only for testing whether an integration method preserves the already-known-rare ones.
