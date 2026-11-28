# Lung fibrosis trio (IPF): Adams 2020 + Habermann 2020 + Vannan 2025

Candidate for the rare-state task. The rare state is **aberrant basaloid cells**: they appear
only in fibrotic (IPF) lungs, and there are about 483 of them in total, roughly 14 per IPF lung.
Two scRNA-seq labs (Adams, Habermann) give a real batch effect. Vannan is spatial (Xenium) and
has no basaloid labels of its own; they must be transferred from the other two.

**Sources and citations required:**
- Adams et al. 2020, *Science Advances*, GEO GSE136831 (Kaminski lab, Yale). 312,928 cells, 78 lungs
  (32 IPF, 18 COPD, 28 control).
- Habermann et al. 2020, *Science Advances*, GEO GSE135893 (Banovich/Kropski labs, Vanderbilt).
  114,396 cells, 30 lungs.
- Vannan et al. 2025, *Nature Genetics*, GEO GSE250346. Xenium, 343 genes, ~1.6M cells, 35 lungs.

Status: **downloaded only** (2026-11-28). No prepare script yet.

## Get it

Each study goes in its own folder under `datasets/raw/`. All files come straight from GEO,
so a script works (no manual browser step). Total is about 4.3 GB. At normal speed it takes 5–15 min.

```
B=https://ftp.ncbi.nlm.nih.gov/geo/series
mkdir -p datasets/raw/adams2020_ipf datasets/raw/habermann2020_ipf datasets/raw/vannan2025_xenium

# Adams 2020 (GSE136831): sparse count matrix + gene IDs + barcodes + cell-type table, 2.15 GB
for f in GSE136831_RawCounts_Sparse.mtx.gz GSE136831_AllCells.GeneIDs.txt.gz \
         GSE136831_AllCells.cellBarcodes.txt.gz GSE136831_AllCells.Samples.CellType.MetadataTable.txt.gz; do
  curl -L --retry 3 -C - -o datasets/raw/adams2020_ipf/$f $B/GSE136nnn/GSE136831/suppl/$f
done

# Habermann 2020 (GSE135893): count matrix + genes + barcodes + metadata (cell types), 1.08 GB
for f in GSE135893_matrix.mtx.gz GSE135893_genes.tsv.gz GSE135893_barcodes.tsv.gz \
         GSE135893_IPF_metadata.csv.gz GSE135893_readme_for_ILD_annotated_fullsize.rds.txt; do
  curl -L --retry 3 -C - -o datasets/raw/habermann2020_ipf/$f $B/GSE135nnn/GSE135893/suppl/$f
done

# Vannan 2025 (GSE250346): processed Seurat object (all samples, with annotations) + its README, 1.0 GB,
# + pathologist H&E region annotations (28 MB; added 2026-11-28 for the spatial task, tasks/spatial_metacells.md)
for f in GSE250346_Seurat_GSE250346_CORRECTED_SEE_RDS_README_082024.rds GSE250346_RDS_README.txt \
         GSE250346_HE_annotations.tar.gz; do
  curl -L --retry 3 -C - -o datasets/raw/vannan2025_xenium/$f $B/GSE250nnn/GSE250346/suppl/$f
done
```

`-C -` resumes a broken download instead of starting over.

**Faster, with a live progress display (aria2c):** aria2c shows percent, speed and time left,
and opens several connections per file (GEO is often slow per connection). Install once with
`winget install aria2.aria2` (then open a new terminal). For each file:

```
aria2c -c -x 8 -s 8 -d datasets/raw/adams2020_ipf $B/GSE136nnn/GSE136831/suppl/GSE136831_RawCounts_Sparse.mtx.gz
```

`-c` continues a partial file, even one started by curl (stop curl first). `-x 8 -s 8` = up to
8 connections per file.

**Deliberately not downloaded** (same counts/labels already in the files above, or not needed):
- `GSE136831_RAW.tar` (17.6 GB): Adams per-sample raw files.
- `GSE135893_ILD_annotated_fullsize.rds.gz` (22.4 GB): Habermann's Seurat object. The mtx +
  metadata above hold the same counts and cell types.
- `GSE250346_RAW.tar` (176 GB): Vannan raw Xenium output per slide. Only needed for
  re-segmentation or image work.
- Vannan H&E images (`*.svs.tiff.gz`).

**Reading the Vannan file:** it is an R (Seurat) `.rds`, so Python cannot open it directly.
Convert it once in R (e.g. with `SeuratDisk` or `zellkonverter`) to `.h5ad`. Read
`GSE250346_RDS_README.txt` first. Two points from it:
- The object holds **nucleus-only** expression (transcripts inside the segmented nucleus),
  not whole-cell counts.
- The preprint-era version had 35 genes swapped by a decoding error (e.g. MRC1 was labelled
  ACTA2). The file above is the corrected 08/2024 upload. Never mix it with an older copy.

## Checksums (sha256, measured on the downloaded files, 2026-11-28)

If yours differ, you do not have the same bytes: download again.

| file (under `datasets/raw/`) | bytes | sha256 |
|---|---|---|
| `adams2020_ipf/GSE136831_RawCounts_Sparse.mtx.gz` | 2,150,633,813 | `266133a964b5c45bf02394acaaff4ea4eb66096ec5a051d92f93dd2747b5850b` |
| `adams2020_ipf/GSE136831_AllCells.GeneIDs.txt.gz` | 323,848 | `e43b3008945c87a6875d4fb1a99a791fdaefeeb3c5a72a5d17589f621eb83af8` |
| `adams2020_ipf/GSE136831_AllCells.cellBarcodes.txt.gz` | 1,392,307 | `0efbb89de618a429e2db0399e25f9b2a0f7cd3b6d6ee9a3da71dfc9318cf1ddc` |
| `adams2020_ipf/GSE136831_AllCells.Samples.CellType.MetadataTable.txt.gz` | 4,308,752 | `147bdd11d720ed30736a459ba62a3d8e2f3f9028a8430e0eb714484f0456049b` |
| `habermann2020_ipf/GSE135893_matrix.mtx.gz` | 1,076,137,992 | `036987d06ed9625a71f5e5be0717c2598e914a6c9cad0628f61a8ebb66e695a6` |
| `habermann2020_ipf/GSE135893_genes.tsv.gz` | 129,668 | `7e8612e72f033ef214de101bb782ab1cb60b8c887be2d3e8a0d637de067b0233` |
| `habermann2020_ipf/GSE135893_barcodes.tsv.gz` | 936,873 | `37a9a98a95d59bb4333c8181c14bb2b384701e6fda1c409e79f3e41c681430d4` |
| `habermann2020_ipf/GSE135893_IPF_metadata.csv.gz` | 3,354,014 | `5c67b3717375859aafc4463024847cf30d7af7bb9d5f4ad9a5364ee6d7726bf5` |
| `habermann2020_ipf/GSE135893_readme_for_ILD_annotated_fullsize.rds.txt` | 642 | `a7f878eb36efe4e549ae00538973da7c51de2cdbcf855058ba5d1ff9fdc788ab` |
| `vannan2025_xenium/GSE250346_Seurat_GSE250346_CORRECTED_SEE_RDS_README_082024.rds` | 1,014,335,925 | `f97b6e20365715fed5e96e831304f86e32e5dbefaeada6d58668a805fd4ebbdd` |
| `vannan2025_xenium/GSE250346_RDS_README.txt` | 5,184 | `3ecdabf1ecd4581a6472e397b21c6ac0d31fbfa9904afff8482f8b03fe07aab6` |
| `vannan2025_xenium/GSE250346_HE_annotations.tar.gz` | 28,290,702 | `902bf4ce42a336b054d0616847c8155c7708bcfb6d9df30f339d6e7d45abe87b` |

Check: `cd datasets/raw && sha256sum adams2020_ipf/* habermann2020_ipf/* vannan2025_xenium/*`
