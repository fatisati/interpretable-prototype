# Dataset options (catalog, no decision implied)

Candidates looked at so far for multi-batch, rare-cell-type work. This file is a living
reference — update it as datasets get audited or new ones get considered. It does not
commit to using any of these; see `docs/decisions/` for actual decisions and their
reasoning.

## Fit for the Task 1 claim

Scored 2026-11-22 against the 7 items in `tasks/rare_cell_preservation.md`, section
"What a dataset needs in order to test this claim". Legend: ✓ meets it, ✗ fails it,
? not checked yet. Cell counts come from our own processed files. Everything marked
"(lit.)" comes from the papers and hasn't been checked by us.

| dataset | 1 sub-state? | 2 rare per batch | 3 marker kept | 4 batch effect | 5 labels independent | 6 raw counts | 7 published loss |
|---|---|---|---|---|---|---|---|
| Heart, broad `cell_type` (already run) | ✗ rare types are separate lineages | ✓ | ? | moderate (14 donors, one study) | ? | ✓ | ✗ |
| Heart, fine `cell_states` | ✓ vCM5: 32 cells inside 5,307 ventricular cardiomyocytes (0.6%); also aCM5, EC8_ln, EC9_FB-like, FB7 | ✓ vCM5 ~3 per donor, 10 donors | ? (EBF1 marker (lit.)) | moderate | ? | ✓ | ✗ |
| HLCA (our capped subsample) | partial: Tuft 165, Neuroendocrine 159, Ionocyte 561 cells inside 13,853 airway epithelium; fairly distinct types | weakened: the 1,000-per-type cap made them ~10x less rare than in the full atlas (Tuft 0.03% → 0.32%) | ? | ✓ strong (14 studies, mixed protocols) | ✗ partly built with scANVI (lit.), which may favor scVI | ✓ | ✓ for Harmony and Seurat RPCA (lit.) |
| PBMC Seurat v4 (not downloaded) | ✓ L3 fine immune subtypes | probably | ? | weak (one lab, one protocol) | ✓ protein data used to define types (lit.) | ✓ | ✗ |
| pancreas (scIB) | partial: epsilon 32 cells (0.2%), hormone cells close to alpha cells | ✓ 8 batches | ? | ✓ strongest (different sequencing technologies) | ? | ✗ | ✗ |
| lung (scIB) | ✗ Ionocytes (46 cells) are a separate type | ✓ 6 batches | ? | ✓ | ? | ✗ | ✗ |
| pbmc_immune (scIB) | ✗ rare types are separate, or sit in only one batch | ✗ | ? | ✓ | ? | ~ (7.9% non-integer) | ✗ |

Reading of the table:
- **Heart fine labels** fit the claim best among the datasets we already have.
  vCM5 is almost exactly the claim's example: a tiny state inside a huge common type.
- **HLCA** has the strongest outside evidence, but that evidence is about Harmony and
  Seurat. As remembered from the paper (needs checking), scANVI/scVI kept these types
  better. Re-subsample with a gentler cap if we use it.
- **pancreas, lung, pbmc_immune**: not worth running for this claim. They fail
  items 1 or 6.

## scIB benchmark trio — pancreas / lung / pbmc_immune

Source: figshare 12420968, Luecken et al. 2022 (*Nature Methods*). Downloaded and
audited already — see `pancreas.md`, `lung.md`, `pbmc_immune.md` in this folder.

| | cells | batches | cell types | counts layer |
|---|---|---|---|---|
| pancreas | 16,382 | 9 (`tech`) | 14 | 42.8% non-integer (mixed UMI / collision-corrected / read counts across batches) |
| lung | 32,472 | 16 (`batch`) | 17 | 91.6% non-integer (SoupX pseudocounts on some samples) |
| pbmc_immune | 33,506 | 5 (`study`) | 16 | 7.9% non-integer (cause not found in the literature) |

Already-normalized `X` is trustworthy for all three (confirmed directly). `counts` is not
trustworthy raw data for NB-loss methods on any of the three, for different reasons each
time — see the individual `.md` files for detail.

## HLCA — Human Lung Cell Atlas

Sikkema et al. 2023, *Nature Medicine*. True raw counts kept in `.raw`. 2.4M cells, 49
source datasets. The paper's own authors document that Harmony and Seurat RPCA fail to
keep 3 named rare cell types (ionocytes, tuft cells, neuroendocrine cells) in their own
clusters during integration. A subsampled version (`hlca_sampled.h5ad`) is referenced in
the old codebase but not yet pulled or audited here.

## Heart Cell Atlas, subsampled

Litviňuková et al. 2020, *Nature*. Built into scvi-tools (`scvi.data.heart_cell_atlas_subsampled`)
specifically for NB-loss models — raw counts by construction. ~18,641 cells (default
subsample), 14 donors as batch, 11 broad cell types / 62 fine-grained states in the full
atlas. Documented rare populations: neuronal cells (<1% of all cardiac cells), and a
specific rare cardiomyocyte subtype (vCM5, ~1%) tied to a known gene (EBF1) and the
heart's conduction system. Not yet downloaded/audited by us — numbers above are from the
literature and scvi-tools docs.

## PBMC Seurat v4 CITE-seq

Hao et al. 2021. Built into scvi-tools (`scvi.data.pbmc_seurat_v4_cite_seq`) — raw
counts by construction. 161,764 cells, 24 batches (8 donors x 3 timepoints, HIV vaccine
trial). Three levels of cell type annotation (L1 broad -> L3 fine-grained), so "rare" has
a principled definition (L3 minus L1/L2), but which L3 types are actually rare and at
what frequency hasn't been checked yet. Not yet downloaded/audited.

## PBMC 10x CITE-seq

`scvi.data.pbmcs_10x_cite_seq`. Raw counts, smallest/simplest option. Only 2 batches, no
documented cell type list or rare-cell narrative found — weakest fit found so far, more
of a smoke-test dataset.

## Open Problems Batch Integration Benchmark

Actively maintained successor to scIB, built on CZ CELLxGENE Census data (6 atlases:
Immune Cell Atlas, Diabetic Kidney, HypoMap, etc.). Raw counts verified by construction.
39k-483k cells per atlas — bigger than what's wanted here, and needs its own
tooling/setup rather than a one-line download. Not explored beyond this initial pass.
