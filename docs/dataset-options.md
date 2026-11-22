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
| NeurIPS 2021 bone marrow, CITE-seq part (downloaded) | ✓ e.g. CD4+ T activated integrinB7+ (1,056) inside CD4+ T activated (6,966); CD8+ T CD49f+; ILC1; MK/E prog | ✓ Plasma cell IGKC+ 6–60 per batch in all 12; Plasmablast IGKC− median 5.5 in 7 of 12 | ? | ✓ strong (4 sites, same donors at several sites) | partial: protein data helped define types (lit.), so RNA may not separate some T-cell labels | ✓ checked: 0% non-integer | ✗ |
| Haber 2017 mouse intestine (downloaded) | ✗ with the labels in the downloaded files (only "Tuft", "Endocrine"); ✓ if fine labels (Tuft-1/2, enteroendocrine subtypes) are found in other GEO files | ✓ Tuft 2–46 per batch, all 10 batches | ? | moderate (10 batches; one set mixes infection conditions) | ✓ per-study clustering (lit.) | ✓ checked: 0% non-integer | ✗ |
| Moffitt 2018 MERFISH hypothalamus, spatial (not downloaded yet) | ✓ ~70 fine neuron subtypes (lit.) | ✓ probably, many animals (lit.) | ✓ automatic: ~155 genes, all kept | moderate (animals, sex) | ? | ✓ transcript counts per cell (lit.) | ✗ |
| Allen whole-brain MERFISH, spatial (backup) | ✓ thousands of fine clusters (lit.) | ✓ (lit.) | ✓ ~500 genes | moderate | ✓ mapped from a separate scRNA atlas (lit.) | ✓ (lit.) | ✗ |
| CosMx lung cancer (He 2022), spatial (backup) | partial: tumor/immune sub-states (lit.) | ✓ 8 samples, 5 patients (lit.) | ✓ 960 genes | ✓ strong (patients) | ✓ reference-based labels (lit.) | ✓ (lit.) | ✗ |
| Tabula Sapiens, 10x part only (backup) | partial (lit.) | ✓ many donors (lit.) | ? | ✓ | ✓ (lit.) | ✓ 10x part; ✗ Smart-seq2 part (read counts) | ✗ |

Reading of the table:
- **Heart fine labels** fit the claim best among the datasets we already have.
  vCM5 is almost exactly the claim's example: a tiny state inside a huge common type.
- **HLCA** has the strongest outside evidence, but that evidence is about Harmony and
  Seurat. As remembered from the paper (needs checking), scANVI/scVI kept these types
  better. Re-subsample with a gentler cap if we use it.
- **pancreas, lung, pbmc_immune**: not worth running for this claim. They fail
  items 1 or 6.
- **Bone marrow (NeurIPS 2021)** is the strongest new candidate: real site effect, raw
  counts, several sub-states present in all 12 batches. Its "visibility" check must pass
  first, because some T-cell labels were defined by protein and may look the same in RNA.
- **Haber** only fits if its fine labels can be found; with coarse labels it is
  the same "separate lineage" case as heart's broad labels.
- **Moffitt MERFISH** is the spatial pick. Rejected spatial options: DLPFC Visium (the most
  common spatial benchmark; spots hold several cells and are labeled only by brain layer,
  so no single rare cells exist) and the MOSAIC cancer cohort (each spot gets its
  most common type's label, which hides rare types).

## Where these candidates came from (literature search, 2026-11-22)

What similar papers use, by kind:

| Kind of paper | Datasets they use |
|---|---|
| Integration benchmarks (scIB, Open Problems, scIB-E) | pancreas, lung, immune, HLCA, NeurIPS 2021 bone marrow, Tabula Sapiens |
| Rare-cell detection (FiRE, GiniClust, scCAD, RareQ) | Jurkat/293T mix with Jurkat diluted, mouse intestine (Grün; Haber), mouse airway (Montoro), Zheng 68k PBMC |
| Metacells (SEACells, MetaQ) | 10x Multiome PBMC / bone marrow, CD34+ bone marrow, mouse gastrulation |
| Spatial integration (iSTBench, STAligner, SpaBatch, MOSAIC) | DLPFC Visium, MERFISH hypothalamus (Moffitt), STARmap, BaristaSeq, MOSAIC cancer cohort |

Rejected beyond the spatial ones above: Jurkat/293T and Kang IFN-β PBMC (rare groups are
whole separate cell lines or lineages, fails item 1); Allen cortex / Tasic 2018 (Smart-seq
read counts, fails item 6).

## NeurIPS 2021 bone marrow (BMMC), CITE-seq part

Luecken et al. 2021, NeurIPS Datasets & Benchmarks. GEO GSE194122. Download: see
`docs/runbook.md`. File: `datasets/raw/neurips2021_bmmc/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad`
(625 MB, unzipped). Counted by us 2026-11-22:

- 90,261 cells × 14,087 genes. Raw counts in `layers['counts']` (0% non-integer on the
  first 2,000 cells). Protein (ADT) data is not in this file's genes.
- Batch: `batch`, 12 values (`s1d1` … `s4d9` = site × donor). 4 sites (`Site`), 9 donors
  (`DonorID`). 3,929–11,473 cells per batch.
- Labels: `cell_type`, 45 types. Largest: CD14+ Mono 21,693 (24%).
- Too rare to use (in only 1–3 batches): cDC1 18, T prog cycling 24, CD8+ T naive CD127+
  CD26− CD101− 42, dnT 56, CD4+ T CD314+ CD45RA+ 93.
- Useful rare groups (cells per batch, median and range): Plasma cell IGKC+ 15 (6–60, all
  12 batches); Plasmablast IGKC− 5.5 (0–37, 7 batches); ILC1 32 (14–99, all 12);
  MK/E prog 37 (8–168, all 12); CD8+ T CD49f+ 53 (15–226, all 12); CD4+ T activated
  integrinB7+ 94 (0–141, 11 batches).
- Also has `is_train` (competition split), donor metadata (age, BMI, sex, smoker).

## Haber 2017 mouse small intestine

Haber et al. 2017, *Nature*. GEO GSE92332. Download: see `docs/runbook.md`. Files in
`datasets/raw/haber2017_intestine/`. Labels and batch are encoded in each cell's column
name, e.g. `B1_AAACCGTGCCAAGT_Tuft` = batch B1, barcode, label Tuft. Counted by us 2026-11-22:

| file | cells | genes | batches | labels | cells per batch |
|---|---|---|---|---|---|
| `atlas_UMIcounts` | 7,216 | 15,972 | 10 (B1–B10) | 15 | 75–1,385 |
| `SalmHelm_UMIcounts` | 9,842 | 15,216 | 10 (B1–B10) | 8 | 200–1,542 |

- Both are raw UMI counts (0% non-integer on 300 genes).
- Atlas rarest: Tuft 166 (2–46 per batch), Enterocyte.Mature.Distal 241, Paneth 260,
  Endocrine 310. All 15 labels appear in all 10 batches.
- SalmHelm adds a condition in the name: Control 3,240, H. polygyrus day 3 2,121, day 10
  2,711, Salmonella 1,770. Conditions change biology (e.g. tuft cells expand with
  H. polygyrus), so batch and biology are mixed in this file.
- **Open issue:** labels here are coarse. The paper's sub-states (Tuft-1/Tuft-2,
  enteroendocrine subtypes) may be in `GSE92332_EEC_UMIcounts` or
  `GSE92332_LargeCellSort_UMIcounts` — not checked yet.

Environment for the counts above (bone marrow and Haber): anndata 0.13.2, pandas 3.0.5,
miniconda Python. Counted with a one-off script, not a repo stage.

## Moffitt 2018 MERFISH hypothalamic preoptic region (spatial)

Moffitt et al. 2018, *Science*. Dryad doi:10.5061/dryad.8t8s248, one CSV (1.03 GB).
Download is manual (Dryad blocks scripts) — see `docs/runbook.md`. From the paper (not
checked by us): ~1.1M cells, ~155 genes, many animals of both sexes, fine neuron
subtypes from the authors' own clustering. Used by spatial integration papers
(STAligner, iSTBench).

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
