# 0001 — Which datasets to build the rare-cell-type baseline comparison on

## Context

Task 1 (`tasks/rare_cell_preservation.md`) needs at least one dataset where: (a) batch
correction / clustering baselines can be run, (b) raw counts are trustworthy enough for
NB-loss methods (scVI, scProto's CVAE), and (c) rare cell types are real, documented, and
ideally tied to known biology — not something we're reading into noise. One dataset isn't
enough to claim the gap is general, so the goal is 2+.

We started from the three datasets this project has historically used (pancreas, lung,
pbmc_immune — all from the same figshare collection). Auditing them directly (see
`datasets/pancreas.md`, `datasets/lung.md`, `datasets/pbmc_immune.md`) surfaced a real
problem: none of their `counts` layers are trustworthy raw counts. That audit is what
opened up this wider search.

## Options considered

### 1. scIB benchmark trio — pancreas / lung / pbmc_immune (figshare 12420968)

Source: Luecken et al. 2022, *Nature Methods*. Already downloaded, audited, and have
working `prepare.py` output (see the three `.md` files in `datasets/`).

| | cells | batches | cell types | counts layer |
|---|---|---|---|---|
| pancreas | 16,382 | 9 (`tech`) | 14 | 42.8% non-integer — mixes true UMI counts (inDrop1–4), UMI-collision-corrected (celseq/celseq2), and plain read counts not UMIs at all (smartseq2) |
| lung | 32,472 | 16 (`batch`) | 17 | 91.6% non-integer — some samples went through SoupX ambient-RNA correction before deposit, producing fractional pseudocounts |
| pbmc_immune | 33,506 | 5 (`study`) | 16 | 7.9% non-integer — cause not found in the literature, not yet explained |

**Pros:** small, already downloaded and fully audited (no surprises left), this project's
own history, real standard benchmark (scIB) with known published metrics.
**Cons:** none of the three can be trusted for NB loss as-is. Lung is fixable in
principle (SoupX's own stochastic-rounding step, not yet implemented). Pancreas has a
genuine unit mismatch (read counts vs. UMI counts) that rounding can't fix. Immune's
cause is unknown. None of the three has a *documented* rare-cell-type story in the
literature — rarity would be something we assert, not something already shown by others.

### 2. HLCA — Human Lung Cell Atlas (Sikkema et al. 2023, *Nature Medicine*)

**Pros:** true raw counts preserved in `.raw`, no audit risk. Canonical, heavily-cited
lung atlas. **Directly documents our exact problem**: the paper's own authors built a
recall/precision metric because Harmony and Seurat RPCA failed to keep 3 named rare cell
types (ionocytes, tuft cells, neuroendocrine cells) in their own clusters during
integration — this is their finding, not our interpretation. 49 source datasets (strong
batch diversity).
**Cons:** 2.4M cells full-size — too big for this task. A subsampled version exists
(`hlca_sampled.h5ad`, already referenced in the old codebase) but we haven't pulled or
audited it ourselves yet, so its exact size/rare-cell-frequency after subsampling is
unconfirmed.

### 3. Heart Cell Atlas, subsampled (`scvi.data.heart_cell_atlas_subsampled`)

Source: Litviňuková et al. 2020, *Nature*. Built into scvi-tools specifically for
NB-loss models — no audit needed by construction.

**Pros:** small by default (~18,641 cells, smaller than our lung/immune datasets), 14
donors as batch, 11 broad cell types (62 fine-grained states in the full atlas). Has a
**documented, literature-grounded rare-cell story**: neuronal cells are <1% of all
cardiac cells, and a specific rare cardiomyocyte subtype (vCM5, ~1%) is tied to a known
gene (EBF1) and the heart's conduction system — the original paper treats this rarity as
biologically important, not incidental.
**Cons:** different tissue from this project's prior work (heart, not blood/pancreas/lung)
— no overlap with past results. We have not yet downloaded/audited it ourselves; all
numbers above are from the literature and scvi-tools docs, not verified firsthand like
the scIB trio was.

### 4. PBMC Seurat v4 CITE-seq (`scvi.data.pbmc_seurat_v4_cite_seq`)

Source: Hao et al. 2021 — the reference Seurat v4 CITE-seq PBMC atlas, built into
scvi-tools for NB-loss models.

**Pros:** true raw counts, no audit needed. Very standard / widely cited reference.
Strong batch structure: 24 batches (8 donors × 3 timepoints from an HIV vaccine trial) —
a harder integration stress test than the scIB trio's 5–16 batches. Three levels of cell
type annotation (L1 broad → L3 fine-grained), so a clear, principled definition of "rare"
is available (L3 minus L1/L2). Bonus CITE-seq protein channel, unused for now.
**Cons:** 161,764 cells — ~9x the scIB trio's size, though still far smaller than HLCA.
We have not confirmed which L3 types are actually rare, nor their frequencies — unlike
the Heart Atlas, no specific rare population is named in what we found so far; this needs
its own audit before relying on it.

### 5. PBMC 10x CITE-seq (`scvi.data.pbmcs_10x_cite_seq`)

**Pros:** true raw counts, smallest/simplest option, quick to set up.
**Cons:** only 2 batches (weak integration stress test), no documented cell type list or
rare-cell narrative found. Weakest fit for this task — more of a smoke-test dataset than
a real benchmark.

### 6. Open Problems Batch Integration Benchmark (scIB's maintained successor)

Built on CZ CELLxGENE Census data (6 atlases: Immune Cell Atlas, Diabetic Kidney,
HypoMap, etc.).

**Pros:** actively maintained, same metric family as scIB, raw counts verified by
construction (census design), real published atlases.
**Cons:** atlases run 39k–483k cells — bigger than what we want here. Needs its own
tooling/setup, not a one-line scvi-tools or figshare download. Not explored beyond this
research pass — would need its own audit cycle before use.

## Decision

Use **Heart Cell Atlas (subsampled)**, **PBMC Seurat v4 CITE-seq**, and **HLCA
(subsampled)** as the three datasets for Task 1. One dataset can't show the gap is
general rather than a quirk of one tissue; three, each from a different organ (heart,
blood, lung), each with its own independently-published rare-cell-type story, is the
target.

Reasoning: all three ship true raw counts by construction or by direct confirmation in
their own paper (no repeat of the scIB counts-layer problem). All three are
genuinely standard/widely-cited references. All three stay well under HLCA's *full*
size (the subsampled version is being used, not the 2.4M-cell original). Each brings its
own literature-documented rare-cell narrative: Heart Atlas (neuronal cells, vCM5), HLCA
(ionocytes, tuft cells, neuroendocrine cells — with a dedicated published recall/precision
metric for exactly this), and PBMC Seurat v4 (rare types expected in its L3 annotation
level, not yet confirmed which ones).

## Not yet done

- Download + run the same integrity audit on all three (real cell/gene/batch counts,
  confirm `counts` really is integer-valued, confirm the rare populations are present at
  the expected frequencies after subsampling, and for PBMC specifically identify which L3
  types are rare and at what frequency) before writing their `datasets/*.md` files.
- Confirm the HLCA subsampled version's exact size and whether ionocytes/tuft/neuroendocrine
  cells survive the subsampling in meaningful numbers — the "rare" story could be
  weakened or destroyed by subsampling if not done carefully.
- The scIB trio is not being discarded — pancreas/lung/pbmc_immune stay in `datasets/` as
  already-audited, MSE-loss-only datasets; they just aren't the NB-loss / rare-cell-type
  dataset set going forward.
