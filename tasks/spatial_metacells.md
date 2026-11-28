# Task: Metacells that respect spatial niche, not just cell type

## Definitions

- **Cell state** (as in SEACells): a cell's true biological condition — the genes
  it is actually running. Two cells are in the same state if all differences in
  their measured expression are technical noise (sparse sampling of RNA), not
  biology.
- **Metacell:** a small group of cells in the same cell state. Averaging them
  removes the noise and keeps the shared signal.
- **Niche:** a cell's spatial neighborhood — which cell types sit around it.
  Example: an immune cell inside a tumor vs. one in healthy tissue.

## Claim

**A cell's niche can change its state.** An immune cell surrounded by cancer
cells may turn on a few genes (a "niche program") that the same immune type
elsewhere does not. These two groups are then *different states* and should
be in *different metacells*.

**Standard methods miss this.** The niche program is a few genes on top of a
large shared cell-type program. Methods that look only at expression (SEACells;
Leiden on scVI embeddings) see the large shared program and put all immune
cells together. Once mixed, the niche genes are averaged away and can't be seen.

**scProto should catch it.** We build scProto's affinity graph (which cells
count as "similar") from spatial context: SEACells' adaptive Gaussian kernel,
computed on a BANKSY embedding (each cell's expression + the average of its
spatial neighbors' expression), or on its PCA. scProto's input is still only
gene expression; the graph just guides which cells it groups. So if the niche
genes exist in expression, the graph points scProto at them and it splits the
cells by niche. Because scProto reconstructs expression, it should not invent
niche splits that expression doesn't support (unlike SEACells, which partitions
whatever graph it's given).

**Example:** fibroblasts at the tumor–stroma border express CXCL9 (a chemokine
that recruits T cells); fibroblasts elsewhere don't. Good metacells keep border
fibroblasts together, so CXCL9 shows up in their average.

## Baselines

1. **SEACells** on expression (PCA) — standard metacell method.
2. **Leiden on scVI embeddings** — clustering on a standard VAE latent space.
3. **scProto, expression-only graph** — the control. It shows the gain comes
   from spatial guidance, not from scProto itself.
4. **scProto, spatial (BANKSY) graph** — our method.

Also useful: SEACells *on the same BANKSY graph*, to show scProto uses the
spatial signal without losing cell-type identity.

## What a dataset needs

1. **Spatial data with cell-type and niche labels** (or enough to derive niches,
   e.g. clustering by neighborhood cell-type mix).
2. **At least one cell type whose expression differs by niche** in a few genes
   (a known niche program; ideally backed by literature).
3. **Baselines 1–2 do not separate these niches** — their metacells mix the
   type across niches. If they already separate them, there's no gap to show.
4. **Preferably several slides or donors**, to show the result is not one
   lucky slide, and to test cross-slide metacells (slides = batches).

Cheap check before training: for the chosen cell type, find genes differing
between niches at single-cell level (ground truth). Then check whether baseline
metacells are mixed across niches for that type.

## Chosen dataset (2026-11-28)

**Vannan 2025 lung fibrosis, Xenium** (GEO GSE250346; download in
`datasets/lung_fibrosis.md`). Why: fibrosis creates niches where cells change
state (aberrant basaloid cells, ~96% in one niche; activated fibroblasts in
fibroblastic foci); 35 lungs / 45 cores, so many samples; and pathologists
marked tissue regions on H&E (fibroblastic focus, fibrosis, normal alveoli, …),
which gives niche labels made *independently* of expression.
Cautions: only 343 genes (the niche genes must be among them); expression is
nucleus-only; the file is an R object and must be converted to `.h5ad` first.

## How we measure it

- **Niche purity** within a cell type: are a metacell's cells from one niche?
- **Niche-program recovery:** correlation between niche-vs-niche fold changes
  computed on metacells and on single cells. High = metacells keep the signal.
- **Niche-DE gene recovery:** of the ground-truth niche genes, how many are
  significant (right direction) at metacell level.
- **Cell-type purity:** must stay high — splitting by niche must not mix
  cell types.
- **Spatial map plot:** metacells drawn on tissue coordinates next to cell-type
  and niche labels, for scProto and baselines.

## Goal (in order)

1. Find a dataset meeting all needs above.
2. Confirm SEACells and Leiden-on-scVI do **not** separate the niche states.
3. Make scProto separate them, and beat its own expression-only version —
   a positive gain from spatial guidance, not just "degrades less".
4. Show it holds across slides/donors.
