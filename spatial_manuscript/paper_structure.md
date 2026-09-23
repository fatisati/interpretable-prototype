# Spatial (biology) paper — structure

Supersedes `spatial_paper_structure.docx` at the repo root. That version is **stale**: it
built the paper around COVET as the novel affinity. We now use **BANKSY** instead, which
changes the contribution claim and rewrites R2. See §"What changed" at the end.

Companion paper: the method paper (NeurIPS submission → ICLR if rejected), cited here as
the ML backbone. No ML novelty is claimed in this paper.

---

## The story, one line

> Cell state is set by identity **and** neighborhood — but metacell methods only see
> identity, so niche-driven programs get averaged away; scProto with a spatial-context
> affinity builds metacells that keep both, and recovers gene programs that
> transcriptomics-only aggregation destroys.

**Working title:** *Niche-Aware Metacells Reveal Microenvironment-Driven Transcriptional
Programs in Spatial Transcriptomics*

**Target:** Nature Methods. Biology-first, method-second. Not NMI — NMI wants ML novelty
as the headline, and the ML novelty lives in the companion paper.

### Terms used throughout

- **metacell** — a group of near-identical cells averaged into one denoised profile; the
  unit that replaces the individual cell in downstream analysis.
- **niche** — a local spatial micro-environment: which cell types surround you, in what
  proportion.
- **affinity** — the cell×cell similarity matrix that decides which cells may be grouped.
  In scProto it is a **plug-in**: swap it and the model's behavior changes, the
  architecture does not.
- **BANKSY** — a published spatial embedding (Nature Genetics 2024) that concatenates a
  cell's own expression with its neighborhood's, mixed by a weight `lambda`. Runs so far
  use the tag `bk08`.
- **DE / DGE** — differential gene expression: which genes are significantly higher in
  group A than group B.

### The concrete failure this paper fixes

```
a macrophage in the tumor core        \
                                       >  near-identical transcriptome
a macrophage at the tumor-stroma edge /   different functional program

transcriptomics-only affinity  ->  same metacell  ->  the two programs average out
                                                      and the difference is GONE from
                                                      the data before analysis starts

spatial-context affinity       ->  separate metacells  ->  program is preserved and
                                                           testable
```

---

## Contribution claim (state it this way, not stronger)

1. **Niche-aware metacells as a concept** — aggregation guided by microenvironment, not
   only transcriptome.
2. **A framework that holds identity and niche simultaneously.** The reconstruction
   objective anchors prototypes to real gene expression, so spatial signal steers grouping
   without overwriting cell-type identity. Everything else on the table gives up one to
   get the other.
3. **The biology** — niche-correlated programs recovered that pooled analysis misses.

Explicitly *not* claimed: a new embedding (we use BANKSY), a new objective (companion
paper), or direct cell–cell communication modelling.

---

## Abstract — 3 sentences

1. Problem: metacells built from transcriptomic similarity average away niche-driven
   programs.
2. What we do: spatial-context affinity (BANKSY) + scProto = niche-aware metacells.
3. What we find: recovers niche-correlated transcriptional programs while preserving
   cell-type identity. End with software availability.

## Introduction — 4 paragraphs

1. **Metacells as a tool.** Compress noisy scRNA-seq into stable cell states; standard
   practice aggregates by transcriptomic similarity; used for denoising, atlases,
   downstream analysis.
2. **The microenvironment shapes cell state.** Established biology, not a hypothesis —
   tumor-microenvironment immune exclusion, cortical layer specialization, fibroblast
   activation at boundaries. Current metacell methods ignore it entirely.
3. **The gap.** MetaCell, SEACells, MetaQ, SuperSpot define similarity from transcriptome
   *or* raw spatial proximity — neither captures niche composition. Concretely: a metacell
   spanning tumor-core and tumor-margin macrophages shows no DE between niches even when
   real programs exist.
4. **What we do.** Spatial-context affinity makes cells in similar microenvironments
   similar regardless of their own transcriptome; scProto's reconstruction anchor keeps
   cell-type identity. Cite companion paper. Open-source.

---

## Results

Each section is a **biological claim**, not a technical description. Every section needs
at least one panel where the method wins.

```
R1  Overview      transcriptomics-only metacells span niches; ours don't   (figure 1)
R2  Affinity      which spatial representation, empirically                (was "why COVET")
R3  Benchmark     trade-off scatter + the same-graph mechanism test
R4  Biology       niche DE programs                                        <- THE PAPER
R5  Downstream    ligand-receptor specificity or cross-slide reproducibility
```

### R1 — Niche-aware metacells resolve spatial structure that transcriptomics-only methods merge

**Claim:** transcriptomics-only metacells merge cells from distinct niches; niche-aware
metacells resolve them. Shown as a visual before/after.

- **Panel A (the problem):** NSCLC spatial map colored by SEACells metacell assignment.
  Highlight 2–3 metacells spanning spatially separate, niche-distinct regions. Inset:
  niche-purity distribution. Headline number: *"X% of SEACells metacells contain cells
  from ≥2 distinct niches."*
- **Panel B (the solution):** same map, scProto + BANKSY. Same regions now niche-coherent.
  Inset: niche-purity distribution shifted right.
- **Panel C (schematic, recommended):** two cells, same transcriptome, different
  neighborhoods — one path merges them, the other keeps them apart.

> **Prerequisite — do not skip.** Under the *old* loss scProto over-merged too (a few giant
> prototypes plus near-singletons), so it would fail its own Figure 1. R1 must be produced
> with the **new loss**. Verify the prototype size distribution on NSCLC before drawing
> anything. The old-loss failure stays internal; it is not a paper panel.

### R2 — Which spatial-context representation

**Claim (empirical, not an invention claim):** representing the microenvironment
explicitly separates niches *within* a cell type, which a cell's own expression cannot.

Compare four affinity inputs on the same cells:

```
(a) own PCA              transcriptome only            -> separates cell types, not niches
(b) BANKSY               own + neighborhood mean       -> ours
(c) COVET                neighbor covariance, 2nd order-> comparison arm
(d) concat(PCA, BANKSY)  both signals                  -> tunable trade-off
```

- UMAP of each, colored **twice** — by cell type, and by niche.
- Quantify: silhouette or kNN purity for the niche label vs. the cell-type label in each
  space.
- Choose `lambda` (BANKSY's neighborhood mixing weight) here; report sensitivity in
  Supplementary rather than in the main text.

*Framing note:* using a published, widely trusted embedding is a strength for this
audience, not a weakness — it removes the "did you tune your own embedding to win?"
objection entirely. Do not oversell BANKSY as a contribution.

### R3 — Superior niche/identity trade-off

**Claim:** scProto + spatial affinity achieves higher niche purity than all baselines while
maintaining cell-type purity. No existing method achieves both.

Benchmark table, ≥3 datasets. Methods:

```
SEACells (transcriptomic affinity)
SEACells + raw spatial proximity
SEACells + BANKSY affinity          <- the crux row, see below
MetaQ
SuperSpot                           (Visium only)
scProto + BANKSY                    (ours)
```

Metrics: niche purity, cell-type purity, compactness, modularity.

**Hero figure — trade-off scatter.** x = cell-type purity, y = niche purity, one point per
method. Ours in the Pareto corner (top-right: nothing beats it on either axis). This is the
single most memorable figure in the paper.

**The crux row: scProto + BANKSY vs. SEACells + BANKSY.** Same graph, different aggregator,
so it isolates *our* contribution from the affinity's. The mechanism, which the text must
state and the experiment must support:

```
a BANKSY graph contains some cross-cell-type edges
   = spatial noise, two adjacent cells of different types, not real niche signal

SEACells   pure archetypal analysis on the fixed graph, no access to expression
           -> cannot tell a noise edge from a real one, cannot correct it

scProto    encoder pretrained on real expression before any graph loss exists
           -> latent space is grounded in transcriptome, so the reconstruction term
              resists edges that contradict it
```

Open and testable: `lambda_proto_recon` currently sits far below `lambda_community` /
`lambda_nassoc`, so how much of this advantage actually materializes is unverified. If the
mechanism does not hold empirically, weaken the claim rather than the experiment.

**Ablation** (isolates affinity from framework): scProto + PCA / + BANKSY / + COVET /
+ concat.

### R4 — Niche-correlated transcriptional programs — THE MOST IMPORTANT RESULT

**Claim:** DE between same-cell-type metacells from different niches recovers known
niche-specific biology and reveals programs missed by standard analysis.

- Volcano or dotplot: DE genes between niche-metacells of one cell type (e.g. macrophages,
  tumor core vs. stroma).
- Annotate known niche markers — hypoxia genes in the core, immune-checkpoint genes at the
  boundary.
- **The negative control, which carries the paper:** these genes are **not** significant in
  standard cell-type DE (all macrophages pooled, niche ignored) but **are** significant in
  niche-metacell DE.
- Heatmap of top niche-correlated genes across niche-metacells, organized by niche.
- Pathway enrichment: hypoxia, immune exclusion, ECM remodeling.
- Spatial validation: one gene with a known spatial pattern, expression map matching the
  metacell assignment.

**Status:** a 6-pair pilot exists from the rebuttal — scProto isolated the (cell type,
niche) pair and recovered the correct genes where neither baseline could, 2 pairs matching
independently described biology.

> **Non-circularity requirement.** The pilot took ground truth from single-cell DE within
> each (cell type, niche) pair. That validates *recovery*, but is circular as a *discovery*
> claim — it only checks for something already visible in the single cells. For this paper
> R4 needs at least one program that is invisible to pooled cell-type DE and visible in
> niche-metacell DE. Find the gene that is (a) known to be spatially regulated in this
> tissue, (b) not significant in standard DE, (c) significant in ours. **That panel alone
> justifies the paper.**

Without R4 there is a better metric but no discovery, and no Nature Methods paper.

### R5 — Downstream analyses benefit

**Claim:** niche-aware metacells improve the sensitivity of downstream spatial analysis.
Pick one:

- **A — Ligand–receptor specificity.** Run CellChat or NicheNet on each method's metacells.
  Measure the within-niche / cross-niche interaction score ratio (higher = more spatially
  specific). Ours should give sharper, better-localized interaction maps.
- **B — Cross-slide reproducibility.** Train on one slide, apply to another; show
  niche-metacell programs reproduce. Connects back to the companion paper's cross-batch
  strength.

---

## Discussion

- What niche-aware metacells enable that was not previously possible.
- When to use spatial vs. transcriptomic affinity — or both, via concat.
- **Limitations:** `K` (number of metacells) must be set in advance; affinity quality
  propagates directly into the prototypes; spatial context degrades in sparse data; a
  single slide gains nothing from cross-batch alignment.
- **Future:** 3D spatial (z-axis scaling partially supported already), ligand–receptor
  informed affinity, integration with single-cell foundation-model embeddings.

## Methods

- **Spatial affinity construction** — BANKSY embedding (own + neighborhood expression,
  mixed by `lambda`), then adaptive-bandwidth RBF kernel on top (`build_seacell_kernel`).
  Parameter choices `k`, `n_pcs`, `lambda`; sensitivity in Supplementary.
- **scProto** — brief summary only, "see companion paper for full details": encoder, soft
  assignment, community loss, reconstruction loss, prototype regularization. **State that
  the corrected prototype-regularization loss is used**, and cite it.
- **Statistics** — DE via pseudobulk per niche-metacell (DESeq2 or edgeR); purity
  comparisons via Mann-Whitney U reported with effect sizes.

## Datasets

```
NSCLC CosMx        single-cell res   HAVE    tumor microenvironment, 9 niches — hero
Brain MERFISH      single-cell res   NEED    cortical layers = clean ground-truth niches
  or Xenium
Visium             spot-based        NEED    required to compare against SuperSpot
2nd tumor/tissue   any               NEED    generalizability across tissue and platform
```

A second dataset is not optional: reviewers of the companion paper already flagged that the
spatial claim rests on one subsampled NSCLC slide.

---

## Execution order — differs from presentation order

```
step 0   confirm the NEW LOSS removes over-merging on NSCLC specifically
         (prototype size distribution + niche purity)
         blocks both papers; do it first regardless of priority

step 1   R1 sanity: are our metacells niche-coherent, and do SEACells' really span
         niches? cheap, and it is figure 1

step 2   R4 expansion — grow the 6-pair pilot, satisfy the non-circularity requirement

step 3   second spatial dataset

step 4   R2 / R3 tables — mechanical once the above holds
```

**R4 before R3 on purpose.** A benchmark always produces a table; biology might produce
nothing. If the DE analysis yields no genuine program, there is no paper — and that should
surface in weeks, not after a full benchmark suite is built around it.

---

## What changed vs. `spatial_paper_structure.docx`

| | old plan | now |
|---|---|---|
| affinity | COVET, presented as the novel contribution | **BANKSY**; COVET demoted to an R2 comparison arm |
| R2 | "why second-order beats first-order" | "which representation, empirically" |
| contribution | new embedding + framework + biology | framework + biology; **no embedding claim** |
| loss | not mentioned | **new corrected loss is a prerequisite for R1** |
| R4 | planned | 6-pair pilot exists; needs a non-circular discovery panel |
| R3 | method comparison | adds the same-graph crux row and its mechanism |
| order | R1→R5 | R4 pulled early — it is the risk |
