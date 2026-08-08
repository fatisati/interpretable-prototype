# scProto vs. SEACells niche recovery: what's the real advantage? (s28nsc)

Working notes from `notebooks/ctx_pca_product_experiment.ipynb` (bk08 affinity, N_PROTOTYPES=200)
plus cross-checks against `other-papers/nscl.pdf` (Pentimalli et al., the actual source paper for
this 3D NSCLC CosMx dataset). Collapse diagnosis (effective_n≈7-9/200, gini≈0.9+, both hard and
untruncated-softmax pending re-check) is logged separately in the notebook itself
(`untrunc-collapse-code`, `beats-both-code` cells) — this file is about the DGE-recovery /
"where does scProto actually help" question, not the collapse question.

## Bottom line (current honest claim)

Not "scProto beats SEACells at niche recovery" — the matched/advantage tables are roughly a wash
across most pairs, and collapse means only ~15-20/146 pairs are even testable in soft mode. The
defensible, narrower claim:

> For a small number of niches defined by strongly marker-rich, spatially-organized cell types
> (vascular endothelium, macrophages, cytotoxic T cells) sitting at immune-vascular interfaces,
> scProto's reconstruction-grounded objective recovers a coherent, receptor-level transcriptional
> program that lines up with this exact dataset's own published ligand-receptor findings —
> something the purely graph-based archetypal baseline does not recover as precisely, even when
> given the identical spatial graph.

Note "does not recover as precisely," not "does not recover at all" — see correction below.

## Flagship candidate: Vascular endothelium × T cell aggregates (ACKR1)

scProto (`ema`, soft) top genes for this pair (pearson_r=0.942 in one run of the notebook):

| gene | mc_logfc | sc_logfc | direction |
|---|---|---|---|
| ACKR1 | 3.569 | 2.748 | match |
| KDR | -2.338 | -1.831 | match |
| CLU | 2.128 | 2.472 | match |
| HLA-DRA | 1.881 | 1.912 | match |
| CD74 | 1.842 | 3.004 | match |
| FLT1 | -1.781 | -1.981 | match |
| HLA-DPA1 / HLA-DRB1 / HLA-DRB5 / HLA-DQB1 | — | — | match (MHC-II block) |

`nscl.pdf` Fig 4 (their own CellChat-based ligand-receptor analysis, independent method, same
dataset): *"the T cell chemoattractant CCL5 (RANTES)... was enriched in T cell niches."* ACKR1 is
the canonical endothelial receptor/decoy for CCL5-class chemokines, expressed on venules that
mediate leukocyte transmigration — i.e. the receptor-side counterpart to a ligand-side finding the
paper reached via a completely different analysis. This is a real independent-method
cross-validation, not just "plausible general biology."

**Correction — check before citing as an exclusive scProto win:** an earlier (pre-edit) run of the
advantage table in this same notebook showed `SEACells` also scoring **pearson_r=0.851** on this
exact (Vascular endothelium, T cell aggregates) pair — meaningful recovery, just lower than
scProto's 0.942. So the correct claim is "scProto recovers it somewhat more precisely," not
"SEACells doesn't find it at all." **Not yet checked: whether SEACells' own top-gene list for this
pair also centers on ACKR1**, or recovers the correlation via a different set of genes entirely —
this matters a lot for how strong the claim can be, and needs `top_genes_for_pair` run on
`SEACells_arbf`/`SEACells_bk08` for this specific pair before the flagship claim goes in the
rebuttal text.

Also fragile on robustness grounds: only 2 soft-labeled metacells (sizes ~15, ~14 cells),
purity=0.38. Needs to survive the `beats-both-code` cell (appended to the notebook, not yet run)
before being called a robust result rather than a single lucky pair.

## Negative control / calibration (proves the lit-check isn't cherry-picked)

`SEACells_bk08`'s own top candidate, **Fibroblasts × Tumor surface** (`IGFBP5`, `COL11A1`, `FN1`,
`INHBA` up) also matches `nscl.pdf` directly — their fibroblast section (p.14) names exactly `FN1`,
`COL11A1`, `INHBA` together as the hypoxic/pro-invasive CAF signature, citing a separate colorectal
CAF paper for the same gene triad. This is a **SEACells win, not scProto's** — good to keep in the
writeup so the literature-matching method reads as honest rather than selectively applied only to
scProto's hits.

## Why, and in which scenarios, could scProto plausibly beat *both* baselines

Mechanistic argument, not yet fully empirically confirmed:

- `SEACells_arbf` (PCA-only graph, no spatial info) structurally **cannot** separate niches within
  a cell type — any within-cell-type niche signal is invisible to its graph by construction. So
  scProto has an inherent capability edge over `arbf` specifically whenever the niche-specific
  signal is (a) real, and (b) not already fully explained by pure transcriptional (non-spatial)
  cell-type identity.
- `SEACells_bk08` gets the *same* spatial graph scProto trains on, but as a pure archetypal method
  it has no mechanism to tell a genuine spatial-niche edge apart from a same-cell-type-boundary
  edge that's merely spatially close (graph construction noise). scProto's `L_rec`
  (reconstruction against real gene expression) is, in principle, the one thing that could let it
  use the same noisy graph more selectively — filtering graph structure that isn't also consistent
  with real transcriptional differences.
- **Caveat on how strong this mechanism actually is right now**: `lambda_proto_recon=0.01` here,
  vs. `lambda_umap`(=`L_community`)`=1` and `lambda_nassoc=1` — a 100x weight disadvantage. So this
  "self-correcting" capacity is structurally present but likely weak in the current hyperparameter
  regime — consistent with why the advantage over `bk08` specifically is narrow/inconsistent
  across pairs rather than a clean, broad win.

**Scenario where a real win is most plausible** (matches the recurring pairs seen across every
version of the advantage table so far — Cytotoxic T cells/Desmoplastic stroma, Macrophages/
Macrophage islands, Fibroblasts/Desmoplastic stroma, Vascular endothelium/T cell aggregates &
Vascular stroma):
1. Cell type has a strong, unambiguous baseline transcriptional identity that survives cVAE
   pretraining as a tight, well-separated cluster (vascular endothelium, macrophages, cytotoxic T
   cells — not, e.g., a fibroblast subtype continuum).
2. The niche-specific transcriptional response has large effect size (big logFC, strong marker
   genes) — subtle/quantitative niche gradients are unlikely to survive dilution through the
   collapsed, hub-dominated soft assignment.
3. Enough real cells in the pair (roughly n_real_cells>=100) that even a diluted soft-pseudobulk
   accumulates detectable signal, AND the bk08 graph has enough genuine boundary noise there that
   archetypal fitting measurably suffers from not being able to ignore it.

## Open TODOs
- [ ] Rerun the notebook fresh, top to bottom (several cells were appended/edited this session:
      `untrunc-collapse-code`, `beats-both-code`, `beats-both-genes-code` — none executed yet).
- [ ] Pull `SEACells_arbf`/`SEACells_bk08`'s own `top_genes_for_pair` output for Vascular
      endothelium × T cell aggregates specifically — confirm whether ACKR1 shows up there too.
- [ ] Confirm the flagship pair survives `beats-both-code`'s tightened filter
      (min_pos_mc_size>=5, n_real_cells>=100, n_pos_mc>=3, margin>0.03 vs. BOTH baselines).
- [ ] If it survives, repeat the same `nscl.pdf` cross-check for whatever else clears that filter.
