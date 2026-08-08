# Plan 1 — Spatial transcriptional program discovery: evaluation design

## Claim being tested

scProto's metacells recover niche-associated transcriptional programs, **within a cell
type**, better than a transcriptomics-only metacell baseline (SEACells on PCA).

This is deliberately scoped to *within-cell-type* recovery, not a claim that scProto
discovers programs shared *across* cell types. Niche effects are known to be
cell-type-specific (different cell types respond to the same niche with different
genes), and scProto's current affinity graph is not cell-type-conditioned, and its
reconstruction loss's gradient is stopped before the encoder — both make a
cross-cell-type discovery claim mechanically harder to support right now. That
reasoning is our own, not from a paper, and is why the claim is scoped this way.

No change to the affinity graph or retraining is required for this plan. Everything
below runs on the current scProto NSCLC output as-is.

---

## Setup (shared by everything below)

**A. Ground truth.** Using single cells only, with their true annotated labels: for
each cell type, run niche-X-vs-rest differential expression (same cell type only).
This produces the real, full-power gene list per (cell type, niche).
Already built: `files/celltype_niches.csv`.

> Paper: **Niche-DE** (Mason et al., *Genome Biology*) and **NCEM** (Fischer, D.S.,
> Schaar, A.C., Theis, F.J. "Modeling intercellular communication in tissues using
> spatial graphs of cells." *Nature Biotechnology*, 2023) — both establish that niche
> effects must be tested conditional on cell type, not pooled across cell types. NCEM
> is from Theis's own lab, using cell-type-to-niche interaction terms ("type-coupled"
> model) for exactly this reason.

**B. Label each metacell.** Default: plain majority vote — whichever (cell type,
niche) is most common among a metacell's real member cells becomes its label.
Alternative, if the metacell method exposes per-cell membership weights (soft
assignment rather than hard): score-weighted labeling using those weights directly,
instead of collapsing to a hard vote.

> Paper (majority vote): **SuperCell** (Bilous, M., Tran, L., Cianciaruso, C., et al.
> "Metacells untangle large and complex single-cell transcriptome networks." *BMC
> Bioinformatics*, 2022) — originated this exact majority-label/purity definition.
> **MetaQ** (Li, Y., Li, H., Lin, Y., et al., *Nature Communications*, 2025) — reuses
> the identical definition for its purity and balanced-accuracy metrics.
> Paper (score-weighted alternative): **SEACells** (Persad, S., Choo, Z-N., Dien, C.,
> et al. "SEACells infers transcriptional and epigenomic cellular states from
> single-cell genomics data." *Nature Biotechnology*, 2023) — provides per-cell
> weights across metacells (soft archetype assignment) rather than hard membership;
> a weighted label would use these directly instead of a hard vote.

**C. (Optional, cheap) Size-confound spot-check.** The real fix is upstream: confirm
metacell construction aggregates raw counts before normalizing, not averaged
pre-lognormed values. Once that's confirmed, a formal per-(cell type, niche) balance
test is not necessary — skip it unless a specific pair looks off on inspection.

> Paper: **LSMetacell** (Zhang, T., Zhu, H., *PLOS Computational Biology*, 2025) —
> proves library-size variance creates false-positive signal specifically when
> normalization happens *before* aggregation; their mechanism is about WGCNA
> networks pooled across many metacells, not a two-group DE test like ours, so
> residual risk here is low once raw-count-first aggregation is confirmed.

---

## Branch 1 — MetaQ-style continuous recovery (primary evidence)

This branch never uses a significance threshold, which matters because our
(cell type × niche) groups are small (often 3–5 metacells) — too small for a
p-value-based test to reliably fire either way.

1. Compute logFC for every gene at the metacell level, comparing niche-positive vs.
   niche-control metacells (same cell type). Add a small pseudocount to both group
   means before taking the ratio, so a near-zero denominator can't blow up the ratio.
2. Restrict to the genes already established as real hits in Setup A (the ground
   truth list) — we are not asking the small metacell groups to discover which genes
   matter, only to confirm a value for genes already known to matter.
3. **Metric (a)**: Pearson correlation between each gene's metacell-level logFC and
   its single-cell-level logFC.
4. **Metric (b)**: Kendall's tau rank-consistency between the metacell-level and
   single-cell-level ranking of those same genes.

> Paper: **MetaQ** (Li, Y., Li, H., Lin, Y., et al. "MetaQ: fast, scalable and
> accurate metacell inference via single-cell quantization." *Nature Communications*,
> 2025), Fig. 5c/5e — exactly these two checks (Pearson correlation of logFC values,
> Kendall's tau rank consistency), restricted to a known top-gene set, with **no
> significance gate** anywhere in the comparison. Pseudocount step: **MetaCell1**
> (Baran, Y., Bercovich, A., Sebé-Pedrós, A., et al. "MetaCell: analysis of single-cell
> RNA-seq data using K-nn graph partitions." *Genome Biology*, 2019) — their `lfp`
> formula (`log2((p_gk + ε) / median_k(p_gk + ε))`) is the source of the buffer-before-
> ratio idea; MetaQ's own methods don't add this beyond scanpy's default, so this one
> piece is borrowed from a different paper than the rest of Branch 1.

## Branch 2 — SuperCell-style discrete recovery (secondary, reported alongside)

1. From the ground-truth list (Setup A), keep genes that are p<0.05 significant at
   single-cell level (SuperCell's own filter).
2. At metacell level, test the same genes with a **sample-weighted t-test** —
   weighted by each metacell's member-cell count, not a plain/unweighted t-test —
   significance at p<0.05.
3. **Metric**: TPR = fraction of the ground-truth list that is also significant at
   the metacell level. (No secondary ranking step — TPR is a plain fraction.)

> Paper: **SuperCell** (Bilous, M., Tran, L., Cianciaruso, C., et al. "Metacells
> untangle large and complex single-cell transcriptome networks." *BMC
> Bioinformatics*, 2022) — this is their exact recipe: sample-weighted t-test (weighted
> by metacell size, since metacells pool unequal numbers of cells), TPR as a plain
> recovery fraction, no ranking step. Unmodified.

**Why this is secondary, not the headline**: SuperCell validated this recipe on
datasets where each group has dozens-to-hundreds of metacells. Ours has 3–5. We
already observed, on this exact NSCLC data, that a significance-gated recovery metric
(`eval_niches` / `kv_niches.csv` / `kr_niches.csv`) returned 0.0 across every real cell
type — almost certainly an underpowered-test artifact, not evidence of zero recovery.
Running Branch 2 anyway and reporting it honestly (likely lower than Branch 1) lets us
pre-empt "why didn't you just use SuperCell's own test" with a direct, cited answer.

## Additional metrics

**GSEA / pathway concordance** (secondary/backup): even where individual genes don't
match exactly, check whether the single-cell-level and metacell-level gene lists
point to the same enriched pathways.

> Paper: metacell review, citing Squair, Q.W. et al. ("Confronting false discoveries
> in single-cell differential expression." *Nature Communications*, 2021) — their
> recommended fallback for when gene-level comparison is noisy. Also doubles as the
> pathway-enrichment evidence reviewer F5RB explicitly requested.

**Do not use niche silhouette as evidence** (already computed in `ct_niche_sill.csv`,
but not to be cited as support).

> Paper: **"Metrics Matter: Why We Need to Stop Using Silhouette in Single-Cell
> Benchmarking"** (Rautenstrauch, P., Ohler, U., bioRxiv, 2025) — shows silhouette-
> style scores can look near-perfect even when true structure recovery fails,
> especially with more than two groups or non-convex clusters — the exact shape of
> our niche-within-cell-type setup.

---

## Baseline

Run Setup A–C and both branches identically on **SEACells-on-PCA** metacells (same K,
same labeling approach from Setup B) — the fair comparison, since it has zero spatial
signal going into its construction.

> General practice across every paper above (SuperCell vs. MetaCell, MetaQ vs.
> SEACells/SuperCell, etc.) — every one benchmarks against a competing metacell
> method, not just against itself. Not a single specific citation, but the shared norm
> across all of them.

---

## Reporting

Report both branches together, explicitly:

> "Using MetaQ's continuous recovery metric (logFC correlation + rank consistency, no
> significance gate needed), scProto shows X. Using the stricter SuperCell
> significance-gated overlap test, the number is Y — lower, as expected, since
> SuperCell's test was validated on far larger per-group metacell counts than ours."

Aggregate both branches' numbers the same way: macro-average across niches within a
cell type, then macro-average across cell types.

> Not from a paper — kept consistent with how this manuscript already reports purity,
> so it isn't a new averaging convention to defend.

---

## Data

Run everything on the **current** scProto NSCLC output, not the stale
`results/scproto/*.csv` snapshot (dated Jan 22 2026, predates this design, and uses an
inconsistent cell-type label scheme across its own files).

> Not from a paper — data hygiene.

---

## Reference list

1. Mason et al. Niche-DE: niche-differential gene expression analysis in spatial
   transcriptomics data identifies context-dependent cell-cell interactions. *Genome
   Biology*.
2. Fischer, D.S., Schaar, A.C., Theis, F.J. Modeling intercellular communication in
   tissues using spatial graphs of cells. *Nature Biotechnology*, 2023.
3. Bilous, M., Hérault, L., Gabriel, A.A.G., Teleman, M., Gfeller, D. Building and
   analyzing metacells in single-cell genomics data. *Molecular Systems Biology*,
   2024.
4. Zhang, T., Zhu, H. Library size-stabilized metacells construction enhances
   co-expression network analysis in single-cell data. *PLOS Computational Biology*,
   2025.
5. Li, Y., Li, H., Lin, Y., et al. MetaQ: fast, scalable and accurate metacell
   inference via single-cell quantization. *Nature Communications*, 2025.
6. Baran, Y., Bercovich, A., Sebé-Pedrós, A., et al. MetaCell: analysis of single-cell
   RNA-seq data using K-nn graph partitions. *Genome Biology*, 2019.
7. Bilous, M., Tran, L., Cianciaruso, C., et al. Metacells untangle large and complex
   single-cell transcriptome networks. *BMC Bioinformatics*, 2022 (SuperCell).
8. Squair, J.W., et al. Confronting false discoveries in single-cell differential
   expression. *Nature Communications*, 2021.
9. Rautenstrauch, P., Ohler, U. Metrics Matter: Why We Need to Stop Using Silhouette
   in Single-Cell Benchmarking. bioRxiv, 2025.
10. Persad, S., Choo, Z-N., Dien, C., et al. SEACells infers transcriptional and
    epigenomic cellular states from single-cell genomics data. *Nature
    Biotechnology*, 2023.
