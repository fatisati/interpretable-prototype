# Niche-Aware Metacells — Prep Doc for Expert Review

Built incrementally, section by section.

---

## Part 1 — Does spatial microenvironment shape cell state?

**Source:** [Pentimalli et al. — *Combining spatial transcriptomics and ECM imaging in 3D for mapping cellular interactions in the tumor microenvironment*](file:///C:/Users/USER/Desktop/scproto/nscl.pdf) (Rajewsky lab, MDC Berlin). One lung tumor (NSCLC), profiled at single-cell resolution and reconstructed in 3D (340,644 cells).

**Setup, in plain terms:** cells were sorted into 18 **cell types** using gene expression alone (a fibroblast is a fibroblast because of what genes it turns on). Separately, cells were sorted into 10 **niches** using only *where they physically sit in the tissue* — what other cell types surround them (tumor core, tumor edge, immune-cell clusters, scar-like tissue, etc.) — with no reference to their own gene expression. Because these two groupings are done independently, we can directly ask: *for cells of the same type, does living in a different niche change what genes they turn on?*

### Yes — niche changes cell behavior (same type, different program)

| Cell type | What we see |
|---|---|
| **Tumor cells** | Cells at the tumor's outer edge gradually switch on an invasion-like program — losing "stay-put" epithelial genes, gaining mobility genes — even before physically leaving the tumor mass. Two genes turn on almost exclusively in this edge region, nowhere else. |
| **Fibroblasts** | The same broad cell type splits into 6 distinct sub-states, and each one is found almost only in one kind of surrounding tissue (scar-like stroma, immune-rich patches, normal-looking tissue, etc.) — not scattered randomly. |
| **Macrophages** | Macrophages sitting at the invasive tumor edge switch on tissue-repair genes; macrophages sitting next to dendritic cells instead switch on immune-suppressing genes. Same cell type, opposite behavior, driven by neighborhood. |
| **T cells** | Cancer-killing T cells cluster in immune-rich patches but are almost entirely blocked from entering the tumor core — same cell, kept out by location, not by its own state. |
| **Signaling genes** | Several communication genes switch on only in one specific niche (e.g. one immune-attracting signal appears only near T cells, a different one only near macrophages). |

### No — niche does not override what type a cell is

| Observation | Evidence |
|---|---|
| **Core identity is stable** | Which "type" a cell is (fibroblast, T cell, epithelial cell...) is decided from gene expression alone, and this holds up no matter where in the tissue the cell sits — niche doesn't relabel a cell's identity, it modulates behavior on top of it. |
| **Some genes ignore niche** | A few basic structural/adhesion genes in blood-vessel cells show up the same way in more than one niche type, unlike genes that are strictly niche-specific. |
| **Some sub-states are niche-shared** | Three of the six fibroblast sub-states still share the same "normal tissue" signature even though they end up in different neighborhoods further downstream. |

**Bottom line:** where a cell sits can genuinely change what genes it expresses — a purely-transcriptomics metacell method would blend these different states together and lose that signal. But it doesn't change what type of cell it fundamentally is. So our cell-cell similarity graph should treat spatial context as an *extra* signal added on top of gene expression, not a replacement for it (→ Part 2).

---

## Part 2 — A better cell-cell similarity graph

Same dataset as Part 1 (NSCLC_3D_section_28, 58,423 cells, 18 cell types, 10–11 niches).

### Exp. 1 — best spatial-context vector ([niche_vector_comparison.ipynb](../notebooks/niche_vector_comparison.ipynb), [metric code](../interpretable_ssl/evaluation/niche_composition_comparison.py))

Each is a per-cell summary of its k=35 spatial neighbors, built from their PCA (expression) embeddings:
- **V1 mean-PCA** — average of neighbors' PCA vectors ("what the neighborhood looks like on average"). *Why:* cheapest, most direct proxy — if a niche is defined by which cell types sit nearby, the average neighbor expression should already shift toward that niche's signature.
- **V2 COVET** — covariance of neighbors' PCA vectors, PCA-reduced ("how much neighbors vary/co-vary," not their location). *Why:* averaging can cancel out a mixed neighborhood (50% type A + 50% type B ≈ neither); covariance keeps that heterogeneity — this is also the descriptor the source paper itself uses to define niches.
- **V4 concat(mean, COVET)** — V1 and V2 stacked into one vector (variance-balanced). *Why:* mean gives "where," COVET gives "how spread" — combining both should in theory capture more of the neighborhood than either alone.

**Ground truth**: the *true* cell-type composition of the same k=35 neighbors (% of each cell type present) — the exact definition the source paper uses to define a niche.

**What each metric measures** — "does this vector's neighborhood structure match the true-composition neighborhood structure":
- **kNN purity** — cluster the true composition vectors (k-means) into niche labels; for each cell, what fraction of its nearest neighbors *in the candidate vector's space* share that same true-niche label.
- **Composition Jaccard** — overlap between "my nearest neighbors in the candidate vector's space" and "my nearest neighbors in true-composition space," as a set (not just same label — stricter than purity).
- **Cosine sim** — average cosine similarity between a cell's true composition vector and the true composition vectors of its neighbors *in the candidate space*. How close in raw composition, not just same cluster.
- **Mean rank** — average of each method's rank (1=best) across all metrics; single overall score.

| Vector | Moment | kNN purity | Jaccard | Cosine sim | Mean rank |
|---|---|---|---|---|---|
| **V1 mean-PCA** | 1st | **0.810** | **0.186** | **0.971** | **1.0 (best)** |
| V2 COVET (cov) | 2nd | 0.713 | 0.135 | 0.940 | 4.0 (worst) |
| V3 soft-cluster avg | 1st, discretized | 0.798 | 0.164 | 0.965 | 2.5 |
| V4 concat(mean, COVET) | 1st+2nd | 0.795 | 0.177 | 0.969 | 2.5 |

Mean alone wins every metric. Sweeping the mean/COVET mix in V4 (α = mean-side weight, 0.1→0.9) never beats pure mean (best blend 0.807 < 0.810). **Adding COVET never helps here, despite carrying more information in theory.**

### Exp. 2 — does spatial context help the actual affinity graph? ([affinity_comparison.ipynb](../notebooks/affinity_comparison.ipynb), [affinity code](../interpretable_ssl/augmenters/graph_generator.py))

Each is a cell-cell kNN graph (k=50) — the actual similarity graph a metacell method would aggregate on:
- **PCA only** — kNN graph from each cell's own PCA (expression) alone, no spatial info. *Why:* baseline — the standard scRNA-seq approach, shows what's lost by ignoring niche entirely.
- **BANKSY** — concatenate own PCA with V1 (mean-neighbor-PCA) into one vector, one RBF kernel on top. *Why:* the established published approach for niche-aware graphs — simplest way to inject spatial info without changing the kernel.
- **mean product** — two separate RBF graphs (own PCA, V1 context), edge weights multiplied. *Why:* concatenation (BANKSY) lets a strong signal in one space compensate for a weak one; multiplying instead requires similarity in **both** spaces independently (AND logic), which should avoid merging cells that only look similar by chance in one space.
- **mean+COVET product** — same, with a third RBF kernel on V2 (COVET) multiplied in. *Why:* if COVET carries information the mean misses (Exp. 1), adding it as a third independent constraint should sharpen niche resolution further.

**What each metric measures**:
- **Cell-type purity** (`ct_weighted_purity`) — weighted fraction of a cell's affinity-graph edges that connect to another cell of the *same cell type*. High = the graph mostly links within a cell type (independent of niche).
- **Niche purity (2D / 3D)** (`niche_purity_2D/3D`) — restricted to same-cell-type edges only, the weighted fraction that *also* connect to a cell in the same niche. **2D** = niches defined from single-section neighbors (what the graph was actually built from — fair, in-sample score). **3D** = the biologically "true" cross-section niches from the source paper (out-of-sample check: does expression alone leak real 3D niche info, or is the graph just fitting flat 2D geometry).

4 graphs, scored on cell-type purity (niche-independent) and within-cell-type niche purity vs. 2D (built-on) and 3D (biological) ground truth:

| Method | Cell-type purity | Niche purity (2D) | Niche purity (3D) |
|---|---|---|---|
| PCA only (no spatial) | **0.766** | 0.360 | 0.352 |
| BANKSY (concat mean-context) | 0.744 | 0.607 | 0.585 |
| mean product (own × mean-context) | 0.599 | 0.716 | 0.710 |
| mean+COVET product | 0.570 | **0.772** | **0.774** |

Trade-off: pure transcriptomics keeps cell-type purity but is niche-blind. Every spatial term buys niche purity at the cost of cell-type purity. mean+COVET generalizes best to real 3D niches (3D score ≥ 2D score) but costs the most cell-type purity.

Per-cell-type: COVET's extra lift over mean-only is ~0 for niche-homogeneous epithelial types (Respiratory epithelium 0.992→0.996, Basal epithelial 0.984→0.986) but large for niche-heterogeneous immune/stromal types (Dendritic cells 0.792→0.847, Cytotoxic T cells 0.716→0.778, Macrophages 0.699→0.763).

### Takeaway

- **Mean is the strongest single lever**: best context vector alone (Exp. 1) and ~2× niche-purity lift over PCA-only at moderate cell-type-purity cost (Exp. 2).
- **COVET alone is the weakest option** (Exp. 1) and only earns its cost as an add-on for niche-heterogeneous cell types (Exp. 2) — never as a replacement for mean.
- Confirms the design choice: mean spatial context as the primary additive term; COVET optional/secondary, applied selectively.

---

## Part 3 — A better algorithm on top of that graph: SEACells archetypal analysis, then scProto

Part 2 established *which* similarity graph to build (expression + mean spatial context).
Part 3 asks: given that graph, what algorithm turns it into metacells?

### SEACells: kernel-archetypal analysis

SEACells turns the Part 2 similarity graph into metacells via **archetypal analysis**: it
finds a small number of "archetypes" such that (1) each archetype is a **weighted
combination of real cells' similarity profiles**, weights ≥ 0 and summing to 1 (a
*convex* combination), and (2) every real cell is, in turn, a convex combination of the
archetypes. Restricting archetypes to convex combinations means they can only fall
*inside or on the edge* of the shape traced out by the real data — geometrically, its
**convex hull** — never an extrapolation beyond it. That's the **convex-hull assumption**:
a metacell can be a genuine blend of real cell states, but it can never invent a state
that isn't already bracketed by real ones. This is what rules out a "hallucinated"
metacell that a raw cluster centroid could otherwise produce.

**The formula.** Let `M` be the `n × n` cell-cell similarity matrix from Part 2 (`n` =
number of cells). SEACells fits:

```
M  ≈  M · B · A
```

- **`B`** (`n × k`, `k` = number of metacells): each **column** picks out one archetype —
  which real cells it's a combination of, and how much of each. Columns are non-negative
  and sum to 1, so `Z = M·B` (each archetype's own similarity profile) is a convex
  combination of *columns of `M`* — i.e. a blend of real cells' similarity profiles, never
  anything outside their convex hull.
- **`A`** (`k × n`): each **column** is one real cell's soft assignment — how much of each
  archetype that cell is made of. Also non-negative, summing to 1 per cell.
- Multiplying them, `M·B·A`, is "rebuild every cell's full similarity row from a
  combination of only `k` archetypes." Training fits `A` and `B` to make that
  reconstruction as close to the real `M` as possible:

```
min_{A, B} ‖M − M·B·A‖²        subject to: A, B ≥ 0, each column sums to 1
```

That squared-error term is graded **per cell** (every entry of the `n × n` matrix), which
is exactly why an archetype that secretly mixes two unrelated groups of cells gets caught
— it will reconstruct many individual cells' rows badly, not just look fine "on average."

### scProto: the same idea, trained end-to-end

scProto adopts the same convex-combination logic, but instead of solving it once as a
closed-form convex optimization, it folds it into end-to-end training: a set of trainable
"prototype" vectors sit in the model's own learned embedding space, every cell is softly
assigned to them, and several of scProto's training losses exist specifically to pull
each prototype toward being explainable as a combination of its assigned cells' real
embeddings — the same archetype-as-convex-combination idea, enforced by gradient descent
instead of by construction. Source: [`paper/sections/method.tex`](../paper/sections/method.tex).

**Encoder** — cell → shared embedding, conditioned on batch (so batch-specific shift
doesn't have to be absorbed into the prototypes themselves):
```
z_i = f_θ(x_i, b_i)
```

**Prototypes** — `K` trainable vectors `c_k` in that same embedding space (unlike
SEACells' `B`, these are not built from a convex combination of real cells — they're
free parameters, pulled into place by the losses below).

**Soft assignment** — how much cell `i` belongs to prototype `k` (`z_i`, `c_k`
unit-normalized, `τ` = temperature):
```
s_ik = softmax_k( z_i · c_k / τ )
```

**Community loss** — the main clustering signal, matching assignment-similarity to the
Part-2 input graph `P`:
```
q_ij = s_i · s_j
L_community = − Σ_ij [ P_ij·log(q_ij) + (1−P_ij)·log(1−q_ij) ]
```
Cells strongly connected in `P` (expression + spatial context) get pulled toward sharing
the same prototypes; unconnected cells get pushed apart.

**Reconstruction loss** — scProto's direct analog of SEACells' `M ≈ M·B·A`, but rebuilding
each cell's *own gene expression* instead of its *similarity row*:
```
x̂_k  = decode(c_k, batch)                  ← each prototype decoded to gene expression
x̃_i  = Σ_k s_ik · x̂_k                      ← cell i rebuilt as a soft combination of prototypes
L_rec = mean_i ‖x_i − x̃_i‖²
```
This is the literal "cell ≈ convex combination of metacells" check, done in gene-expression
space. The encoder gets no gradient from this term — only the decoder and prototypes
update from it, so reconstruction accuracy can't distort the clustering geometry.

**Prototype regularization** — keeps each prototype a tight, distinct community,
batch-aware (`vol` = degree sums from `P`, restricted to edges touching batch `b`):
```
M_kj  = (Sᵀ P^(b) S)_kj / sqrt(vol_k^(b) · vol_j^(b)),   aggregated as max over b
L_nassoc = mean_k (M_kk − 1)²  +  α · mean_(k≠j) M_kj²
```
Diagonal → 1: prototype `k` is a tight community in *at least one* batch (so a rare,
batch-specific state can still earn its own prototype). Off-diagonal → 0: no two
prototypes secretly overlap, in *any* batch. This is the coarse, pair-of-prototypes-level
check discussed below — the thing `sim_recon_loss` tries to sharpen into a per-cell check.

**Usage loss** — prevents dead prototypes (`m` = EMA momentum):
```
u_k ← m·u_k + (1−m)·max_i(s_ik)
L_usage = mean_k[ −log(u_k) ]
```

**Total objective:**
```
L = L_community + λ·L_rec + λ_n·L_nassoc + λ_u·L_usage
```

**Training, 2 stages:** (1) pretrain encoder+decoder as a batch-conditional cVAE (scPoli),
plain reconstruction, no prototypes yet; (2) initialize `K` prototypes from the pretrained
embeddings using SEACells' own waypoint method (greedy MaxMin in diffusion-map space — the
one piece scProto reuses directly from SEACells), calibrate `τ`, then jointly optimize the
full loss above.

### Why this can be better

**1. Grounding in gene expression prevents over-fragmenting niche-invariant cell types.**
The Part 2 graph blends real expression with spatial context, so for cell types whose
state genuinely doesn't depend on niche (Part 1's "niche does not override identity"
cases — core cell-type identity, shared adhesion/fibroblast programs) the expression
signal dominates and those cells still collapse into one coherent metacell. Spatial
context only pulls cells apart where the graph shows a *real* expression difference
between niches. A method that used spatial position directly, instead of through this
graph, would risk fragmenting every niche-invariant cell type into artificial per-niche
pieces it doesn't actually have.

**2. The convex-combination assumption, applied in embedding space, matches how real
cell-state variation is structured — and this is independently validated in the
literature, not just a convenient modeling choice.** [Crowley, Alon & Quake — *Pareto
optimality reveals an atlas of cellular archetypes*, PNAS 2026](../literature/crowley-et-al-2026-pareto-optimality-reveals-an-atlas-of-cellular-archetypes.pdf)
tested exactly this assumption on the Tabula Sapiens whole-body atlas (309,193 cells, 173
cell types, 24 tissues, 14 donors after QC), working in **PCA/embedding space**, not raw
counts. They found 75% of well-sampled cell types (82/110) are well fit by a
low-dimensional polytope (a simplex — line, triangle, tetrahedron, ...) in that space,
and that the polytope's vertices — its archetypes — have significantly enriched,
functionally distinct gene programs (FDR=0.10, confirmed against a shuffled-data null).
Their framing: cells trade off performance on competing tasks, and Pareto-optimal
phenotypes are mathematically confined to a polytope's vertices and edges. That's the
same convex-hull assumption SEACells/scProto make about metacells, tested independently
and confirmed on real biology, in embedding space specifically — exactly where SEACells
and scProto build their own archetypes.

**3. The shared, learned embedding removes batch effects, in principle enabling
cross-batch metacells.** scProto's encoder is a conditional VAE (scPoli-style) that
already takes each cell's batch/section as an explicit conditioning input (`condition_key
='section'` in every training run so far), so the embedding space cells get assigned to
prototypes *in* is trained to be batch-invariant before the archetypal step even happens
— combined with the per-batch/per-section handling already built into scProto's losses,
this gives a structural path to metacells that span multiple batches/sections sharing the
same underlying biology. SEACells has no equivalent: it decomposes one similarity matrix
built from whichever cells it's given, with no batch-conditioning step at all.
**Caveat:** this is a real, implemented capability (`batch_entropy` in the evaluation
metrics, multi-batch dataset configs already in the repo — `pbmc-immune`, `bmpbmc-immune`,
`lung`, `cd34`), but every spatial/niche experiment in this doc so far uses a single
tissue section, so it hasn't yet been empirically demonstrated in this niche-aware
context — a real capability, not yet a measured result here.

---

What follows picks up from there: once we know what scProto is *trying* to do, how well
does it actually do it in practice, and where does it fall short of SEACells?

### The core difference: checking every cell, vs. checking a summary

Both SEACells and scProto build a small set of "archetypes" or "prototypes" — the
metacells — and both use the cell-cell similarity graph from Part 2 to do it. The
difference that matters most is *how carefully each method checks its own work*.

SEACells checks itself as strictly as possible: for **every single cell**, it asks "does
my current set of archetypes correctly predict this exact cell's full similarity profile
— i.e. which other cells it's actually close to?" (this is the `M ≈ MBA` reconstruction
mentioned above — `M` is the full cell-cell similarity table, and the method is graded on
how well it can rebuild that *entire* table from a small number of archetypes). If one
archetype has quietly become a mix of two unrelated groups of cells, that shows up
immediately: some cells assigned to it will have their true neighbors predicted
completely wrong, and the method is penalized for it, cell by cell.

scProto's equivalent check (`nassoc_loss`) is much coarser: instead of checking every
cell, it only checks **one summary number per pair of prototypes** — roughly, "on
average, how strongly do these two groups' cells connect to each other?" That's the
statistical equivalent of judging a neighborhood by its average income instead of asking
every household — two genuinely different groups of cells can still average out to look
fine together, even though, cell by cell, one prototype is secretly gluing together two
things that don't belong together. Summing throws away *which* cells the edges connect,
and that's exactly the information that would have caught the problem.

`sim_recon_loss` is scProto's attempt to close that gap: it makes each prototype predict
every individual cell's similarity profile too — not just the group-level summary — so it
has to pass the same cell-by-cell test SEACells applies natively. It's the closest thing
scProto has to SEACells' own reconstruction objective.
(Full technical derivation: [`seacells_kernel_archetypal_vs_scproto_losses.md`](seacells_kernel_archetypal_vs_scproto_losses.md).)

### Real measured results

All runs below: `s28nsc` dataset, 58,423 cells, 800 prototypes, `arbf` affinity, identical
architecture/config except the one flagged variable — a clean single-variable ablation.
Sources: [`train_eval_sim_recon.ipynb`](../notebooks/train_eval_sim_recon.ipynb)
(baseline, `+full`) and
[`train_eval_sim_recon_diffusion.ipynb`](../notebooks/train_eval_sim_recon_diffusion.ipynb)
(`+diffusion_t0.5`, the variant mathematically equivalent to rank-*k* MSE on the affinity
matrix itself — see [`sim_recon_global_vs_local_compaction.md`](sim_recon_global_vs_local_compaction.md)
for why `t=0.5` is exactly that, not just "similar to it").

| config | purity | niche purity | modularity | scgraph corr. |
|---|---|---|---|---|
| `arbf` (no sim_recon) | 0.862 | 0.856 | 0.465 | 0.354 |
| `arbf` + `sim_recon` (`full` target) | 0.899 | 0.875 | 0.454 | 0.368 |
| `arbf` + `sim_recon` (`diffusion`, `t=0.5`) | **0.904** | **0.904** | 0.450 | **0.645** |

Both purity and niche purity move **monotonically upward** as the reconstruction target
gets closer to literally what SEACells' RSS optimizes — `t=0.5` is the Eckart-Young-optimal
rank-*k* affinity reconstruction, and it also nearly doubles `scgraph_corr_avg` (how well
the metacells' structure tracks real gene-expression relationships). This is a real,
measured lever, not just a theoretical one.

**But the aggregate gain hides a real, measured cost**, exactly where the theory
predicted it would (fine/rare local structure lives in the low-eigenvalue directions
`t=0.5` deprioritizes). One concrete pairing, tracked across all three runs:

| config | Macrophages × Macrophage-islands purity | same, coverage |
|---|---|---|
| `arbf` | 0.246 | 0.678 |
| `arbf+full` | 0.220 | 0.672 |
| `arbf+diffusion_t0.5` | 0.116 | **0.006** |

Coverage for that specific niche pairing collapses from 0.68 to 0.006 under `t=0.5` — the
run that wins on every aggregate number is quietly failing almost completely on one
concrete rare/local pattern. This confirms the trade-off empirically, not just
theoretically.

**How big is the scProto-vs-SEACells gap to begin with?**
[`scproto_spatial_comparison.ipynb`](../notebooks/scproto_spatial_comparison.ipynb)
(completed run, a different — per-cell median — purity formula, not directly numerically
comparable to the aggregate table above) compares baseline scProto variants (no
`sim_recon` yet) against SEACells (PCA) on per-cell-type purity:

| cell type | scProto (BANKSY) | SEACells (PCA) | gap |
|---|---|---|---|
| Tumor cells (common) | 0.99 | 1.00 | ~0 |
| Fibroblasts (common) | 0.87 | 0.94 | small |
| Macrophages | 0.63 | 0.83 | moderate |
| Vascular endothelium | 0.73 | 0.94 | moderate |
| Dendritic cells (rare) | 0.09 | 0.61 | **large** |
| B cells (rare) | 0.01 | 0.18 | **large** |
| Alveolar cells (rare) | 0.21 | 0.86 | **large** |

The gap is small for common, spatially-simple cell types and large specifically for rare
or spatially-heterogeneous ones — precisely the population `nassoc`'s single-scalar
summary is weakest on, and precisely what `sim_recon` was built to help with.

### Why the gap is structural, not (only) a tuning problem

`sim_recon` never trains alone — it's one of five loss terms simultaneously reshaping the
same `soft_assign`/prototype parameters every step, three of them (`umap`, `nassoc`,
`sim_recon`) at equal λ=1, versus `proto_recon`'s λ=0.01 as the only real-expression
grounding force. The most plausible direct antagonist is `proto_usage_loss`: it rewards
*even* usage across prototypes regardless of whether real population sizes support an
even split — and in real scRNA-seq/spatial data they never do. (Full breakdown:
[`sim_recon_competing_losses.md`](sim_recon_competing_losses.md).)

The deeper, structural point: SEACells gets purity **and** no-dead-archetypes as automatic
consequences of *one* hard, simplex-constrained convex optimization (Frank-Wolfe over
`A`/`B`) — an archetype that doesn't help reduce `‖M−MBA‖²` is wasted budget the
optimization itself wants to reallocate. scProto instead approximates those same two
properties with **two separate, soft, competing losses** (`nassoc` for purity,
`proto_usage` for anti-collapse) that actively fight each other whenever population sizes
are uneven. Adding `sim_recon` adds a third voice to the purity side of that tug-of-war;
it doesn't touch the actual source of the conflict.
(Full argument: [`seacells_kernel_archetypal_vs_scproto_losses.md`](seacells_kernel_archetypal_vs_scproto_losses.md).)

Two concrete, already-scoped fixes to that specific conflict, not yet run:
`lambda_proto_usage=0` with `sim_recon` on (cheapest test), and
`proto_decoupled=True, gmm_resurrect=True` — an EMA-based hard-anchoring mechanism that
sets prototype position directly from assigned cells' embeddings (no gradient, no
competing-loss seat) and resurrects dead prototypes explicitly, a much closer structural
analogue of SEACells' own `B`-matrix update.
(Details: [`proto_anchoring_vs_proto_usage.md`](proto_anchoring_vs_proto_usage.md).)

### Does scProto work as well as, or better than, SEACells? Why?

**Matching SEACells on graph-purity: plausible, and already trending that way.**
`sim_recon` trains scProto toward literally the same target SEACells' RSS optimizes, and
the measured numbers above move monotonically toward it as the target gets closer to that
literal objective (baseline → `full` → `diffusion_t0.5`: purity 0.862 → 0.899 → 0.904,
niche purity 0.856 → 0.875 → 0.904). Nothing about the objective is out of reach in
principle.

**Beating SEACells outright, as currently weighted: unlikely** — for a structural reason,
not a training-budget one. SEACells' purity and anti-collapse guarantee come for free from
a single hard convex constraint; scProto approximates the same two properties with two
separately-tuned soft losses that fight each other whenever real population sizes are
uneven, which is always. Piling more graph-topology loss terms onto that stack (which is
what `sim_recon` is) produces diminishing, not compounding, returns — visible directly in
the numbers: `full` bought +0.037 purity, `diffusion_t0.5` bought only +0.005 more purity
while quietly collapsing coverage on a real rare-niche pairing from 0.68 to 0.006. Until
the `proto_usage`-vs-purity conflict itself is resolved (`proto_decoupled`+`gmm_resurrect`
is the concrete candidate), scProto is trying to out-tune a soft approximation of
something SEACells gets structurally for free.

**Where scProto has a genuine structural edge SEACells cannot match:**
- **Real gene-expression grounding.** `proto_recon_loss` keeps prototypes tied to actual
  expression; SEACells' objective never looks at raw/PCA expression at all — only the
  kernel/affinity matrix. A scProto prototype can in principle stay graph-pure *and*
  decode to a biologically realistic expression profile, a property SEACells doesn't
  optimize for at all.
- **Cross-batch/cross-section metacells.** scProto's per-batch loop (already used for
  `nassoc` and `sim_recon`) and shared embedding space are built to reconcile multiple
  sections into one prototype set. SEACells' kernel-archetypal analysis has no native
  batch-integration mechanism — it's a single-matrix decomposition with no equivalent of
  a batch-conditional term.

**Bottom line:** scProto can plausibly match SEACells' own graph-purity numbers once the
`proto_usage` structural conflict is fixed — `sim_recon` already gets it partway there,
measurably. Beating SEACells outright is more likely to come from the two axes SEACells
structurally can't touch (expression grounding, batch integration) than from further
tuning of the affinity-reconstruction losses SEACells already optimizes optimally by
construction.

### Challenges hit along the way (briefly)

- **Loss collapsed to ~0 in 1-2 epochs (`diffusion` target).** Not a real vanishing
  gradient — an `eigsh` scaling artifact made near-zero output trivially near-optimal.
  Fixed by rescaling eigenvectors to unit RMS and dropping the trivial leading one.
  ([`sim_recon_diffusion_coordinates.md`](sim_recon_diffusion_coordinates.md))
- **Loss collapsed to ~0 again (`full` target).** Different cause: the target row is
  >99.8% zero, so plain MSE's trivial optimum is "predict zero everywhere." Fixed with
  class-balanced weighted MSE (upweight the rare true-neighbor entries) plus a `softplus`
  decoder output (targets are never negative).
  ([`sim_recon_sparsity_decision.md`](sim_recon_sparsity_decision.md))
- **`full` target too expensive at 58k cells** (every cell reconstructed against every
  other cell in its batch). Fixed with column-subsampling (`sim_recon_neg_sample`): always
  keep real neighbor columns, freshly resample the zero columns each step.
  ([`sim_recon_full_column_subsampling.md`](sim_recon_full_column_subsampling.md))

### What's left open (scoped, not yet run)

- `lambda_proto_usage=0` + `sim_recon` on, and `proto_decoupled=True, gmm_resurrect=True`
  as its replacement — both proposed, neither run yet.
- `full` + `sim_recon_neg_sample`, run in isolation (other graph-topology losses zeroed),
  to measure `sim_recon`'s ceiling on its own rather than diluted by four other losses.
- A direct, same-formula, same-run comparison of scProto+`sim_recon` against SEACells —
  every notebook set up for this (`train_eval_sim_recon_negsample.ipynb`,
  `train_eval_sim_recon_only_vs_seacell.ipynb`, `train_eval_seacell_diffusion.ipynb`) was
  interrupted mid-run (Colab disconnects) before producing a completed result; the numbers
  above are the real, completed subset available today.
- Batch-effect removal / cross-batch metacells (Part 3, argument 3) — implemented, never
  run on a multi-section spatial dataset.

---

## Looking for feedback on

1. **Part 1/2** — is mean spatial-context (over COVET, or a mix) the right call given our
   labels and metrics, or are we missing a case where covariance should dominate?
2. **Part 3, convex-hull argument** — does grounding archetypal analysis in embedding
   space, backed by the Crowley et al. Pareto-optimality result, hold up as a biological
   argument, or is it a weaker analogy than we think?
3. **scProto vs. SEACells** — is the structural read (soft competing losses vs. one hard
   convex constraint) the right diagnosis, or is there a simpler explanation for the gap
   we're missing?
4. Any blind spot in the overall approach that isn't visible from inside this codebase.
