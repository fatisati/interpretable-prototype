# What `sim_recon` still misses, why SEACells wins, and can scProto close the gap?

*(Continues after the "better algorithm" section — SEACells kernel-archetypal analysis on
the improved similarity graph, then scProto, convex-hull assumption, why that can be
biologically better. This section covers what scProto's per-cell resolution loss
(`sim_recon`) does and doesn't fix, with real measured numbers, and closes with a direct
verdict on whether scProto can match or beat SEACells.)*

## 1. The gap `sim_recon` exists to close

SEACells' archetypal analysis reconstructs each cell's **full row** of the cell-cell
kernel/affinity matrix (`M ≈ MBA`, hard simplex constraints on `A`/`B`) — every cell's
actual similarity profile is fit, so an archetype that secretly glues together two
disconnected sub-communities is directly penalized. scProto's `nassoc_loss` only ever
sees a single **summed scalar per prototype pair** (in-cluster edge weight / volume) — it
can't tell a genuinely homogeneous prototype from one that quietly straddles two
unrelated cell groups, because summing throws away *which* cells the edges connect.
`sim_recon_loss` was added to give scProto that missing per-cell pressure: it decodes
each prototype into a predicted per-cell profile (either the cell's real affinity row, or
a compressed diffusion-map coordinate of it) and reconstructs it through the soft
assignment — the closest scProto analogue of SEACells' own `‖M − MBA‖²` objective.
(Full derivation: [`seacells_kernel_archetypal_vs_scproto_losses.md`](seacells_kernel_archetypal_vs_scproto_losses.md).)

## 2. Real measured results

All runs below: `s28nsc` dataset, 58,423 cells, 800 prototypes, `arbf` affinity, identical
architecture/config except the one flagged variable — a clean single-variable ablation.
Sources: [`notebooks/train_eval_sim_recon.ipynb`](../notebooks/train_eval_sim_recon.ipynb)
(baseline, `+full`) and
[`notebooks/train_eval_sim_recon_diffusion.ipynb`](../notebooks/train_eval_sim_recon_diffusion.ipynb)
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
`t=0.5` deprioritizes — [`sim_recon_global_vs_local_compaction.md`](sim_recon_global_vs_local_compaction.md)).
One concrete pairing, tracked across all three runs:

| config | Macrophages × Macrophage-islands purity | same, coverage |
|---|---|---|
| `arbf` | 0.246 | 0.678 |
| `arbf+full` | 0.220 | 0.672 |
| `arbf+diffusion_t0.5` | 0.116 | **0.006** |

Coverage for that specific niche pairing collapses from 0.68 to 0.006 under `t=0.5` — the
run that wins on every aggregate number is quietly failing almost completely on one
concrete rare/local pattern. This is the same trade-off `sim_recon_global_vs_local_compaction.md`
predicted from first principles, now confirmed empirically rather than just argued
theoretically.

### How big is the scProto-vs-SEACells gap to begin with?

[`notebooks/scproto_spatial_comparison.ipynb`](../notebooks/scproto_spatial_comparison.ipynb)
(completed run, different — per-cell median — purity formula, not directly numerically
comparable to the aggregate table above) compares baseline scProto variants (no
`sim_recon` yet) against SEACells (PCA) on per-cell-type purity. Selected rows
(median `celltype_purity`, scProto's best baseline variant = BANKSY affinity, vs. SEACells):

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

## 3. Why the gap is structural, not (only) a tuning problem

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
it doesn't touch the actual source of the conflict. (Full argument:
[`seacells_kernel_archetypal_vs_scproto_losses.md`](seacells_kernel_archetypal_vs_scproto_losses.md).)

Two concrete, already-scoped fixes to that specific conflict, not yet run:
`lambda_proto_usage=0` with `sim_recon` on (cheapest test), and
`proto_decoupled=True, gmm_resurrect=True` — an EMA-based hard-anchoring mechanism that
sets prototype position directly from assigned cells' embeddings (no gradient, no
competing-loss seat) and resurrects dead prototypes explicitly, a much closer structural
analogue of SEACells' own `B`-matrix update. (Details:
[`proto_anchoring_vs_proto_usage.md`](proto_anchoring_vs_proto_usage.md).)

## 4. Does scProto work as well as, or better than, SEACells? Why?

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
  `nassoc` and `sim_recon`, [`sim_recon_sparsity_decision.md`](sim_recon_sparsity_decision.md))
  and shared embedding space are built to reconcile multiple sections into one prototype
  set. SEACells' kernel-archetypal analysis has no native batch-integration mechanism —
  it's a single-matrix decomposition with no equivalent of a batch-conditional term.

**Bottom line:** scProto can plausibly match SEACells' own graph-purity numbers once the
`proto_usage` structural conflict is fixed — `sim_recon` already gets it partway there,
measurably. Beating SEACells outright is more likely to come from the two axes SEACells
structurally can't touch (expression grounding, batch integration) than from further
tuning of the affinity-reconstruction losses SEACells already optimizes optimally by
construction.

## What's left open (scoped, not yet run)

- `lambda_proto_usage=0` + `sim_recon` on, and `proto_decoupled=True, gmm_resurrect=True`
  as its replacement — both proposed, neither run yet.
- `full` + `sim_recon_neg_sample`, run in isolation (other graph-topology losses zeroed),
  to measure `sim_recon`'s ceiling on its own rather than diluted by four other losses.
- A direct, same-formula, same-run comparison of scProto+`sim_recon` against SEACells —
  every notebook set up for this (`train_eval_sim_recon_negsample.ipynb`,
  `train_eval_sim_recon_only_vs_seacell.ipynb`, `train_eval_seacell_diffusion.ipynb`)
  was interrupted mid-run (Colab disconnects) before producing a completed result; the
  numbers in this doc are the real, completed subset available today.
