# Why scProto+sim_recon still underperforms SEACells — which other loss terms might be fighting it

Context: `notebooks/train_eval_sim_recon_diffusion.ipynb` shows scProto, even
with `sim_recon` (+ `diffusion` target) added, still scoring notably worse
than SEACells on niche/celltype purity. `sim_recon` doesn't train in
isolation — it's one of five active loss terms sharing the same
`soft_assign` / prototype / encoder parameters every step, under the
notebook's config: `LAMBDA_PROTO_UMAP_PRECON | {'nassoc_agg': 'max'}` plus
`lambda_sim_recon=1.0` for the sim-recon run. This doc walks through each
active term, what it optimizes, and how plausible it is as a source of
gradient conflict with `sim_recon`.

## The five active loss terms

All from `interpretable_ssl/trainers/scproto.py`, `_run_umap_epoch` (loss
assembly around lines 1439–1679):

| term | λ | what it does |
|---|---|---|
| `umap_loss` | 1 | Contrastive edge loss: for each affinity-graph positive edge, pulls the two cells' `soft_assign` vectors (softmax of `z @ proto^T`) toward higher dot-product similarity; negative-sampled pairs pushed apart. Builds the embedding/routing structure. |
| `proto_recon_loss` | 0.01 | Decodes each prototype to gene-expression space, MSEs against assigned cells' actual expression (`scores` detached — gradient goes to decoder/prototypes, not encoder). Grounds prototypes in real expression. |
| `nassoc_loss` | 1 | "Normalized association," per-batch then max-aggregated. Pushes the prototype-block affinity matrix's diagonal toward 1 (dense in-prototype edges) and off-diagonal toward 0 (no cross-prototype edge sharing) — a coarse, per-prototype-scalar purity signal. |
| `proto_usage_loss` | 0.1 (EMA mode) | Penalizes prototypes with low aggregate `soft_assign` mass. Pure anti-collapse — has no term for *where* a prototype sits, only *how much* mass it wins. |
| `sim_recon_loss` | 1.0 | Decodes each prototype to a per-cell profile (diffusion coords or full affinity row) and MSEs against the real per-cell target; `soft_assign` keeps gradient here (unlike `proto_recon`). Meant to catch prototypes that glue together two disconnected sub-communities — the fine-resolution signal `nassoc`'s single scalar can't see. |

Everything else (`lambda_swav`, `lambda_kl`, `lambda_recon`, `lambda_r1r2`,
`lambda_proto_attract`) is 0 in this config, confirmed against
`interpretable_ssl/configs/defaults.py`.

## Per-term fight assessment

- **`umap_loss` — low risk.** Contrastive routing loss operating on the
  same affinity graph `nassoc`/`sim_recon` use. In principle aligned;
  no strong reason to expect it to pull `soft_assign` in a genuinely
  different direction than `sim_recon` wants.

- **`proto_recon_loss` — mostly low risk, one real edge case.** λ=0.01 is
  100x smaller than the graph-topology terms, so it shouldn't dominate.
  The plausible conflict: two rare cell types that look similar under the
  graph's adaptive-bandwidth kernel (hence `sim_recon`/`nassoc` want to
  treat them as related) but whose actual gene-expression profiles differ
  more than the kernel's local-density correction implies — `proto_recon`
  would then pull the shared prototype's decoded profile toward a
  compromise expression pattern while `sim_recon`/`nassoc` want one
  prototype only if the graph-affinity structure genuinely supports it.
  Real but likely minor at current weight.

- **`nassoc_loss` — aligned in principle, resolution mismatch in practice.**
  Both `nassoc` and `sim_recon` are reconstructing structure from the same
  affinity graph, so their optima should broadly coincide (see
  `seacells_kernel_archetypal_vs_scproto_losses.md` — this pairing is
  the actual analogue of SEACells' own kernel-matrix objective). But
  `nassoc` only ever sees a single scalar per prototype pair (summed edge
  weight / volume), while `sim_recon` sees per-cell profiles. A prototype
  that looks pure to `nassoc` (good diagonal) but is actually two
  disconnected sub-communities glued together is exactly the case where
  the two losses' *gradients* disagree even though their *global optima*
  don't — `nassoc` has already been satisfied and stops pushing, while
  `sim_recon` is still trying to split. Short-term gradient conflict, not
  a structural one.

- **`proto_usage_loss` — most plausible direct antagonist.** It rewards
  *even* usage across prototypes regardless of whether the data naturally
  splits that way. If real cell-population sizes are uneven (typical for
  scRNA-seq, especially with rare types), `proto_usage` is actively
  fighting any loss — `nassoc` or `sim_recon` — that wants an uneven,
  purity-optimal split. It's also purely about aggregate softmax mass, not
  about prototype *position*, so it doesn't even deliver the "stay on the
  data manifold" property one might hope a usage constraint would give.
  See `seacells_kernel_archetypal_vs_scproto_losses.md` and
  `proto_anchoring_vs_proto_usage.md` for the deeper dive and the
  alternative (`proto_decoupled` + `gmm_resurrect`) this points toward.

## The weight-imbalance point

Three terms sit at λ=1 (`umap`, `nassoc`, `sim_recon` when added) — all
graph-topology objectives, all reshaping the same `soft_assign` matrix
simultaneously — while the only real-expression grounding force
(`proto_recon`) sits at λ=0.01, two orders of magnitude weaker. SEACells'
entire objective is graph/kernel-matrix reconstruction (see
`seacells_kernel_archetypal_vs_scproto_losses.md`) with no separate
expression term at all, so this isn't necessarily wrong by comparison — but
it does mean scProto currently has three same-weight losses simultaneously
pulling on `soft_assign` (`umap`, `nassoc`, `sim_recon`) plus a fourth
(`proto_usage`) actively working against purity, versus SEACells' single
structurally-constrained convex objective. That's a plausible reason
`sim_recon`'s marginal fine-resolution correction gets diluted rather than
decisively winning.

## Suggested diagnostic ablations (not yet run)

1. `lambda_proto_usage=0` with `sim_recon` on — cheapest test of the most
   suspected antagonist.
2. `proto_decoupled=True, gmm_resurrect=True, lambda_proto_usage=0` — the
   structural alternative to `proto_usage`; see
   `proto_anchoring_vs_proto_usage.md`.
3. `full` + neighbor-subsampling, run with `lambda_umap`/`lambda_nassoc`/
   `lambda_proto_recon` all zeroed, to see `sim_recon`'s ceiling in
   isolation — already flagged as an open question in the index.
