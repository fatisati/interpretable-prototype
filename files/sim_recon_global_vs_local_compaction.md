# Why we don't weight `sim_recon`'s diffusion loss by eigenvalue (variance)

> **Update**: the follow-up below ("Refinement — `sqrt(eigenvalue)` weighting
> is exactly MSE-on-the-affinity-matrix") sharpens the conclusion here. The
> "don't do it" framing still holds as the *default*, but weighting turned
> out to be a precise, real dial rather than simply a mistake — it's now
> implemented (`sim_recon_diffusion_t`) and worth reading to the end.

## The question this resolves

If diffusion coordinates decay slowly (`sim_recon_full_column_subsampling.md`
— 734-882 of them needed for 90-99% of the graph's energy), and every kept
dimension is rescaled to the same O(1) RMS, an obvious next idea is: weight
each dimension's contribution to the loss by how much variance/energy it
explains, so the decoder is pushed hardest to get the big, important
directions right. Does that make sense, and would it cost us the rare/fine
cell-type patterns `sim_recon` exists to protect?

## Short answer: don't do it — and the reason is more fundamental than "wrong weighting"

**This is literally the `diffusion_t` (eigenvalue^t) knob**, already built,
already removed — see `sim_recon_diffusion_coordinates.md`:

> `diffusion_t>0` would upweight coarse/global eigenvectors over fine/local
> ones, but the fine/local directions are exactly what let this loss catch
> a prototype that's secretly gluing together two disconnected
> sub-communities... so unweighted (every eigenvector equal) isn't a
> default waiting to be tuned, it's the only setting that fits what the
> loss is for.

Eigenvalue ranks a direction by **population-wide** variance explained. A
rare cell type, by definition, contributes little population variance — so
weighting the loss by eigenvalue would deprioritize reconstructing exactly
the directions a rare/fine pattern is most likely to live in.

## Why reweighting the loss doesn't actually fix this (the deeper point)

The damage isn't only in the loss weighting — it already happened one step
earlier, at **which eigenvectors get kept at all**. `n_eigs` is a hard
top-k cutoff on population-wide variance (`_compute_sim_recon_diffusion_targets`
keeps the top `n_eigs` eigenvalues by magnitude). A rare pattern that
doesn't carry much aggregate variance has weak influence on whether its
defining direction even survives that cutoff, regardless of anything you do
to the loss afterward. Reweighting only changes how hard the decoder tries
on dimensions *already selected* — it can't resurrect a direction the top-k
truncation already dropped. Loss-side weighting is a fix applied one stage
too late.

**Second trap, if you tried the opposite (upweight the *fine* tail instead,
to protect rare patterns):** eigenvalue magnitude can't distinguish "genuine
rare biological signal" from "numerical noise" — both live at the low end
of the spectrum, and `eigsh`'s own `tol=1e-2` makes the deepest part of that
tail the least numerically trustworthy to begin with. Upweighting the tail
amplifies real rare signal and noise indiscriminately, with no way to tell
them apart from the eigenvalue alone.

## The actual fix: compact locally (per-cell), not globally (shared basis)

The distinction that matters: **diffusion coordinates compress globally** —
one fixed, shared basis for the whole dataset, chosen by population-wide
variance. Any rare pattern is structurally disadvantaged at the moment that
basis is selected, no matter how later stages weight it.

`sim_recon_target='full'` with `sim_recon_neg_sample` (column-subsampling —
see `sim_recon_full_column_subsampling.md`) **compacts locally, per cell**,
instead:

- Every cell's own real graph neighbors (`pos_cols`) are always kept, with
  certainty, regardless of how common or rare that cell's pattern is
  dataset-wide — "is this common across the whole population" never enters
  the decision.
- Only the *zero* population (genuine non-neighbors) gets subsampled, and
  freshly re-drawn every step, so nothing about which negatives are shown
  is permanently fixed either.
- The compaction ratio (~75 kept neighbor-columns out of 58k, plus however
  many sampled negatives) comes from each row's own sparsity, not from a
  population-level importance ranking.

This is the structural reason `full` + neighbor-subsampling is the safer
"compact but don't lose rare patterns" answer, rather than any per-dimension
reweighting scheme on top of a fixed global eigenbasis.

## If you still want a reweighting-style fix

The concern here is really about *cells* (rare cell types), not *dimensions*
— so if a compaction scheme needs some kind of reweighting, it should be
per-row (e.g. upweight cells from small/rare local neighborhoods, similar
in spirit to the class-balanced positive/negative weighting already used
for `full`'s zero-flood problem in `sim_recon_sparsity_decision.md`), not
per-eigen-dimension. Not implemented — noted here as the more targeted
alternative if `full` + neighbor-subsampling alone turns out to still
under-serve rare cell types in practice.

## Refinement — `sqrt(eigenvalue)` weighting is exactly MSE-on-the-affinity-matrix

Follow-up question that came up after the above: "if we weight by eigenvalue,
would that be similar to MSE on the real similarity matrix?" Yes — and
precisely, not just approximately, which sharpens (doesn't overturn) the
"don't do it" conclusion above into "here's the exact knob, and here's
exactly what it costs."

**The math.** `L_sym = V Λ Vᵀ = Σₖ λₖ vₖ vₖᵀ`. By the Eckart-Young theorem,
the best possible rank-`k` approximation of a symmetric matrix (lowest
Frobenius/squared error) is its own top-`k` eigendecomposition,
`L_sym_k = Σₖ λₖ vₖ vₖᵀ`. If each eigenvector is scaled by `sqrt(λₖ)` (not
`λₖ` itself — the exponent matters) before being used as a per-cell
coordinate, then two cells' weighted-coordinate dot product is:

```
weighted_i · weighted_j = Σₖ λₖ · vₖ[i] · vₖ[j]  =  (L_sym_k)[i, j]
```

i.e. exactly the rank-`k`-optimal reconstructed similarity value between
those two cells. So per-cell MSE on `sqrt(λ)`-weighted diffusion coordinates
is mathematically the *same objective* as MSE on a reconstructed similarity
matrix (up to the error from keeping only `n_eigs` of the ~58k possible
directions) — not merely similar to it. This is the classical fact behind
kernel PCA / classical multidimensional scaling: unweighted eigenvectors are
just directions, and `sqrt(eigenvalue)` scaling is exactly what turns them
into a metric-preserving embedding of the original similarity structure.
In this codebase's terms, that's `diffusion_t=0.5` specifically (not `t=1`,
which reconstructs `L_sym²`, a smoothed/2-step version, not `L_sym` itself).

**This doesn't overturn "don't weight by default" — it explains exactly what you
trade for what.** `t=0.5` gets you the closest `diffusion` mode can come to
behaving like `sim_recon_target='full'` (or SEACells' own RSS): a real
mathematical equivalence, not a vague "upweight the important stuff." But
`sqrt(λ)`-weighting is *why* that equivalence holds — it's pulling the loss
toward exactly what raw-affinity MSE already does (prioritize big, common,
coarse relationships), which is the same reason `t=0` was chosen for
`sim_recon` in the first place: fine/rare sub-community structure lives
disproportionately in the lower-`λ` directions, and any `t>0` shrinks their
influence back down. There's no setting that gets both properties at once,
because a rank-`k` approximation is optimal for *aggregate* squared error
precisely by spending its budget on the highest-variance directions, at the
expense of the low-variance ones — the trade-off is not a bug to route
around, it's what "optimal rank-k approximation" means.

**Implemented as a real, tunable dial** — `sim_recon_diffusion_t`
(`configs/defaults.py`, abbreviated `srdt` in `constants.py`,
`_compute_sim_recon_diffusion_targets` in `trainers/scproto.py`). Default
`0.0` (unchanged behavior, everything before this refinement). Set to `0.5`
to run the SEACells-equivalent version. `notebooks/train_eval_sim_recon_diffusion.py`
now trains both (`arbf+diffusion` at `t=0`, `arbf+diffusion_t0.5`) alongside
the `arbf` baseline and SEACells, so the trade-off predicted above (closer
to SEACells overall, at some cost to rare-cell-type purity specifically) is
directly checkable in that notebook's per-cell-type purity table rather than
staying a theoretical claim.

## Where this is tracked

- `sim_recon_diffusion_coordinates.md` — the original `diffusion_t` removal
  decision this revisits and confirms (then the refinement above partially
  reverses, as a tunable experiment rather than a restored default).
- `sim_recon_full_column_subsampling.md` — the neighbor-subsampling
  mechanism this recommends leaning on instead, for the default/no-tuning
  case.
- `sim_recon_sparsity_decision.md` — the row-wise (not dimension-wise)
  class-balanced weighting pattern a future per-cell rarity weight would
  follow.
- `notebooks/train_eval_sim_recon_diffusion.py` (`.ipynb`) — trains and
  compares `t=0` vs `t=0.5` vs the `arbf` baseline vs SEACells.
