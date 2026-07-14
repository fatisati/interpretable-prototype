# `sim_recon_target='diffusion'`: what it is, why we use it, what broke, how we fixed it

## What a diffusion coordinate is (short version)

Take the cell-cell affinity graph, row-normalize it into a random walk
(`P = D⁻¹A`, "probability of hopping from cell i to cell j"), and symmetrize
it (`L_sym = D^{-1/2} A D^{-1/2}`). Diagonalize `L_sym` instead of actually
running the walk: `L_sym = V Λ Vᵀ`. Each cell's row in the top few columns of
`V` — a handful of numbers per cell — is its **diffusion coordinate**.
Euclidean distance between two cells' coordinates approximates how hard it is
for a random walk to get from one to the other ("diffusion distance"), not
raw similarity.

This is the same object as a **Laplacian eigenmap / spectral embedding**.
The textbook "diffusion map" is this plus an `eigenvalue^t` weighting
(`t` = diffusion time) that upweights coarse/global eigenvectors over
fine/local ones; at `t=0` that weighting is a no-op, so what we compute is
the `t=0` corner of the same family — see "`diffusion_t` removed" below for
why we deliberately stayed there.

Mathematically it's close to what you'd get from PCA on the similarity
matrix directly (since the affinity matrix is symmetric, its principal
components *are* its eigenvectors) — the row-normalization just prevents
high-degree cells from dominating the components purely by having more/
stronger edges.

## Why it's a good compression target

`sim_recon_loss` exists to give scProto per-cell resolution pressure that
`nassoc` can't provide: `nassoc`'s diagonal only sees a summed scalar per
prototype (in-cluster edge weight / volume), so it can't tell a genuinely
homogeneous prototype from one that secretly glues together two internally-
cohesive but mutually disconnected sub-communities — summing throws away
*which* cells the edges connect. Reconstructing each cell's actual affinity
row (`sim_recon_target='full'`) fixes that, but the target is the full
n_cells-wide row: >99.8% zero, most of it uninformative for a given cell,
and expensive to decode.

The affinity graph's real information content is much lower-rank than
n_cells — diffusion coordinates are exactly a way to compress "this cell's
similarity pattern" into a small, dense, non-sparse vector (`n_eigs`
dimensions, e.g. 10–128) without hand-picking a projection: the eigenbasis
is the graph's own natural low-rank summary, not an arbitrary one. If a
prototype's cells don't actually share a similarity pattern, they won't
share diffusion coordinates either, so the resolution pressure survives the
compression in principle — this notebook-pair (`train_eval_sim_recon.ipynb`
for `full`, `train_eval_sim_recon_diffusion.ipynb` for `diffusion`) exists to
check that it survives in practice too.

## The challenge: `sim_recon_loss` collapsed to ~0 within 1-2 epochs

Symptom: loss looked like it converged almost instantly, which read like
vanishing gradient. It wasn't a gradient problem — it was a **target-scale**
problem specific to raw eigenvector output:

- `eigsh` returns each eigenvector unit-L2-norm *over the whole batch*
  (`Σ vᵢ² = 1`). Spread over `N_batch` cells, that puts the average entry at
  `~1/√N_batch` — for a few thousand cells, individual target values are
  already tiny (`~0.01–0.03`).
- A decoder that just outputs near-zero for every prototype already sits
  almost exactly on the MSE floor (`mean(target²) ≈ 1/N_batch`), so "loss
  ≈ 0 fast" didn't mean the model learned the similarity structure — it
  meant it learned "output ≈ 0" and stopped.

Two smaller issues rode along with the same function:
- The top eigenvector (eigenvalue ≈1 for a connected graph, direction
  ~`√degree`) is a trivial "how well-connected is this cell overall"
  signal, not discriminative between cells — it was being kept and wasting
  one of the `n_eigs` target dimensions.
- A `sim_recon_diffusion_t` config option existed but was never actually
  passed into the target-computation call — every run silently used
  unweighted eigenvectors regardless of what was configured.

## How we solved it

In `_compute_sim_recon_diffusion_targets`
(`interpretable_ssl/trainers/scproto.py`):

1. **Rescale to O(1) RMS entry.** Multiply each batch's eigenvectors by
   `√N_batch`, converting `Σvᵢ²=1` to `mean(vᵢ²)≈1` — target scale now
   reflects the graph, not an `eigsh` bookkeeping convention, so a
   near-zero decoder output is no longer trivially near-optimal.
2. **Drop the trivial leading eigenvector.** Request `n_eigs+1`
   eigenvectors, sort descending by eigenvalue (`eigsh(..., which='LM')`
   returns them ascending, not by magnitude — a re-sort is required to
   reliably find the largest), drop the first, keep the next `n_eigs`.
3. **Removed the `diffusion_t` knob entirely**, rather than fixing its
   wiring. `sim_recon`'s whole purpose is catching a prototype that splits
   into disconnected sub-communities, and that signal lives in the
   fine/local (small-eigenvalue) directions. `diffusion_t>0` would
   upweight the coarse/global directions instead, actively burying the
   signal this loss needs — so unweighted (`t=0`, every eigenvector equal)
   isn't a default waiting to be tuned, it's the only setting that fits
   what the loss is for. One less footgun, one less bug surface.
4. **Added a cheap collapse check.** Each training epoch now prints
   `sim_recon=... [pred_std=... target_std=...]`. If `pred_std` stays near
   0 while `target_std` doesn't, the decoder has collapsed to a
   near-constant output — visible immediately instead of inferred from the
   loss curve.

Confirmed *not* a bug, while investigating: the target is computed exactly
once per training run (inside `_setup_umap_edges`, which is guarded against
re-running mid-training) and cached — not recomputed per epoch or
iteration.

## Where this is tracked

`notebooks/train_eval_sim_recon_diffusion.ipynb` — trains the `arbf`
baseline + `arbf+diffusion` sim-recon run, loads SEACells, and compares all
three (purity tables, niche heatmaps, diff-vs-baseline heatmap, trade-off
plot, UMAPs). Kept separate from `train_eval_sim_recon.ipynb`, which is
dedicated to the `full` target so the two don't clobber each other's runs.
