# Is `proto_usage_loss` needed, and what should replace/complement it?

> **Update**: option A below is now implemented as `lambda_proto_anchor`
> (`configs/defaults.py`, loss block in `trainers/scproto.py`'s
> `_run_umap_epoch`, right after `proto_attract_loss`). Default `0.0` (off,
> unchanged behavior). Mutually exclusive with `proto_decoupled=True` — the
> loss is skipped (with a startup-log note) if both are set, since they're
> two different mechanisms for the same job. Not yet run against option B.

Follow-up to `seacells_kernel_archetypal_vs_scproto_losses.md`. That doc
established SEACells gets purity *and* no-dead-archetypes for free from one
structurally-constrained convex optimization on the kernel matrix
(`min_{A,B} ||M-MBA||²`, `A`/`B` column-stochastic), while scProto currently
tries to approximate those two properties with two separate, competing soft
losses: `nassoc_loss` (purity-like) and `proto_usage_loss` (anti-collapse).
This doc works through whether `proto_usage_loss` earns its keep, and
compares two candidate replacements/complements.

## What `proto_usage_loss` actually regularizes

`scproto.py:1420-1437`. All three modes (`max`, `ema`, `nk`) are functions
of `soft_assign.sum(dim=0)` or `soft_assign.max(dim=0)` — aggregate softmax
*mass* per prototype. There is **no term for prototype position** anywhere
in it. It only affects position indirectly: winning more mass requires a
higher `dot(z, proto)` for some cells, so the gradient nudges a prototype
toward whichever region lets it pick up more weight — which is not
necessarily the center of a real, coherent cluster. A prototype can satisfy
`proto_usage` by owning a diffuse, low-confidence smear across a contested
boundary region between two real clusters.

It also actively fights purity-seeking losses when real cell-population
sizes are uneven (the normal case): it wants even usage regardless of
whether the data supports an even split.

## Two candidate alternatives considered

### A. Soft MSE anchor in latent space (proposed in discussion)

Reuse the existing `logits = z @ proto^T` (already computed for
`soft_assign`), but softmax over the **cell** axis per prototype column
instead of over the **prototype** axis per cell — giving `W[N,K]`,
column-stochastic (`Σ_i W[i,k] = 1`). This is a direct latent-space
analogue of SEACells' `B` matrix (`Z = MB`, archetype = convex combination
of kernel-matrix columns). Then:

```
recon_proto_k = W[:, k]^T @ z        # (D,) — combination of this batch's cell embeddings
loss = MSE(proto_k, recon_proto_k)   # pulls the free trainable proto toward it
```

`proto` stays a free, gradient-trained parameter; this just adds a pull
toward being explainable as a combination of real embeddings.

**Required implementation detail:** `W` must be `.detach()`ed before
computing `recon_proto`, otherwise gradient flows into `proto` through both
the target side (`W = softmax(z·proto)`) and the loss side at once,
producing unpredictable dynamics — same stop-gradient pattern already used
for `scores` in `proto_recon_loss` (`scproto.py:1414`).

### B. `proto_decoupled` + `gmm_resurrect` (already implemented, currently unused)

`_update_protos_ema`, `scproto.py:1037-1108`. Sets prototype position
directly (no gradient) to an EMA of the weighted mean of assigned cell
embeddings: `new_mu_k = Σ_i S[i,k]·z_i / Σ_i S[i,k]`, accumulated across
minibatches with a per-prototype adaptive forgetting factor
`eta_k = eta^usage_frac` (barely-used prototypes freeze instead of getting
overwritten by one noisy batch). `gmm_resurrect` explicitly splits an
over-dominant prototype into whichever one is most unused, placed near the
dominant one with small noise — an explicit, position-aware anti-collapse
mechanism.

Currently off in every run in this notebook: `proto_decoupled` defaults to
`False` (`configs/defaults.py:223`) and isn't set in
`LAMBDA_PROTO_UMAP_PRECON`.

## Working out the relationship between A and B

`_update_protos_ema`'s update rule (`Σ_i S[i,k]·z_i / Σ_i S[i,k]`) and
option A's `recon_proto_k` (`Σ_i W[i,k]·z_i`, `W` column-normalized) are
**the same combination rule** — dividing by the column sum makes any
nonnegative weight column-stochastic-equivalent, so reusing the existing
row-stochastic `soft_assign` (already normalized per-cell) and dividing by
its column sum achieves the same thing a fresh column-softmax would. Option
A is not a different idea from option B mathematically — it's the same
"proto = weighted combination of cell embeddings" rule, computed and
applied differently:

| | A: soft MSE anchor | B: EMA (`proto_decoupled`) |
|---|---|---|
| proto stays a free gradient param | yes — soft pull, competes for gradient | no — position set directly, no gradient |
| adds to the already-crowded loss stack (see `sim_recon_competing_losses.md`) | yes — a 6th competing term | no — fully decoupled from the shared gradient budget |
| target computed from | current minibatch only (~1024 cells) | running accumulator over *all* minibatches seen, EMA-weighted |
| rare-cell-type protection | none — a rare prototype's true cells may not even be sampled into a given minibatch, so its target that step can be dominated by whichever majority cells happen to be closest | built in — `eta_k = eta^usage_frac` freezes a barely-used prototype's running stats instead of overwriting with single-batch noise |
| anti-collapse | none by itself | `gmm_resurrect` — explicit split-dominant-into-dead-proto step |
| circularity | target `W` depends on `proto` itself (must detach to avoid a moving-target feedback loop) | none — running stats are accumulated from realized assignments, not fed back through the still-updating `proto` in the same step |

## Conclusion / recommendation

`proto_usage_loss` as currently implemented is a comparatively weak,
indirect stand-in for what SEACells' simplex constraint gets structurally:
it only reacts to a prototype that's *already* losing, does nothing for
prototype position, and directly fights purity-seeking losses whenever
real population sizes are uneven. It shouldn't be dropped to zero with
nothing in its place, though — scProto's softmax routing has no built-in
guarantee against dead prototypes the way archetypal analysis does.

Between the two replacements, **B (`proto_decoupled` + `gmm_resurrect`)
is the stronger candidate to try first**: it's already implemented, it
doesn't add a 6th term to a loss stack already suspected of
over-crowding `soft_assign` (see `sim_recon_competing_losses.md`), and its
adaptive forgetting factor specifically protects the rare/low-usage
prototypes that a single-minibatch version (A) would be noisiest for —
which is precisely the failure mode (rare cell types) this whole
investigation cares most about. Option A remains a reasonable fallback if
B's non-gradient, offline-EM-style update turns out to interact badly with
the rest of the training loop in practice.

## Suggested experiment

`proto_decoupled=True, gmm_resurrect=True, lambda_proto_usage=0`, compared
against the existing baseline / `+sim_recon` pair. Not yet run — tracked in
`sim_recon_investigation_index.md`'s open questions.

Now that option A is implemented, a second, cheaper comparison is possible
in the same pass: `lambda_proto_anchor=<something>, lambda_proto_usage=0`
(prototype stays a free gradient param, no `proto_decoupled`) against both
the option-B run above and the existing `proto_usage`-only baseline. Three-
way comparison, not two — also not yet run.
