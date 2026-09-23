# The DGE story: why scProto should beat both SEACells baselines (some niches)

## The problem, as framed

- The `bk08` affinity graph connects cells that share a similar spatial context (and similar cell
  type).
- Using `bk08` as scProto's training signal corrects the pretrained embedding space: until now,
  that embedding mostly groups cells by cell type alone, and does not necessarily capture subtle
  but important spatial (niche) programs.
- What scProto does: the signal says cells of the same (cell type, niche) should be assigned to
  the same prototype. This changes both the latent space and the prototypes to encode this.
- What we want to show: by decoding the prototypes and running the plan1-style DGE analysis,
  there are some (cell type, niche) pairs where scProto recovers well-known genes that match what
  is already known at the single-cell level — while:
  - SEACells on PCA (`arbf`) cannot, because it has no spatial info at all.
  - SEACells on the same `bk08` graph also cannot, because it has no access to transcriptional
    info. Some `bk08` edges are cross-cell-type edges (spatial noise, not real niche signal) that
    shouldn't count — but archetypal analysis doesn't know this and does not correct for it.

## Why scProto could beat SEACells (`arbf`) — no spatial info at all

- `arbf` is built purely from transcriptional similarity (PCA space). It has zero spatial
  information.
- So it cannot separate two niches within the same cell type when the niche's gene expression
  change is subtle — i.e. not a main axis of variance in the transcriptome. That distinction
  isn't in its input, unless the niche signal is already large enough to show up in PCA on its
  own.

## Why scProto could beat SEACells (`bk08`) — same graph, archetypal method

- `bk08` gives SEACells the exact same graph scProto trains on.
- But SEACells is a pure archetypal method: it fits convex combinations directly on that fixed
  graph. It has no access to transcriptional info to check the graph against.
- So when some `bk08` edges are really cross-cell-type edges (spatial noise, not real niche
  signal) and shouldn't count, SEACells has no way to know this and does not correct for it.
- scProto's encoder, by contrast, is pretrained with a real reconstruction objective before
  prototypes or any graph-based loss even exist — so its latent space starts out grounded in real
  gene expression, not only in the graph. SEACells never has this, at any stage.
- Confirmed from the code: the encoder is **not frozen** during prototype training — it keeps
  training alongside the prototypes, so this is a head-start advantage, not a permanent guarantee.

## What this does NOT claim (kept out on purpose)

- It does not claim scProto beats SEACells broadly — most pairs are a wash.
- It does not name specific genes/pairs (e.g. candidates like ACKR1) — those are promising but
  unverified, and belong in the separate results notes once actually re-checked.
- It does not claim the `bk08` mechanism above is strong in practice yet — `lambda_proto_recon`
  is weighted far below `lambda_community`/`lambda_nassoc` in the current config, so how much this
  advantage actually shows up is an open, testable question.
