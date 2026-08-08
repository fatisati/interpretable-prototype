# SEACells' actual objective — kernel archetypal analysis, not expression-space RSS

Correction to an assumption made earlier in this investigation (loss-diagnosis
discussion, 2026-07-14): SEACells does **not** minimize reconstruction error
in raw/PCA expression space. Confirmed from `literature/s41587-023-01716-9.pdf`
(Persad et al., *Nat Biotechnol* 2023), Methods section "Kernel archetypal
analysis."

## The actual objective

1. Build a KNN graph in PCA/SVD space, convert distances to an
   adaptive-Gaussian-kernel affinity/similarity matrix `M ∈ R^{n×n}` — the
   same idea as scProto's `arbf` affinity graph.
2. Archetypal analysis is applied to **M itself**, not to the raw
   expression/PCA matrix. Decomposition: `M ≈ M B A`, where
   - `B ∈ R^{n×s}`: archetype weight matrix — each archetype is `Z = MB`, a
     convex combination of *columns of the kernel matrix* (i.e. a combination
     of cells' similarity profiles, not raw expression vectors).
   - `A ∈ R^{s×n}`: membership matrix — each cell is a convex combination of
     the `s` archetypes.
   - Both `A` and `B` are **column-stochastic**: entries ≥ 0, columns sum to
     1 — a hard simplex constraint, not a soft penalty.
3. Objective: minimize squared reconstruction error (SRE) of the kernel
   matrix itself:
   `min_{A,B} ||M - MBA||_F^2`
4. Optimized via alternating Frank-Wolfe updates (convex in `A` given `B` and
   vice versa).
5. Archetypes are literally vertices of a convex polytope approximating the
   data's convex hull in kernel space — by construction they cannot leave the
   region spanned by real cells.

## What this means for the scProto loss comparison

- The "reconstruct the affinity/kernel matrix" objective is what
  `nassoc_loss` and `sim_recon_loss` are each partially doing —
  **not** `proto_recon_loss` (which reconstructs raw gene expression,
  something SEACells' core objective has no equivalent of at all).
  - `nassoc_loss` ≈ reconstructing a coarse (per-prototype-pair scalar)
    summary of `M`'s block structure.
  - `sim_recon_loss` ≈ reconstructing `M` at finer (per-cell) resolution —
    this is the closer match to SEACells' actual `||M - MBA||^2`.
- SEACells gets **purity** (tight block-diagonal `M`) and **no dead
  archetypes** as *automatic consequences* of one structurally-constrained
  convex optimization — not from two separate soft losses the way scProto
  splits it into `nassoc` (purity-like) + `proto_usage` (anti-collapse). The
  simplex constraint on `A`/`B` does this work: an archetype that
  contributes nothing to reducing `||M - MBA||^2` is wasted "budget" the
  objective actively wants to reallocate, so there's no equivalent of a
  "dead prototype" the way scProto's free-floating softmax routing can
  independently produce one.
- SEACells' `B` constraint (archetypes = convex combos of kernel-matrix
  columns) is a **hard**, structural on-manifold constraint — much stronger
  than a soft MSE penalty pulling a free-floating prototype parameter toward
  a combination of cell embeddings. It's enforced by the optimization
  algorithm itself (Frank-Wolfe over the simplex), not by an extra loss term
  competing for gradient against everything else.

## Open question this raises for scProto

Whether `proto_usage_loss` (soft, competes with `nassoc`/`sim_recon`/`umap`
for gradient every step) is a weak substitute for what SEACells gets
structurally for free — and whether `proto_decoupled=True` +
`gmm_resurrect=True` (EMA-of-cluster-mean prototype position updates,
decoupled from the graph-topology losses' gradient entirely, with an
explicit split-dominant-into-dead-proto resurrect step) is a closer
structural analogue: it hard-anchors prototype position to real
assigned-cell embeddings *and* provides anti-collapse, without adding
another competing soft loss term to the stack. Not yet tested — see
`sim_recon_investigation_index.md` open questions.
