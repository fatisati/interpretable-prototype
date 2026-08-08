import scanpy as sc
import SEACells
import pandas as pd
import numpy as np
import os
import json
import hashlib
import scipy.sparse as sp

# Fix SEACells GPU bug: inject cupyx into SEACells.core namespace
try:
    import cupy as cp
    import cupyx
    import cupyx.scipy.sparse
    if cp.cuda.is_available():
        SEACells.core.cupyx = cupyx
        SEACells.core.cp = cp
except ImportError:
    pass

# Fix SEACells/anndata version drift: summarize_by_SEACell (SEACells/core.py) calls
# sc.AnnData(matrix, dtype=matrix.dtype) -- `dtype` was removed from AnnData.__init__ in
# newer anndata releases. Whether this breaks depends on whichever anndata version pip
# resolves at install time (unpinned) -- not fixable with a version pin without risking a
# *different* break (scarches/scvi-tools need a newer anndata, with the `anndata.io`
# submodule, than the last one that still accepts `dtype=`; there may be no anndata version
# that satisfies both).
#
# NOT using inspect.signature() to detect whether patching is needed -- tried that first
# and it silently failed to patch, because the installed AnnData.__init__ is wrapped by
# `legacy_api_wrap` (visible in the actual traceback: .../legacy_api_wrap/__init__.py in
# fn_compatible), which exists specifically to keep old/deprecated call signatures
# introspectable for compatibility tooling -- inspect.signature() can report `dtype` as a
# valid parameter even though calling with it still raises TypeError at runtime.
#
# A module-level-only patch (apply once, at import time) also proved unreliable in
# practice -- it didn't survive across cell reruns in a session with `%autoreload 2`
# active, most likely because autoreload reliably reloads function/class *bodies* but
# doesn't guarantee re-running bare, side-effecting module-level statements (like a
# monkeypatch sitting outside any function) on every detected change. So this is wrapped
# in an idempotent function, applied both here at import time AND freshly inside
# compute_seacells() immediately before the actual risky call -- removing any dependence
# on import/reload timing.
#
# CAUTION, learned the hard way (RecursionError): a plain module-level
# `_RealAnnData = sc.AnnData` is NOT actually "captured once" the way the name implies.
# get_trainer() reloads every interpretable_ssl.* module (including this one) on every
# call via importlib.reload(), which re-executes this file's top-level code in the SAME,
# persistent module namespace -- `scanpy` itself is not reloaded, so by the second such
# reload, `sc.AnnData` is already the wrapper from the first pass. `_RealAnnData = sc.AnnData`
# then rebinds `_RealAnnData` to that wrapper, not the true original -- and because
# `_AnnDataDtypeSafe`'s body resolves `_RealAnnData` by name AT CALL TIME (a global lookup,
# not a captured closure value), the moment `_RealAnnData` and `_AnnDataDtypeSafe` end up
# bound to the identical function object, calling it calls itself: infinite recursion.
# Fix: stash the true original on the `scanpy` module itself (which is never reloaded),
# guarded so it's captured exactly once no matter how many times THIS module reloads.
if not hasattr(sc, '_scproto_real_anndata'):
    sc._scproto_real_anndata = sc.AnnData

def _AnnDataDtypeSafe(*args, **kwargs):
    try:
        return sc._scproto_real_anndata(*args, **kwargs)
    except TypeError as e:
        if 'dtype' in kwargs and 'dtype' in str(e):
            kwargs = {k: v for k, v in kwargs.items() if k != 'dtype'}
            return sc._scproto_real_anndata(*args, **kwargs)
        raise


def _ensure_seacells_anndata_patch():
    """Idempotent -- safe (and cheap) to call every time before anything that might hit
    SEACells' dtype= incompatibility. See the module-level comment above for the full
    reasoning; this exists as a function specifically so it can be reapplied right before
    the risky call, not just once at import."""
    sc.AnnData = _AnnDataDtypeSafe
    # Patch whichever attribute SEACells.core actually uses to reach AnnData -- different
    # SEACells versions import it differently (`import scanpy as sc` + `sc.AnnData(...)`,
    # `from anndata import AnnData` + `AnnData(...)`, `import scanpy` + `scanpy.AnnData(...)`),
    # so patch every name that's actually present rather than assuming one specific style.
    if hasattr(SEACells.core, 'sc') and hasattr(SEACells.core.sc, 'AnnData'):
        SEACells.core.sc.AnnData = _AnnDataDtypeSafe
    if hasattr(SEACells.core, 'AnnData'):
        SEACells.core.AnnData = _AnnDataDtypeSafe
    if hasattr(SEACells.core, 'scanpy') and hasattr(SEACells.core.scanpy, 'AnnData'):
        SEACells.core.scanpy.AnnData = _AnnDataDtypeSafe


_ensure_seacells_anndata_patch()

from collections import Counter
from interpretable_ssl.evaluation.mc_metric_utils import *


def preprocess(ad, n_top_genes):
    raw_ad = sc.AnnData(ad.X)
    raw_ad.obs_names, raw_ad.var_names = ad.obs_names, ad.var_names
    ad.raw = raw_ad
    # Normalize cells, log transform and compute highly variable genes
    # sc.pp.normalize_per_cell(ad)
    # sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=n_top_genes)
    # Compute principal components -
    # Here we use 50 components. This number may also be selected by examining variance explaint
    sc.tl.pca(ad, n_comps=50, use_highly_variable=True)
    return ad


def _seacells_backend(n_cells=None, dense_cell_limit=30000):
    """Return (use_gpu, use_sparse) based on available hardware and dataset size.

    use_sparse keeps K sparse throughout and avoids materializing the full
    n_cells×n_cells dense matrix -- but SEACells only supports use_sparse on
    CPU (use_gpu=True forces a dense kernel_matrix @ kernel_matrix.T, which for
    n_cells=58423 is a ~58423x58423 dense array -- tens of GB -- and reliably
    crashes the runtime out of RAM). So above dense_cell_limit, force CPU+sparse
    even when a GPU is available -- trading GPU speed for not crashing.
    """
    gpu_available = False
    try:
        import cupy as cp
        import cupyx.scipy.sparse
        gpu_available = cp.cuda.is_available()
    except Exception:
        pass

    if gpu_available and (n_cells is None or n_cells <= dense_cell_limit):
        print("[SEACells backend] GPU detected → use_gpu=True, use_sparse=False")
        return True, False
    if gpu_available:
        print(f"[SEACells backend] GPU detected but n_cells={n_cells} > "
              f"{dense_cell_limit} -- forcing CPU+sparse anyway (GPU path is dense-only "
              f"and would need a ~{n_cells}x{n_cells} dense matrix, likely OOM)")
    else:
        print("[SEACells backend] No GPU → use_sparse=True (sparse CPU, avoids dense K)")
    return False, True


def _optimized_sparse_seacells_class():
    """Returns a SEACellsCPU subclass that never materializes
    K = kernel_matrix @ kernel_matrix.T explicitly.

    The real SEACells.cpu.SEACellsCPU forms this full (n_cells x n_cells)
    product ONCE in add_precomputed_kernel_matrix() and reuses it in every
    Frank-Wolfe iteration (_updateA/_updateB) and once more at the end of
    fitting (self.Z_ = self.B_.T @ self.K). Even though our own kernel_matrix
    is genuinely sparse (~75 nonzeros/row for a k=50-ish graph), the SQUARED
    product K can be vastly denser -- K[i,j] is nonzero whenever cells i and j
    share ANY common neighbor in the original graph, which for a well-connected
    kNN-style graph over n_cells=58423 reliably produces tens to hundreds of
    millions of nonzeros (or worse) -- exhausting RAM regardless of use_sparse
    being honored, since use_sparse only governs the iteration-loop dtype, not
    whether this one-time product gets formed at all.

    Every real usage of self.K in the original class only ever multiplies it by
    an (n_cells x k) matrix, k = n_SEACells (small, ~200) -- by associativity,
    K @ X == kernel_matrix @ (kernel_matrix.T @ X), which costs
    O(nnz(kernel_matrix) * k) instead of materializing K. This subclass
    overrides every method that touched self.K (add_precomputed_kernel_matrix,
    initialize, _updateA, _updateB, step, _fit) to use that identity instead,
    with every other line copied verbatim from SEACells.cpu.SEACellsCPU so
    behavior is otherwise unchanged. _get_greedy_centers (the other self.K-heavy
    method) is NOT overridden -- it's dead code for our pipeline, since we
    always pass initial_archetypes explicitly (waypoint_archetype_indices),
    which skips initialize_archetypes()/_get_greedy_centers() entirely.
    """
    import SEACells.cpu as seacells_cpu
    from scipy.sparse import csr_matrix
    from sklearn.preprocessing import normalize

    class _SEACellsCPUFast(seacells_cpu.SEACellsCPU):
        def add_precomputed_kernel_matrix(self, K):
            assert K.shape == (self.n_cells, self.n_cells), (
                f"Dimension of kernel matrix must be n_cells = "
                f"({self.n_cells},{self.n_cells}), not {K.shape} "
            )
            self.kernel_matrix = K
            # Deliberately never set self.K -- every real usage below goes
            # through _kdot()/_kdot_left() instead of the full product.

        def _kdot(self, X):
            """K @ X without ever forming K = kernel_matrix @ kernel_matrix.T."""
            return self.kernel_matrix @ (self.kernel_matrix.T @ X)

        def _kdot_left(self, X):
            """X @ K without ever forming K."""
            return (X @ self.kernel_matrix) @ self.kernel_matrix.T

        def initialize(self, initial_archetypes=None, initial_assignments=None):
            if self.kernel_matrix is None:
                raise RuntimeError(
                    "Must first construct kernel matrix before initializing SEACells."
                )
            n = self.kernel_matrix.shape[0]

            if initial_archetypes is not None:
                if self.verbose:
                    print("Using provided list of initial archetypes")
                self.archetypes = initial_archetypes

            if self.archetypes is None:
                self.initialize_archetypes()
            self.k = len(self.archetypes)
            k = self.k

            cols = np.arange(k)
            rows = self.archetypes
            shape = (n, k)
            B0 = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=shape)

            self.B0 = B0
            B = self.B0.copy()

            if initial_assignments is not None:
                A0 = initial_assignments
                assert A0.shape == (k, n), (
                    f"Initial assignment matrix should be of shape (k={k} x n={n})"
                )
                A0 = csr_matrix(A0)
                A0 = normalize(A0, axis=0, norm="l1")
            else:
                archetypes_per_cell = int(k * 0.25)
                rows = np.random.randint(0, k, size=(n, archetypes_per_cell)).reshape(-1)
                columns = np.repeat(np.arange(n), archetypes_per_cell)
                A0 = csr_matrix(
                    (np.random.random(len(rows)), (rows, columns)), shape=(k, n)
                )
                A0 = normalize(A0, axis=0, norm="l1")
                if self.verbose:
                    print("Randomly initialized A matrix.")

            self.A0 = A0
            A = self.A0.copy()
            A = self._updateA(B, A)

            self.A_ = A
            self.B_ = B

            RSS = self.compute_RSS(A, B)
            self.RSS_iters.append(RSS)

            if self.convergence_threshold is None:
                self.convergence_threshold = self.convergence_epsilon * RSS
                if self.verbose:
                    print(f"Setting convergence threshold at {self.convergence_threshold:.5f}")

        def _updateA(self, B, A_prev):
            n, k = B.shape
            A = A_prev
            t = 0
            t2 = self._kdot(B).T
            t1 = t2 @ B
            while t < self.max_FW_iter:
                G = 2.0 * np.array(t1 @ A - t2)
                amins = np.argmin(G, axis=0)
                amins = np.array(amins).reshape(-1)
                e = csr_matrix((np.ones(len(amins)), (amins, np.arange(n))), shape=A.shape)
                A += 2.0 / (t + 2.0) * (e - A)
                t += 1
            return A

        def _updateB(self, A, B_prev):
            k, n = A.shape
            B = B_prev
            t = 0
            t1 = A @ A.T
            t2 = self._kdot(A.T)
            while t < self.max_FW_iter:
                G = 2.0 * np.array(self._kdot(B @ t1) - t2)
                amins = np.argmin(G, axis=0)
                amins = np.array(amins).reshape(-1)
                e = csr_matrix((np.ones(len(amins)), (amins, np.arange(k))), shape=B.shape)
                B += 2.0 / (t + 2.0) * (e - B)
                t += 1
            return B

        def compute_RSS(self, A=None, B=None):
            """||X - XBA||^2 without ever forming the (n_cells x n_cells)
            reconstruction matrix XBA.

            The original compute_reconstruction()/compute_RSS() forms
            (kernel_matrix.dot(B)).dot(A) explicitly -- shape (n_cells,
            n_cells) by construction (X is n_cells x n_cells, B is n_cells x
            k, A is k x n_cells). Even though A/B are individually sparse,
            k=n_SEACells is small enough (~200) that A's/B's nonzero-archetype
            subsets overlap for nearly every (row, col) pair once multiplied
            through X, so this reconstruction comes out essentially dense at
            n_cells=58423 -- called every single Frank-Wolfe iteration via
            step(), this is likely the actual remaining OOM even after
            avoiding self.K.

            Expand ||X - XBA||_F^2 = ||X||_F^2 - 2<X, XBA> + ||XBA||_F^2 and
            rewrite every term via the trace-cyclic identity tr(M @ N) ==
            tr(N @ M) so nothing larger than an (n_cells x k) or (k x k)
            matrix is ever formed:
              Y      = X @ B                     (n_cells x k)
              XAT    = X @ A.T                   (n_cells x k)
              <X,XBA> = tr(XAT.T @ Y)             (k x k, then trace)
              ||XBA||^2 = tr((Y.T @ Y) @ (A @ A.T))   (k x k, then trace)
              ||X||^2 = sum of squares of X's own (already sparse) entries
            """
            if A is None:
                A = self.A_
            if B is None:
                B = self.B_
            X = self.kernel_matrix

            Y = X.dot(B)             # (n_cells, k)
            XAT = X.dot(A.T)         # (n_cells, k)
            AAT = A.dot(A.T)         # (k, k)
            YTY = Y.T.dot(Y)         # (k, k)

            def _dense(M):
                # k x k (or smaller) either way -- trivially cheap to densify,
                # but np.asarray() on a scipy.sparse matrix doesn't reliably
                # produce a real dense array (may just wrap the sparse object),
                # so go through .toarray() explicitly whenever M is sparse.
                return M.toarray() if sp.issparse(M) else np.asarray(M)

            X_sq = X.multiply(X).sum()
            cross_term = _dense(XAT.T.dot(Y)).trace()
            recon_sq = _dense(YTY.dot(AAT)).trace()

            return float(X_sq - 2.0 * cross_term + recon_sq)

        def step(self):
            A = self.A_
            B = self.B_
            if self.kernel_matrix is None:
                raise RuntimeError(
                    "Kernel matrix has not been computed. Run model.construct_kernel_matrix() first."
                )
            if A is None:
                raise RuntimeError(
                    "Cell to SEACell assignment matrix has not been initialised. Run model.initialize() first."
                )
            if B is None:
                raise RuntimeError(
                    "Archetype matrix has not been initialised. Run model.initialize() first."
                )
            A = self._updateA(B, A)
            B = self._updateB(A, B)
            self.RSS_iters.append(self.compute_RSS(A, B))
            self.A_ = A
            self.B_ = B
            labels = self.get_hard_assignments()
            self.ad.obs["SEACell"] = labels["SEACell"]

        def _fit(self, max_iter=50, min_iter=10, initial_archetypes=None, initial_assignments=None):
            self.initialize(
                initial_archetypes=initial_archetypes,
                initial_assignments=initial_assignments,
            )
            converged = False
            n_iter = 0
            while (not converged and n_iter < max_iter) or n_iter < min_iter:
                n_iter += 1
                if n_iter == 1 or (n_iter) % 10 == 0:
                    if self.verbose:
                        print(f"Starting iteration {n_iter}.")
                self.step()
                if n_iter == 1 or (n_iter) % 10 == 0:
                    if self.verbose:
                        print(f"Completed iteration {n_iter}.")
                if np.abs(self.RSS_iters[-2] - self.RSS_iters[-1]) < self.convergence_threshold:
                    if self.verbose:
                        print(f"Converged after {n_iter} iterations.")
                    converged = True

            # self.Z_ = self.B_.T @ self.K, without ever forming self.K
            self.Z_ = self._kdot_left(self.B_.T)

            labels = self.get_hard_assignments()
            self.ad.obs["SEACell"] = labels["SEACell"]

            if not converged:
                raise RuntimeWarning(
                    "Warning: Algorithm has not converged - you may need to increase the maximum number of iterations"
                )

    return _SEACellsCPUFast


def _make_seacells_model(ad, **kwargs):
    """SEACells.core.SEACells(...) constructor.

    For the sparse-CPU path (use_sparse=True, use_gpu=False), constructs our
    own optimized subclass instead of the vanilla one -- see
    _optimized_sparse_seacells_class()'s docstring for why the vanilla
    SEACellsCPU reliably OOMs at n_cells=58423 regardless of use_sparse.

    For every other path, calls the real SEACells.core.SEACells(...) factory
    directly, dropping `use_sparse` only if that call actually raises a
    TypeError mentioning it (NOT using inspect.signature() to predict this --
    SEACells.core.SEACells is a plain function, not a class, so
    inspect.signature(SEACells.core.SEACells.__init__) inspects the generic
    object.__init__ every function object has, not the factory's own real,
    explicit signature -- a permanent false negative that always concluded
    use_sparse was unsupported even when it wasn't; same failure mode already
    hit and fixed for AnnData.__init__ elsewhere in this file, wrapped by
    legacy_api_wrap).
    """
    if kwargs.get("use_sparse") and not kwargs.get("use_gpu", False):
        print("[SEACells backend] using our own optimized sparse-CPU SEACells "
              "(never materializes kernel_matrix @ kernel_matrix.T)")
        cls = _optimized_sparse_seacells_class()
        fit_kwargs = {k: v for k, v in kwargs.items() if k not in ("use_gpu", "use_sparse")}
        return cls(ad, **fit_kwargs)

    try:
        return SEACells.core.SEACells(ad, **kwargs)
    except TypeError as e:
        if "use_sparse" in kwargs and "use_sparse" in str(e):
            print("[SEACells backend] installed SEACells rejected `use_sparse` "
                  f"({e}) — dropping it and retrying")
            kwargs.pop("use_sparse")
            return SEACells.core.SEACells(ad, **kwargs)
        raise


def compute_seacells(ad, n_SEACells, build_kernel_on="X_pca", k=50):
    n_waypoint_eigs = 10
    use_gpu, use_sparse = _seacells_backend(n_cells=ad.n_obs)

    model = _make_seacells_model(
        ad,
        build_kernel_on=build_kernel_on,
        n_SEACells=n_SEACells,
        n_waypoint_eigs=n_waypoint_eigs,
        use_gpu=use_gpu,
        use_sparse=use_sparse,
    )

    model.construct_kernel_matrix()
    model.initialize_archetypes()
    model.fit()

    if "counts" not in ad.layers:
        ad.layers["counts"] = ad.X.copy()

    _ensure_seacells_anndata_patch()  # reapply fresh -- see comment where this is defined
    SEACell_ad = SEACells.core.summarize_by_SEACell(
        ad, SEACells_label="SEACell", summarize_layer="counts"
    )
    # SEACell_soft_ad = SEACells.core.summarize_by_soft_SEACell(ad, model.A_, celltype_label='celltype',summarize_layer='raw', minimum_weight=0.05)
    return ad, SEACell_ad, model


def suggest_n_seacells(aff, cells_per_mc=50, n_eigs=150, gap_z_thresh=1.0):
    """Suggest n_SEACells from affinity via four estimates:

    - Ratio-based:       n_cells // cells_per_mc  (granularity target / upper bound)
    - Spectral gap:      largest eigenvalue drop after λ1  (major groups only)
    - Multi-scale k:     last significant gap across all scales (major + substructure)
    - Participation ratio (PR): 1/sum(p_i^2), p_i=λ_i/sum(λ)  (effective dimensionality)

    A gap is 'significant' if it exceeds mean + gap_z_thresh * std of all gaps.

    Args:
        aff:           scipy sparse affinity matrix (N x N).
        cells_per_mc:  target cells per metacell for ratio-based estimate (default 50).
        n_eigs:        number of top eigenvalues to inspect (default 150).
        gap_z_thresh:  z-score threshold for a gap to count as significant (default 1.0).

    Returns:
        int: suggested n_SEACells = max(multi_scale_k, pr).
    """
    from scipy.sparse.linalg import eigsh

    n_cells = aff.shape[0]
    ratio_based = n_cells // cells_per_mc

    vals = eigsh(aff, k=min(n_eigs, n_cells - 1), which='LM', return_eigenvectors=False)
    vals = np.sort(vals)[::-1]

    # skip λ1 (constant/mean mode, trivially dominant in affinity matrices)
    gaps = np.abs(np.diff(vals[1:]))

    # major groups: first (largest) gap
    spectral_k = int(np.argmax(gaps)) + 2

    # substructure: last gap that exceeds mean + z*std
    threshold = gaps.mean() + gap_z_thresh * gaps.std()
    significant = np.where(gaps > threshold)[0]
    multi_scale_k = int(significant[-1]) + 2 if len(significant) > 0 else spectral_k

    # participation ratio: effective number of active eigenmodes (all scales)
    vals_pos = vals[vals > 0]
    p = vals_pos / vals_pos.sum()
    pr = int(round(1.0 / (p ** 2).sum()))

    suggestion = max(multi_scale_k, pr)
    print(f"n_cells:             {n_cells}")
    print(f"Ratio-based:         {ratio_based}  (1 per {cells_per_mc} cells, upper bound)")
    print(f"Spectral gap:        {spectral_k}  (major groups only)")
    print(f"Multi-scale k:       {multi_scale_k}  (major + substructure, last significant gap)")
    print(f"Participation ratio: {pr}  (effective eigenmodes)")
    print(f"Suggestion:          max({multi_scale_k}, {pr}) = {suggestion}")
    return suggestion


def compute_diffusion_embedding(aff, n_eigs=1024, diffusion_t=0.5):
    """Diffusion coordinates of an affinity graph, as an (N, n_eigs) array
    suitable for ad.obsm[...] + compute_seacells(build_kernel_on=...).

    Identical construction to SCProtoTrainer._compute_sim_recon_diffusion_targets
    (interpretable_ssl/trainers/scproto.py), kept in sync deliberately, run
    globally instead of per-batch (fine for a single-section dataset; pass
    per-batch and concatenate yourself if that matters for your case):
      1. top eigenvectors of the symmetric-normalized Laplacian
         L_sym = D^{-1/2} A D^{-1/2}.
      2. each eigenvector weighted by eigenvalue**diffusion_t. At the
         default diffusion_t=0.5 (sqrt(eigenvalue)), the Gram matrix of the
         weighted top-n_eigs eigenvectors is exactly the Eckart-Young-optimal
         rank-n_eigs reconstruction of L_sym — i.e. per-cell MSE on these
         coordinates is mathematically equivalent (up to the rank-n_eigs
         truncation) to MSE on a reconstructed affinity matrix, matching
         sim_recon_target='full'/SEACells' own kernel-RSS objective as
         closely as a compact per-cell embedding can.
         diffusion_t=0.0 (unweighted) does NOT have this property — it's a
         Laplacian-eigenmap embedding, not an affinity-matrix reconstruction.
         See files/sim_recon_global_vs_local_compaction.md for the full
         derivation and what diffusion_t=0.5 trades away (fine/rare-pattern
         resolution) to get this equivalence.
      3. the trivial leading eigenvector (eigenvalue ~1, ~sqrt(degree)
         direction, non-discriminative) is dropped.
      4. rescaled from eigsh's unit-L2-norm convention to O(1) RMS entry
         (unrelated to the eigenvalue weighting — a separate batch-size-
         artifact fix, applied after it).

    Args:
        aff:          scipy sparse affinity matrix (N x N), not necessarily symmetric.
        n_eigs:        number of diffusion coordinates to return (default 1024).
        diffusion_t:   eigenvalue**t weighting (default 0.5 — see above).

    Returns:
        (N, n_eigs) float32 array.
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh

    N = aff.shape[0]
    A = (aff + aff.T) / 2
    d = np.array(A.sum(axis=1)).ravel()
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(d + 1e-8))
    L_sym = D_inv_sqrt @ A @ D_inv_sqrt

    # request one extra eigenvector so we can drop the trivial leading one
    # below and still keep n_eigs discriminative ones
    k_request = min(n_eigs + 1, N - 2)
    print(f"[diffusion embedding] eigsh on {N}x{N} affinity, k={k_request}, diffusion_t={diffusion_t} ...")
    vals, vecs = eigsh(L_sym, k=k_request, which='LM', tol=1e-2)
    order = np.argsort(-vals)  # descending: largest eigenvalue first
    vals_sorted = vals[order][1:]  # drop the trivial leading eigenvalue (largest)
    vecs = vecs[:, order][:, 1:]   # drop the trivial leading eigenvector

    if diffusion_t:
        # clip: L_sym can return tiny-negative eigenvalues at float precision
        # near 0, which would NaN under a fractional power
        weights = np.clip(vals_sorted, 0, None) ** diffusion_t
        vecs = vecs * weights[np.newaxis, :]

    vecs = vecs * np.sqrt(N)  # undo eigsh's unit-norm convention -> O(1) RMS entry
    print(f"[diffusion embedding] done — {vecs.shape[1]} eigenvectors")
    return vecs.astype(np.float32)


def compute_seacells_diffusion(
    ad, n_SEACells, aff, n_eigs=1024, diffusion_t=0.5,
    n_waypoint_eigs=10, build_kernel_on='X_diffusion', edge_chunk=200_000,
):
    """Run SEACells' archetypal analysis directly on a rank-n_eigs diffusion
    reconstruction of `aff`, instead of on `aff` itself (SEACells(PCA)'s own
    kernel) or on a *different*, freshly-built kernel over diffusion
    coordinates.

    Passing build_kernel_on='X_diffusion' to compute_seacells() would be
    wrong for this purpose: it routes through construct_kernel_matrix(),
    which builds a brand-new adaptive-RBF kernel from kNN distances in
    diffusion-coordinate space — an uncontrolled second kernel construction,
    not "archetypal analysis on a compacted version of the same M". Instead:

      1. compute_diffusion_embedding(aff, ...) -> Z, the diffusion embedding
         of aff's symmetric-normalized Laplacian L_sym = D^-1/2 A D^-1/2.
      2. Reconstruct M_hat, the rank-n_eigs approximation of `aff` itself
         (not of L_sym — degree-renormalized back to aff's own scale via
         sqrt(d_i * d_j)), evaluated only at aff's own nonzero (i, j) pairs.
         Z @ Z.T is dense and infeasible at N~58k (~13.6TB); we only need
         values at the ~4M edges that already exist.
      3. Inject M_hat directly as `self.kernel_matrix` via
         model.add_precomputed_kernel_matrix(), skipping
         construct_kernel_matrix() entirely — mirrors
         compute_seacells_from_affinity()'s pattern for a precomputed .pkl
         affinity, just computed in-memory here. Confirmed against
         SEACells/gpu.py: compute_RSS() reconstructs ||self.kernel_matrix -
         self.kernel_matrix @ B @ A||, so whatever is injected here *is* the
         matrix archetypal analysis actually optimizes against — self.K =
         kernel_matrix @ kernel_matrix.T is just a derived Gram matrix for
         the Frank-Wolfe update steps, not a second independent input.

    ad.obsm[build_kernel_on] is set to Z, but only because SEACells' own
    waypoint-sampling init (_get_waypoint_centers) always reads
    ad.obsm[build_kernel_on] to run its own separate (cheap) diffusion-map
    call for picking spread-out initial archetype seeds — unrelated to the
    kernel matrix injected above.

    Args:
        ad:            AnnData object (cells x genes), preprocessed.
        n_SEACells:    number of SEACells (metacells) to compute.
        aff:           scipy sparse affinity matrix (N x N) — e.g. the same
                       'arbf' kernel SEACells(PCA) itself is built on.
        n_eigs:        diffusion embedding dimensionality (default 1024).
        diffusion_t:   eigenvalue**t weighting (default 0.5 — see
                       compute_diffusion_embedding / files/sim_recon_global_vs_local_compaction.md).
        n_waypoint_eigs: passed through to SEACells for waypoint init (default 10).
        build_kernel_on: ad.obsm key to store Z under (default 'X_diffusion').
        edge_chunk:    edges processed per batch when evaluating Z@Z.T at
                       aff's nonzero pairs, to bound peak memory.

    Returns:
        (ad, SEACell_ad, model)
    """
    import scipy.sparse as sp

    N = aff.shape[0]
    A = (aff + aff.T) / 2
    d = np.array(A.sum(axis=1)).ravel()
    sqrt_d = np.sqrt(d)

    Z = compute_diffusion_embedding(aff, n_eigs=n_eigs, diffusion_t=diffusion_t)
    ad.obsm[build_kernel_on] = Z

    coo = A.tocoo()
    row, col = coo.row, coo.col
    n_edges = len(row)
    vals = np.empty(n_edges, dtype=np.float32)
    print(f"[seacells diffusion] reconstructing M_hat at {n_edges} existing edges ...")
    for start in range(0, n_edges, edge_chunk):
        end = min(start + edge_chunk, n_edges)
        r, c = row[start:end], col[start:end]
        dot = np.einsum('ij,ij->i', Z[r], Z[c])
        vals[start:end] = dot * sqrt_d[r] * sqrt_d[c]
    M_hat = sp.csr_matrix((vals, (row, col)), shape=(N, N))

    use_gpu, use_sparse = _seacells_backend(n_cells=ad.n_obs)
    model = _make_seacells_model(
        ad,
        build_kernel_on=build_kernel_on,
        n_SEACells=n_SEACells,
        n_waypoint_eigs=n_waypoint_eigs,
        use_gpu=use_gpu,
        use_sparse=use_sparse,
    )
    model.add_precomputed_kernel_matrix(M_hat)
    model.initialize_archetypes()
    model.fit()

    if "counts" not in ad.layers:
        ad.layers["counts"] = ad.X.copy()

    _ensure_seacells_anndata_patch()  # reapply fresh -- see comment where this is defined
    SEACell_ad = SEACells.core.summarize_by_SEACell(
        ad, SEACells_label="SEACell", summarize_layer="counts"
    )
    return ad, SEACell_ad, model


def waypoint_archetype_indices(aff, k, n_eigs=10, seed=None):
    """Topology-aware archetype seed selection computed directly from an
    affinity matrix, instead of SEACells' own initialize_archetypes(), which
    needs ad.obsm[build_kernel_on] and runs a separate palantir diffusion-map
    call unrelated to whatever kernel you injected via
    add_precomputed_kernel_matrix(). Use this together with
    compute_seacells_own_affinity() so both the kernel *and* the archetype
    seeding come from the same graph.

    Adapted from SCProtoTrainer._init_prototypes_waypoint (scproto.py), which
    does the same diffusion-map + greedy-MaxMin selection but then encodes the
    chosen cells into scProto's own prototype vectors; here we stop at the
    cell indices themselves so they can be passed straight to
    model.fit(initial_archetypes=...).

    Steps:
      1. Symmetrize aff and compute its normalized diffusion map (top n_eigs
         eigenvectors of D^-1/2 A D^-1/2 — same normalization SEACells' own
         waypoint init uses internally).
      2. Greedy MaxMin in diffusion space: iteratively pick the cell farthest
         (in diffusion coordinates) from all already-chosen cells. Guarantees
         coverage of every topological region of the graph, including rare
         populations — unlike k-means-style init, which over-represents dense
         clusters.

    Args:
        aff:     scipy sparse affinity matrix (N x N) — e.g. your mean_product
                 graph. This *is* the graph the seeds are chosen from; no
                 other embedding is used.
        k:       number of archetypes (= n_SEACells) to select.
        n_eigs:  number of diffusion-map eigenvectors (default 10 — matches
                 SEACells' own n_waypoint_eigs default).
        seed:    optional int, seeds the random first pick for reproducibility.

    Returns:
        np.ndarray of shape (k,) — cell indices, ready for
        model.fit(initial_archetypes=...).
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh
    from tqdm import tqdm

    N = aff.shape[0]
    A = sp.csr_matrix(aff)
    A = (A + A.T) / 2

    d = np.array(A.sum(axis=1)).ravel()
    print(f"[waypoint init] N={N}  k={k}  n_eigs={n_eigs}  "
          f"nnz={A.nnz}  nnz/row={A.nnz / N:.1f}")

    d_inv_sqrt = 1.0 / np.sqrt(d + 1e-8)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    L_sym = D_inv_sqrt @ A @ D_inv_sqrt

    n_eigs = min(n_eigs, N - 2)
    print("[waypoint init] computing diffusion map ...")
    _, vecs = eigsh(L_sym, k=n_eigs, which='LM', tol=1e-2)
    print(f"[waypoint init] diffusion map done — {n_eigs} eigenvectors")

    rng = np.random.RandomState(seed)
    chosen = [int(rng.randint(0, N))]
    min_dists = np.full(N, np.inf)
    for _ in tqdm(range(k - 1), desc="waypoint MaxMin", unit="archetype"):
        last = chosen[-1]
        d_last = ((vecs - vecs[last]) ** 2).sum(axis=1)
        min_dists = np.minimum(min_dists, d_last)
        chosen.append(int(min_dists.argmax()))

    chosen = np.array(chosen, dtype=int)
    print(f"[waypoint init] selected {k} archetype seed cells")
    return chosen


def compute_seacells_own_affinity(
    ad, n_SEACells, aff, n_waypoint_eigs=10, build_kernel_on='X_pca',
    max_iter=100, min_iter=10, seed=None,
):
    """Run SEACells' archetypal (Frank-Wolfe) analysis entirely on your own
    affinity graph — both the kernel matrix *and* the archetype seeding come
    from `aff`, unlike compute_seacells_from_affinity() (kernel from `aff`,
    but seeding still via SEACells' native palantir/ad.obsm[build_kernel_on]
    waypoint init) or scproto.py's eval_seacells() (kernel from our affinity,
    but random archetype init, no waypoint init at all).

    Skips both construct_kernel_matrix() (kernel comes from `aff` via
    add_precomputed_kernel_matrix) and SEACells' own initialize_archetypes()
    (seeds come from waypoint_archetype_indices(aff, ...), passed via
    model.fit(initial_archetypes=...), which — per SEACells' own
    initialize()/cpu.py — causes it to skip calling initialize_archetypes()
    altogether, so ad.obsm[build_kernel_on] is never read).

    Args:
        ad:               AnnData (cells x genes), preprocessed.
        n_SEACells:       number of SEACells (metacells) / archetypes.
        aff:              scipy sparse affinity matrix (N x N) — e.g. your
                          mean_product graph (see graph_generator.rbf_product /
                          generate_affinity(..., affinity_type='mean_product')).
        n_waypoint_eigs:  diffusion-map eigenvectors for archetype seeding
                          (default 10, matches SEACells' own default).
        build_kernel_on:  unused for computation — SEACells' constructor
                          requires *some* value, but it's never read since
                          both construct_kernel_matrix() and
                          initialize_archetypes() are skipped here.
        max_iter/min_iter: passed through to model.fit().
        seed:             optional int for waypoint init's random first pick.

    Returns:
        (ad, SEACell_ad, model)
    """
    archetype_idx = waypoint_archetype_indices(
        aff, n_SEACells, n_eigs=n_waypoint_eigs, seed=seed
    )

    use_gpu, use_sparse = _seacells_backend(n_cells=ad.n_obs)
    model = _make_seacells_model(
        ad,
        build_kernel_on=build_kernel_on,
        n_SEACells=n_SEACells,
        n_waypoint_eigs=n_waypoint_eigs,
        use_gpu=use_gpu,
        use_sparse=use_sparse,
    )
    model.add_precomputed_kernel_matrix(aff)
    model.fit(max_iter=max_iter, min_iter=min_iter, initial_archetypes=archetype_idx)

    if "counts" not in ad.layers:
        ad.layers["counts"] = ad.X.copy()

    _ensure_seacells_anndata_patch()  # reapply fresh -- see comment where this is defined
    SEACell_ad = SEACells.core.summarize_by_SEACell(
        ad, SEACells_label="SEACell", summarize_layer="counts"
    )
    return ad, SEACell_ad, model


def compute_seacells_from_affinity(
    ad, n_SEACells, ds_name, affinity_type,
    n_components=50, k_neighbors=50, graph_dir='./graphs', n_waypoint_eigs=10,
    build_kernel_on='X_pca',
):
    """Run SEACells archetypal analysis using a precomputed affinity matrix.

    Skips construct_kernel_matrix() entirely by injecting the loaded affinity
    via model.add_precomputed_kernel_matrix(). The affinity path is constructed
    from parameters to match the convention used by save_affinity().

    Args:
        ad:               AnnData object (cells x genes), preprocessed.
        n_SEACells:       number of SEACells (metacells) to compute.
        ds_name:          dataset name string (e.g. 's28nsc').
        affinity_type:    affinity type tag (e.g. 'covet', 'arbf').
        n_components:     must match what was used when saving (default 50).
        k_neighbors:      must match what was used when saving (default 50).
        graph_dir:        directory where affinity .pkl files are stored (default './graphs').
        n_waypoint_eigs:  number of waypoint eigenvectors (default 10).
        build_kernel_on:  ad.obsm key used for waypoint initialization (default 'X_pca').
                          Should match the embedding space the affinity was built on.

    Returns:
        (ad, SEACell_ad, model)
    """
    import pickle

    n_cells = len(ad)
    fname = f"affinity_{ds_name}{n_cells}_ncomp{n_components}_kneighbors{k_neighbors}_{affinity_type}.pkl"
    aff_path = os.path.join(graph_dir, fname)
    print(f"Loading affinity from {aff_path} ...")

    with open(aff_path, 'rb') as f:
        aff = pickle.load(f)

    use_gpu, use_sparse = _seacells_backend(n_cells=ad.n_obs)

    model = _make_seacells_model(
        ad,
        build_kernel_on=build_kernel_on,
        n_SEACells=n_SEACells,
        n_waypoint_eigs=n_waypoint_eigs,
        use_gpu=use_gpu,
        use_sparse=use_sparse,
    )

    model.add_precomputed_kernel_matrix(aff)
    model.initialize_archetypes()
    model.fit()

    if "counts" not in ad.layers:
        ad.layers["counts"] = ad.X.copy()

    _ensure_seacells_anndata_patch()  # reapply fresh -- see comment where this is defined
    SEACell_ad = SEACells.core.summarize_by_SEACell(
        ad, SEACells_label="SEACell", summarize_layer="counts"
    )
    return ad, SEACell_ad, model


def save_seacell_df(named_dfs, p):
    for name, df in named_dfs.items():
        df.to_csv(f"{p}/{name}.csv", index=False)


def agg_obs(SEACell_ad, adata, obs_key):
    SEACell_ad.obs[obs_key] = (
        adata.obs.groupby("SEACell")[obs_key]
        .agg(lambda x: x.mode()[0])
        .reindex(SEACell_ad.obs_names)
    )
    return SEACell_ad


def _hash_bytes(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _hash_matrix(m):
    if sp.issparse(m):
        m = m.tocsr()
        return _hash_bytes(m.data, m.indices, m.indptr, np.array(m.shape))
    return _hash_bytes(np.asarray(m))


def _hash_series(s):
    return _hash_bytes(pd.util.hash_pandas_object(s, index=False).values)


def _hash_df(df):
    return _hash_bytes(pd.util.hash_pandas_object(df, index=True).values)


def save_seacell(ad, SEACell_ad, ds_id, build_kernel_on="X_pca", num_prototypes=None):
    """Save per-cell + aggregated metacell AnnData for this tag.

    ad (X, layers, var, and most of obs/obsm/obsp) is identical across every
    tag ('seacell', 'seacell_X_harmony', 'seacell_X_covet', ...) computed for
    the same ds_id -- only obs['SEACell'] (the cluster assignment) and
    occasionally one tag-specific obsm/obsp key actually differ. Instead of
    writing a full independent multi-hundred-MB copy per tag, the first save
    for a ds_id writes a shared 'seacell_sc_base.h5ad' (+ a manifest of
    per-field content hashes) directly under the dataset dir; every later
    save for the same ds_id compares each field (X, each layer, var, each
    varm/obsm/obsp key, each obs column) against that manifest by hash and
    ONLY writes fields that are new or different into this tag's own
    'seacell_sc.h5ad', which becomes a small delta file. Any field that
    doesn't hash-match the base (including things that are supposed to be
    shared but happen to differ, e.g. non-deterministic PCA) is kept in the
    delta rather than deduped -- so a save always reproduces the same ad
    load_seacell() used to return before this change, just split across two
    files. See _reconstruct_from_delta (embedding_metrics.load_seacell).
    """
    from interpretable_ssl.configs.paths import get_seacell_model_dir, get_dataset_model_dir

    seacell_dir = get_seacell_model_dir(ds_id, build_kernel_on, num_prototypes=num_prototypes)
    print("saving to: ", seacell_dir)
    os.makedirs(seacell_dir, exist_ok=True)

    ds_dir = get_dataset_model_dir(ds_id)
    base_path = os.path.join(ds_dir, "seacell_sc_base.h5ad")
    manifest_path = os.path.join(ds_dir, "seacell_sc_base.manifest.json")

    if not os.path.exists(base_path):
        # first tag for this ds_id: it *becomes* the shared base verbatim
        # (everything except the tag-specific 'SEACell' assignment).
        os.makedirs(ds_dir, exist_ok=True)
        base_ad = ad.copy()
        if "SEACell" in base_ad.obs.columns:
            del base_ad.obs["SEACell"]
        base_ad.write(base_path, compression="gzip")

        manifest = {
            "X": _hash_matrix(ad.X) if ad.X is not None else None,
            "layers": {k: _hash_matrix(v) for k, v in ad.layers.items()},
            "var": _hash_df(ad.var),
            "varm": {k: _hash_matrix(v) for k, v in ad.varm.items()},
            "obs": {c: _hash_series(ad.obs[c]) for c in ad.obs.columns if c != "SEACell"},
            "obsm": {k: _hash_matrix(v) for k, v in ad.obsm.items()},
            "obsp": {k: _hash_matrix(v) for k, v in ad.obsp.items()},
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)
        print(f"  created shared base at {base_path}")
    else:
        with open(manifest_path) as f:
            manifest = json.load(f)

    # field-by-field: identical to base -> omit from this tag's file;
    # new or different -> keep in this tag's own delta file
    x_matches = (
        manifest["X"] is not None and ad.X is not None
        and manifest["X"] == _hash_matrix(ad.X)
    )
    var_matches = manifest["var"] == _hash_df(ad.var)
    extra_layers = {k: v for k, v in ad.layers.items()
                     if manifest["layers"].get(k) != _hash_matrix(v)}
    extra_varm = {k: v for k, v in ad.varm.items()
                   if manifest["varm"].get(k) != _hash_matrix(v)}
    extra_obsm = {k: v for k, v in ad.obsm.items()
                   if manifest["obsm"].get(k) != _hash_matrix(v)}
    extra_obsp = {k: v for k, v in ad.obsp.items()
                   if manifest["obsp"].get(k) != _hash_matrix(v)}
    extra_obs_cols = [c for c in ad.obs.columns
                       if c == "SEACell" or manifest["obs"].get(c) != _hash_series(ad.obs[c])]

    delta = sc.AnnData(
        X=None if x_matches else ad.X,
        obs=ad.obs[extra_obs_cols].copy(),
        var=ad.var.copy() if not var_matches else pd.DataFrame(index=ad.var_names),
    )
    for k, v in extra_layers.items():
        delta.layers[k] = v
    for k, v in extra_varm.items():
        delta.varm[k] = v
    for k, v in extra_obsm.items():
        delta.obsm[k] = v
    for k, v in extra_obsp.items():
        delta.obsp[k] = v

    delta.uns["_seacell_delta"] = True
    delta.uns["_seacell_base_path"] = os.path.relpath(base_path, seacell_dir)
    delta.write(os.path.join(seacell_dir, "seacell_sc.h5ad"), compression="gzip")
    SEACell_ad.write(os.path.join(seacell_dir, "seacell_agg.h5ad"), compression="gzip")
    print(f"  delta kept: X={'no (deduped)' if x_matches else 'yes'}, "
          f"{len(extra_layers)} layer(s), {len(extra_varm)} varm, "
          f"{len(extra_obsm)} obsm, {len(extra_obsp)} obsp, "
          f"obs cols {extra_obs_cols}")


def _reconstruct_from_delta(delta, seacell_dir):
    """Inverse of the dedup in save_seacell: merge a delta ad (as loaded
    straight from seacell_sc.h5ad) back onto its shared base to reproduce
    exactly the ad that used to be saved as one full file. Any field present
    in delta overrides/extends the base; everything else comes from base.
    """
    base_path = os.path.join(seacell_dir, delta.uns["_seacell_base_path"])
    base = sc.read_h5ad(base_path)

    if not base.obs_names.equals(delta.obs_names):
        raise ValueError(
            f"seacell base ({base_path}) and delta obs_names don't match -- "
            "can't safely reconstruct ad. This shouldn't happen unless the "
            "base was overwritten by a different dataset load order."
        )

    full = base.copy()
    if delta.X is not None:
        full.X = delta.X
    for c in delta.obs.columns:
        full.obs[c] = delta.obs[c].reindex(full.obs_names).values
    for k in delta.layers.keys():
        full.layers[k] = delta.layers[k]
    for k in delta.varm.keys():
        full.varm[k] = delta.varm[k]
    for k in delta.obsm.keys():
        full.obsm[k] = delta.obsm[k]
    for k in delta.obsp.keys():
        full.obsp[k] = delta.obsp[k]
    if delta.var.shape[1] > 0:
        for c in delta.var.columns:
            full.var[c] = delta.var[c].values
    return full
