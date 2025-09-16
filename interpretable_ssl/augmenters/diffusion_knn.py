import numpy as np
from scipy.sparse import csr_matrix, issparse, diags

def _to_csr_no_self(K):
    """Ensure CSR and remove self-similarities on the diagonal."""
    if not issparse(K):
        K = csr_matrix(K)
    K = K.tocsr(copy=True)
    K.setdiag(0.0)
    K.eliminate_zeros()
    return K

def _row_normalize(W):
    """Row-stochastic normalization: P = D^{-1} W (rows sum to 1)."""
    rowsum = np.array(W.sum(axis=1)).ravel()
    rowsum[rowsum == 0] = 1.0
    Dinv = diags(1.0 / rowsum)
    return Dinv @ W

def _power_sparse(P, t=2):
    """Compute P^t by repeated sparse multiplications."""
    X = P
    for _ in range(t - 1):
        X = X @ P
    return X

def _topk_per_row_dense(M, k):
    """Return top-k values and indices per row from a dense matrix."""
    n = M.shape[0]
    k = min(k, M.shape[1]-1)  # safety
    # argpartition for O(n) top-k, then sort those k
    idx_part = np.argpartition(M, -k, axis=1)[:, -k:]
    row_idx = np.arange(n)[:, None]
    top_vals = M[row_idx, idx_part]
    order = np.argsort(-top_vals, axis=1)
    idx = idx_part[row_idx, order]
    vals = top_vals[row_idx, order]
    return vals, idx

def diffusion_knn_from_affinity(K, k=15, t=2, return_affinity=True):
    """
    Input:
      K: (N x N) kernel/affinity (dense or sparse). Larger = more similar.
      k: neighbors per node.
      t: diffusion steps (t=1 ~ original one-step neighbors; t>=2 = similar-to-similar).
    Output:
      I: (N x k) neighbor indices per node
      A: (N x k) diffusion affinities (optional)
      D: (N x k) distances (1 - affinity), same shape
    """
    # 1) Make it sparse & remove self-loops
    W = _to_csr_no_self(K)

    # 2) Build diffusion operator P = D^{-1} W (row-stochastic)
    P = _row_normalize(W)

    # 3) Multi-step diffusion affinity: P^t
    Pt = _power_sparse(P, t=t)

    # 4) Convert to dense for simple top-k selection (OK for small/medium N)
    #    For very large N, consider keeping it sparse and doing per-row selection.
    M = Pt.toarray()

    # 5) Zero diagonal again (no self as neighbor)
    np.fill_diagonal(M, 0.0)

    # 6) Pick top-k neighbors per row by diffusion affinity
    A, I = _topk_per_row_dense(M, k=k)

    # 7) Turn affinities into distances in [0,1] (since rows of P^t sum ~1)
    D = 1.0 - A

    return (I, A, D) if return_affinity else (I, D)
