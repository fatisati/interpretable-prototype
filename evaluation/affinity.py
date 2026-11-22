"""Cell-cell affinity graph, shared by every baseline's cohesion metric.

Same recipe SEACells itself uses (F:\\codes\\SEACells\\SEACells\\build_graph.py):
an adaptive-bandwidth RBF kernel on a symmetrized kNN graph built in PCA
space -- the bandwidth for each cell is its own distance to its (k//2)-th
nearest neighbor, not one fixed value for every cell, so dense and sparse
regions of the embedding get different-sized neighborhoods.

Computed sparse-native instead of SEACells' per-row Python loop over the full
n x n dense distance matrix. SEACells computes, for every cell, its squared
distance to *all* n cells and then masks down to the ~2k it keeps -- here we
only ever compute distances for the pairs the kNN graph actually keeps, which
is the same output through a cheaper path, not a different method.
"""
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.neighbors import NearestNeighbors


def build_affinity_graph(embedding, k=15, graph_construction="union"):
    """Adaptive RBF affinity graph over `embedding` (n_cells x n_dims, e.g. PCA).

    k: neighborhood size, same meaning as SEACells' own `k` param (includes
       the cell itself, so there are k-1 real neighbors per cell).
    graph_construction: 'union' or 'intersection' -- same two options
       SEACells exposes, with the same meaning (OR vs AND of each cell's
       kNN membership before computing weights).

    Returns a sparse (n_cells, n_cells) CSR matrix. Diagonal is 1 (every
    cell is its own nearest neighbor at distance 0, so its RBF weight is
    exp(0) = 1) -- same as SEACells' own kernel_matrix, which is what makes
    this matrix usable directly as a precomputed SEACells kernel.
    """
    n = embedding.shape[0]

    nn = NearestNeighbors(n_neighbors=k).fit(embedding)
    distances, indices = nn.kneighbors(embedding)  # (n, k); column 0 is the cell itself (distance 0)

    # Bandwidth = distance to the (k//2)-th real neighbor, i.e. column k//2 --
    # column 0 is self, so this already skips self, matching SEACells exactly.
    bandwidth = distances[:, k // 2]

    rows = np.repeat(np.arange(n), k)
    cols = indices.ravel()
    knn_graph = coo_matrix((np.ones(n * k), (rows, cols)), shape=(n, n)).tocsr()

    if graph_construction == "union":
        sym_graph = (knn_graph + knn_graph.T) > 0
    elif graph_construction in ("intersection", "intersect"):
        sym_graph = knn_graph.multiply(knn_graph.T) > 0
    else:
        raise ValueError(f"graph_construction must be 'union' or 'intersection', got {graph_construction!r}")
    sym_graph = sym_graph.tocoo()

    # RBF weight for every edge the symmetrized graph kept, computed only for
    # those (row, col) pairs -- not a full dense n x n grid.
    row_idx, col_idx = sym_graph.row, sym_graph.col
    sq_dist = np.sum((embedding[row_idx] - embedding[col_idx]) ** 2, axis=1)
    denom = bandwidth[row_idx] * bandwidth[col_idx]
    weights = np.exp(-sq_dist / denom)

    return csr_matrix((weights, (row_idx, col_idx)), shape=(n, n))
