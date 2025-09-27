import faiss
import torch
import numpy as np
from scipy.sparse import lil_matrix
from tqdm import tqdm


def local_bandwidth_lnn(z: torch.Tensor, l: int) -> torch.Tensor:
    """
    Compute local bandwidth = distance to l-th nearest neighbor using FAISS.

    Args:
        z (torch.Tensor): (N, d) tensor of features
        l (int): neighbor index (>= 1)

    Returns:
        torch.Tensor: (N,) tensor with distances to l-th neighbor
    """
    # make sure input is float32 on CPU for FAISS
    z_np = z.detach().cpu().numpy().astype("float32")

    # build index (exact L2)
    index = faiss.IndexFlatL2(z_np.shape[1])
    index.add(z_np)

    # query self against index
    D, I = index.search(z_np, l + 1)  # D: (N, l+1)

    # D is squared L2 distance by default → take sqrt
    knn_dists = np.sqrt(D)

    # take distance to l-th neighbor (skip self at position 0)
    lnn = knn_dists[:, l]

    # back to torch
    return torch.from_numpy(lnn).clamp_min(1e-8)


def cell_proto_affinity(z, protos, sigma):
    # if normalize:
    #     z, protos = F.normalize(z, dim=1), F.normalize(protos, dim=1)
    d = torch.cdist(z, protos, p=2)
    return torch.exp(-(d**2) / (sigma[:, None] ** 2))


def kth_neighbor_distance(distances, k, i):
    """Returns distance to kth nearest neighbor.

    Distances: sparse CSR matrix
    k: kth nearest neighbor
    i: index of row
    .
    """
    # convert row to 1D array
    row_as_array = distances[i, :].toarray().ravel()

    # number of nonzero elements
    num_nonzero = np.sum(row_as_array > 0)

    # argsort
    kth_neighbor_idx = np.argsort(np.argsort(-row_as_array)) == num_nonzero - k
    return np.linalg.norm(row_as_array[kth_neighbor_idx])


def rbf_for_row(G, data, median_distances, i):
    """Helper function for computing radial basis function kernel for each row of the data matrix.

    :param G: (array) KNN graph representing nearest neighbour connections between cells
    :param data: (array) data matrix between which euclidean distances are computed for RBF
    :param median_distances: (array) radius for RBF - the median distance between cell and k nearest-neighbours
    :param i: (int) data row index for which RBF is calculated
    :return: sparse matrix containing computed RBF for row
    """
    # convert row to binary numpy array
    row_as_array = G[i, :].toarray().ravel()

    # compute distances ||x - y||^2 in PC/original X space
    numerator = np.sum(np.square(data[i, :] - data), axis=1, keepdims=False)

    # compute radii - median distance is distance to kth nearest neighbor
    denominator = median_distances[i] * median_distances

    # exp
    full_row = np.exp(-numerator / denominator)

    # masked row - to contain only indices captured by G matrix
    masked_row = np.multiply(full_row, row_as_array)

    return lil_matrix(masked_row)


def compute_median_distances(z, k, verbose=True):
    z_np = z.detach().cpu().numpy().astype("float32")

    # build index (exact L2)
    index = faiss.IndexFlatL2(z_np.shape[1])
    index.add(z_np)

    # query self against index
    D, I = index.search(z_np, k + 1)  # D: (N, l+1)

    # D is squared L2 distance by default → take sqrt
    knn_graph_distances = np.sqrt(D)
    
    # self.knn_graph = knn_graph
    if verbose:
        print("Computing radius for adaptive bandwidth kernel...")

    median = k // 2
    median_distances = []
    n = knn_graph_distances.shape[0]
    for i in range(n):
        d = kth_neighbor_distance(knn_graph_distances, median, i)
        median_distances.append(d)

    # convert to numpy array
    median_distances = np.array(median_distances)
    return median_distances


def rbf(
    knn_graph_distances, x, median_distances, graph_construction="union", verbose=True
):
    """Initialize adaptive bandwith RBF kernel (as described in C-isomap).

    :param k: (int) number of nearest neighbors for RBF kernel
    :return: (sparse matrix) constructed RBF kernel
    """
    # import scanpy as sc

    # if self.verbose:
    #     print("Computing kNN graph using scanpy NN ...")

    # # compute kNN and the distance from each point to its nearest neighbors
    # sc.pp.neighbors(self.ad, use_rep=self.build_on, n_neighbors=k, knn=True)
    # knn_graph_distances = self.ad.obsp["distances"]

    # Binarize distances to get connectivity
    knn_graph = knn_graph_distances.copy()
    knn_graph[knn_graph != 0] = 1
    # Include self as neighbour
    knn_graph.setdiag(1)

    # median_distances = compute_median_distances(knn_graph_distances)

    if verbose:
        print("Making graph symmetric...")

    print(
        f"Parameter graph_construction = {graph_construction} being used to build KNN graph..."
    )
    if graph_construction == "union":
        sym_graph = (knn_graph + knn_graph.T > 0).astype(float)
    elif graph_construction in ["intersect", "intersection"]:
        knn_graph = (knn_graph > 0).astype(float)
        sym_graph = knn_graph.multiply(knn_graph.T)
    else:
        raise ValueError(
            f"Parameter graph_construction = {graph_construction} is not valid. \
         Please select `union` or `intersection`"
        )

    # self.sym_graph = sym_graph
    if verbose:
        print("Computing RBF kernel...")

    n = knn_graph_distances.shape[0]
    similarity_matrix_rows = [
        rbf_for_row(sym_graph, x, median_distances, i) for i in tqdm(range(n))
    ]

    if verbose:
        print("Building similarity LIL matrix...")

    similarity_matrix = lil_matrix((n, n))
    for i in tqdm(range(n)):
        similarity_matrix[i] = similarity_matrix_rows[i]

    if verbose:
        print("Constructing CSR matrix...")

    M = (similarity_matrix).tocsr()
    return M
