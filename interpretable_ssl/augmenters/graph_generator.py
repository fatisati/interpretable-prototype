import numpy as np
import scanpy as sc
from interpretable_ssl.augmenters.diffusion_knn import *
import numpy as np, scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


def compute_affinities(adata_path, affinity_type, batch_key, n_comps, k):
    adata = sc.read_h5ad(adata_path)
    sc.tl.pca(adata, n_comps=n_comps)
    A = generate_affinity(adata, k, batch_key, affinity_type)
    if sp.issparse(A):
        A.setdiag(0)
    else:
        np.fill_diagonal(A, 0)
    return A


def generate_affinity(ad, k, bk, affinity_type="inverse_dist"):
    print(affinity_type)
    if affinity_type == "inverse_dist":
        ind, dist = faiss_knn(ad, k)
        inv_dist = 1.0 / (dist + 1e-8)  # avoid div by zero
        return inv_dist

    elif affinity_type in ["arbf", "coaff", "ncoaff", "sym-coaff"]:
        import SEACells

        print("calculating seacell affinity")
        kernel_model = SEACells.build_graph.SEACellGraph(ad, "X_pca", verbose=True)
        if affinity_type.startswith("sym"):
            graph_construction = "intersect"
        else:
            graph_construction = "union"
        M = kernel_model.rbf(k, graph_construction=graph_construction)
        if affinity_type == "coaff":
            return M @ M.T
        elif affinity_type == "ncoaff":
            # --- L2 normalize rows to remove degree bias ---
            row_norms = np.sqrt(M.multiply(M).sum(axis=1)).A1  # vector of ||M_i||
            row_norms[row_norms == 0] = 1e-12  # avoid division by 0
            M_norm = M.multiply(1.0 / row_norms[:, None])  # each row -> unit L2 norm

            # --- compute normalized co-affinity (cosine between rows of M) ---
            C = M_norm @ M_norm.T  # still sparse
            return C
        else:  # arbf
            return M

    elif affinity_type == "umap":
        sc.pp.neighbors(ad, n_neighbors=k, use_rep="X_pca")
        return ad.obsp["connectivities"]

    elif affinity_type in ["spatial", 'scoaff']:
        s_aff = multi_batch_aff(ad, bk, lambda x: spatial_affinity(x, k))
        if affinity_type == 'spatial':
            return s_aff
        else:
            return s_aff @ s_aff.T

    elif affinity_type in ["st", 'stcoaff']:
        st_aff = multi_batch_aff(ad, bk, lambda x: st_affinity(x, k))
        if affinity_type == 'st':
            return st_aff
        else:
            return st_aff @ st_aff.T

def faiss_knn(b_adata, k):
    import faiss

    batch_pca = b_adata.obsm["X_pca"]
    index = faiss.IndexFlatL2(batch_pca.shape[1])
    index.add(batch_pca)
    D, I = index.search(batch_pca, k + 1)  # search k+1
    return D[:, 1:], I[:, 1:]


def diffusion_knn(batch_adata, k, n_proto):
    import SEACells

    model = SEACells.core.SEACells(
        batch_adata,
        build_kernel_on="X_pca",
        n_SEACells=n_proto,
        n_waypoint_eigs=10,
        convergence_epsilon=1e-5,
    )
    model.construct_kernel_matrix()
    km = model.kernel_matrix
    I, A, D = diffusion_knn_from_affinity(km, k)
    return D, I


def multi_batch_aff(ad, bk, fn):
    import scipy.sparse as sp

    rows_all = []
    cols_all = []
    vals_all = []
    for b in ad.obs[bk].unique():
        idx = np.where(ad.obs[bk].values == b)[0]
        A_b = fn(ad[idx]).tocoo()
        rows_all.append(idx[A_b.row])
        cols_all.append(idx[A_b.col])
        vals_all.append(A_b.data)
    rows = np.concatenate(rows_all)
    cols = np.concatenate(cols_all)
    vals = np.concatenate(vals_all)
    return sp.csr_matrix((vals, (rows, cols)), shape=(ad.n_obs, ad.n_obs))


def spatial_affinity(ad, k):
    return build_seacell_kernel(ad.obsm["spatial"], ad.obsm["spatial"], k=k)


def st_affinity(ad, k=50):
    return build_seacell_kernel(ad.obsm["spatial"], ad.obsm["X_pca"], k=k)


def build_seacell_kernel(X_knn, X_aff, k=50, graph_construction="union"):
    nn = NearestNeighbors(n_neighbors=k).fit(X_knn)
    _, idxs = nn.kneighbors(X_knn)  # (n, k)

    diff = X_aff[:, None, :] - X_aff[idxs]  # (n, k, d)
    dists = np.linalg.norm(diff, axis=2)  # (n, k)

    # from scipy.spatial.distance import cdist

    # dists = cdist(X_aff, X_aff[idxs].reshape(-1, X_aff.shape[1])).reshape(n, k)

    sigma = np.median(dists, axis=1, keepdims=True)  # (n, 1)
    sigma_i = sigma  # (n,1)
    sigma_j = sigma[idxs]  # (n,k,1)
    sigma_prod = sigma_i * sigma_j.squeeze(-1)  # (n,k)

    A_vals = np.exp(-(dists**2) / sigma_prod)  # (n,k)

    rows = np.repeat(np.arange(len(X_knn)), k)
    cols = idxs.reshape(-1)
    A = sp.csr_matrix(
        (A_vals.reshape(-1), (rows, cols)), shape=(len(X_knn), len(X_knn))
    )

    if graph_construction == "union":
        A = A + A.T - A.multiply(A.T)
    elif graph_construction in ["intersection", "intersect"]:
        A = A.multiply(A.T)
    else:
        raise ValueError("graph_construction must be 'union' or 'intersection'")

    return A


import sys
import pickle
import os

if __name__ == "__main__":
    config_file, save_path, lock_path = sys.argv[1:4]

    with open(config_file, "rb") as f:
        args = pickle.load(f)

    # Unpack into build_graph
    aff = compute_affinities(*args)

    tmp_path = save_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(aff, f)
    os.replace(tmp_path, save_path)  # atomic swap
    os.remove(lock_path)
    print(f"{lock_path} removed")
