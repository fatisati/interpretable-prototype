import numpy as np
import scanpy as sc
from interpretable_ssl.augmenters.diffusion_knn import *
import numpy as np, scipy.sparse as sp

def faiss_knn(b_adata, k):
    import faiss

    batch_pca = b_adata.obsm["X_pca"]
    index = faiss.IndexFlatL2(batch_pca.shape[1])
    index.add(batch_pca)
    D, I = index.search(batch_pca, k + 1)   # search k+1
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


def generate_affinity(batch_ad, k, affinity_type="inverse_dist"):
    print(affinity_type)
    if affinity_type == "inverse_dist":
        ind, dist = faiss_knn(batch_ad, k)
        inv_dist = 1.0 / (dist + 1e-8)  # avoid div by zero
        return inv_dist

    elif affinity_type in ["arbf", "coaff", "ncoaff", 'sym-coaff']:
        import SEACells
        print('calculating seacell affinity')
        kernel_model = SEACells.build_graph.SEACellGraph(
            batch_ad, "X_pca", verbose=True
        )
        if affinity_type.startswith('sym'):
            graph_construction = 'intersect'
        else:
            graph_construction = 'union'
        M = kernel_model.rbf(k, graph_construction=graph_construction)
        if affinity_type == "coaff":
            return M @ M.T
        elif affinity_type == "ncoaff":
            # --- L2 normalize rows to remove degree bias ---
            row_norms = np.sqrt(M.multiply(M).sum(axis=1)).A1  # vector of ||M_i||
            row_norms[row_norms == 0] = 1e-12                  # avoid division by 0
            M_norm = M.multiply(1.0 / row_norms[:, None])      # each row -> unit L2 norm

            # --- compute normalized co-affinity (cosine between rows of M) ---
            C = M_norm @ M_norm.T                              # still sparse
            return C
        else: #arbf
            return M

    elif affinity_type == "umap":
        sc.pp.neighbors(batch_ad, n_neighbors=k, use_rep="X_pca")
        return batch_ad.obsp["connectivities"]


def compute_batch_affinities(adata_path, affinity_type, batch_key, n_comps, k):
    adata = sc.read_h5ad(adata_path)
    sc.tl.pca(adata, n_comps=n_comps)
    A = generate_affinity(adata, k, affinity_type)
    if sp.issparse(A):
            A.setdiag(0)
    else:
        np.fill_diagonal(A, 0)
    return A


import sys
import pickle
import os

if __name__ == "__main__":
    config_file, save_path, lock_path = sys.argv[1:4]

    with open(config_file, "rb") as f:
        args = pickle.load(f)

    # Unpack into build_graph
    aff = compute_batch_affinities(*args)

    tmp_path = save_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(aff, f)
    os.replace(tmp_path, save_path)  # atomic swap
    os.remove(lock_path)
    print(f'{lock_path} removed')
