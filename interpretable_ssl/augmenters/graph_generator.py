import numpy as np
import scanpy as sc
from interpretable_ssl.augmenters.diffusion_knn import *


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
    if affinity_type == "inverse_dist":
        ind, dist = faiss_knn(batch_ad, k)
        inv_dist = 1.0 / (dist + 1e-8)  # avoid div by zero
        return inv_dist

    elif affinity_type in ["arbf", "coaffinity"]:
        import SEACells

        kernel_model = SEACells.build_graph.SEACellGraph(
            batch_ad, "X_pca", verbose=True
        )
        M = kernel_model.rbf(k)
        if affinity_type == "coaffinity":
            return M @ M.T
        else:
            return M

    elif affinity_type == "umap":
        sc.pp.neighbors(batch_ad, n_neighbors=k, use_rep="X_pca")
        return batch_ad.obsp["connectivities"]


def compute_batch_affinities(adata_path, affinity_type, batch_key, n_comps, k):
    adata = sc.read_h5ad(adata_path)
    ds_affinities = {}
    for batch_id in adata.obs[batch_key].unique():
        print(batch_id)
        mask = adata.obs[batch_key] == batch_id
        b_adata = adata[mask].copy()
        sc.tl.pca(b_adata, n_comps=n_comps)
        ds_affinities[batch_id] = generate_affinity(b_adata, k, affinity_type)

    return ds_affinities


import sys
import pickle
import os

if __name__ == "__main__":
    config_file, save_path = sys.argv[1:3]

    with open(config_file, "rb") as f:
        args = pickle.load(f)

    # Unpack into build_graph
    ds_affinities = compute_batch_affinities(*args)

    tmp_path = save_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(ds_affinities, f)
    os.replace(tmp_path, save_path)  # atomic swap
