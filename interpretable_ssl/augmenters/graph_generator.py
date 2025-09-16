import numpy as np
import scanpy as sc
from interpretable_ssl.augmenters.diffusion_knn import *

def faiss_knn(b_adata, k, _n_proto):
    import faiss
    batch_pca = b_adata.obsm['X_pca']
    index = faiss.IndexFlatL2(batch_pca.shape[1])
    index.add(batch_pca)
    D, I = index.search(batch_pca, k + 1)
    return D[:, 1:], I[:, 1:]

def diffusion_knn(batch_adata, k, n_proto):
    import SEACells
    model = SEACells.core.SEACells(batch_adata, 
                  build_kernel_on='X_pca', 
                  n_SEACells=n_proto, 
                  n_waypoint_eigs=10,
                  convergence_epsilon = 1e-5)
    model.construct_kernel_matrix()
    km = model.kernel_matrix
    I, A, D = diffusion_knn_from_affinity(km, k)
    return D, I
    
def knn_within_batches(adata, knn_func, batch_key, n_comps, k, *args):
    n = len(adata)
    indices = np.zeros((n, k), dtype=int)
    distances = np.zeros((n, k), dtype=float)

    for batch_id in adata.obs[batch_key].unique():
        print(batch_id)
        mask = adata.obs[batch_key] == batch_id
        
        b_adata = adata[mask]
        sc.tl.pca(b_adata, n_comps=n_comps)

        D, I = knn_func(b_adata, k, *args)

        global_idx = np.where(mask)[0]
        indices[global_idx] = global_idx[I]
        distances[global_idx] = D

    return indices, distances

def knn_indices_to_edge_set(indices):
    edges = set()
    for i, row in enumerate(indices):
        for j in row:
            if i != j:  # optional, to avoid self-loops
                edges.add((i, j))  # directed edge
    return edges

def build_graph(adata_path, knn_method, *args):
    adata = sc.read_h5ad(adata_path)
    if knn_method == 'faiss':
        return knn_within_batches(adata, faiss_knn, *args)
    elif knn_method == 'diffusion':
        return knn_within_batches(adata, diffusion_knn, *args)
    
    
import sys
import pickle

if __name__ == "__main__":
    config_file, save_path = sys.argv[1:3]

    with open(config_file, "rb") as f:
        args = pickle.load(f)

    # Unpack into build_graph
    graph = build_graph(*args)

    with open(save_path, "wb") as f:
        pickle.dump(graph, f)