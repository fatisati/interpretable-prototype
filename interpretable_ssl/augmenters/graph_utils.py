import numpy as np
import faiss
import scanpy as sc

def faiss_knn_within_batches(adata, batch_key, n_comps, k):
    n = len(adata)
    indices = np.zeros((n, k), dtype=int)
    distances = np.zeros((n, k), dtype=float)

    for batch_id in adata.obs[batch_key].unique():
        
        mask = adata.obs[batch_key] == batch_id
        b_adata = adata[mask]
        sc.tl.pca(b_adata, n_comps=n_comps)
        batch_pca = b_adata.obsm['X_pca']
        
        global_idx = np.where(mask)[0]

        index = faiss.IndexFlatL2(batch_pca.shape[1])
        index.add(batch_pca)
        D, I = index.search(batch_pca, k + 1)

        indices[global_idx] = global_idx[I[:, 1:]]
        distances[global_idx] = D[:, 1:]

    return indices, distances

def knn_indices_to_edge_set(indices):
    edges = set()
    for i, row in enumerate(indices):
        for j in row:
            if i != j:  # optional, to avoid self-loops
                edges.add((i, j))  # directed edge
    return edges
