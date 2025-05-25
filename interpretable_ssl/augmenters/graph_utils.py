import numpy as np
import faiss

def faiss_knn_within_batches(data, batch_ids, k):
    n = data.shape[0]
    indices = np.zeros((n, k), dtype=int)
    distances = np.zeros((n, k), dtype=float)

    for batch in np.unique(batch_ids):
        mask = batch_ids == batch
        batch_data = data[mask]
        global_idx = np.where(mask)[0]

        index = faiss.IndexFlatL2(batch_data.shape[1])
        index.add(batch_data)
        D, I = index.search(batch_data, k + 1)

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
