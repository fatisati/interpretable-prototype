from collections import defaultdict
import numpy as np
from sklearn.neighbors import NearestNeighbors
import logging
import faiss
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_knn_graph_per_batch(spatial_coords, batch_ids, k=10):
    from sklearn.neighbors import NearestNeighbors
    import numpy as np

    all_edges = set()
    unique_batches = np.unique(batch_ids)

    for batch in unique_batches:
        # mask for current batch
        mask = batch_ids == batch
        coords = spatial_coords[mask]
        global_indices = np.where(mask)[0]  # index in full adata

        # # build kNN on this batch
        # nn = NearestNeighbors(n_neighbors=k + 1)
        # nn.fit(coords)
        # _, indices = nn.kneighbors(coords)
        # Initialize the FAISS index
        logger.info("Initializing FAISS index.")
        faiss_index = faiss.IndexFlatL2(coords.shape[1])
        faiss_index.add(coords)

        # Perform the kNN search
        logger.info("Performing kNN search with FAISS.")
        distances, indices = faiss_index.search(coords, k + 1)

        # Exclude the self-loop (first column corresponds to the point itself)
        distances = distances[:, 1:]
        indices = indices[:, 1:]

        # add edges using global indices
        for i, row in enumerate(indices):
            for j in row[1:]:  # skip self
                u = global_indices[i]
                v = global_indices[j]
                # keep edge as directed (do NOT sort)
                all_edges.add((u, v))

    return all_edges


def get_graph_intersection(edges1, edges2):
    return edges1 & edges2


def count_nodes_with_less_than(edges, num_nodes, threshold):
    out_degree = defaultdict(int)
    for u, v in edges:
        out_degree[u] += 1  # only count u → v
    degree_array = np.array([out_degree[i] for i in range(num_nodes)])
    count = np.sum(degree_array < threshold)
    print(f"Number of nodes with fewer than {threshold} outgoing neighbors: {count}")
    return count


def fill_in_missing_edges(intersect_edges, candidate_edges, k, n_nodes):
    # Track out-degree
    out_degree = defaultdict(int)
    for u, v in intersect_edges:
        out_degree[u] += 1

    # Candidate neighbors: directed u → v
    candidate_neighbors = defaultdict(list)
    for u, v in candidate_edges:
        candidate_neighbors[u].append(v)

    updated_edges = set(intersect_edges)

    for node in range(n_nodes):
        needed = k - out_degree[node]
        if needed <= 0:
            continue

        tried = set()
        for neighbor in candidate_neighbors[node]:
            if neighbor == node or (node, neighbor) in updated_edges:
                continue
            updated_edges.add((node, neighbor))
            out_degree[node] += 1
            tried.add(neighbor)
            if out_degree[node] >= k:
                break

        # Fallback: connect to any other nodes if still not enough
        if out_degree[node] < k:
            for neighbor in range(n_nodes):
                if (
                    neighbor == node
                    or (node, neighbor) in updated_edges
                    or neighbor in tried
                ):
                    continue
                updated_edges.add((node, neighbor))
                out_degree[node] += 1
                if out_degree[node] >= k:
                    break

    return updated_edges


def compare_graphs(reference_edges, test_edges, name=""):
    intersection = reference_edges & test_edges
    union = reference_edges | test_edges

    jaccard = len(intersection) / len(union) if len(union) > 0 else 0
    precision = len(intersection) / len(test_edges) if len(test_edges) > 0 else 0
    recall = len(intersection) / len(reference_edges) if len(reference_edges) > 0 else 0

    print(f"📊 Comparison to {name} graph:")
    print(f"  Shared edges     : {len(intersection)}")
    print(f"  Jaccard similarity: {jaccard:.3f}")
    print(
        f"  Precision         : {precision:.3f}  (how much of test overlaps with {name})"
    )
    print(
        f"  Recall            : {recall:.3f}  (how much of {name} is recovered in test)"
    )
    print()
    return jaccard, precision, recall


def get_combined_features(adata):
    # Combined features
    lambda_ = 0.5
    spatial_scaled = StandardScaler().fit_transform(adata.obs[["x", "y"]])
    expr_scaled = StandardScaler().fit_transform(adata.obsm["X_pca"])
    spatial_weight = lambda_
    expr_weight = (1 - lambda_) * (spatial_scaled.shape[1] / expr_scaled.shape[1])

    combined_features = np.concatenate(
        [spatial_scaled * spatial_weight, expr_scaled * expr_weight], axis=1
    )
    return combined_features


def build_indices_and_distances_from_edges(edges, combined_features, k):
    neighbor_dict = defaultdict(list)

    for u, v in edges:
        # Compute Euclidean distance between combined feature vectors
        d = np.linalg.norm(combined_features[u] - combined_features[v])
        neighbor_dict[u].append((v, d))
        neighbor_dict[v].append((u, d))  # assuming undirected

    n = combined_features.shape[0]
    indices = np.full((n, k), -1, dtype=int)
    distances = np.full((n, k), np.inf, dtype=float)

    for i in neighbor_dict:
        # Sort neighbors by distance
        sorted_neighbors = sorted(neighbor_dict[i], key=lambda x: x[1])[:k]
        for j, (nbr, dist) in enumerate(sorted_neighbors):
            indices[i, j] = nbr
            distances[i, j] = dist

    return indices, distances


def generate_spatio_transcriptional_graph(adata, k, min_k, batch_label="batch"):
    print("using new method:)")
    spatial = adata.obs[["x", "y"]].to_numpy()
    batch_ids = adata.obs[batch_label].values
    spatial_edges = build_knn_graph_per_batch(spatial, batch_ids, k)

    expr = adata.obsm["X_pca"]  # or your embedding
    expr_edges = build_knn_graph_per_batch(expr, batch_ids, k)

    # Intersect graphs
    intersect_edges = get_graph_intersection(spatial_edges, expr_edges)

    # Print results
    print(f"Spatial edges: {len(spatial_edges)}")
    print(f"Expression edges: {len(expr_edges)}")
    print(f"Intersection edges: {len(intersect_edges)}")

    num_nodes = adata.n_obs  # or len(adata)
    count_nodes_with_less_than(intersect_edges, num_nodes, min_k)

    combined_features = get_combined_features(adata)
    candidate_knn_edges = build_knn_graph_per_batch(
        combined_features, batch_ids, k=min_k
    )
    filled_edges = fill_in_missing_edges(
        intersect_edges, candidate_knn_edges, k=min_k, n_nodes=adata.n_obs
    )

    count_nodes_with_less_than(filled_edges, num_nodes, min_k)

    # Example: compare filled graph to spatial and expression graphs
    compare_graphs(spatial_edges, filled_edges, name="spatial")
    compare_graphs(expr_edges, filled_edges, name="expression")

    indices, distances = build_indices_and_distances_from_edges(
        filled_edges, combined_features, k
    )
    return indices, distances
