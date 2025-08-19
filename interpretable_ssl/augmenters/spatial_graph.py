from collections import defaultdict
import numpy as np
from sklearn.neighbors import NearestNeighbors
import logging
import faiss
from sklearn.preprocessing import StandardScaler
from interpretable_ssl.augmenters.graph_utils import *

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


from collections import defaultdict


def fill_missing_edges(intersect_edges, knn_indices, k):
    """
    Return a new set of edges that fills from combined features
    only where intersection edges are insufficient.
    """
    from collections import defaultdict

    n_nodes = knn_indices.shape[0]
    out_degree = defaultdict(int)
    for u, v in intersect_edges:
        out_degree[u] += 1

    # Collect candidate edges from combined_knn
    added_edges = set()
    for u in range(n_nodes):
        needed = k - out_degree[u]
        if needed <= 0:
            continue
        for v in knn_indices[u]:
            if (u, v) not in intersect_edges and (u, v) not in added_edges and u != v:
                added_edges.add((u, v))
                out_degree[u] += 1
                if out_degree[u] >= k:
                    break
    return added_edges



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
    if 'x' in adata.obs:
        spatial_scaled = StandardScaler().fit_transform(adata.obs[["x", "y"]])
    else:
        spatial_scaled = StandardScaler().fit_transform(adata.obsm['spatial'])
    expr_scaled = StandardScaler().fit_transform(adata.obsm["X_pca"])
    spatial_weight = lambda_
    expr_weight = (1 - lambda_) * (spatial_scaled.shape[1] / expr_scaled.shape[1])

    combined_features = np.concatenate(
        [spatial_scaled * spatial_weight, expr_scaled * expr_weight], axis=1
    )
    return combined_features

def build_neighbor_dict_from_edges(edges, combined_features):
    neighbor_dict = defaultdict(list)
    for u, v in edges:
        d = np.linalg.norm(combined_features[u] - combined_features[v])
        neighbor_dict[u].append((v, d))

    # Sort each neighbor list by distance
    sorted_dict = {
        u: sorted(neighbors, key=lambda x: x[1])
        for u, neighbors in neighbor_dict.items()
    }
    return sorted_dict

def neighbor_dict_to_array(neighbor_dict, max_k=None):
    n = len(neighbor_dict)
    max_neighbors = max(len(v) for v in neighbor_dict.values()) if max_k is None else max_k
    indices = np.full((n, max_neighbors), -1, dtype=int)
    distances = np.full((n, max_neighbors), np.inf, dtype=float)

    for i in range(n):
        for j, (v, d) in enumerate(neighbor_dict.get(i, [])[:max_neighbors]):
            indices[i, j] = v
            distances[i, j] = d

    return indices, distances

def generate_spatio_transcriptional_graph(adata, k, min_k, batch_label="batch"):
    batch_ids = adata.obs[batch_label].values
    num_nodes = adata.n_obs

    # Spatial edges
    if 'x' in adata.obs:
        spatial = adata.obs[["x", "y"]].to_numpy()
    else:
        spatial = adata.obsm['spatial']
    spatial_knn, _ = faiss_knn_within_batches(spatial, batch_ids, k)
    spatial_edges = knn_indices_to_edge_set(spatial_knn)

    # Expression edges
    expr = adata.obsm["X_pca"]
    expr_knn, _ = faiss_knn_within_batches(expr, batch_ids, k)
    expr_edges = knn_indices_to_edge_set(expr_knn)

    # Intersect graphs
    intersect_edges = get_graph_intersection(spatial_edges, expr_edges)
    print(f"Spatial edges: {len(spatial_edges)}")
    print(f"Expression edges: {len(expr_edges)}")
    print(f"Intersection edges: {len(intersect_edges)}")

    count_nodes_with_less_than(intersect_edges, num_nodes, min_k)

    # Use combined features to fill missing edges
    combined = get_combined_features(adata)
    combined_knn, combined_distances = faiss_knn_within_batches(
        combined, batch_ids, min_k
    )

    added_edges = fill_missing_edges(intersect_edges, combined_knn, min_k)
    final_edges = intersect_edges | added_edges

    count_nodes_with_less_than(final_edges, num_nodes, min_k)
    compare_graphs(spatial_edges, final_edges, name="spatial")
    compare_graphs(expr_edges, final_edges, name="expression")

    neighbor_dict = build_neighbor_dict_from_edges(final_edges, combined)
    indices, distances = neighbor_dict_to_array(neighbor_dict)
    return indices, distances
