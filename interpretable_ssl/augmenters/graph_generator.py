import numpy as np
import scanpy as sc
from interpretable_ssl.augmenters.diffusion_knn import *
import numpy as np, scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


def compute_affinities(adata_path, affinity_type, batch_key, n_comps, k, graph_mode):
    adata = sc.read_h5ad(adata_path)
    sc.tl.pca(adata, n_comps=n_comps)
    A = generate_affinity(adata, k, batch_key, affinity_type, graph_mode)
    if sp.issparse(A):
        A.setdiag(0)
    else:
        np.fill_diagonal(A, 0)
    return A


# spatial gated
def build_sg_aff(pca_aff, spatial, cutoff=0.05):
    A = pca_aff.tocsr()
    A.setdiag(0)
    A.eliminate_zeros()

    r, c = A.nonzero()
    d = np.linalg.norm(spatial[r] - spatial[c], axis=1)

    order = np.argsort(r)
    r = r[order]
    c = c[order]
    d = d[order]
    data = A.data[order]

    split = np.flatnonzero(np.diff(r)) + 1
    groups = np.split(d, split)
    rows = np.unique(r)

    sigma = np.zeros(A.shape[0])
    min_d = np.zeros(A.shape[0])
    sigma[rows] = np.fromiter((np.median(g) for g in groups), float)
    sigma[sigma == 0] = np.median(d)
    # min_d[rows] = np.fromiter((np.quantile(g, 0.9) for g in groups), float)
    min_d[rows] = np.fromiter((np.quantile(g, cutoff) for g in groups), float)

    w = np.exp(-(d**2) / (sigma[r] * sigma[c] + 1e-12))
    data = data * w

    keep = d <= min_d[r]

    A = sp.csr_matrix((data[keep], (r[keep], c[keep])), shape=A.shape)
    A = A.maximum(A.T)
    A.eliminate_zeros()
    return A


def spatial_context_aff(ad, pca_aff, beta=0.5):
    import SEACells
    from sklearn.preprocessing import normalize

    sp_model = SEACells.build_graph.SEACellGraph(ad, "spatial", verbose=True)
    sp_aff = sp_model.rbf(50)

    # sp_aff = sp_aff.multiply(pca_aff > 0)
    pca_aff.setdiag(0)
    sp_aff.setdiag(0)
    sp_aff.eliminate_zeros()

    pca_aff = normalize(pca_aff, norm="l1", axis=1)
    sp_aff = normalize(sp_aff, norm="l1", axis=1)

    A = (1 - beta) * pca_aff + beta * sp_aff
    A = A.maximum(A.T)
    A.eliminate_zeros()
    return A


def generate_affinity(ad, k, bk, affinity_type="inverse_dist", graph_mode=None):
    print(affinity_type)

    if "spatial" in ad.obsm:
        x = ad.obsm["spatial"].copy()
        if x.shape[1] == 3:
            x[:, 2] *= 30.0
        ad.obsm["spatial"] = x

    if affinity_type == "inverse_dist":
        ind, dist = faiss_knn(ad, k)
        inv_dist = 1.0 / (dist + 1e-8)  # avoid div by zero
        return inv_dist

    elif affinity_type in [
        "arbf",
        "coaff",
        "ncoaff",
        "icoaff",
        "iarbf",
        "sg",
        "sarbf",
        "scoaff",
    ]:
        import SEACells

        print("calculating seacell affinity")
        kernel_model = SEACells.build_graph.SEACellGraph(ad, "X_pca", verbose=True)
        if affinity_type.startswith("i"):
            graph_construction = "intersect"
        else:
            graph_construction = "union"

        if affinity_type == "sg":
            M = kernel_model.rbf(50, graph_construction=graph_construction)
            if k == 50:
                k = 0.05
            A = build_sg_aff(M, ad.obsm["spatial"], k)
            return A
        else:
            M = kernel_model.rbf(k, graph_construction=graph_construction)

        if affinity_type == "sarbf":
            return spatial_context_aff(ad, M)

        if affinity_type == "scoaff":
            return spatial_context_aff(ad, M @ M.T)

        if affinity_type == "ncoaff":
            # --- L2 normalize rows to remove degree bias ---
            row_norms = np.sqrt(M.multiply(M).sum(axis=1)).A1  # vector of ||M_i||
            row_norms[row_norms == 0] = 1e-12  # avoid division by 0
            M_norm = M.multiply(1.0 / row_norms[:, None])  # each row -> unit L2 norm

            # --- compute normalized co-affinity (cosine between rows of M) ---
            C = M_norm @ M_norm.T  # still sparse
            return C
        elif affinity_type.endswith("coaff"):
            return M @ M.T
        else:  # arbf
            return M

    elif affinity_type == "umap":
        sc.pp.neighbors(ad, n_neighbors=k, use_rep="X_pca")
        return ad.obsp["connectivities"]

    elif affinity_type == "ctx_umap":
        ad.obsm["X_ctx"] = spatial_context_pca(ad, k)
        sc.pp.neighbors(ad, n_neighbors=k, use_rep="X_ctx")
        return ad.obsp["connectivities"]

    elif affinity_type == "ctx":
        print('using new aff')
        import SEACells
        if 'X_ctx' not in ad.obsm:
            sc.tl.pca(ad)
            ad.obsm["X_ctx"] = build_context(ad, 7.5)
        else:
            print('using existing X_ctx in adata')
        kernel_model = SEACells.build_graph.SEACellGraph(ad, "X_ctx", verbose=True)
        return kernel_model.rbf(k, graph_construction="union")

    elif affinity_type == "cpca":
        X_ctx = spatial_context_pca(ad, k)
        ad.obsm["X_ctx"] = X_ctx
        return build_seacell_kernel(X_ctx, ad.obsm["X_pca"], k=k, graph_mode=graph_mode or "knn")

    elif affinity_type in ["spatial", "scoaff"]:

        # s_aff = multi_batch_aff(ad, bk, lambda x: spatial_affinity(x, k))
        s_aff = spatial_affinity(ad, k, graph_mode=graph_mode)
        if affinity_type == "spatial":
            return s_aff
        else:
            return s_aff @ s_aff.T

    elif affinity_type in ["st", "stcoaff"]:
        # st_aff = multi_batch_aff(ad, bk, lambda x: st_affinity(x, k))
        st_aff = st_affinity(ad, k, graph_mode=graph_mode)

        if affinity_type == "st":
            return st_aff
        else:
            return st_aff @ st_aff.T


def spatial_context_pca(ad, k):
    """Average PCA of k nearest spatial neighbors per cell."""
    import faiss

    spatial = ad.obsm["spatial"].astype(np.float32)
    if spatial.shape[1] == 3:
        spatial[:, 2] *= 30.0
    index = faiss.IndexFlatL2(spatial.shape[1])
    index.add(spatial)
    _, I = index.search(spatial, k + 1)
    I = I[:, 1:]  # exclude self
    pca = ad.obsm["X_pca"]
    return pca[I].mean(axis=1)  # (N, d)


def build_context(ad, radius):
    Xsp = ad.obsm["spatial"]
    Xpca = ad.obsm["X_pca"]
    nn_sp = NearestNeighbors(radius=radius).fit(Xsp)
    neigh = nn_sp.radius_neighbors(Xsp, return_distance=False)
    neigh = [idx[idx != i] for i, idx in enumerate(neigh)]
    ctx = np.stack([Xpca[idx].mean(0) if len(idx) else Xpca[i] for i, idx in enumerate(neigh)])
    return ctx

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


def spatial_affinity(ad, k, graph_mode):
    return build_seacell_kernel(
        ad.obsm["spatial"], ad.obsm["spatial"], k=k, graph_mode=graph_mode
    )


def st_affinity(ad, k, graph_mode):
    return build_seacell_kernel(
        ad.obsm["spatial"], ad.obsm["X_pca"], k=k, graph_mode=graph_mode
    )


from sklearn.neighbors import radius_neighbors_graph


def symmetrize_graph(G, mode="union"):
    if mode == "union":
        return (G + G.T > 0).astype(float)
    elif mode in ["intersect", "intersection"]:
        G = (G > 0).astype(float)
        return G.multiply(G.T)
    else:
        raise ValueError


def ensure_radius_has_neighbors(x, radius, step=10, max_radius=500):
    while radius <= max_radius:
        G = radius_neighbors_graph(
            x, radius=radius, mode="connectivity", include_self=False
        ).tocsr()

        G = symmetrize_graph(G)

        if (np.diff(G.indptr) > 0).all():
            return G

        radius += step

    raise RuntimeError("Radius exceeded max_radius without finding neighbors")


def build_graph(x, radius=None, k=None, mode="knn"):
    print("using new code")
    n = x.shape[0]

    if mode == "knn":
        nn = NearestNeighbors(n_neighbors=k).fit(x)
        _, idxs = nn.kneighbors(x)

        rows = np.repeat(np.arange(n), k)
        cols = idxs.reshape(-1)
        G = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        G = symmetrize_graph(G)
    elif mode == "radius":
        G = ensure_radius_has_neighbors(x, radius)

    else:
        raise ValueError
    # always union

    idxs = np.zeros((n, k), dtype=int)

    for i in range(n):
        neigh = G.indices[G.indptr[i] : G.indptr[i + 1]]
        if len(neigh) == 0:
            idxs[i] = i
        elif len(neigh) >= k:
            idxs[i] = neigh[:k]
        else:
            idxs[i] = np.pad(neigh, (0, k - len(neigh)), constant_values=neigh[0])

    return idxs


def build_seacell_kernel(x_graph, X_aff, k=50, radius=50.0, graph_mode="knn"):

    idxs = build_graph(x_graph, radius, k, graph_mode)
    diff = X_aff[:, None, :] - X_aff[idxs]  # (n, k, d)
    dists = np.linalg.norm(diff, axis=2)  # (n, k)

    # from scipy.spatial.distance import cdist

    # dists = cdist(X_aff, X_aff[idxs].reshape(-1, X_aff.shape[1])).reshape(n, k)

    sigma = np.median(dists, axis=1, keepdims=True)  # (n, 1)
    sigma = np.maximum(sigma, 1e-8)
    sigma_i = sigma  # (n,1)
    sigma_j = sigma[idxs]  # (n,k,1)
    sigma_prod = sigma_i * sigma_j.squeeze(-1)  # (n,k)

    A_vals = np.exp(-(dists**2) / sigma_prod)  # (n,k)

    rows = np.repeat(np.arange(len(x_graph)), k)
    cols = idxs.reshape(-1)
    A = sp.csr_matrix(
        (A_vals.reshape(-1), (rows, cols)), shape=(len(x_graph), len(x_graph))
    )

    # if graph_construction == "union":
    #     A = A + A.T - A.multiply(A.T)
    # elif graph_construction in ["intersection", "intersect"]:
    #     A = A.multiply(A.T)
    # else:
    #     raise ValueError("graph_construction must be 'union' or 'intersection'")

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
