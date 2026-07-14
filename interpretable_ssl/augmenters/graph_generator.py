import numpy as np
import scanpy as sc
from interpretable_ssl.augmenters.diffusion_knn import *
import numpy as np, scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


def _knn_sym_sigma(ad, build_on, k, graph_construction='union'):
    """Shared core: kNN → binarize → sigma → symmetrize. Returns (sym, sigma, n).

    sigma_i = distance to the (k//2)-th nearest neighbour (adaptive bandwidth).
    sym is the symmetrized binary adjacency matrix.
    """
    sc.pp.neighbors(ad, use_rep=build_on, n_neighbors=k, knn=True)
    knn_dist = ad.obsp['distances'].tocsr()
    n = knn_dist.shape[0]

    knn_bin = knn_dist.copy()
    knn_bin.data[:] = 1.0
    knn_bin.setdiag(1)

    asc_idx = max(k // 2 - 1, 0)
    row_counts = np.diff(knn_dist.indptr)
    if (row_counts == k).all():
        sigma = np.partition(knn_dist.data.reshape(n, k), asc_idx, axis=1)[:, asc_idx]
    else:
        sigma = np.empty(n)
        for i in range(n):
            s, e = knn_dist.indptr[i], knn_dist.indptr[i + 1]
            d = knn_dist.data[s:e]
            idx = min(asc_idx, max(len(d) - 1, 0))
            sigma[i] = np.partition(d, idx)[idx] if len(d) else 1e-8
    sigma = np.maximum(sigma, 1e-8)

    if graph_construction == 'union':
        sym = (knn_bin + knn_bin.T > 0).astype(float).tocsr()
    else:
        kg = (knn_bin > 0).astype(float)
        sym = kg.multiply(kg.T).tocsr()

    return sym, sigma, n


def _rbf_on_graph(X, sym, sigma, chunk=2_000_000):
    """Compute adaptive-bandwidth RBF values for all edges in sym. Returns CSR matrix."""
    n = X.shape[0]
    row_idx, col_idx = sym.nonzero()
    n_edges = len(row_idx)
    rbf_vals = np.empty(n_edges, dtype=np.float64)
    for s in range(0, n_edges, chunk):
        e = min(s + chunk, n_edges)
        r, c = row_idx[s:e], col_idx[s:e]
        diff = X[r] - X[c]
        sq = np.einsum('ij,ij->i', diff, diff)
        rbf_vals[s:e] = np.exp(-sq / (sigma[r] * sigma[c]))
    return sp.csr_matrix((rbf_vals, (row_idx, col_idx)), shape=(n, n))


def _sigma_from_graph(X, union_csr, k):
    """Compute adaptive bandwidth sigma_i from distances in X over union-graph neighbors.

    sigma_i = k//2-th nearest distance in X among the union-graph neighbors of cell i.
    Using the union as the reference set makes sigma consistent across both spaces:
    boundary cells (close in one space, far in the other) get sigma calibrated to
    the full candidate set rather than to each space's own tight kNN.
    """
    n = X.shape[0]
    asc_idx = max(k // 2 - 1, 0)
    sigma = np.empty(n)
    for i in range(n):
        s, e = int(union_csr.indptr[i]), int(union_csr.indptr[i + 1])
        jj = union_csr.indices[s:e]
        jj = jj[jj != i]               # exclude self-loop
        if len(jj) == 0:
            sigma[i] = 1e-8
            continue
        diff = X[i] - X[jj]
        d = np.sqrt(np.einsum('ij,ij->i', diff, diff))
        idx = min(asc_idx, len(d) - 1)
        sigma[i] = np.partition(d, idx)[idx]
    return np.maximum(sigma, 1e-8)


def _topk_per_row(M_csr, k):
    """Prune a CSR matrix to the top-k entries per row by value, then re-symmetrize.

    Rows with ≤ k nonzeros are unchanged.  After per-row pruning the matrix is
    re-symmetrized as (M + M.T) / 2 so:
      - edges kept by both directions retain their full weight
      - edges kept by only one direction get half weight (soft inclusion)
    """
    M = M_csr.tocsr().copy()
    counts = np.diff(M.indptr)
    for i in np.where(counts > k)[0]:
        s, e = int(M.indptr[i]), int(M.indptr[i + 1])
        row_data = M.data[s:e]
        n_drop = len(row_data) - k
        drop_pos = np.argpartition(row_data, n_drop)[:n_drop]
        M.data[s + drop_pos] = 0.0
    M.eliminate_zeros()
    M = (M + M.T).multiply(0.5)
    M.eliminate_zeros()
    return M.tocsr()


def rbf_product(ad, build_on_list, k=50, per_space_sigma=False):
    """Product of N adaptive RBF kernels on the union of their kNN graphs.

    Edges: union of all per-space kNN graphs.
    Sigma (two modes, controlled by per_space_sigma):
      False (default): sigma recomputed from union-graph neighbors in each space —
        consistent for boundary cells but can inflate bandwidth for cross-space edges.
      True: sigma computed from each space's own kNN (before union) — tighter
        per-space bandwidth, so edges that enter via another space's kNN get a
        low RBF value in this space and are pruned out. Enforces AND logic more strictly.
    Weight: w_ij = rbf_1(i,j) × rbf_2(i,j) × ... × rbf_N(i,j)  — soft AND logic.

    After the product, top-k pruning per row removes near-zero edges and
    restores original kNN density (~k nnz/row vs ~N*k in the raw union graph).

    Args:
        build_on_list:    list of obsm keys, one per space (e.g. ['X_pca', 'X_covet'])
        per_space_sigma:  if True, use per-space sigma (from own kNN) instead of
                          recomputing sigma from the union graph
    """
    import time
    t0 = t_step = time.time()
    def _t():
        nonlocal t_step
        elapsed = time.time() - t_step
        t_step = time.time()
        return f"{elapsed:.1f}s"

    spaces = list(build_on_list)
    print(f"[rbf_product] n={ad.n_obs}  k={k}  spaces={spaces}")

    # Step 1: kNN sym per space (also keep sigma if per_space_sigma=True)
    syms = []
    sigmas = []
    n = None
    for i, key in enumerate(spaces):
        print(f"[rbf_product] step 1.{i+1}: kNN → sym  {key}  (sc.pp.neighbors) ...")
        sym, sigma, n = _knn_sym_sigma(ad, key, k)
        syms.append(sym)
        sigmas.append(sigma)
        print(f"[rbf_product] step 1.{i+1} done ({_t()})  nnz={sym.nnz}  nnz/row={sym.nnz/n:.1f}")

    # Step 2: union
    union = syms[0].copy()
    for sym in syms[1:]:
        union = union + sym
    union = (union > 0).astype(float).tocsr()
    print(f"[rbf_product] step 2: union done ({_t()})  nnz={union.nnz}  nnz/row={union.nnz/n:.1f}")

    sigma_mode_str = "per-space (own kNN)" if per_space_sigma else "union-graph"
    print(f"[rbf_product] step 3: RBF product over {len(spaces)} spaces  sigma={sigma_mode_str} ...")
    M = None
    for i, key in enumerate(spaces):
        X = ad.obsm[key].astype(np.float64)
        sigma = sigmas[i] if per_space_sigma else _sigma_from_graph(X, union, k)
        print(f"[rbf_product]   {key}: sigma [{sigma.min():.3e}, {sigma.mean():.3e}, {sigma.max():.3e}]")
        M_i = _rbf_on_graph(X, union, sigma)
        M = M_i if M is None else M.multiply(M_i)
    M.eliminate_zeros()
    print(f"[rbf_product] step 3 done ({_t()})  nnz={M.nnz}")

    # Step 4: top-k pruning + re-symmetrize
    print(f"[rbf_product] step 4: top-{k} pruning per row + re-symmetrize ...")
    nnz_before = M.nnz
    M = _topk_per_row(M, k)
    M.setdiag(0)
    M.eliminate_zeros()
    vals = M.data
    print(f"[rbf_product] step 4 done ({_t()})  nnz={M.nnz} (was {nnz_before}, "
          f"{100*(1 - M.nnz/nnz_before):.0f}% removed)  nnz/row={M.nnz/n:.1f}  "
          f"rbf [{vals.min():.3e}, {vals.mean():.3e}, {vals.max():.3e}]")
    print(f"[rbf_product] total {time.time()-t0:.1f}s")
    return M


def rbf_optimized(ad, build_on='X_pca', k=50, graph_construction='union'):
    """Exact SEACells rbf() logic, fully vectorized.

    Matches SEACells step-by-step:
      1. sc.pp.neighbors kNN (pynndescent — identical to SEACells)
      2. binarize distances to 0/1 + setdiag(1)
      3. sigma_i = kth_neighbor_distance(dist, k//2, i) — vectorized via np.partition
      4. symmetrize binary graph (union or intersect — identical to SEACells)
      5. RBF exp(-||xi-xj||²/(σi·σj)) for edges only — O(nnz·d) not O(n²·d)
      6. build CSR directly — no slow LIL assembly
    """
    X_check = ad.obsm[build_on]
    print(f"[rbf_optimized] n={ad.n_obs}  k={k}  build_on={build_on}  graph_construction={graph_construction}")
    print(f"[rbf_optimized] feature stats: shape={X_check.shape}  min={X_check.min():.4e}  "
          f"mean={np.abs(X_check).mean():.4e}  max={X_check.max():.4e}  std={X_check.std():.4e}")

    print("[rbf_optimized] step 1-4: kNN → sigma → sym ...")
    sym, sigma, n = _knn_sym_sigma(ad, build_on, k, graph_construction)
    print(f"[rbf_optimized] sym nnz={sym.nnz}  nnz/row={sym.nnz/n:.1f}  "
          f"sigma min={sigma.min():.4e}  mean={sigma.mean():.4e}  max={sigma.max():.4e}")

    print("[rbf_optimized] step 5: computing RBF ...")
    X = ad.obsm[build_on].astype(np.float64)
    M = _rbf_on_graph(X, sym, sigma)
    print(f"[rbf_optimized] done  M.nnz={M.nnz}  rbf min={M.data.min():.4e}  "
          f"mean={M.data.mean():.4e}  max={M.data.max():.4e}")
    return M


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


def compute_banksy_embedding(ad, lambda_param=0.2, num_neighbours=15,
                             n_components=50, obsm_key='X_banksy_orig'):
    """Compute BANKSY embedding using the original pybanksy package.

    Requires: pip install pybanksy

    Args:
        ad:             AnnData with .X (normalized gene expression) and obsm['spatial']
        lambda_param:   balance own (0) vs neighbour expression (1).
                        0.2 = cell-typing default from the paper.
        num_neighbours: spatial kNN for neighbourhood computation
        n_components:   PCA dims to keep from the augmented BANKSY matrix
        obsm_key:       where to store the resulting embedding
    """
    try:
        from banksy.initialize_banksy import initialize_banksy
        from banksy.embed_banksy import generate_banksy_matrix
        from sklearn.decomposition import PCA as SklearnPCA
    except ImportError:
        raise ImportError(
            "pybanksy not found. Run in your notebook:\n"
            "    !pip install pybanksy"
        )

    if 'spatial' not in ad.obsm:
        raise ValueError("compute_banksy_embedding requires ad.obsm['spatial']")

    spatial_2d = ad.obsm['spatial'][:, :2].astype(np.float32)
    ad.obsm['_bky_spatial_2d'] = spatial_2d
    ad.obs['_bky_x'] = spatial_2d[:, 0].astype(float)
    ad.obs['_bky_y'] = spatial_2d[:, 1].astype(float)

    print(f"[banksy_orig] n={ad.n_obs}  genes={ad.n_vars}  "
          f"lambda={lambda_param}  k={num_neighbours}  n_components={n_components}")
    print("[banksy_orig] step 1: initializing BANKSY neighbourhood graph ...")
    banksy_dict = initialize_banksy(
        ad,
        coord_keys=('_bky_x', '_bky_y', '_bky_spatial_2d'),
        num_neighbours=num_neighbours,
        nbr_weight_decay='scaled_gaussian',
        plt_edge_hist=False,
        plt_nbr_weights=False,
        plt_agf_angles=False,
        plt_theta=False,
    )

    # generate_banksy_matrix builds the augmented matrix and stores it as an
    # AnnData in banksy_dict['scaled_gaussian'][lambda_param]['adata'].
    # We then run PCA ourselves — no Leiden clustering, no nonspatial pass.
    print("[banksy_orig] step 2: building augmented matrix ...")
    banksy_dict, _ = generate_banksy_matrix(
        ad, banksy_dict,
        lambda_list=[lambda_param],
        max_m=1,
        verbose=False,
    )

    print("[banksy_orig] step 3: PCA ...")
    inner_ad = banksy_dict['scaled_gaussian'][lambda_param]['adata']
    X_aug = inner_ad.X
    if sp.issparse(X_aug):
        X_aug = X_aug.toarray()
    X_aug = np.asarray(X_aug, dtype=np.float32)

    pca = SklearnPCA(n_components=n_components, random_state=0)
    X_emb = pca.fit_transform(X_aug).astype(np.float32)

    ad.obsm[obsm_key] = X_emb
    print(f"[banksy_orig] done  shape={X_emb.shape}  stored in obsm['{obsm_key}']")

    # Clean up temporary keys
    ad.obs.drop(columns=['_bky_x', '_bky_y'], inplace=True, errors='ignore')
    if '_bky_spatial_2d' in ad.obsm:
        del ad.obsm['_bky_spatial_2d']


def generate_affinity(ad, k, bk, affinity_type="inverse_dist", graph_mode=None,
                      spatial_radius=12.57, per_space_sigma=False, weighted_context=False):
    print(affinity_type)

    if affinity_type == "inverse_dist":
        ind, dist = faiss_knn(ad, k)
        inv_dist = 1.0 / (dist + 1e-8)  # avoid div by zero
        return inv_dist

    elif affinity_type == 'arbf_val':
        import SEACells
        print("arbf_val: recalculating PCA (n_comps=50) then SEACells rbf ...")
        sc.tl.pca(ad, n_comps=50)
        kernel_model = SEACells.build_graph.SEACellGraph(ad, "X_pca", verbose=True)
        return kernel_model.rbf(k, graph_construction="union")

    elif affinity_type == 'arbf_opt':
        print("arbf_opt: recalculating PCA (n_comps=50) then optimized rbf ...")
        sc.tl.pca(ad, n_comps=50)
        return rbf_optimized(ad, build_on='X_pca', k=k, graph_construction='union')

    elif affinity_type.startswith('banksy_orig'):
        # Original BANKSY embedding (gene expression level) + adaptive RBF.
        # Requires: pip install pybanksy
        # Suffix controls lambda: 'banksy_orig' -> 0.2 (cell-typing default),
        # 'banksy_orig0.5' -> lambda=0.5, 'banksy_orig0.8' -> domain segmentation.
        suffix = affinity_type[len('banksy_orig'):]
        lambda_param = float(suffix) if suffix else 0.2
        print(f"banksy_orig: lambda={lambda_param}  k={k}  "
              f"(requires pybanksy — run !pip install pybanksy if not installed)")
        if 'X_banksy_orig' not in ad.obsm:
            compute_banksy_embedding(ad, lambda_param=lambda_param,
                                     num_neighbours=15, n_components=50,
                                     obsm_key='X_banksy_orig')
        return rbf_optimized(ad, build_on='X_banksy_orig', k=k, graph_construction='union')

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
            A = build_sg_aff(M, ad.obsm["spatial"][:, :2], k)
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

    elif affinity_type == 'arbf_custom':
        # PCA affinity using our build_seacell_kernel — for comparing against SEACells arbf.
        # Uses symmetrize=True (union, same as SEACells) to verify kernels are equivalent.
        n_pca_dims = ad.obsm['X_pca'].shape[1]
        print(f"[generate_affinity] arbf_custom: pca_dims={n_pca_dims}  "
              f"aff_k={k}  kernel=custom_arbf  symmetrize=True")
        X_pca = ad.obsm['X_pca']
        return build_seacell_kernel(X_pca, X_pca, k=k, graph_mode=graph_mode or 'knn',
                                    symmetrize=True)

    elif affinity_type == 'ctx_mean':
        # Pure first-order spatial context: distance-weighted mean of spatial neighbours' PCA.
        # Uses k=100 spatial neighbours for mean (matching COVET's covariance k),
        # then adaptive RBF kernel with union symmetrization.
        ctx_k, aff_k = 100, 50
        n_pca_dims = ad.obsm['X_pca'].shape[1]
        print(f"[generate_affinity] ctx_mean: spatial_k={ctx_k}  pca_dims={n_pca_dims}  "
              f"aff_k={aff_k}  kernel=rbf_optimized  graph_construction=union")
        X_ctx = spatial_context_pca_weighted(ad, k=ctx_k)
        ad.obsm['X_ctx_mean'] = X_ctx
        return rbf_optimized(ad, build_on='X_ctx_mean', k=aff_k, graph_construction='union')

    elif affinity_type.startswith("banksy_radius"):
        # BANKSY with calibrated 50 µm radius neighbourhood instead of fixed-k kNN.
        # Suffix controls own/context balance: 'banksy_radius' -> alpha=0.5.
        suffix = affinity_type[len('banksy_radius'):]
        alpha = float(suffix) if suffix else 0.5
        print(f"[generate_affinity] banksy_radius: alpha={alpha:.2f}  "
              f"radius={spatial_radius:.2f}  aff_k={k}")
        X_pca = ad.obsm['X_pca']
        X_ctx = spatial_context_pca_weighted_radius(ad, spatial_radius)
        V_pca = X_pca.var(axis=0).sum()
        V_ctx = X_ctx.var(axis=0).sum()
        lam = np.sqrt(alpha * V_pca / ((1 - alpha) * V_ctx + alpha * V_pca))
        print(f"banksy_radius: V_pca={V_pca:.3f}  V_ctx={V_ctx:.3f}  lambda={lam:.3f}")
        ad.obsm['X_banksy_radius'] = np.concatenate(
            [np.sqrt(1 - lam**2) * X_pca, lam * X_ctx], axis=1
        )
        return rbf_optimized(ad, build_on='X_banksy_radius', k=k, graph_construction='union')

    elif affinity_type.startswith("banksy"):
        # BANKSY-style: concat own PCA with distance-weighted spatial-neighbour mean PCA, then arbf.
        # alpha (suffix) = desired fraction of distance variance from X_ctx (default 0.5 = equal).
        # lambda is derived from alpha and the empirical variances so the balance is data-adaptive:
        #   lambda = sqrt(alpha * V_pca / ((1-alpha) * V_ctx + alpha * V_pca))
        # This compensates for the variance shrinkage caused by neighbourhood averaging without
        # artificially rescaling either part.
        # spatial_k=35 is fixed to match the paper's ~50µm neighbourhood (same as COVET).
        suffix = affinity_type[len("banksy"):]
        alpha = float(suffix) if suffix else 0.5
        X_pca = ad.obsm["X_pca"]
        X_ctx = spatial_context_pca_weighted(ad, 35, weighted=weighted_context)
        V_pca = X_pca.var(axis=0).sum()
        V_ctx = X_ctx.var(axis=0).sum()
        lam = np.sqrt(alpha * V_pca / ((1 - alpha) * V_ctx + alpha * V_pca))
        print(f"banksy: alpha={alpha:.2f}  V_pca={V_pca:.3f}  V_ctx={V_ctx:.3f}  lambda={lam:.3f}  "
              f"weighted_context={weighted_context}")
        X_banksy = np.concatenate(
            [np.sqrt(1 - lam**2) * X_pca, lam * X_ctx], axis=1
        )
        ad.obsm["X_banksy"] = X_banksy
        return build_seacell_kernel(X_banksy, X_banksy, k=k, graph_mode=graph_mode or "knn")

    elif affinity_type == 'covet_product':
        if 'spatial' not in ad.obsm:
            raise ValueError("covet_product requires ad.obsm['spatial']")
        print(f"[generate_affinity] covet_product: feature_k=35  n_pcs=10  aff_k={k}  "
              f"weight=rbf_pca×rbf_covet  per_space_sigma={per_space_sigma}")
        compute_covet_features(ad, k=35, n_pcs=10, alpha=1.0, n_comps=None, obsm_key='X_covet')
        return rbf_product(ad, ['X_pca', 'X_covet'], k=k, per_space_sigma=per_space_sigma)

    elif affinity_type == 'mean_pca_only':
        if 'spatial' not in ad.obsm:
            raise ValueError("mean_pca_only requires ad.obsm['spatial']")
        print(f"[generate_affinity] mean_pca_only: feature_k=35  aff_k={k}  "
              f"weight=rbf_mean_pca (no own-cell PCA)  weighted_context={weighted_context}")
        compute_mean_pca_context(ad, k=35, obsm_key='X_mean_pca', weighted=weighted_context)
        return rbf_optimized(ad, build_on='X_mean_pca', k=k, graph_construction='union')

    elif affinity_type == 'mean_product':
        if 'spatial' not in ad.obsm:
            raise ValueError("mean_product requires ad.obsm['spatial']")
        print(f"[generate_affinity] mean_product: feature_k=35  aff_k={k}  "
              f"weight=rbf_pca×rbf_mean_pca  per_space_sigma={per_space_sigma}  "
              f"weighted_context={weighted_context}")
        compute_mean_pca_context(ad, k=35, obsm_key='X_mean_pca', weighted=weighted_context)
        return rbf_product(ad, ['X_pca', 'X_mean_pca'], k=k, per_space_sigma=per_space_sigma)

    elif affinity_type == 'mean_covet_product':
        if 'spatial' not in ad.obsm:
            raise ValueError("mean_covet_product requires ad.obsm['spatial']")
        print(f"[generate_affinity] mean_covet_product: feature_k=35  n_pcs=10  aff_k={k}  "
              f"weight=rbf_pca×rbf_mean_pca×rbf_covet  per_space_sigma={per_space_sigma}  "
              f"weighted_context={weighted_context}")
        compute_mean_pca_context(ad, k=35, obsm_key='X_mean_pca', weighted=weighted_context)
        compute_covet_features(ad, k=35, n_pcs=10, alpha=1.0, n_comps=None, obsm_key='X_covet')
        return rbf_product(ad, ['X_pca', 'X_mean_pca', 'X_covet'], k=k, per_space_sigma=per_space_sigma)

    elif affinity_type == 'covet_radius':
        if 'spatial' not in ad.obsm:
            raise ValueError("covet_radius requires ad.obsm['spatial']")
        print(f"[generate_affinity] covet_radius: radius={spatial_radius:.2f}  n_pcs=10  aff_k={k}")
        compute_covet_features_radius(ad, radius_units=spatial_radius, n_pcs=10,
                                      alpha=1.0, obsm_key='X_covet_radius')
        return rbf_optimized(ad, build_on='X_covet_radius', k=k, graph_construction='union')

    elif affinity_type == 'covet' or affinity_type.startswith('covet_a'):
        if affinity_type == 'covet':
            alpha = 1.0
        else:
            alpha = int(affinity_type.split('_a')[1]) / 10.0
        if 'spatial' not in ad.obsm:
            raise ValueError("COVET affinity requires ad.obsm['spatial']")
        print(f"[generate_affinity] covet: feature_k=100  n_pcs=25  alpha={alpha}  "
              f"aff_k=50  kernel=rbf_optimized  graph_construction=union")
        compute_covet_features(ad, k=100, n_pcs=25, alpha=alpha, n_comps=None, obsm_key='X_covet')
        return rbf_optimized(ad, build_on='X_covet', k=50, graph_construction='union')

    elif affinity_type == 'covet_pca_direct':
        # Fair apples-to-apples comparison with BANKSY:
        #   BANKSY:          concat(own_PCA, mean_neighbour_PCA)  → RBF
        #   covet_pca_direct: concat(own_PCA, covet_PCA)          → RBF   (no extra reduction)
        # Uses identical lambda variance-balancing and the same 100-dim → RBF pipeline as BANKSY.
        # The only difference is the neighbourhood summary: mean (BANKSY) vs covariance PCA (COVET).
        if 'spatial' not in ad.obsm:
            raise ValueError("covet_pca_direct requires ad.obsm['spatial']")
        compute_covet_features(ad, k=35, n_pcs=10, alpha=1.0, n_comps=None, obsm_key='X_covet')
        X_pca   = ad.obsm['X_pca']
        X_covet = ad.obsm['X_covet'].astype(np.float32)
        V_pca   = float(X_pca.var(axis=0).sum())
        V_covet = float(X_covet.var(axis=0).sum())
        lam = np.sqrt(0.5 * V_pca / (0.5 * V_covet + 0.5 * V_pca))
        X_concat = np.concatenate([np.sqrt(1 - lam**2) * X_pca, lam * X_covet], axis=1)
        print(f"[generate_affinity] covet_pca_direct: feature_k=35  n_pcs=10  "
              f"V_pca={V_pca:.2f}  V_covet={V_covet:.2f}  lambda={lam:.3f}  "
              f"concat_dims={X_concat.shape[1]}  aff_k={k}")
        ad.obsm['X_covet_pca_direct'] = X_concat
        return rbf_optimized(ad, build_on='X_covet_pca_direct', k=k, graph_construction='union')

    elif affinity_type.startswith('covet_pca'):
        # Concat(variance-normalized X_pca, X_covet) → PCA(n_comps) → arbf union.
        # Suffix controls transcript/niche balance: 'covet_pca' -> alpha=0.5,
        # 'covet_pca0.3' -> alpha=0.3 (less transcript), 'covet_pca0.7' -> alpha=0.7.
        from sklearn.decomposition import PCA as _PCA
        suffix = affinity_type[len('covet_pca'):]
        alpha = float(suffix) if suffix else 0.5
        n_comps = 50
        if 'spatial' not in ad.obsm:
            raise ValueError("covet_pca affinity requires ad.obsm['spatial']")

        compute_covet_features(ad, k=100, n_pcs=25, alpha=1.0, n_comps=None, obsm_key='X_covet')
        X_covet = ad.obsm['X_covet'].astype(np.float32)
        X_pca   = ad.obsm['X_pca']

        # alpha directly controls variance fraction: alpha from PCA, (1-alpha) from COVET.
        # Each part is scaled to unit total variance first, then multiplied by sqrt(fraction).
        V_pca   = float(X_pca.var(axis=0).sum())
        V_covet = float(X_covet.var(axis=0).sum())
        X_pca_scaled   = X_pca   / np.sqrt(V_pca)   * np.sqrt(alpha)
        X_covet_scaled = X_covet / np.sqrt(V_covet) * np.sqrt(1 - alpha)
        X_concat = np.concatenate([X_pca_scaled, X_covet_scaled], axis=1)

        pca_model = _PCA(n_components=n_comps)
        X_reduced = pca_model.fit_transform(X_concat).astype(np.float32)
        evr = pca_model.explained_variance_ratio_.sum()

        print(f"[generate_affinity] covet_pca: feature_k=100  n_pcs=25  alpha={alpha}  "
              f"V_pca={V_pca:.2f}  V_covet={V_covet:.2f}  "
              f"concat_dims={X_concat.shape[1]}  pca_comps={n_comps}  "
              f"explained_var={evr:.3f}  aff_k=50  kernel=rbf_optimized  graph_construction=union")

        ad.obsm['X_covet_pca'] = X_reduced
        return rbf_optimized(ad, build_on='X_covet_pca', k=50, graph_construction='union')

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

    spatial = ad.obsm["spatial"][:, :2].astype(np.float32)
    index = faiss.IndexFlatL2(spatial.shape[1])
    index.add(spatial)
    _, I = index.search(spatial, k + 1)
    I = I[:, 1:]  # exclude self
    pca = ad.obsm["X_pca"]
    return pca[I].mean(axis=1)  # (N, d)


def spatial_context_pca_weighted(ad, k, weighted=False):
    """Average PCA of k nearest spatial neighbors (BANKSY-style context).

    weighted=False (default): plain unweighted mean (1/k per neighbor) — matches
        the original BANKSY paper's neighbourhood-mean feature.
    weighted=True: distance-weighted average using an adaptive Gaussian kernel,
        w_ij = exp(-d²_ij / (σ_i * σ_j)) where σ_i is the median spatial distance
        to cell i's k neighbors (same bandwidth logic as build_seacell_kernel).
    """
    if not weighted:
        return spatial_context_pca(ad, k)

    import faiss

    spatial = ad.obsm["spatial"][:, :2].astype(np.float32)
    index = faiss.IndexFlatL2(spatial.shape[1])
    index.add(spatial)
    D_sq, I = index.search(spatial, k + 1)  # squared L2 distances
    D_sq = D_sq[:, 1:].astype(np.float64)   # (N, k), exclude self
    I = I[:, 1:]                              # (N, k)

    sigma = np.median(np.sqrt(D_sq), axis=1, keepdims=True)  # (N, 1)
    sigma = np.maximum(sigma, 1e-8)
    sigma_j = sigma[I].squeeze(-1)                            # (N, k)
    sigma_prod = sigma * sigma_j                              # (N, k)

    W = np.exp(-D_sq / sigma_prod)           # (N, k) Gaussian weights
    W /= W.sum(axis=1, keepdims=True)        # row-normalise

    pca = ad.obsm["X_pca"]
    # weighted sum: (N, k, d) * (N, k, 1) -> (N, d)
    return (pca[I] * W[:, :, None]).sum(axis=1)


def build_context(ad, radius):
    Xsp = ad.obsm["spatial"][:, :2]
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
    sp = ad.obsm["spatial"][:, :2]
    return build_seacell_kernel(sp, sp, k=k, graph_mode=graph_mode)


def st_affinity(ad, k, graph_mode):
    sp = ad.obsm["spatial"][:, :2]
    return build_seacell_kernel(sp, ad.obsm["X_pca"], k=k, graph_mode=graph_mode)


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


def build_seacell_kernel(x_graph, X_aff, k=50, radius=50.0, graph_mode="knn",
                         symmetrize=False):
    """Adaptive-bandwidth RBF kernel matching SEACells logic."""
    import time
    n = len(x_graph)
    print(f"  [build_seacell_kernel] n={n}  k={k}  "
          f"x_graph.shape={x_graph.shape}  X_aff.shape={X_aff.shape}  "
          f"symmetrize={symmetrize}")

    t0 = time.time()
    # k+1 because sklearn includes self as the first neighbor (dist=0) when querying on training data
    nn = NearestNeighbors(n_neighbors=k + 1).fit(x_graph)
    knn_dists_all, idxs_all = nn.kneighbors(x_graph)
    knn_dists = knn_dists_all[:, 1:]   # drop self (col 0, dist=0)
    idxs      = idxs_all[:, 1:]        # drop self
    print(f"  [build_seacell_kernel] kNN done in {time.time()-t0:.1f}s  "
          f"idxs.shape={idxs.shape}  (self excluded)")

    median_idx = max(k // 2 - 1, 0)
    sigma = np.maximum(knn_dists[:, median_idx], 1e-8)

    t1 = time.time()
    diff  = X_aff[:, None, :] - X_aff[idxs]
    dists = np.linalg.norm(diff, axis=2)
    sigma_prod = sigma[:, None] * sigma[idxs]
    A_vals = np.exp(-(dists ** 2) / sigma_prod)
    print(f"  [build_seacell_kernel] RBF done in {time.time()-t1:.1f}s  "
          f"A_vals min={A_vals.min():.4f}  max={A_vals.max():.4f}  "
          f"mean={A_vals.mean():.4f}")

    rows = np.repeat(np.arange(n), k)
    cols = idxs.reshape(-1)
    A = sp.csr_matrix((A_vals.reshape(-1), (rows, cols)), shape=(n, n))
    print(f"  [build_seacell_kernel] sparse matrix built  nnz={A.nnz}  "
          f"nnz_per_row={A.nnz/n:.1f}")

    if symmetrize:
        A = A + A.T
        print(f"  [build_seacell_kernel] symmetrized  nnz={A.nnz}")

    return A


def compute_banksy_features(ad, k, alpha=0.5, weighted=False):
    """Compute BANKSY joint embedding on the full dataset and store in ad.obsm["X_banksy"].

    Must be called on the full AnnData before sampling so every cell's spatial
    context vector reflects its true neighbourhood in the tissue.

    Args:
        ad:       AnnData with X_pca and obsm["spatial"]
        k:        spatial neighbours used for the context average
        alpha:    target fraction of distance variance from spatial context [0, 1]
        weighted: False (default) = plain average, True = distance-weighted Gaussian
                  average. See spatial_context_pca_weighted.
    """
    X_pca = ad.obsm["X_pca"]
    X_ctx = spatial_context_pca_weighted(ad, k, weighted=weighted)
    V_pca = X_pca.var(axis=0).sum()
    V_ctx = X_ctx.var(axis=0).sum()
    lam = np.sqrt(alpha * V_pca / ((1 - alpha) * V_ctx + alpha * V_pca))
    print(f"banksy features: alpha={alpha:.2f}  V_pca={V_pca:.3f}  V_ctx={V_ctx:.3f}  lambda={lam:.3f}")
    ad.obsm["X_banksy"] = np.concatenate(
        [np.sqrt(1 - lam**2) * X_pca, lam * X_ctx], axis=1
    )


def compute_mean_pca_context(ad, k=35, obsm_key='X_mean_pca', weighted=False):
    """Mean of k spatial-neighbour PCA vectors (BANKSY context only).

    Stores the context vector in ad.obsm[obsm_key].  Unlike compute_banksy_features,
    this does NOT concat with own PCA — it's the pure neighbourhood summary used as
    input to rbf_product.

    weighted: False (default) = plain average, True = distance-weighted Gaussian
        average. See spatial_context_pca_weighted.
    """
    ad.obsm[obsm_key] = spatial_context_pca_weighted(ad, k, weighted=weighted).astype(np.float32)
    V = float(ad.obsm[obsm_key].var(axis=0).sum())
    mode = "distance-weighted" if weighted else "plain avg"
    print(f"compute_mean_pca_context: k={k}  dims={ad.obsm[obsm_key].shape[1]}  V={V:.3f}  mode={mode}")



def _covet_spatial_knn(ad, k):
    """Compute spatial kNN indices for COVET. Returns (N, k) int array."""
    import faiss
    spatial = ad.obsm["spatial"][:, :2].astype(np.float32)
    index = faiss.IndexFlatL2(spatial.shape[1])
    index.add(spatial)
    _, I = index.search(spatial, k + 1)
    return I[:, 1:]  # exclude self


def _covet_cov_flat(ad, I, n_pcs):
    """Compute flattened upper-triangle covariance of neighbour PCA.

    Args:
        ad:    AnnData with obsm["X_pca"]
        I:     (N, k) neighbour index array from _covet_spatial_knn
        n_pcs: number of PCA dims to use as covariance input

    Returns:
        cov_flat: (N, n_pcs*(n_pcs+1)/2) float array
    """
    k = I.shape[1]
    X_pca_ctx = ad.obsm["X_pca"][:, :n_pcs]
    X_nbr = X_pca_ctx[I]                                # (N, k, n_pcs)
    X_nbr = X_nbr - X_nbr.mean(axis=1, keepdims=True)   # centre per cell
    cov = np.einsum('ijk,ijl->ikl', X_nbr, X_nbr) / max(k - 1, 1)
    ti, tj = np.triu_indices(n_pcs, k=0)
    return cov[:, ti, tj]                               # (N, n_pcs*(n_pcs+1)/2)


def _covet_apply(ad, cov_flat, n_comps, alpha, obsm_key):
    """PCA-reduce cov_flat and lambda-balance with own PCA. Stores result in ad.obsm[obsm_key].

    Args:
        ad:       AnnData with obsm["X_pca"]
        cov_flat: (N, D) precomputed flattened covariance
        n_comps:  PCA components to keep (None = auto)
        alpha:    variance fraction from covariance side [0,1]; 1.0 = covet-only
        obsm_key: where to store result
    """
    import scanpy as sc
    import anndata as ann

    n_pca_own = ad.obsm["X_pca"].shape[1]
    n_cov_dims = cov_flat.shape[1]
    max_comps = n_cov_dims - 1                           # arpack: n_comps < n_features
    n_comps = min(n_pca_own, max_comps) if n_comps is None else min(n_comps, max_comps)

    tmp = ann.AnnData(cov_flat)
    sc.tl.pca(tmp, n_comps=n_comps)
    X_cov = tmp.obsm["X_pca"]

    if n_comps < n_pca_own:
        X_cov = np.concatenate([X_cov, np.zeros((len(ad), n_pca_own - n_comps))], axis=1)

    if alpha == 1.0:
        ad.obsm[obsm_key] = X_cov
        return n_comps, n_cov_dims

    X_own = ad.obsm["X_pca"]
    V_own = float(X_own.var(axis=0).sum())
    V_cov = float(X_cov.var(axis=0).sum())
    lam = np.sqrt(alpha * V_own / ((1 - alpha) * V_cov + alpha * V_own))
    ad.obsm[obsm_key] = np.concatenate([np.sqrt(1 - lam**2) * X_own, lam * X_cov], axis=1)
    return n_comps, n_cov_dims


def compute_covet_features(ad, k=35, n_pcs=10, alpha=0.5, obsm_key="X_covet", n_comps=None):
    """Compute COVET-style niche vector: own X_pca + covariance of neighbour X_pca.

    For each cell, computes the covariance matrix of its k spatial neighbours'
    PCA coordinates (k x n_pcs → n_pcs x n_pcs), flattens the upper triangle
    (n_pcs*(n_pcs+1)/2 values), applies PCA to decorrelate and match X_pca
    dimensionality, then lambda-balances both parts by variance contribution.

    Statistical requirement: k >= 3 * n_pcs for stable covariance estimation.
    Paper-grounded defaults: k=35 covers ~50µm neighbourhood in 2D (median 32
    cells per 50µm in this NSCLC CosMx dataset).

    Args:
        ad:       AnnData with obsm["X_pca"] and obsm["spatial"]
        k:        spatial neighbours for covariance (default 35)
        n_pcs:    PCA dims used as input to covariance (default 10, ratio k/n_pcs=3.5)
        alpha:    target fraction of total variance from microenv covariance [0,1]
        n_comps:  PCA components to keep from the flattened covariance matrix.
                  None = auto: min(n_pca_own, n_cov_dims - 1).
                  Set explicitly (e.g. 10, 20) to keep only dominant neighbourhood axes.

    Stores:
        ad.obsm[obsm_key]: ready for sample_and_affinity
    """
    I = _covet_spatial_knn(ad, k)
    cov_flat = _covet_cov_flat(ad, I, n_pcs)
    n_comps_used, n_cov_dims = _covet_apply(ad, cov_flat, n_comps, alpha, obsm_key)

    if alpha == 1.0:
        print(f"covet: k={k}  n_pcs={n_pcs}  n_comps={n_comps_used}  "
              f"alpha=1.0 (covet-only)  cov_dims={n_cov_dims}→{n_comps_used}")
    else:
        print(f"covet: k={k}  n_pcs={n_pcs}  n_comps={n_comps_used}  alpha={alpha:.2f}  "
              f"cov_dims={n_cov_dims}→{n_comps_used}")


def _covet_radius_neighbors(ad, radius_units):
    """Spatial neighbors within radius_units for each cell. Returns list of int arrays (excl. self)."""
    from sklearn.neighbors import NearestNeighbors as _NearestNeighbors
    spatial = ad.obsm['spatial'][:, :2].astype(np.float32)
    nn = _NearestNeighbors(radius=radius_units).fit(spatial)
    dists_list, inds_list = nn.radius_neighbors(spatial, return_distance=False)
    return [inds[inds != i].astype(int) for i, inds in enumerate(inds_list)]


def _covet_cov_flat_variable(ad, neighbors, n_pcs):
    """Flattened upper-triangle covariance of neighbour PCA for variable neighbourhood sizes.

    Cells with fewer than 2 neighbours get a zero vector (not enough for covariance estimation).
    """
    X_pca = ad.obsm['X_pca'][:, :n_pcs]
    n = len(ad)
    d = n_pcs * (n_pcs + 1) // 2
    cov_flat = np.zeros((n, d), dtype=np.float32)
    ti, tj = np.triu_indices(n_pcs)
    for i, idx in enumerate(neighbors):
        if len(idx) < 2:
            continue
        X_nbr = X_pca[idx]
        X_nbr = X_nbr - X_nbr.mean(0)
        cov = (X_nbr.T @ X_nbr) / (len(idx) - 1)
        cov_flat[i] = cov[ti, tj]
    return cov_flat


def compute_covet_features_radius(ad, radius_units, n_pcs=10, alpha=1.0,
                                   n_comps=None, obsm_key='X_covet_radius'):
    """COVET with calibrated 50 µm radius neighbourhood instead of fixed-k kNN.

    Args:
        ad:           AnnData with obsm['spatial'] and obsm['X_pca']
        radius_units: 50 µm in coordinate units (calibrated from spatial_radius_diagnostic)
        n_pcs:        PCA dims used as input to covariance
        alpha:        variance fraction from covariance side (1.0 = covet-only)
        n_comps:      PCA components from flattened covariance (None = auto)
        obsm_key:     where to store the result
    """
    neighbors = _covet_radius_neighbors(ad, radius_units)
    counts = np.array([len(nb) for nb in neighbors])
    print(f'covet_radius: radius={radius_units:.2f}  n_pcs={n_pcs}  '
          f'median_neighbours={np.median(counts):.0f}  cells_with_<2={(counts < 2).sum()}')
    cov_flat = _covet_cov_flat_variable(ad, neighbors, n_pcs)
    _covet_apply(ad, cov_flat, n_comps, alpha, obsm_key)
    print(f'covet_radius: stored {obsm_key}  shape={ad.obsm[obsm_key].shape}')


def spatial_context_pca_weighted_radius(ad, radius_units):
    """Distance-weighted mean of neighbour PCA within radius (BANKSY-style, radius variant).

    Each cell's context vector is a Gaussian-weighted average of PCA over all cells
    within radius_units.  Bandwidth sigma_i = median distance to those neighbours
    (adaptive, same logic as spatial_context_pca_weighted).
    Cells with no neighbours fall back to their own PCA.
    """
    from sklearn.neighbors import NearestNeighbors as _NearestNeighbors
    spatial = ad.obsm['spatial'][:, :2].astype(np.float32)
    nn = _NearestNeighbors(radius=radius_units).fit(spatial)
    dists_list, inds_list = nn.radius_neighbors(spatial, return_distance=True)

    n = len(ad)
    sigma = np.full(n, 1e-8)
    for i, (dists, inds) in enumerate(zip(dists_list, inds_list)):
        d = dists[inds != i]
        if len(d) > 0:
            sigma[i] = max(float(np.median(d)), 1e-8)

    X_pca = ad.obsm['X_pca']
    X_ctx = np.zeros_like(X_pca)
    for i, (dists, inds) in enumerate(zip(dists_list, inds_list)):
        mask = inds != i
        d, idx = dists[mask], inds[mask]
        if len(idx) == 0:
            X_ctx[i] = X_pca[i]
            continue
        W = np.exp(-d ** 2 / (sigma[i] * sigma[idx]))
        W /= W.sum() + 1e-12
        X_ctx[i] = (X_pca[idx] * W[:, None]).sum(0)
    return X_ctx


def sample_and_affinity(ad, k=30, n=5000, stratify_by=None,
                        graph_mode=None, random_state=0, obsm_key="X_banksy"):
    """Stratified-sample cells then compute arbf affinity on a joint embedding.

    Call compute_banksy_features or compute_covet_features first to populate
    the embedding key in ad.obsm.

    Args:
        ad:           AnnData with ad.obsm[obsm_key] already computed
        k:            neighbours for arbf affinity
        n:            total cells to sample
        stratify_by:  list of obs column names to stratify on,
                      e.g. ["cell_type", "niche"].  None = uniform random.
        graph_mode:   "knn" or "radius"
        random_state: RNG seed
        obsm_key:     obsm slot to use as embedding (default "X_banksy")

    Returns:
        ad_sub:  subsampled AnnData
        aff:     sparse arbf affinity matrix (n_sub x n_sub)
    """
    assert obsm_key in ad.obsm, f"'{obsm_key}' not in ad.obsm. Run compute_banksy_features or compute_covet_features first."

    rng = np.random.default_rng(random_state)
    if stratify_by is not None:
        strata = ad.obs[stratify_by].astype(str).apply(
            lambda row: "__".join(row.values), axis=1
        )
        groups = {s: np.where(strata == s)[0] for s in strata.unique()}
        per_stratum = max(1, n // len(groups))
        idx = np.concatenate([
            rng.choice(g, size=min(per_stratum, len(g)), replace=False)
            for g in groups.values()
        ])
        idx = rng.permutation(idx)
    else:
        idx = rng.choice(ad.n_obs, size=min(n, ad.n_obs), replace=False)

    ad_sub = ad[idx].copy()
    X = ad_sub.obsm[obsm_key]
    aff = build_seacell_kernel(X, X, k=k, graph_mode=graph_mode or "knn")
    aff.setdiag(0)
    aff.eliminate_zeros()
    return ad_sub, aff


def save_affinity(aff, ds_name, n_cells, affinity_type,
                  n_components=50, k_neighbors=50, graph_dir='./graphs'):
    """Save affinity with the exact filename the trainer's load_affinities() expects.

    Filename: affinity_{ds_name}{n_cells}_ncomp{n_components}_kneighbors{k_neighbors}_{affinity_type}.pkl

    Args:
        aff:           scipy sparse affinity matrix
        ds_name:       dataset name string (e.g. 's28nsc') — must match DATASETS key
        n_cells:       len(ad) — number of cells in the full dataset
        affinity_type: string tag (e.g. 'covet') — must match what you pass to find_metacells
        n_components:  must match MultiCropsDataset n_components (default 50)
        k_neighbors:   must match MultiCropsDataset k_neighbors (default 50)
        graph_dir:     must match trainer save_dir (default './graphs')
    """
    import os
    import pickle
    os.makedirs(graph_dir, exist_ok=True)
    fname = f"affinity_{ds_name}{n_cells}_ncomp{n_components}_kneighbors{k_neighbors}_{affinity_type}.pkl"
    fpath = os.path.join(graph_dir, fname)
    with open(fpath, 'wb') as f:
        pickle.dump(aff, f)
    print(f"Saved: {fpath}")
    return fpath


def diagnose_embedding(ad, rep_key='X_covet', niche_col='niches_3D', k=50, sample_n=2000):
    """Diagnose an obsm embedding: variance, NN distances, niche separation, sigma_b.

    Also checks for the z-double-scaling bug: _covet_spatial_knn always re-scales z by 30,
    so if generate_affinity already scaled ad.obsm['spatial'][:, 2] (and set _spatial_z_scaled),
    any subsequent compute_covet_features call will use z * 900 for spatial kNN.

    Args:
        ad:        AnnData with ad.obsm[rep_key]
        rep_key:   obsm key to inspect (default 'X_covet')
        niche_col: obs column with niche labels (default 'niches_3D')
        k:         neighbours for sigma_b computation (default 50)
        sample_n:  cells to subsample for niche-separation distance matrix (default 2000)
    """
    from sklearn.neighbors import NearestNeighbors as _NearestNeighbors

    print(f"{'='*60}")
    print(f"  diagnose_embedding: rep_key='{rep_key}'")
    print(f"{'='*60}")

    # --- spatial audit ---
    if 'spatial' in ad.obsm:
        sp = ad.obsm['spatial']
        xy_range = (sp[:, 0].max() - sp[:, 0].min(),
                    sp[:, 1].max() - sp[:, 1].min())
        print(f"\n[spatial]  shape={sp.shape}  "
              f"x_range={xy_range[0]:.1f}  y_range={xy_range[1]:.1f}")
        if sp.shape[1] == 3:
            z_unique = np.unique(sp[:, 2])
            print(f"  z unique values ({len(z_unique)}): {z_unique[:10]}"
                  f"{'...' if len(z_unique) > 10 else ''}")
            print(f"  (only x,y used for spatial kNN — z ignored)")
    else:
        print("\n[spatial]  not present in ad.obsm")

    # --- embedding stats ---
    if rep_key not in ad.obsm:
        print(f"\nERROR: '{rep_key}' not in ad.obsm. Available: {list(ad.obsm.keys())}")
        return

    X = ad.obsm[rep_key]
    print(f"\n[{rep_key}]  shape={X.shape}")

    var_per_dim = X.var(0)
    print(f"  per-dim variance:  min={var_per_dim.min():.4e}  "
          f"mean={var_per_dim.mean():.4e}  max={var_per_dim.max():.4e}")
    print(f"  near-zero-var dims (< 1e-6): {(var_per_dim < 1e-6).sum()}")

    # --- nearest-neighbour distances ---
    nn = _NearestNeighbors(n_neighbors=2).fit(X)
    dists, _ = nn.kneighbors(X)
    nn_dist = dists[:, 1]
    n_dup = (nn_dist < 1e-6).sum()
    print(f"\n[NN distances]")
    print(f"  min={nn_dist.min():.4e}  median={np.median(nn_dist):.4e}  "
          f"max={nn_dist.max():.4e}")
    print(f"  near-duplicate cells (nn_dist < 1e-6): {n_dup}  "
          f"{'<-- PROBLEM: will zero-out sigma_b' if n_dup > 0 else 'ok'}")

    # --- niche separation ---
    if niche_col in ad.obs.columns:
        niche_labels = ad.obs[niche_col].astype(str).values
        idx = np.random.choice(len(ad), min(sample_n, len(ad)), replace=False)
        X_sub = X[idx]
        niches_sub = niche_labels[idx]
        D = np.sqrt(((X_sub[:, None] - X_sub[None]) ** 2).sum(-1))
        same = niches_sub[:, None] == niches_sub[None]
        np.fill_diagonal(same, False)
        w_mean = D[same].mean();   w_med = np.median(D[same])
        a_mean = D[~same].mean();  a_med = np.median(D[~same])
        ratio = a_mean / max(w_mean, 1e-12)
        flag = 'ok' if ratio > 1.3 else 'weak — may not separate niches well'
        print(f"\n[niche separation]  (sample n={len(idx)}, col='{niche_col}')")
        print(f"  within-niche:  mean={w_mean:.4f}  median={w_med:.4f}")
        print(f"  across-niche:  mean={a_mean:.4f}  median={a_med:.4f}")
        print(f"  ratio (across/within): {ratio:.2f}  {'✓' if ratio > 1.3 else '⚠'} {flag}")
    else:
        print(f"\n[niche separation]  skipped — '{niche_col}' not in ad.obs")

    # --- sigma_b from kNN ---
    sc.pp.neighbors(ad, use_rep=rep_key, n_neighbors=k)
    knn_dist = ad.obsp['distances'].tocsr()
    asc_idx = max(k // 2 - 1, 0)
    sigma_b = np.array([
        np.partition(knn_dist.data[knn_dist.indptr[i]:knn_dist.indptr[i+1]], asc_idx)[asc_idx]
        if knn_dist.indptr[i+1] > knn_dist.indptr[i] else 1e-8
        for i in range(len(ad))
    ])
    n_zero_sigma = (sigma_b < 1e-4).sum()
    print(f"\n[sigma_b]  (k={k}, bandwidth = dist to k//2-th neighbour)")
    print(f"  min={sigma_b.min():.4e}  median={np.median(sigma_b):.4e}  "
          f"max={sigma_b.max():.4e}")
    print(f"  cells with sigma < 1e-4: {n_zero_sigma}  "
          f"{'<-- PROBLEM: NaN/inf in RBF kernel' if n_zero_sigma > 0 else 'ok'}")
    print(f"{'='*60}\n")


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
