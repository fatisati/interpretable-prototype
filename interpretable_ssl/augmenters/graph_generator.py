import numpy as np
import scanpy as sc
from interpretable_ssl.augmenters.diffusion_knn import *
import numpy as np, scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


def _knn_sym_sigma(ad, build_on, k, graph_construction='union', bandwidth_mode='median'):
    """Shared core: kNN → binarize → sigma → symmetrize. Returns (sym, sigma, n).

    bandwidth_mode='median' (default): sigma_i = distance to the (k//2)-th nearest
        neighbour — the convention used everywhere else in this file.
    bandwidth_mode='max_gap': sigma_i = distance just before the largest
        consecutive gap in cell i's own sorted kNN distances — an elbow-style,
        per-cell-adaptive choice instead of always using the same fixed
        percentile position, which may or may not land on a real boundary
        between "true local neighbors" and "everything else".
    sym is the symmetrized binary adjacency matrix.
    """
    sc.pp.neighbors(ad, use_rep=build_on, n_neighbors=k, knn=True)
    knn_dist = ad.obsp['distances'].tocsr()
    n = knn_dist.shape[0]

    knn_bin = knn_dist.copy()
    knn_bin.data[:] = 1.0
    knn_bin.setdiag(1)

    if bandwidth_mode == 'median':
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
    elif bandwidth_mode == 'max_gap':
        # Delegates to _bandwidth_from_neighbor_dists (defined later in this module,
        # resolved at call time -- fine for two top-level functions in the same file)
        # rather than a second, separately-maintained copy of the same Kneedle-style
        # logic, which is what caused this to still have the raw-largest-gap bug
        # after it was fixed in the other copy.
        sigma = np.empty(n)
        for i in range(n):
            s, e = knn_dist.indptr[i], knn_dist.indptr[i + 1]
            d = np.sort(knn_dist.data[s:e])
            sigma[i] = _bandwidth_from_neighbor_dists(d, 'max_gap') if len(d) else 1e-8
    else:
        raise ValueError(f"unknown bandwidth_mode: {bandwidth_mode!r} (expected 'median' or 'max_gap')")
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


def rbf_product(ad, build_on_list, k=50, per_space_sigma=False, bandwidth_mode="median"):
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
        bandwidth_mode:   'median' (default, same convention _sigma_from_graph also
                          uses) or 'max_gap' (elbow-style: sigma at each row's
                          largest consecutive distance gap). Only affects the
                          per_space_sigma=True path — _sigma_from_graph (used when
                          per_space_sigma=False) is unchanged and still median-only.
    """
    import time
    t0 = t_step = time.time()
    def _t():
        nonlocal t_step
        elapsed = time.time() - t_step
        t_step = time.time()
        return f"{elapsed:.1f}s"

    spaces = list(build_on_list)
    print(f"[rbf_product] n={ad.n_obs}  k={k}  spaces={spaces}  bandwidth_mode={bandwidth_mode!r}")

    # Step 1: kNN sym per space (also keep sigma if per_space_sigma=True)
    syms = []
    sigmas = []
    n = None
    for i, key in enumerate(spaces):
        print(f"[rbf_product] step 1.{i+1}: kNN → sym  {key}  (sc.pp.neighbors) ...")
        sym, sigma, n = _knn_sym_sigma(ad, key, k, bandwidth_mode=bandwidth_mode)
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
    import SEACells.build_graph  # bare `import SEACells` doesn't reliably expose the
    # build_graph submodule as an attribute -- see the 'ctx' branch of generate_affinity
    # for the full explanation. Same fix, applied here too.
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
        import SEACells.build_graph  # bare `import SEACells` doesn't reliably expose the
        # build_graph submodule as an attribute -- see the 'ctx' branch below for the
        # full explanation. Same fix, applied here too.
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

    elif affinity_type == 'bk32':
        # Exact original pybanksy package (own expression + neighbor-mean expression
        # + AGF, z-scored per block, lambda-weighted, concatenated, THEN PCA'd down
        # to n_components -- the real published pipeline, not our own PCA-concat
        # approximation), with num_neighbours=32 (not the hardcoded 15 'banksy_orig'
        # uses) to match the median-32 niche-density target _ensure_X_ctx already
        # calibrates ctx/ctxm/ctxg/ctxb to, and lambda=0.5 (balance spatial context
        # vs. cell identity, chosen over the paper's cell-typing default of 0.2
        # since niche recovery sits between "cell typing" and "domain segmentation").
        # Then ordinary arbf (adaptive-bandwidth RBF, union symmetrization) on the
        # resulting 50-dim embedding. Deliberately not 'banksy_orig0.5' -- anything
        # starting with 'banksy' truncates to 'bank' under model_name.py's [:4]
        # rule, colliding with the existing plain 'banksy0.5' runs already on disk.
        print(f"bk32: pybanksy (exact package), lambda=0.5, num_neighbours=32, "
              f"n_components=50, then arbf (k={k}) on top")
        if 'X_bk32' in ad.obsm:
            print(f"[bk32] using cached X_bk32 embedding already in adata.obsm "
                  f"(shape={ad.obsm['X_bk32'].shape}) -- NOT recomputing. This must have "
                  f"been precomputed on the FULL tissue before any cell-type filtering "
                  f"(spatial_subsets.build_celltype_subset_with_context's banksy_configs) "
                  f"for spatial context to be correct on a filtered subset.")
        else:
            print(f"[bk32] WARNING: X_bk32 not in adata.obsm -- computing it fresh HERE, "
                  f"on the {ad.n_obs} cells currently in `ad`. If `ad` is already filtered "
                  f"to one cell type, the neighbour-mean term will only see same-type "
                  f"neighbours, not the true mixed-cell-type spatial context -- precompute "
                  f"X_bk32 on the full tissue first if that's not what you want.")
            compute_banksy_embedding(ad, lambda_param=0.5, num_neighbours=32,
                                     n_components=50, obsm_key='X_bk32')
        return rbf_optimized(ad, build_on='X_bk32', k=k, graph_construction='union')

    elif affinity_type == 'bk08':
        # Same pybanksy embedding as 'bk32', but lambda=0.8 -- the paper's own
        # domain-segmentation default (vs. 0.5 for 'bk32', vs. 0.2 for cell
        # typing) -- and the kernel is built with the REAL SEACells package's own
        # SEACellGraph.rbf() (SEACells/build_graph.py), not this file's
        # rbf_optimized reimplementation, so results are directly comparable to
        # the actual package's own graph construction, not just our own version.
        print(f"bk08: pybanksy (exact package), lambda=0.8, num_neighbours=32, "
              f"n_components=50, then REAL SEACells build_graph.rbf (k={k}) on top")
        if 'X_bk08' in ad.obsm:
            print(f"[bk08] using cached X_bk08 embedding already in adata.obsm "
                  f"(shape={ad.obsm['X_bk08'].shape}) -- NOT recomputing. This must have "
                  f"been precomputed on the FULL tissue before any cell-type filtering "
                  f"(spatial_subsets.build_celltype_subset_with_context's banksy_configs) "
                  f"for spatial context to be correct on a filtered subset.")
        else:
            print(f"[bk08] WARNING: X_bk08 not in adata.obsm -- computing it fresh HERE, "
                  f"on the {ad.n_obs} cells currently in `ad`. If `ad` is already filtered "
                  f"to one cell type, the neighbour-mean term will only see same-type "
                  f"neighbours, not the true mixed-cell-type spatial context -- precompute "
                  f"X_bk08 on the full tissue first if that's not what you want.")
            compute_banksy_embedding(ad, lambda_param=0.8, num_neighbours=32,
                                     n_components=50, obsm_key='X_bk08')
        from SEACells.build_graph import SEACellGraph
        sg = SEACellGraph(ad, build_on='X_bk08', verbose=True)
        return sg.rbf(k=k, graph_construction='union')

    elif affinity_type == 'bkmg':
        # Same pybanksy embedding as 'bk08' (lambda=0.8, reuses X_bk08 if already
        # built), same max_gap (Kneedle) bandwidth 'ctxg' uses -- via the SAME
        # function ctxg calls, pca_weight_on_existing_topology, not a separate
        # reimplementation. Topology is a plain kNN graph on X_bk08 (median-mode
        # _knn_sym_sigma, but only its binary `sym` is kept -- the sigma from
        # that call is discarded); max_gap bandwidth is then computed on X_bk08
        # distances restricted to THAT fixed topology, exactly like ctxg computes
        # its max_gap bandwidth on X_pca restricted to the ctx-kernel's topology.
        # Earlier version computed both the kNN topology AND the max_gap sigma
        # directly from X_bk08 in one pass (_knn_sym_sigma(bandwidth_mode='max_gap')
        # + _rbf_on_graph) -- on BANKSY's heavily neighbor-smoothed lambda=0.8
        # embedding, local kNN distances are so compressed that Kneedle kept
        # snapping to a near-zero-distance early knee, underflowing the RBF
        # weight on almost every edge and leaving a near-singleton graph
        # (58313/58423 Leiden "communities"). Deriving the bandwidth over a
        # topology fixed independently of that same knee-detection step is what
        # keeps ctxg stable, so bkmg now does the same.
        print(f"bkmg: pybanksy (exact package), lambda=0.8, num_neighbours=32, "
              f"n_components=50, then max_gap bandwidth (ctxg's pca_weight_on_existing_topology) "
              f"on a fixed kNN topology (k={k})")
        if 'X_bk08' not in ad.obsm:
            compute_banksy_embedding(ad, lambda_param=0.8, num_neighbours=32,
                                     n_components=50, obsm_key='X_bk08')
        sym, _, _ = _knn_sym_sigma(ad, 'X_bk08', k, graph_construction='union',
                                    bandwidth_mode='median')
        return pca_weight_on_existing_topology(sym, ad.obsm['X_bk08'], bandwidth_mode='max_gap')

    elif affinity_type == 'ctxb':
        # BANKSY-style feature concatenation, but built entirely from OUR OWN
        # X_pca/X_ctx (calibrated radius, median-32 target) instead of the pybanksy
        # package -- no reason to give up that calibration for pybanksy's own
        # default spatial-kNN neighbourhood definition (k=15) when all that
        # actually matters is the *structure*: concatenate, variance-balance, run
        # one kernel. Same weighting formula the existing banksy/banksy_radius
        # branches already use (own-expression coefficient sqrt(1-lam**2), context
        # coefficient lam, lam = sqrt(alpha*V_pca / ((1-alpha)*V_ctx + alpha*V_pca))
        # -- letting Lambda := lam**2, this solves exactly Lambda = alpha*V_pca /
        # ((1-alpha)*V_ctx + alpha*V_pca), i.e. sqrt(1-Lambda)/sqrt(Lambda) in
        # BANKSY's own published notation, just solved directly for the coefficient
        # instead of for the paper's lambda first -- same formula), but with
        # alpha=0.5 (equal variance contribution from both blocks) instead of the
        # paper's hand-picked 0.2 -- a data-driven, symmetric default rather than a
        # borrowed hyperparameter.
        #
        # No second PCA after concatenating: arbf's kernel is Euclidean-distance
        # kNN, which is rotation-invariant, so a non-dimensionality-reducing PCA on
        # the 100-dim concatenated space would change nothing about neighbor
        # structure, and dropping dimensions only risks losing real signal for no
        # computational need (100 dims is trivial for kNN). Concatenate, then arbf,
        # directly.
        _ensure_X_ctx(ad)
        X_pca = ad.obsm['X_pca']
        X_ctx = ad.obsm['X_ctx']
        alpha = 0.5
        V_pca = X_pca.var(axis=0).sum()
        V_ctx = X_ctx.var(axis=0).sum()
        lam = np.sqrt(alpha * V_pca / ((1 - alpha) * V_ctx + alpha * V_pca))
        print(f"[ctxb] alpha={alpha:.2f}  V_pca={V_pca:.3f}  V_ctx={V_ctx:.3f}  lambda={lam**2:.3f}  "
              f"(coefficients: pca={np.sqrt(1 - lam**2):.3f}, ctx={lam:.3f})")
        ad.obsm['X_ctxb'] = np.concatenate(
            [np.sqrt(1 - lam**2) * X_pca, lam * X_ctx], axis=1
        )
        return rbf_optimized(ad, build_on='X_ctxb', k=k, graph_construction='union')

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
        import SEACells.build_graph  # bare `import SEACells` doesn't reliably expose the
        # build_graph submodule as an attribute -- see the 'ctx' branch below for the
        # full explanation. Same fix, applied here too (this is the branch that hit it
        # first, on a freshly-registered dataset whose affinity had never been built
        # before -- every prior 'arbf' run had a cached .pkl and never actually called
        # generate_affinity fresh).

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
        import time
        import SEACells.build_graph  # bare `import SEACells` doesn't reliably expose the
        # build_graph submodule as an attribute -- depends on SEACells' own __init__.py,
        # which apparently changed (unpinned `pip install git+...`) or was affected by
        # --no-deps. Scoped to just this branch; not touching the other 4 occurrences of
        # the same bare-import pattern elsewhere in this file for now.
        _ensure_X_ctx(ad)
        M = _get_or_build_ctx_kernel(ad, k)

        # Diagnostic: for each cell, the edge-weighted fraction of its neighbours that
        # share its cell type / niche label -- then summary stats of that fraction across
        # cells. Direct visibility into what the graph is actually connecting, rather than
        # inferring it after training from purity metrics. Skips gracefully if these obs
        # columns aren't present (this branch is spatial-specific but not guaranteed to
        # always run on NSCLC-labelled data). Shared helper (see weighted_same_label_frac /
        # log_weighted_same_label_fracs above) so ctx_pca_median/ctx_pca_maxgap below reuse
        # the exact same diagnostic instead of a re-derived copy.
        log_weighted_same_label_fracs(M, ad, "ctx affinity")

        return M

    elif affinity_type in ("ctxm", "ctxg"):
        # Same X_ctx-based topology as the "ctx" branch above (SEACells-style
        # radius-averaged PCA, radius calibrated by _ensure_X_ctx), but the edge
        # weight is the product of the X_ctx-arbf kernel AND a second, PCA-based
        # arbf kernel computed on the SAME edges -- "PCA distance calculated on top
        # of the kNN graph built on spatial", via pca_weight_on_existing_topology,
        # NOT rbf_product's separate-PCA-kNN-then-union approach (a different graph
        # from the one actually being tested here; reverted after building it once).
        # Since PCA distance is large between different cell types (identity
        # dominates PCA variance) and small within one type, this softly suppresses
        # cross-cell-type edges without ever touching a cell-type label -- purely a
        # function of expression similarity.
        #
        # Names are deliberately exactly 4 characters, not "ctx_pca_median" /
        # "ctx_pca_maxgap": model_name.py's generate_model_name() truncates every
        # string-valued param (including affinity_type) to [:4] for the run
        # directory name, so anything longer than 4 chars sharing a 4-char prefix
        # would collide and silently overwrite the other run's checkpoint dir.
        #   'ctxm' — median bandwidth (same convention as everywhere else in this
        #            file: distance to the k//2-th nearest neighbor).
        #   'ctxg' — max_gap bandwidth (Kneedle-style knee detection -- see
        #            _bandwidth_from_neighbor_dists).
        _ensure_X_ctx(ad)
        s_ctx = _get_or_build_ctx_kernel(ad, k)

        bandwidth_mode = "median" if affinity_type == "ctxm" else "max_gap"
        print(f"[{affinity_type}] pca_weight_on_existing_topology(bandwidth_mode={bandwidth_mode!r}) "
              f"on the same {s_ctx.nnz} edges ...")
        s_pca = pca_weight_on_existing_topology(s_ctx, ad.obsm["X_pca"], bandwidth_mode)

        M = s_ctx.multiply(s_pca).tocsr()
        print(f"[{affinity_type}] ctx-only nnz={s_ctx.nnz}  mean_weight={s_ctx.data.mean():.4f}  "
              f"-> combined nnz={M.nnz}  mean_weight={M.data.mean() if M.nnz else float('nan'):.4f}")

        log_weighted_same_label_fracs(M, ad, affinity_type)
        return M

    elif affinity_type in ("ctmr", "ctgr"):
        # Same construction as ctxm/ctxg, then RECALIBRATED via a second adaptive-
        # bandwidth pass (_recalibrate_combined_kernel) instead of ctxm/ctxg's raw
        # Hadamard product. Motivation: ctxm/ctxg's mean edge weight measured ~2-2.6x
        # lower than arbf's, which matters because scproto's epsilon calibration
        # (calibrate_epsilon) binary-searches epsilon so the latent soft-assignment
        # similarity on positive edges matches this graph's own mean edge weight
        # (aff_raw, NOT row-normalized) -- a much lower target forces a much softer
        # (larger) epsilon than arbf gets, plausibly over-smoothing the 800-way
        # softmax into the few-giant-prototypes collapse we measured.
        #
        # A single global rescale-to-match-ctx's-mean was tried first and rejected:
        # it patches the mean but not the shape (per-row degree, variance, tails), and
        # isn't how the rest of this file calibrates kernels. _recalibrate_combined_
        # kernel instead re-derives a proper per-row adaptive bandwidth from the
        # PRODUCT's own combined distance (see its docstring for the full derivation)
        # -- the identical recipe 'arbf' itself uses, just applied to the joint
        # ctx+PCA distance instead of PCA distance alone. Only aff_raw's mean (and
        # therefore the epsilon calibration target) changes -- normalize_aff() row-
        # normalizes self.aff regardless of absolute input scale, so the *relative*
        # same-type-vs-cross-type suppression pattern is unaffected.
        #   'ctmr' -- median bandwidth (matches ctxm), recalibrated.
        #   'ctgr' -- max_gap bandwidth (matches ctxg), recalibrated.
        _ensure_X_ctx(ad)
        s_ctx = _get_or_build_ctx_kernel(ad, k)

        bandwidth_mode = "median" if affinity_type == "ctmr" else "max_gap"
        print(f"[{affinity_type}] pca_weight_on_existing_topology(bandwidth_mode={bandwidth_mode!r}) "
              f"on the same {s_ctx.nnz} edges ...")
        s_pca = pca_weight_on_existing_topology(s_ctx, ad.obsm["X_pca"], bandwidth_mode)

        M_raw = s_ctx.multiply(s_pca).tocsr()
        M = _recalibrate_combined_kernel(M_raw, bandwidth_mode)
        print(f"[{affinity_type}] ctx-only mean_weight={s_ctx.data.mean():.4f}  "
              f"raw product mean_weight={M_raw.data.mean() if M_raw.nnz else float('nan'):.4f}  "
              f"-> recalibrated mean_weight={M.data.mean() if M.nnz else float('nan'):.4f}")

        log_weighted_same_label_fracs(M, ad, affinity_type)
        return M

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


def _bandwidth_from_neighbor_dists(d_sorted, mode):
    """Per-row bandwidth from one row's own sorted (ascending) neighbor distances.

    mode='median': distance at the SEACells convention position (len//2 - 1) --
        same rule build_seacell_kernel/_knn_sym_sigma already use.
    mode='max_gap': Kneedle-style knee detection (Satopaa, Lehman, Xu, Raglin &
        Kohler, "Finding a 'Kneedle' in a Haystack: Detecting Knee Points in System
        Behavior", ICDCSW 2011) -- NOT the raw largest consecutive gap. Sorted kNN
        distances typically look like a hockey stick: flat/slow for the first many
        (genuinely close) neighbors, then an accelerating rise once you run out of
        close neighbors and start reaching into the background -- i.e. a convex
        increasing curve, sitting below the chord connecting its first and last
        points. Kneedle's knee is the point of maximum distance BELOW that chord
        (after min-max normalizing both axes to [0,1], the scale factor between
        indices and distances no longer matters).
        This is what actually fixes the near-duplicate-neighbor bug, not just an
        ad hoc workaround for it: a jump right at the very start of the sorted list
        sits close to the chord's own starting point almost by construction (small
        x, small normalized y), so it contributes only a small deviation regardless
        of how large that jump is in raw distance terms -- unlike a raw-gap search,
        which picks exactly that spurious early jump. Confirmed empirically: the
        earlier raw-largest-gap version produced sigma hitting the 1e-8 floor for a
        large fraction of cells on a real run, collapsing the RBF kernel and
        cascading into rows with zero surviving edges after top-k pruning downstream.
    """
    m = len(d_sorted)
    if mode == 'median':
        idx = max(m // 2 - 1, 0)
        return max(d_sorted[idx], 1e-8)
    elif mode == 'max_gap':
        if m < 3:
            return max(d_sorted[-1], 1e-8)
        x_norm = np.arange(m, dtype=np.float64) / (m - 1)
        y_range = d_sorted[-1] - d_sorted[0]
        y_norm = (d_sorted - d_sorted[0]) / y_range if y_range > 0 else np.zeros(m)
        # Convex increasing curve -> knee = max distance BELOW the y=x chord.
        idx = int(np.argmax(x_norm - y_norm))
        return max(d_sorted[idx], 1e-8)
    else:
        raise ValueError(f"unknown bandwidth mode: {mode!r} (expected 'median' or 'max_gap')")


def pca_weight_on_existing_topology(s_aff, X_pca, bandwidth_mode='median'):
    """Given an already-built sparse affinity matrix (the X_ctx-based spatial kernel),
    compute a second, PCA-based RBF weight for the SAME edges -- same topology,
    different weight, so the two can be multiplied elementwise later.

    PCA distances (and the bandwidth derived from them) are computed strictly over
    each cell's REAL spatial-context neighbors -- i.e. "PCA distance calculated on
    top of the kNN graph built on spatial" -- not a separately-built PCA kNN search
    unioned in afterward (which is what rbf_product does, and is a different graph
    from what's being tested here).

    Returns a sparse matrix with the identical nonzero pattern as s_aff.
    """
    coo = s_aff.tocoo()
    n = s_aff.shape[0]

    row_to_cols = {}
    for r, c in zip(coo.row, coo.col):
        row_to_cols.setdefault(r, []).append(c)

    sigma = np.full(n, np.nan)
    for r, cols in row_to_cols.items():
        cols = np.asarray(cols)
        d = np.linalg.norm(X_pca[r] - X_pca[cols], axis=1)
        sigma[r] = _bandwidth_from_neighbor_dists(np.sort(d), bandwidth_mode)

    pca_data = np.empty_like(coo.data, dtype=np.float64)
    for i, (r, c) in enumerate(zip(coo.row, coo.col)):
        d = np.linalg.norm(X_pca[r] - X_pca[c])
        sig_c = sigma[c] if not np.isnan(sigma[c]) else sigma[r]
        sigma_prod = max(sigma[r] * sig_c, 1e-8)
        pca_data[i] = np.exp(-(d ** 2) / sigma_prod)

    return sp.csr_matrix((pca_data, (coo.row, coo.col)), shape=s_aff.shape)


def _recalibrate_combined_kernel(M, bandwidth_mode='median'):
    """Re-derive a properly-scaled adaptive-bandwidth kernel from a product of two
    RBF kernels sharing the same edge set (e.g. s_ctx.multiply(s_pca)).

    Why this is needed: for two Gaussian/RBF kernels sharing edges, weight_ij =
    exp(-d_ij^2 / bandwidth_ij), so log(w1_ij) + log(w2_ij) = -(d1_ij^2/b1_ij +
    d2_ij^2/b2_ij) -- multiplying the WEIGHTS is mathematically equivalent to
    summing the two kernels' own normalized squared distances into one combined
    distance D_ij = sqrt(-2*log(w1_ij * w2_ij)). That combined distance is exactly
    the right thing to build a joint kernel from -- but the raw product leaves its
    overall bandwidth implicitly fixed at 1, never recalibrated to D's own scale.
    Since D sums two independently-normalized ("each individually order-1")
    quantities, its typical size is roughly double either factor's own typical
    normalized distance -- exactly why the raw product's weights come out
    systematically smaller than either input kernel (measured ~2-2.6x lower mean
    edge weight than arbf on this dataset).

    Fix: treat D_ij as a plain distance and re-run the SAME adaptive per-row
    bandwidth selection this file uses everywhere else (_bandwidth_from_neighbor_
    dists -- median or max_gap), then re-exponentiate: exp(-D_ij^2 / (2*Sigma_i^2)).
    Mechanically identical to how 'arbf' itself derives weights from PCA distance --
    just applied to the joint ctx+PCA distance instead of PCA distance alone, so the
    result's marginal statistics (mean weight, effective degree) should resemble
    arbf's own by construction, rather than an arbitrarily rescaled product.

    Preserves M's exact sparsity pattern; only the weights change.
    """
    M = M.tocsr()
    coo = M.tocoo()

    # D_ij = sqrt(-2*log(w_ij)); clip below to avoid log(0) for numerically-zero
    # survivors (weights can't exceed 1 after multiplying two <=1 kernels).
    w = np.clip(coo.data, 1e-12, 1.0)
    d = np.sqrt(-2.0 * np.log(w))

    row_to_idx = {}
    for i, r in enumerate(coo.row):
        row_to_idx.setdefault(r, []).append(i)

    sigma = np.full(M.shape[0], np.nan)
    for r, idx in row_to_idx.items():
        sigma[r] = _bandwidth_from_neighbor_dists(np.sort(d[idx]), bandwidth_mode)

    new_data = np.empty_like(d)
    for i, r in enumerate(coo.row):
        new_data[i] = np.exp(-(d[i] ** 2) / (2 * sigma[r] ** 2))

    return sp.csr_matrix((new_data, (coo.row, coo.col)), shape=M.shape)


def _get_or_build_ctx_kernel(ad, k):
    """Build (or reuse) the SEACells adaptive-RBF kernel on X_ctx -- the shared base
    step for 'ctx', 'ctxm', and 'ctxg', which otherwise each independently rebuild
    this identical ~4-5 minute computation (kNN -> RBF -> LIL -> CSR) even though it
    depends only on X_ctx and k, not on which of the three is asking for it, or on
    whatever PCA correction gets multiplied on top afterward.

    Stored in ad.obsp['ctx_kernel'], which persists to disk via write_h5ad the same
    way ad.obsm['X_ctx'] already does (see _ensure_X_ctx) -- so once any ONE of
    ctx/ctxm/ctxg has built it and the caller saves the adata back to disk, every
    later call (including run_mc_task reloading a fresh copy of the file internally
    for each separate scProto training run) reuses it instead of rebuilding.
    """
    if 'ctx_kernel' in ad.obsp:
        print("[ctx kernel] using existing ad.obsp['ctx_kernel'] (not recomputed)")
        return ad.obsp['ctx_kernel']

    import time
    import SEACells.build_graph
    t0 = time.time()
    print(f"[ctx kernel] building SEACellGraph + rbf kernel on X_ctx (n={ad.n_obs}, k={k}) ...")
    kernel_model = SEACells.build_graph.SEACellGraph(ad, "X_ctx", verbose=True)
    s_ctx = kernel_model.rbf(k, graph_construction="union")
    print(f"[ctx kernel] done ({time.time()-t0:.1f}s)  nnz={s_ctx.nnz}")
    ad.obsp['ctx_kernel'] = s_ctx
    return s_ctx


def _ensure_X_ctx(ad, target_median_neighbours=None):
    """Ensure ad.obsm['X_ctx'] exists, computing it via a radius CALIBRATED to a target
    median neighbour count (default 32, Pentimalli et al.'s own reported 2D median --
    see datasets/spatial_subsets.py) rather than a hand-picked fixed radius. Shared by
    the 'ctx', 'ctxm', and 'ctxg' branches below so this logic lives in exactly one
    place, not duplicated with its own hardcoded radius per branch (which is what 'ctx'
    and 'ctxm'/'ctxg' each did independently before this helper existed).

    Reuses whatever's already in ad.obsm['X_ctx'] as-is and never recomputes -- e.g. if
    it was pre-baked into the .h5ad on disk (as done for fibnsc.h5ad via
    build_celltype_subset_with_context, and however a caller chooses to do the same for
    any other dataset to avoid recomputing this across multiple training runs that each
    reload the file fresh).
    """
    if 'X_ctx' in ad.obsm:
        print('[X_ctx] using existing X_ctx already in adata (not recomputed)')
        return ad.obsm['X_ctx']

    import time
    from interpretable_ssl.datasets.spatial_subsets import (
        calibrate_radius_for_target_median_neighbours, DEFAULT_TARGET_MEDIAN_NEIGHBOURS,
    )
    if 'X_pca' not in ad.obsm:
        sc.tl.pca(ad)
    target = target_median_neighbours or DEFAULT_TARGET_MEDIAN_NEIGHBOURS
    radius = calibrate_radius_for_target_median_neighbours(ad, target=target)
    t0 = time.time()
    ad.obsm['X_ctx'] = build_context(ad, radius)
    print(f"[X_ctx] build_context(radius={radius:.4f}) done ({time.time()-t0:.1f}s)")
    return ad.obsm['X_ctx']


def weighted_same_label_frac(Mcsr, labels):
    """For each cell, the edge-weighted fraction of its neighbours sharing its
    label. Shared by the 'ctx' branch's own diagnostic print and the new
    ctx_pca_* branches below — pulled out to module level so both can call it
    instead of duplicating it. Returns NaN for isolated cells (row_sum == 0).
    """
    labels = np.asarray(labels)
    row_sums = np.asarray(Mcsr.sum(axis=1)).ravel()
    Mcoo = Mcsr.tocoo()
    match = (labels[Mcoo.row] == labels[Mcoo.col]).astype(np.float64)
    weighted = np.zeros(Mcsr.shape[0])
    np.add.at(weighted, Mcoo.row, Mcoo.data * match)
    with np.errstate(invalid='ignore', divide='ignore'):
        frac = weighted / row_sums
    return frac


def log_weighted_same_label_fracs(M, ad, name, label_specs=(("cell-type", "celltypes"), ("niche", "niches_2D"))):
    """Print weighted same-label neighbour fraction for each (label_name, obs_key)
    pair that exists in ad.obs — same print format the 'ctx' branch already uses,
    now reusable so ctx_pca_median/ctx_pca_maxgap (and anything else) can log the
    same diagnostic without re-deriving it.
    """
    Mcsr = M.tocsr()
    for label_name, obs_key in label_specs:
        if obs_key not in ad.obs.columns:
            continue
        frac = weighted_same_label_frac(Mcsr, ad.obs[obs_key].to_numpy())
        print(f"[{name}] weighted same-{label_name} neighbour fraction "
              f"(obs['{obs_key}']): mean={np.nanmean(frac):.3f}  "
              f"median={np.nanmedian(frac):.3f}  std={np.nanstd(frac):.3f}  "
              f"n_isolated={int(np.isnan(frac).sum())}")


def graph_collapse_diagnostics(M, name="graph", leiden_resolution=1.0, top_n_communities=5,
                                num_prototypes=None):
    """Predicts, from an affinity graph alone -- no scProto training needed -- whether
    training on it is likely to collapse most cells onto very few prototypes (the
    observed failure mode: one metacell absorbing 2702/15309 cells while 93% of
    prototypes get under 10 cells).

    Three independent checks:

    1. Effective degree (effk) vs. raw degree, per cell. effk_i = 1/sum_j(p_ij^2),
       p_ij = row-normalized weight -- already the exact concept this codebase uses
       elsewhere for temperature calibration (appendix/training.tex: "the target
       K_eff is derived from the median effective number of neighbours in the
       affinity graph"). effk close to raw degree means most of a cell's neighbours
       contribute roughly equally to its weight budget -- no real local structure to
       separate on (flat, like a plain X_ctx-topology graph). effk much smaller than
       raw degree means a few neighbours dominate -- real local contrast (sharp, like
       arbf-on-PCA).

    2. Degree SPREAD (std/min/max, not just mean/median): a graph where most cells
       sit around the same degree but a subset are far denser is exactly the
       "few-huge-many-tiny" shape -- the mean/median alone can look fine while the
       spread already gives it away. Same underlying stats `affinity_report()`
       (trainers/scproto.py, printed as `aff_stats` at the start of every real
       training run) already reports for whatever graph a run actually trains on --
       reproduced here so a candidate graph can be checked before committing to a
       training run at all.

    3. Leiden community sizes on the RAW graph, before any neural training. This is
       the more direct test: run a cheap, purely graph-theoretic clustering and check
       whether one community already swallows a large fraction of cells. Since
       scProto's own community-preserving loss (the UMAP loss on graph edges, and
       nassoc when lambda_nassoc>0) pulls toward the same kind of structure this
       graph already encodes, a graph that Leiden-collapses on its own is a strong
       predictor that training will too -- lets you compare candidate graphs (e.g.
       'ctxg' vs. a new design) BEFORE spending time on a full run.

    Also checks the simple rule: if cells are on average connected to effk_mean
    "effective" neighbours, a community-preserving loss will tend to pull each of
    those neighbourhoods toward one shared prototype, so the number of prototypes
    that actually end up used should be roughly n_cells / effk_mean. If
    num_prototypes (K) is passed and is far above that predicted count, K itself is
    oversized relative to what this graph's own connectivity can differentiate --
    excess prototypes have nothing to grab onto and end up near-empty, which is
    exactly the observed "many tiny metacells" half of the failure mode (independent
    of the "few huge" half, which the effk/degree-flatness and Leiden checks above
    speak to).

    Args:
        M:                 sparse affinity matrix (n_cells x n_cells).
        name:              label for the printed output.
        leiden_resolution: passed to leidenalg's RBConfigurationVertexPartition.
        top_n_communities: how many of the largest communities to print sizes for.
        num_prototypes:    if given, compares against the effk-predicted metacell
                            count (n_cells / effk_mean) and raw-degree-predicted
                            count (n_cells / degree_mean).

    Returns:
        dict with 'degree' and 'effk' (per-cell arrays); also 'leiden_sizes' and
        'leiden_top1_frac' if leidenalg/python-igraph are installed.
    """
    Mcsr = M.tocsr()
    row_sums = np.asarray(Mcsr.sum(axis=1)).ravel()
    degree = np.asarray((Mcsr > 0).sum(axis=1)).ravel()

    Mcoo = Mcsr.tocoo()
    with np.errstate(invalid='ignore', divide='ignore'):
        p = Mcoo.data / row_sums[Mcoo.row]
    effk_denom = np.zeros(Mcsr.shape[0])
    np.add.at(effk_denom, Mcoo.row, np.nan_to_num(p) ** 2)
    effk = 1.0 / np.clip(effk_denom, 1e-12, None)

    ratio = effk / np.clip(degree, 1, None)
    print(f"[{name}] n={Mcsr.shape[0]}  nnz={Mcsr.nnz}")
    print(f"[{name}] raw degree:  median={np.median(degree):.1f}  mean={degree.mean():.1f}  "
          f"std={degree.std():.1f}  min={degree.min()}  max={degree.max()}")
    print(f"[{name}] effk:        median={np.median(effk):.1f}  mean={effk.mean():.1f}  "
          f"std={effk.std():.1f}  min={effk.min():.1f}  max={effk.max():.1f}")
    print(f"[{name}] effk/degree ratio (median): {np.median(ratio):.3f}  "
          f"(near 1.0 = flat/no local structure to separate on, near 0 = sharp/discriminative)")

    n_cells = Mcsr.shape[0]
    pred_k_effk = n_cells / max(effk.mean(), 1e-8)
    pred_k_degree = n_cells / max(degree.mean(), 1e-8)
    print(f"[{name}] predicted usable prototype count: n/effk_mean={pred_k_effk:.0f}  "
          f"n/degree_mean={pred_k_degree:.0f}"
          + (f"  vs. requested K={num_prototypes}" if num_prototypes else ""))
    if num_prototypes and num_prototypes > 2 * pred_k_effk:
        print(f"[{name}] WARNING: requested K={num_prototypes} is more than 2x the "
              f"effk-predicted usable count ({pred_k_effk:.0f}) -- most of the excess "
              f"prototypes have no distinguishable neighbourhood to grab onto and will "
              f"end up near-empty, regardless of what happens with the few dense ones.")

    result = {
        'degree': degree, 'effk': effk,
        'pred_k_effk': pred_k_effk, 'pred_k_degree': pred_k_degree,
    }

    try:
        import igraph as ig
        import leidenalg
        g = ig.Graph(n=Mcsr.shape[0], edges=list(zip(Mcoo.row.tolist(), Mcoo.col.tolist())),
                     directed=False)
        g.es['weight'] = Mcoo.data.tolist()
        g.simplify(combine_edges='sum')
        partition = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition,
            weights='weight', resolution_parameter=leiden_resolution, seed=0,
        )
        sizes = sorted(partition.sizes(), reverse=True)
        top = sizes[:top_n_communities]
        frac_top1 = top[0] / Mcsr.shape[0]
        print(f"[{name}] Leiden (resolution={leiden_resolution}): {len(sizes)} communities, "
              f"largest {top_n_communities}: {top}  top-1 fraction of all cells: {frac_top1:.1%}")
        if frac_top1 > 0.3:
            print(f"[{name}] WARNING: {frac_top1:.1%} of cells fall into a single Leiden "
                  f"community on the raw graph, before any training -- strong predictor of "
                  f"prototype collapse (same failure mode as the observed 2702-cell metacell).")
        result['leiden_sizes'] = sizes
        result['leiden_top1_frac'] = frac_top1
    except ImportError:
        print(f"[{name}] leidenalg/python-igraph not installed -- skipping the community-size "
              f"check (pip install python-igraph leidenalg to enable it). The effk numbers "
              f"above still stand on their own.")

    return result


def leiden_resolution_sweep(M, resolutions=(0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0),
                             name="graph", min_community_size=5):
    """Sweep Leiden resolution on a raw affinity graph and report how many communities
    each resolution finds -- lets you pick a prototype count (num_prototypes) directly
    from the graph's own natural structure, instead of committing to a fixed convention
    (K ~= N/75) that may be far more than the graph can actually differentiate. No
    training needed -- purely graph-theoretic, same Leiden call graph_collapse_diagnostics
    already makes at a single fixed resolution, just swept over many resolutions instead.

    Use alongside the true niche count (e.g. `adata.obs[NICHE_KEY].nunique()`) as a
    sanity reference point -- not a target to hit directly (a resolution producing
    exactly as many communities as true niches would badly under-resolve real
    within-niche heterogeneity useful for pseudobulk DGE), but a floor: a resolution
    giving noticeably FEWER communities than true niches means the graph itself can't
    even separate the niches you're trying to recover, regardless of what scProto/
    SEACells do downstream.

    Args:
        M:                  sparse affinity matrix.
        resolutions:        resolution values to try.
        name:               label for printed output.
        min_community_size: communities smaller than this are counted separately (a
                             resolution producing lots of these is likely
                             over-splitting into noise, not real structure).

    Returns:
        list of dicts, one per resolution: resolution, n_communities, median_size,
        frac_below_min_size, top1_frac.
    """
    import igraph as ig
    import leidenalg

    Mcsr = M.tocsr()
    Mcoo = Mcsr.tocoo()
    g = ig.Graph(n=Mcsr.shape[0], edges=list(zip(Mcoo.row.tolist(), Mcoo.col.tolist())),
                 directed=False)
    g.es['weight'] = Mcoo.data.tolist()
    g.simplify(combine_edges='sum')

    results = []
    print(f"[{name}] Leiden resolution sweep (n={Mcsr.shape[0]} cells):")
    header = f"{'resolution':>10}  {'n_comm':>7}  {'median_size':>12}  " \
             f"{'frac<'+str(min_community_size):>8}  {'top1_frac':>9}"
    print(header)
    for res in resolutions:
        partition = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition,
            weights='weight', resolution_parameter=res, seed=0,
        )
        sizes = np.array(sorted(partition.sizes(), reverse=True))
        n_comm = len(sizes)
        median_size = float(np.median(sizes))
        frac_small = float((sizes < min_community_size).mean())
        top1_frac = float(sizes[0] / Mcsr.shape[0])
        print(f"{res:>10}  {n_comm:>7}  {median_size:>12.1f}  {frac_small:>8.1%}  {top1_frac:>9.1%}")
        results.append({
            'resolution': res, 'n_communities': n_comm, 'median_size': median_size,
            'frac_below_min_size': frac_small, 'top1_frac': top1_frac,
        })
    return results


def faiss_knn(b_adata, k):
    import faiss

    batch_pca = b_adata.obsm["X_pca"]
    index = faiss.IndexFlatL2(batch_pca.shape[1])
    index.add(batch_pca)
    D, I = index.search(batch_pca, k + 1)  # search k+1
    return D[:, 1:], I[:, 1:]


def diffusion_knn(batch_adata, k, n_proto):
    import SEACells.core  # explicit submodule import for consistency with the
    # build_graph fix elsewhere in this file -- bare `import SEACells` has not been
    # observed to drop `.core` (unlike `.build_graph`), but there's no reason to rely
    # on that continuing to hold across SEACells versions.

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


def load_or_build_affinity(ad, ds_name, affinity_type, k=50, bk=None,
                            n_components=50, k_neighbors=50, graph_dir='./graphs', **kwargs):
    """save_affinity's filename convention, but as a cache check first: if a matching
    .pkl already exists (e.g. one a training run already produced via the
    adata_augmenter subprocess pipeline, which uses this exact same naming scheme —
    see adata_augmenter.py:set_graph_name), load it instead of recomputing generate_affinity
    from scratch. Same idea as train_seacell(mode='eval')/_resolve_run_dir's idempotency
    elsewhere in this codebase -- avoid paying for expensive graph construction twice.
    """
    import os
    import pickle
    fname = f"affinity_{ds_name}{ad.n_obs}_ncomp{n_components}_kneighbors{k_neighbors}_{affinity_type}.pkl"
    fpath = os.path.join(graph_dir, fname)
    if os.path.exists(fpath):
        print(f"[load_or_build_affinity] found cached graph at {fpath} -- loading (not recomputing)")
        with open(fpath, 'rb') as f:
            return pickle.load(f)
    print(f"[load_or_build_affinity] no cached graph at {fpath} -- building fresh "
          f"(affinity_type={affinity_type}, k={k})")
    aff = generate_affinity(ad, k, bk, affinity_type=affinity_type, **kwargs)
    save_affinity(aff, ds_name, ad.n_obs, affinity_type,
                  n_components=n_components, k_neighbors=k_neighbors, graph_dir=graph_dir)
    return aff


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
