import scanpy as sc
import SEACells
import pandas as pd
import numpy as np
import os

# Fix SEACells GPU bug: inject cupyx into SEACells.core namespace
try:
    import cupy as cp
    import cupyx
    import cupyx.scipy.sparse
    if cp.cuda.is_available():
        SEACells.core.cupyx = cupyx
        SEACells.core.cp = cp
except ImportError:
    pass
from collections import Counter
from interpretable_ssl.evaluation.mc_metric_utils import *


def preprocess(ad, n_top_genes):
    raw_ad = sc.AnnData(ad.X)
    raw_ad.obs_names, raw_ad.var_names = ad.obs_names, ad.var_names
    ad.raw = raw_ad
    # Normalize cells, log transform and compute highly variable genes
    # sc.pp.normalize_per_cell(ad)
    # sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=n_top_genes)
    # Compute principal components -
    # Here we use 50 components. This number may also be selected by examining variance explaint
    sc.tl.pca(ad, n_comps=50, use_highly_variable=True)
    return ad


def _seacells_backend():
    """Return (use_gpu, use_sparse) based on available hardware.

    For large datasets without GPU, use_sparse=True keeps K sparse throughout
    and avoids materializing the full n_cells×n_cells dense matrix.
    """
    try:
        import cupy as cp
        import cupyx.scipy.sparse
        if cp.cuda.is_available():
            print("[SEACells backend] GPU detected → use_gpu=True, use_sparse=False")
            return True, False
    except Exception:
        pass
    print("[SEACells backend] No GPU → use_sparse=True (sparse CPU, avoids dense K)")
    return False, True


def compute_seacells(ad, n_SEACells, build_kernel_on="X_pca", k=50):
    n_waypoint_eigs = 10
    use_gpu, use_sparse = _seacells_backend()

    model = SEACells.core.SEACells(
        ad,
        build_kernel_on=build_kernel_on,
        n_SEACells=n_SEACells,
        n_waypoint_eigs=n_waypoint_eigs,
        use_gpu=use_gpu,
        use_sparse=use_sparse,
    )

    model.construct_kernel_matrix()
    model.initialize_archetypes()
    model.fit()

    if "counts" not in ad.layers:
        ad.layers["counts"] = ad.X.copy()

    SEACell_ad = SEACells.core.summarize_by_SEACell(
        ad, SEACells_label="SEACell", summarize_layer="counts"
    )
    # SEACell_soft_ad = SEACells.core.summarize_by_soft_SEACell(ad, model.A_, celltype_label='celltype',summarize_layer='raw', minimum_weight=0.05)
    return ad, SEACell_ad, model


def suggest_n_seacells(aff, cells_per_mc=50, n_eigs=150, gap_z_thresh=1.0):
    """Suggest n_SEACells from affinity via four estimates:

    - Ratio-based:       n_cells // cells_per_mc  (granularity target / upper bound)
    - Spectral gap:      largest eigenvalue drop after λ1  (major groups only)
    - Multi-scale k:     last significant gap across all scales (major + substructure)
    - Participation ratio (PR): 1/sum(p_i^2), p_i=λ_i/sum(λ)  (effective dimensionality)

    A gap is 'significant' if it exceeds mean + gap_z_thresh * std of all gaps.

    Args:
        aff:           scipy sparse affinity matrix (N x N).
        cells_per_mc:  target cells per metacell for ratio-based estimate (default 50).
        n_eigs:        number of top eigenvalues to inspect (default 150).
        gap_z_thresh:  z-score threshold for a gap to count as significant (default 1.0).

    Returns:
        int: suggested n_SEACells = max(multi_scale_k, pr).
    """
    from scipy.sparse.linalg import eigsh

    n_cells = aff.shape[0]
    ratio_based = n_cells // cells_per_mc

    vals = eigsh(aff, k=min(n_eigs, n_cells - 1), which='LM', return_eigenvectors=False)
    vals = np.sort(vals)[::-1]

    # skip λ1 (constant/mean mode, trivially dominant in affinity matrices)
    gaps = np.abs(np.diff(vals[1:]))

    # major groups: first (largest) gap
    spectral_k = int(np.argmax(gaps)) + 2

    # substructure: last gap that exceeds mean + z*std
    threshold = gaps.mean() + gap_z_thresh * gaps.std()
    significant = np.where(gaps > threshold)[0]
    multi_scale_k = int(significant[-1]) + 2 if len(significant) > 0 else spectral_k

    # participation ratio: effective number of active eigenmodes (all scales)
    vals_pos = vals[vals > 0]
    p = vals_pos / vals_pos.sum()
    pr = int(round(1.0 / (p ** 2).sum()))

    suggestion = max(multi_scale_k, pr)
    print(f"n_cells:             {n_cells}")
    print(f"Ratio-based:         {ratio_based}  (1 per {cells_per_mc} cells, upper bound)")
    print(f"Spectral gap:        {spectral_k}  (major groups only)")
    print(f"Multi-scale k:       {multi_scale_k}  (major + substructure, last significant gap)")
    print(f"Participation ratio: {pr}  (effective eigenmodes)")
    print(f"Suggestion:          max({multi_scale_k}, {pr}) = {suggestion}")
    return suggestion


def compute_seacells_from_affinity(
    ad, n_SEACells, ds_name, affinity_type,
    n_components=50, k_neighbors=50, graph_dir='./graphs', n_waypoint_eigs=10,
    build_kernel_on='X_pca',
):
    """Run SEACells archetypal analysis using a precomputed affinity matrix.

    Skips construct_kernel_matrix() entirely by injecting the loaded affinity
    via model.add_precomputed_kernel_matrix(). The affinity path is constructed
    from parameters to match the convention used by save_affinity().

    Args:
        ad:               AnnData object (cells x genes), preprocessed.
        n_SEACells:       number of SEACells (metacells) to compute.
        ds_name:          dataset name string (e.g. 's28nsc').
        affinity_type:    affinity type tag (e.g. 'covet', 'arbf').
        n_components:     must match what was used when saving (default 50).
        k_neighbors:      must match what was used when saving (default 50).
        graph_dir:        directory where affinity .pkl files are stored (default './graphs').
        n_waypoint_eigs:  number of waypoint eigenvectors (default 10).
        build_kernel_on:  ad.obsm key used for waypoint initialization (default 'X_pca').
                          Should match the embedding space the affinity was built on.

    Returns:
        (ad, SEACell_ad, model)
    """
    import pickle

    n_cells = len(ad)
    fname = f"affinity_{ds_name}{n_cells}_ncomp{n_components}_kneighbors{k_neighbors}_{affinity_type}.pkl"
    aff_path = os.path.join(graph_dir, fname)
    print(f"Loading affinity from {aff_path} ...")

    with open(aff_path, 'rb') as f:
        aff = pickle.load(f)

    use_gpu, use_sparse = _seacells_backend()

    model = SEACells.core.SEACells(
        ad,
        build_kernel_on=build_kernel_on,
        n_SEACells=n_SEACells,
        n_waypoint_eigs=n_waypoint_eigs,
        use_gpu=use_gpu,
        use_sparse=use_sparse,
    )

    model.add_precomputed_kernel_matrix(aff)
    model.initialize_archetypes()
    model.fit()

    if "counts" not in ad.layers:
        ad.layers["counts"] = ad.X.copy()

    SEACell_ad = SEACells.core.summarize_by_SEACell(
        ad, SEACells_label="SEACell", summarize_layer="counts"
    )
    return ad, SEACell_ad, model


def save_seacell_df(named_dfs, p):
    for name, df in named_dfs.items():
        df.to_csv(f"{p}/{name}.csv", index=False)


def agg_obs(SEACell_ad, adata, obs_key):
    SEACell_ad.obs[obs_key] = (
        adata.obs.groupby("SEACell")[obs_key]
        .agg(lambda x: x.mode()[0])
        .reindex(SEACell_ad.obs_names)
    )
    return SEACell_ad


def save_seacell(ad, SEACell_ad, ds_id, build_kernel_on="X_pca"):
    from interpretable_ssl.configs.paths import get_seacell_model_dir
    seacell_dir = get_seacell_model_dir(ds_id, build_kernel_on)
    print("saving to: ", seacell_dir)
    os.makedirs(seacell_dir, exist_ok=True)
    ad.write(os.path.join(seacell_dir, "seacell_sc.h5ad"))
    SEACell_ad.write(os.path.join(seacell_dir, "seacell_agg.h5ad"))
