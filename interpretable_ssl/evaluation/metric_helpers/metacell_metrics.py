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


def compute_seacells(ad, n_SEACells, build_kernel_on="X_pca", k=50):
    n_waypoint_eigs = 10

    # Auto-detect GPU: use if cupy is available and working
    use_gpu = False
    try:
        import cupy as cp
        import cupyx.scipy.sparse
        if cp.cuda.is_available():
            use_gpu = True
    except Exception:
        pass

    model = SEACells.core.SEACells(
        ad,
        build_kernel_on=build_kernel_on,
        n_SEACells=n_SEACells,
        n_waypoint_eigs=n_waypoint_eigs,
        use_gpu=use_gpu,
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
