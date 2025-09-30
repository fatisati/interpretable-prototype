import scanpy as sc
import SEACells
import pandas as pd
import numpy as np
import os


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


def compute_seacells(ad, n_SEACells, build_kernel_on="X_pca"):
    # ad = preprocess(ad, n_top_genes)
    ## Additional parameters
    n_waypoint_eigs = (
        10  # Number of eigenvalues to consider when initializing metacells
    )

    model = SEACells.core.SEACells(
        ad,
        build_kernel_on=build_kernel_on,
        n_SEACells=n_SEACells,
        n_waypoint_eigs=n_waypoint_eigs,
        convergence_epsilon=1e-5,
    )

    model.construct_kernel_matrix()
    # M = model.kernel_matrix
    # Initialize archetypes
    model.initialize_archetypes()
    model.fit(min_iter=10, max_iter=50)

    SEACell_ad = SEACells.core.summarize_by_SEACell(
        ad, SEACells_label="SEACell", summarize_layer="X"
    )
    # SEACell_soft_ad = SEACells.core.summarize_by_soft_SEACell(ad, model.A_, celltype_label='celltype',summarize_layer='raw', minimum_weight=0.05)
    return ad, SEACell_ad, model


def mc_quality_metrics(ad=None, cell_type_key="celltype"):
    if "X_pca" not in ad.obsm:
        sc.tl.pca(ad)
    purity = SEACells.evaluate.compute_celltype_purity(ad, cell_type_key)
    compactness = SEACells.evaluate.compactness(ad, "X_pca")
    separation = SEACells.evaluate.separation(ad, "X_pca", nth_nbr=1)
    res_dict = {"purity": purity, "compactness": compactness, "separation": separation}
    summary = {}

    for metric, df in res_dict.items():
        vals = df.iloc[:, -1].dropna().values
        median = np.median(vals)
        q25, q75 = np.percentile(vals, [25, 75])
        iqr = q75 - q25

        summary[metric] = f"{median:.3f} ± {iqr:.3f}"
    summary_df = pd.DataFrame([summary])
    return summary_df, res_dict  # single row


def save_seacell_df(named_dfs, p):
    for name, df in named_dfs.items():
        df.to_csv(f"{p}/{name}.csv", index=False)


def scproto_metacell_metrics(t, path):
    model = t.load_model()
    z = t.encode_adata(t.ref.adata, model)
    proto_ids = t.get_proto_assignments(z, model).argmax(axis=1)
    adata = t.ref.adata
    adata.obs["SEACell"] = proto_ids
    metacell_metrics = mc_quality_metrics(adata, t.dataset.cell_type_key)
    save_seacell_df(metacell_metrics, path)
    return metacell_metrics


def agg_obs(SEACell_ad, adata, obs_key):
    SEACell_ad.obs[obs_key] = (
        adata.obs.groupby("SEACell")[obs_key]
        .agg(lambda x: x.mode()[0])
        .reindex(SEACell_ad.obs_names)
    )
    return SEACell_ad


def save_seacell(ad, SEACell_ad, ds_id):
    home = "/home/icb/fatemehs.hashemig/"
    ad.write(f"{home}/models/{ds_id}/seacell_sc.h5ad")
    SEACell_ad.write(f"{home}/models/{ds_id}/seacell_agg.h5ad")
