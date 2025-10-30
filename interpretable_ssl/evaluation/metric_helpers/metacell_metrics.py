import scanpy as sc
import SEACells
import pandas as pd
import numpy as np
import os
from collections import Counter


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
        ad, SEACells_label="SEACell", summarize_layer="counts"
    )
    # SEACell_soft_ad = SEACells.core.summarize_by_soft_SEACell(ad, model.A_, celltype_label='celltype',summarize_layer='raw', minimum_weight=0.05)
    return ad, SEACell_ad, model

def avg_mc_quality_metrics(ad, bk, lk):
    batch_res = {}
    for b in ad.obs[bk].unique():
        b_ad = ad[ad.obs[bk] == b]
        _, res_dict = mc_quality_metrics(
                ad=b_ad, cell_type_key=lk
            )
        batch_res[b] = pd.concat([df for key, df in res_dict.items()], axis=1)
    
    # Combine all dfs (outer join on seacell, align columns by name)
    merged = pd.concat(batch_res.values(), axis=1, keys=batch_res.keys())
    
    # 2. Identify numeric and categorical inner columns (level=1)
    inner_cols = merged.columns.get_level_values(1)
    numeric_inner_cols = merged.select_dtypes(include=[np.number]).columns.get_level_values(1).unique()
    categoric_inner_cols = [c for c in inner_cols.unique() if c not in numeric_inner_cols]
    
    # 3. Average numeric columns (use xs to slice by inner level)
    if len(numeric_inner_cols) > 0:
        numeric_part = merged.loc[:, merged.columns.get_level_values(1).isin(numeric_inner_cols)]
        avg_numeric = numeric_part.groupby(level=1, axis=1).mean()
    else:
        avg_numeric = pd.DataFrame(index=merged.index)
    # 4. Majority vote for categorical columns
    def majority_vote(series):
        vals = series.dropna().tolist()
        if not vals:
            return np.nan
        return Counter(vals).most_common(1)[0][0]
    
    cat_frames = []
    for col in categoric_inner_cols:
        if (None, col) in merged.columns or any(c[1] == col for c in merged.columns):
            cols = merged.xs(col, level=1, axis=1)  # all batch columns for this feature
            maj = cols.apply(majority_vote, axis=1)
            cat_frames.append(maj.rename(col))
    
    if cat_frames:
        avg_categoric = pd.concat(cat_frames, axis=1)
        avg_df = pd.concat([avg_numeric, avg_categoric], axis=1)
    else:
        avg_df = avg_numeric
    return avg_df, batch_res

def mc_quality_metrics(ad, cell_type_key="celltype"):
    if "X_pca" not in ad.obsm:
        sc.tl.pca(ad)
    purity = SEACells.evaluate.compute_celltype_purity(ad, cell_type_key)
    compactness = SEACells.evaluate.compactness(ad, "X_pca")
    separation = SEACells.evaluate.separation(ad, "X_pca", nth_nbr=1)
    res_dict = {"purity": purity, "compactness": compactness, "separation": separation}
    summary = {}

    for metric, df in res_dict.items():
        vals = df.iloc[:, -1].dropna().values
        if metric == 'purity':
            center = np.mean(vals)
        else:
            center = np.median(vals)
        q25, q75 = np.percentile(vals, [25, 75])
        iqr = q75 - q25

        summary[metric] = f"{center:.3f} ± {iqr:.3f}"
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
    print('saving to: ', f"{home}/models/{ds_id}/seacell")
    os.makedirs(f"{home}/models/{ds_id}/seacell", exist_ok=True)
    ad.write(f"{home}/models/{ds_id}/seacell/seacell_sc.h5ad")
    SEACell_ad.write(f"{home}/models/{ds_id}/seacell/seacell_agg.h5ad")
