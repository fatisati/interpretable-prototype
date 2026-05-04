import json
import numpy as np
import pandas as pd
import scipy.sparse as sp

from interpretable_ssl.evaluation.mc_metric_utils import *
from interpretable_ssl.datasets.dataset_configs import *
from interpretable_ssl.configs.paths import get_seacell_model_dir


def get_seacell_path(ds_id, build_kernel_on="X_pca"):
    return get_seacell_model_dir(ds_id, build_kernel_on)


def _compute_modularity(ad, mc_idx):
    """Newman weighted modularity using ad.obsp['connectivities']."""
    if 'connectivities' not in ad.obsp:
        sc.pp.neighbors(ad, use_rep='X_pca')
    A = sp.csr_matrix(ad.obsp['connectivities'])
    A = (A + A.T) / 2
    degrees = np.array(A.sum(axis=1)).ravel()
    two_m = degrees.sum()
    if two_m == 0:
        return {'modularity': 0.0, 'per_cluster_contribution': {}}
    cluster_ids = np.unique(mc_idx)
    Q = 0.0
    per_cluster = {}
    for c in cluster_ids:
        mask = (mc_idx == c)
        e_k = A[mask][:, mask].sum()
        d_k = degrees[mask].sum()
        contrib = float((e_k - d_k * d_k / two_m) / two_m)
        per_cluster[int(c)] = contrib
        Q += contrib
    return {'modularity': float(Q), 'per_cluster_contribution': per_cluster}


def _save_seacell_umap_data(ds_id, ad, mc_ad, build_kernel_on="X_pca"):
    """Compute joint cell+metacell UMAP and save umap_cells.csv + umap_protos.csv +
    cell_assignments.csv + proto_vectors.npy."""
    from collections import Counter

    save_path = get_seacell_path(ds_id, build_kernel_on)
    bk = DATASETS[ds_id].get('batch_key', None)
    lk = DATASETS[ds_id]['label_key']

    mc_idx = np.array([int(i.split('-')[-1]) for i in ad.obs['SEACell'].values])
    n_mc = len(mc_ad)
    cell_pca = ad.obsm['X_pca']

    # Metacell PCA = mean of assigned cells' PCA coordinates
    mc_pca = np.zeros((n_mc, cell_pca.shape[1]))
    for p in range(n_mc):
        mask = mc_idx == p
        if mask.sum() > 0:
            mc_pca[p] = cell_pca[mask].mean(axis=0)

    # Joint UMAP: cells + metacell mean-PCA stacked
    import scanpy as sc_local
    combined = np.vstack([cell_pca, mc_pca])
    tmp_ad = sc_local.AnnData(combined)
    sc_local.pp.neighbors(tmp_ad, use_rep='X', n_neighbors=15, metric='cosine', random_state=42)
    sc_local.tl.umap(tmp_ad, random_state=42)
    z_umap = tmp_ad.obsm['X_umap'][:len(ad)]
    proto_umap = tmp_ad.obsm['X_umap'][len(ad):]

    # --- umap_cells.csv ---
    cells_df = pd.DataFrame({'cell_id': ad.obs_names, 'umap_1': z_umap[:, 0], 'umap_2': z_umap[:, 1], 'metacell_id': mc_idx})
    for col in [lk, bk]:
        if col and col in ad.obs.columns:
            cells_df[col] = ad.obs[col].values
    cells_df.to_csv(os.path.join(save_path, 'umap_cells.csv'), index=False)

    # --- cell_assignments.csv (all cells, for spatial figure join) ---
    assign_df = pd.DataFrame({'cell_id': ad.obs_names, 'metacell_id': mc_idx})
    for col in [lk, bk]:
        if col and col in ad.obs.columns:
            assign_df[col] = ad.obs[col].values
    assign_df.to_csv(os.path.join(save_path, 'cell_assignments.csv'), index=False)

    # --- umap_protos.csv ---
    nk = DATASETS[ds_id].get('niche_key', None)
    lk_vals = ad.obs[lk].values if lk and lk in ad.obs.columns else None
    nk_vals = ad.obs[nk].values if nk and nk in ad.obs.columns else None
    proto_rows = []
    for p in range(n_mc):
        mask = mc_idx == p
        n = int(mask.sum())
        row = {'proto_id': p, 'umap_1': proto_umap[p, 0], 'umap_2': proto_umap[p, 1], 'n_cells': n}
        if lk_vals is not None:
            row[f'majority_{lk}'] = Counter(lk_vals[mask]).most_common(1)[0][0] if n > 0 else None
        if nk_vals is not None:
            row[f'majority_{nk}'] = Counter(nk_vals[mask]).most_common(1)[0][0] if n > 0 else None
        proto_rows.append(row)
    pd.DataFrame(proto_rows).to_csv(os.path.join(save_path, 'umap_protos.csv'), index=False)

    # --- proto_vectors.npy: PCA of metacell gene expression ---
    import scanpy as sc_local
    mc_tmp = mc_ad.copy()
    if sp.issparse(mc_tmp.X):
        max_val = mc_tmp.X.max()
    else:
        max_val = mc_tmp.X.max()
    if max_val > 20:
        sc_local.pp.normalize_total(mc_tmp)
        sc_local.pp.log1p(mc_tmp)
    sc_local.tl.pca(mc_tmp)
    np.save(os.path.join(save_path, 'proto_vectors.npy'), mc_tmp.obsm['X_pca'])

    print(f"SEACell UMAP data saved to {save_path}")


def eval_seacell_task1(ds_id, build_kernel_on="X_pca"):
    """Load saved SEACell files and compute metacell quality metrics comparable to
    t_proto.eval_metacell_quality(): purity, batch entropy, and modularity.

    Returns dict with keys 'purity', 'batch_entropy', 'modularity'.
    """
    from interpretable_ssl.evaluation.metric_helpers.embedding_metrics import load_seacell

    save_path = get_seacell_path(ds_id, build_kernel_on)
    sc_file = os.path.join(save_path, 'seacell_sc.h5ad')
    if not os.path.exists(sc_file):
        raise FileNotFoundError(f"SEACell files not found at {save_path}. Run train mode first.")

    print(f"Loading SEACell from {save_path} ...")
    ad, SEACell_ad = load_seacell(ds_id, build_kernel_on=build_kernel_on)

    bk = DATASETS[ds_id].get('batch_key', None)
    lk = DATASETS[ds_id]['label_key']

    # integer mc index
    mc_idx = np.array([int(i.split('-')[-1]) for i in ad.obs['SEACell'].values])
    obs = ad.obs.copy()
    obs['_mc'] = mc_idx

    # --- Unused prototypes ---
    n_total_mc = mc_idx.max() + 1
    n_unused = int(n_total_mc - len(np.unique(mc_idx)))
    unused_ratio = n_unused / n_total_mc
    print(f"[seacell] unused protos: {n_unused}/{n_total_mc} ({unused_ratio:.2%})")

    # --- Metacell sizes (save once, reuse for weighted stats) ---
    mc_sizes = obs['_mc'].astype(str).value_counts().rename('size')
    mc_sizes.index.name = 'metacell'
    mc_sizes.to_csv(os.path.join(save_path, 'size_per_mc.csv'))

    # --- Cell-type purity ---
    purity_per_mc = calc_purity(obs, label_key=lk, mc_key='_mc', return_per_mc=True)
    if purity_per_mc is not None:
        purity_per_mc.index.name = 'metacell'
        purity_per_mc.to_csv(os.path.join(save_path, 'purity_per_mc.csv'))
        weights_p = mc_sizes.reindex(purity_per_mc.index).fillna(0)
        w_sum_p = weights_p.sum()
        weighted_mean_purity = float((purity_per_mc * weights_p).sum() / w_sum_p)
        weighted_std_purity = float(np.sqrt(((purity_per_mc - weighted_mean_purity) ** 2 * weights_p).sum() / w_sum_p))
        print(f"[seacell] mean cell-type purity: {purity_per_mc.mean():.4f}  (size-weighted: {weighted_mean_purity:.4f} ± {weighted_std_purity:.4f})")
    else:
        weighted_mean_purity = None
        weighted_std_purity = None
        print(f"[seacell] cell-type purity: label key '{lk}' not in obs, skipped")

    # --- Niche purity ---
    nk = DATASETS[ds_id].get('niche_key')
    niche_purity_per_mc = calc_purity(obs, label_key=nk, mc_key='_mc', return_per_mc=True) if nk else None
    if niche_purity_per_mc is not None:
        niche_purity_per_mc.index.name = 'metacell'
        niche_purity_per_mc.to_csv(os.path.join(save_path, 'niche_purity_per_mc.csv'))
        weights_n = mc_sizes.reindex(niche_purity_per_mc.index).fillna(0)
        w_sum_n = weights_n.sum()
        weighted_mean_niche_purity = float((niche_purity_per_mc * weights_n).sum() / w_sum_n)
        weighted_std_niche_purity = float(np.sqrt(((niche_purity_per_mc - weighted_mean_niche_purity) ** 2 * weights_n).sum() / w_sum_n))
        print(f"[seacell] mean niche purity: {niche_purity_per_mc.mean():.4f}  (size-weighted: {weighted_mean_niche_purity:.4f} ± {weighted_std_niche_purity:.4f})")

    # --- Batch entropy ---
    entropy_per_mc = None
    if bk is not None:
        entropy_per_mc = calc_batch_entropy(obs, batch_key=bk, mc_key='_mc', return_per_mc=True)
    if entropy_per_mc is not None:
        entropy_per_mc.index.name = 'metacell'
        entropy_per_mc.to_csv(os.path.join(save_path, 'batch_entropy_per_mc.csv'))
        weights = mc_sizes.reindex(entropy_per_mc.index).fillna(0)
        w_sum = weights.sum()
        weighted_mean_entropy = float((entropy_per_mc * weights).sum() / w_sum)
        weighted_std_entropy = float(np.sqrt(((entropy_per_mc - weighted_mean_entropy) ** 2 * weights).sum() / w_sum))
        print(f"[seacell] mean batch entropy: {entropy_per_mc.mean():.4f}  (size-weighted: {weighted_mean_entropy:.4f} ± {weighted_std_entropy:.4f})")
    else:
        weighted_mean_entropy = None
        weighted_std_entropy = None
        print(f"[seacell] batch entropy: batch key not found, skipped")

    # --- Modularity ---
    mod_result = _compute_modularity(ad, mc_idx)
    print(f"[seacell] modularity: {mod_result['modularity']:.4f}")
    def _convert(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        raise TypeError(f'Not serializable: {type(o)}')
    with open(os.path.join(save_path, 'modularity.json'), 'w') as f:
        json.dump(mod_result, f, indent=2, default=_convert)

    # --- Per-batch modularity ---
    mean_batch_mod = std_batch_mod = None
    if bk is not None and bk in ad.obs.columns:
        from interpretable_ssl.evaluation.mc_metric_utils import calc_modularity_per_batch
        A = sp.csr_matrix(ad.obsp['connectivities'])
        batch_mod_s = calc_modularity_per_batch(A, mc_idx, ad.obs[bk].values)
        batch_mod_s.to_csv(os.path.join(save_path, 'modularity_per_batch.csv'))
        mean_batch_mod = float(batch_mod_s.mean())
        std_batch_mod = float(batch_mod_s.std())
        print(f"[seacell] per-batch modularity: mean={mean_batch_mod:.4f}, std={std_batch_mod:.4f}")

    # --- Summary metrics.json (read-then-update to preserve task2 metrics) ---
    metrics_path = os.path.join(save_path, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        metrics = {}
    if purity_per_mc is not None:
        metrics['mean_cell_type_purity'] = float(purity_per_mc.mean())
        metrics['weighted_mean_cell_type_purity'] = weighted_mean_purity
        metrics['weighted_std_cell_type_purity'] = weighted_std_purity
    if niche_purity_per_mc is not None:
        metrics['mean_niche_purity'] = float(niche_purity_per_mc.mean())
    if entropy_per_mc is not None:
        metrics['mean_batch_entropy'] = float(entropy_per_mc.mean())
        metrics['weighted_mean_batch_entropy'] = weighted_mean_entropy
        metrics['weighted_std_batch_entropy'] = weighted_std_entropy
    metrics['modularity'] = mod_result['modularity']
    metrics['n_unused_protos'] = n_unused
    metrics['unused_proto_ratio'] = unused_ratio
    if mean_batch_mod is not None:
        metrics['mean_modularity_batch'] = mean_batch_mod
        metrics['std_modularity_batch'] = std_batch_mod

    # --- Aff-DC compactness (same method as trainer) ---
    try:
        import pickle
        from interpretable_ssl.evaluation.mc_metric_utils import compute_aff_dc_compactness

        n_components = 50   # augmenter default
        k_neighbors  = 50   # augmenter default
        affinity_type = 'arbf'
        graph_dir = './graphs'
        ds_name = ds_id  # str(sc_ds) = sc_ds.name = ds_id for known datasets
        n_cells = len(ad)
        graph_name = f"affinity_{ds_name}{n_cells}_ncomp{n_components}_kneighbors{k_neighbors}_{affinity_type}.pkl"
        graph_path = os.path.join(graph_dir, graph_name)
        print(f"[aff_dc_compactness] looking for graph at: {graph_path}")

        if not os.path.exists(graph_path):
            print(f"[aff_dc_compactness] graph not found, skipping compactness. "
                  f"Run trainer with affinity_type='arbf' first to generate it.")
        else:
            with open(graph_path, 'rb') as f:
                aff = pickle.load(f)
            aff = sp.csr_matrix(aff)
            # diagonal is 0 in aff_raw; set to 1 for diffusion maps
            aff_for_dc = aff.copy()
            aff_for_dc.setdiag(1)

            mc_ids_arr = ad.obs['SEACell'].astype(str).values
            batches_arr = ad.obs[bk].values if bk is not None else np.zeros(len(ad), dtype=str)

            comp_df, counts_df = compute_aff_dc_compactness(aff_for_dc, mc_ids_arr, batches_arr)
            valid_counts = counts_df.where(comp_df.notna(), 0)
            per_mc_mean = (comp_df.fillna(0) * valid_counts).sum(axis=1) / valid_counts.sum(axis=1)
            per_batch_mean = (comp_df.fillna(0) * valid_counts).sum(axis=0) / valid_counts.sum(axis=0)

            out_df = comp_df.copy()
            out_df['weighted_mean'] = per_mc_mean
            csv_path = os.path.join(save_path, 'aff_dc_compactness.csv')
            out_df.to_csv(csv_path)

            metrics['aff_compactness_mean'] = float(per_mc_mean.mean())
            metrics['aff_compactness_per_batch'] = {str(b): float(v) for b, v in per_batch_mean.items()}
            print(f"[aff_dc_compactness] mean={metrics['aff_compactness_mean']:.4f} | saved to {csv_path}")
    except Exception as e:
        import traceback
        print(f"Warning: aff_dc_compactness failed: {e}")
        traceback.print_exc()

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics to {save_path}")
    _save_seacell_umap_data(ds_id, ad, SEACell_ad, build_kernel_on)

    return {
        'purity': purity_per_mc,
        'niche_purity': niche_purity_per_mc,
        'batch_entropy': entropy_per_mc,
        'modularity': mod_result,
        'n_unused_protos': n_unused,
        'unused_proto_ratio': unused_ratio,
    }


def eval_seacell_task2(ds_id, build_kernel_on="X_pca"):
    """Compute and save task 2 metrics for SEACell metacells.

    Mirrors eval_seacell_quality() for task 1. Loads saved SEACell files,
    then computes coverage, DGE consistency, and scGraph.

    Returns:
        dict with scalar summaries: coverage, dge_rbo_avg, dge_kendall_avg,
        dge_jaccard_avg, scgraph_corr_avg.
    """
    import scanpy as sc
    from interpretable_ssl.evaluation.metric_helpers.embedding_metrics import load_seacell

    save_path = get_seacell_path(ds_id, build_kernel_on)
    sc_file = os.path.join(save_path, 'seacell_sc.h5ad')
    if not os.path.exists(sc_file):
        raise FileNotFoundError(f"SEACell files not found at {save_path}. Run train mode first.")

    print(f"Loading SEACell from {save_path} ...")
    ad, mc_ad = load_seacell(ds_id, build_kernel_on=build_kernel_on)

    bk = DATASETS[ds_id].get('batch_key', None)
    lk = DATASETS[ds_id]['label_key']

    # PCA on mc_ad for scGraph
    sc.tl.pca(mc_ad)
    mc_ad.obsm['seacell_mc_pca'] = mc_ad.obsm['X_pca']

    scalars = calc_task2_metrics(ad, mc_ad, lk, bk, ['seacell_mc_pca'], 'seacell', save_path)

    for metric_name, value in scalars.items():
        if value is not None:
            print(f"[seacell task2] {metric_name}: {value:.4f}")

    metrics_path = os.path.join(save_path, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        metrics = {}
    metrics.update({k: v for k, v in scalars.items() if v is not None})
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    return scalars


def eval_seacell_task3(ds_id, build_kernel_on="X_pca"):
    """Compute and save task 3 spatial metrics (niche DGE consistency) for SEACells.

    Only runs if dataset has a niche_key defined.

    Returns:
        dict with key 'ct_niche_rbo_avg', or empty dict if no niche_key.
    """
    nk = DATASETS[ds_id].get('niche_key')
    if not nk:
        print(f"[seacell task3] no niche_key for '{ds_id}', skipped")
        return {}

    from interpretable_ssl.evaluation.metric_helpers.embedding_metrics import load_seacell
    from interpretable_ssl.evaluation.de_helper import celltype_niche_dge
    from interpretable_ssl.evaluation.mc_metric_utils import calc_purity

    save_path = get_seacell_path(ds_id, build_kernel_on)
    sc_file = os.path.join(save_path, 'seacell_sc.h5ad')
    if not os.path.exists(sc_file):
        raise FileNotFoundError(f"SEACell files not found at {save_path}. Run train mode first.")

    print(f"Loading SEACell from {save_path} ...")
    ad, mc_ad = load_seacell(ds_id, build_kernel_on=build_kernel_on)
    lk = DATASETS[ds_id]['label_key']

    # Propagate niche labels to metacells via majority vote
    obs = ad.obs.copy()
    mc_idx = np.array([int(i.split('-')[-1]) for i in ad.obs['SEACell'].values])
    obs['_mc'] = mc_idx
    _, major_niche = calc_purity(obs, label_key=nk, mc_key='_mc',
                                  return_per_mc=True, return_major_label=True)
    mc_ad.obs[nk] = mc_ad.obs.index.map(
        lambda x: major_niche.get(str(int(x.split('-')[-1])))
    )

    summary, _ = celltype_niche_dge(ad, mc_ad, lk, nk, 'seacell', save_path)
    ct_niche_rbo_avg = float(summary.values.mean())
    print(f"[seacell task3] mean ct-niche RBO: {ct_niche_rbo_avg:.4f}")

    metrics_path = os.path.join(save_path, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        metrics = {}
    metrics['ct_niche_rbo_avg'] = ct_niche_rbo_avg
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    return {'ct_niche_rbo_avg': ct_niche_rbo_avg}


def load_dataset(ds_id):
    conf = DATASETS[ds_id]
    return (
        sc.read_h5ad(conf["path"]),
        conf.get('batch_key', None),
        conf["label_key"],
        conf["num_prototypes"],
    )


def train_seacell(ds_id, mode, k=50, build_kernel_on="X_pca"):
    seacell_exists = os.path.exists(get_seacell_path(ds_id, build_kernel_on) + "/seacell_sc.h5ad")
    if mode == "train" or not (seacell_exists):
        from interpretable_ssl.evaluation.metric_helpers.metacell_metrics import (
            compute_seacells,
            agg_obs,
            save_seacell,
        )

        os.makedirs(get_seacell_path(ds_id, build_kernel_on), exist_ok=True)
        # use the current dataset id, not a fixed string
        ad, bk, lk, n_proto = load_dataset(ds_id)
        print(len(ad))
        if ad.X.max() > 20:
            sc.pp.normalize_total(ad)
            sc.pp.log1p(ad)

        if build_kernel_on == "X_pca" and "X_pca" not in ad.obsm:
            sc.tl.pca(ad)
        ad, SEACell_ad, model = compute_seacells(ad, n_proto, k=k, build_kernel_on=build_kernel_on)
        agg_obs(SEACell_ad, ad, lk)
        if bk is not None:
            agg_obs(SEACell_ad, ad, bk)
        save_seacell(ad, SEACell_ad, ds_id, build_kernel_on)
    else:
        print("eval mode, seacell file found. Call eval_seacell_quality() for metrics.")
