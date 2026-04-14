import json
import numpy as np
import pandas as pd
import scipy.sparse as sp

from interpretable_ssl.evaluation.mc_metric_utils import *
from interpretable_ssl.datasets.dataset_configs import *
from interpretable_ssl.configs.paths import get_seacell_model_dir


def get_seacell_path(ds_id):
    return get_seacell_model_dir(ds_id)


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


def eval_seacell_task1(ds_id):
    """Load saved SEACell files and compute metacell quality metrics comparable to
    t_proto.eval_metacell_quality(): purity, batch entropy, and modularity.

    Returns dict with keys 'purity', 'batch_entropy', 'modularity'.
    """
    from interpretable_ssl.evaluation.metric_helpers.embedding_metrics import load_seacell

    save_path = get_seacell_path(ds_id)
    sc_file = os.path.join(save_path, 'seacell_sc.h5ad')
    if not os.path.exists(sc_file):
        raise FileNotFoundError(f"SEACell files not found at {save_path}. Run train mode first.")

    print(f"Loading SEACell from {save_path} ...")
    ad, SEACell_ad = load_seacell(ds_id)

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

    # --- Cell-type purity ---
    purity_per_mc = calc_purity(obs, label_key=lk, mc_key='_mc', return_per_mc=True)
    if purity_per_mc is not None:
        purity_per_mc.index.name = 'metacell'
        purity_per_mc.to_csv(os.path.join(save_path, 'purity_per_mc.csv'))
        print(f"[seacell] mean cell-type purity: {purity_per_mc.mean():.4f}")
    else:
        print(f"[seacell] cell-type purity: label key '{lk}' not in obs, skipped")

    # --- Niche purity ---
    nk = DATASETS[ds_id].get('niche_key')
    niche_purity_per_mc = calc_purity(obs, label_key=nk, mc_key='_mc', return_per_mc=True) if nk else None
    if niche_purity_per_mc is not None:
        niche_purity_per_mc.index.name = 'metacell'
        niche_purity_per_mc.to_csv(os.path.join(save_path, 'niche_purity_per_mc.csv'))
        print(f"[seacell] mean niche purity: {niche_purity_per_mc.mean():.4f}")

    # --- Batch entropy ---
    entropy_per_mc = None
    if bk is not None:
        entropy_per_mc = calc_batch_entropy(obs, batch_key=bk, mc_key='_mc', return_per_mc=True)
    if entropy_per_mc is not None:
        entropy_per_mc.index.name = 'metacell'
        entropy_per_mc.to_csv(os.path.join(save_path, 'batch_entropy_per_mc.csv'))
        print(f"[seacell] mean batch entropy: {entropy_per_mc.mean():.4f}")
    else:
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

    # --- Summary metrics.json (read-then-update to preserve task2 metrics) ---
    metrics_path = os.path.join(save_path, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        metrics = {}
    if purity_per_mc is not None:
        metrics['mean_cell_type_purity'] = float(purity_per_mc.mean())
    if niche_purity_per_mc is not None:
        metrics['mean_niche_purity'] = float(niche_purity_per_mc.mean())
    if entropy_per_mc is not None:
        metrics['mean_batch_entropy'] = float(entropy_per_mc.mean())
    metrics['modularity'] = mod_result['modularity']
    metrics['n_unused_protos'] = n_unused
    metrics['unused_proto_ratio'] = unused_ratio
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics to {save_path}")
    return {
        'purity': purity_per_mc,
        'niche_purity': niche_purity_per_mc,
        'batch_entropy': entropy_per_mc,
        'modularity': mod_result,
        'n_unused_protos': n_unused,
        'unused_proto_ratio': unused_ratio,
    }


def eval_seacell_task2(ds_id):
    """Compute and save task 2 metrics for SEACell metacells.

    Mirrors eval_seacell_quality() for task 1. Loads saved SEACell files,
    then computes coverage, DGE consistency, and scGraph.

    Returns:
        dict with scalar summaries: coverage, dge_rbo_avg, dge_kendall_avg,
        dge_jaccard_avg, scgraph_corr_avg.
    """
    import scanpy as sc
    from interpretable_ssl.evaluation.metric_helpers.embedding_metrics import load_seacell

    save_path = get_seacell_path(ds_id)
    sc_file = os.path.join(save_path, 'seacell_sc.h5ad')
    if not os.path.exists(sc_file):
        raise FileNotFoundError(f"SEACell files not found at {save_path}. Run train mode first.")

    print(f"Loading SEACell from {save_path} ...")
    ad, mc_ad = load_seacell(ds_id)

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


def eval_seacell_task3(ds_id):
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

    save_path = get_seacell_path(ds_id)
    sc_file = os.path.join(save_path, 'seacell_sc.h5ad')
    if not os.path.exists(sc_file):
        raise FileNotFoundError(f"SEACell files not found at {save_path}. Run train mode first.")

    print(f"Loading SEACell from {save_path} ...")
    ad, mc_ad = load_seacell(ds_id)
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


def train_seacell(ds_id, mode, k=50):
    seacell_exists = os.path.exists(get_seacell_path(ds_id) + "/seacell_sc.h5ad")
    if mode == "train" or not (seacell_exists):
        from interpretable_ssl.evaluation.metric_helpers.metacell_metrics import (
            compute_seacells,
            agg_obs,
            save_seacell,
        )

        os.makedirs(get_seacell_path(ds_id), exist_ok=True)
        # use the current dataset id, not a fixed string
        ad, bk, lk, n_proto = load_dataset(ds_id)
        print(len(ad))
        if ad.X.max() > 20:
            sc.pp.normalize_total(ad)
            sc.pp.log1p(ad)

        sc.tl.pca(ad)
        ad, SEACell_ad, model = compute_seacells(ad, n_proto, k=k)
        agg_obs(SEACell_ad, ad, lk)
        if bk is not None:
            agg_obs(SEACell_ad, ad, bk)
        save_seacell(ad, SEACell_ad, ds_id)
    else:
        print("eval mode, seacell file found. Call eval_seacell_quality() for metrics.")
