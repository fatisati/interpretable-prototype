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


def eval_seacell_quality(ds_id):
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

    # --- Purity ---
    purity_per_mc = calc_purity(obs, label_key=lk, mc_key='_mc', return_per_mc=True)
    if purity_per_mc is not None:
        purity_per_mc.index.name = 'metacell'
        purity_per_mc.to_csv(os.path.join(save_path, 'purity_per_mc.csv'))
        print(f"[seacell] mean cell-type purity: {purity_per_mc.mean():.4f}")
    else:
        print(f"[seacell] cell-type purity: label key '{lk}' not in obs, skipped")

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

    # --- Summary metrics.json ---
    metrics = {}
    if purity_per_mc is not None:
        metrics['mean_cell_type_purity'] = float(purity_per_mc.mean())
    if entropy_per_mc is not None:
        metrics['mean_batch_entropy'] = float(entropy_per_mc.mean())
    metrics['modularity'] = mod_result['modularity']
    with open(os.path.join(save_path, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics to {save_path}")
    return {
        'purity': purity_per_mc,
        'batch_entropy': entropy_per_mc,
        'modularity': mod_result,
    }


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
