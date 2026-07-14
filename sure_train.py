import os
import sys
import json
import numpy as np
import scanpy as sc
import scipy.sparse as sp

from interpretable_ssl.datasets.dataset_configs import DATASETS
from interpretable_ssl.configs.paths import get_sure_model_dir, CODE_DIR
from interpretable_ssl.evaluation.mc_metric_utils import compute_task1_metrics, calc_task2_metrics

_SURE_DIR = os.path.join(CODE_DIR, 'baselines', 'SURE')
if _SURE_DIR not in sys.path:
    sys.path.insert(0, _SURE_DIR)


def _get_counts(ad):
    """Return raw count matrix as a dense float32 numpy array."""
    if 'counts' not in ad.layers:
        raise ValueError("adata.layers['counts'] not found. SURE requires raw counts.")
    X = ad.layers['counts']
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype='float32')


def _build_metacell_adata(ad, mc_ids, xs):
    """Sum raw counts per metacell to produce an aggregated AnnData."""
    n_mc = int(mc_ids.max()) + 1
    mc_X = np.zeros((n_mc, xs.shape[1]), dtype='float32')
    for i in range(n_mc):
        mask = mc_ids == i
        if mask.any():
            mc_X[i] = xs[mask].sum(axis=0)
    mc_ad = sc.AnnData(mc_X)
    mc_ad.var_names = ad.var_names
    mc_ad.obs_names = [f'SURE-{i}' for i in range(n_mc)]
    return mc_ad


def train_sure(ds_id, mode, n_epochs=None, batch_size=None):
    from SURE import SURE

    save_path = get_sure_model_dir(ds_id)
    sure_sc_file = os.path.join(save_path, 'sure_sc.h5ad')

    if mode != 'train' and os.path.exists(sure_sc_file):
        print("eval mode, sure file found. Load sure_sc.h5ad / sure_agg.h5ad for metrics.")
        return

    os.makedirs(save_path, exist_ok=True)

    conf = DATASETS[ds_id]
    ad = sc.read_h5ad(conf['path'])
    n_proto = conf['num_prototypes']
    batch_size = batch_size if batch_size is not None else conf.get('batch_size', 256)
    n_epochs = n_epochs if n_epochs is not None else conf.get('n_epochs', 200)
    lk = conf['label_key']
    bk = conf.get('batch_key', None)

    xs = _get_counts(ad)
    xs_norm = np.log1p(xs / (xs.sum(axis=1, keepdims=True) + 1e-8) * 1e4)
    ad.X = xs

    model = SURE(
        input_dim=xs_norm.shape[1],
        codebook_size=n_proto,
        loss_func='multinomial',
        use_cuda=True,
    )
    model.fit(xs_norm, num_epochs=n_epochs, batch_size=batch_size, use_jax=False)

    # hard_assignments returns one-hot (n_cells, n_proto); argmax gives integer ids
    assignments = model.hard_assignments(xs_norm)
    mc_ids = np.argmax(assignments, axis=1)

    # Store in the same 'SEACell' column used by the eval infrastructure
    ad.obs['SEACell'] = [f'SURE-{i}' for i in mc_ids]

    mc_ad = _build_metacell_adata(ad, mc_ids, xs)

    # Propagate cell-level labels to metacells via majority vote
    for key in [lk, bk]:
        if key and key in ad.obs.columns:
            mc_ad.obs[key] = (
                ad.obs.groupby('SEACell')[key]
                .agg(lambda x: x.mode()[0])
                .reindex(mc_ad.obs_names)
            )

    SURE.save_model(model, os.path.join(save_path, 'sure_model.pth'))
    ad.write(os.path.join(save_path, 'sure_sc.h5ad'))
    mc_ad.write(os.path.join(save_path, 'sure_agg.h5ad'))
    print(f"SURE saved to {save_path}")


def _load_sure(ds_id):
    save_path = get_sure_model_dir(ds_id)
    sc_file = os.path.join(save_path, 'sure_sc.h5ad')
    if not os.path.exists(sc_file):
        raise FileNotFoundError(f"SURE files not found at {save_path}. Run train_sure() first.")
    ad = sc.read_h5ad(sc_file)
    mc_ad = sc.read_h5ad(os.path.join(save_path, 'sure_agg.h5ad'))
    return ad, mc_ad


def eval_sure_task1(ds_id, n_components=50, k_neighbors=50,
                    affinity_type='arbf', graph_dir='./graphs'):
    save_path = get_sure_model_dir(ds_id)
    ad, _ = _load_sure(ds_id)

    bk = DATASETS[ds_id].get('batch_key', None)
    lk = DATASETS[ds_id]['label_key']
    nk = DATASETS[ds_id].get('niche_key')
    mc_idx = np.array([int(i.split('-')[-1]) for i in ad.obs['SEACell'].values])

    if 'connectivities' not in ad.obsp:
        X = ad.X if not sp.issparse(ad.X) else ad.X.toarray()
        if X.max() > 20:
            sc.pp.normalize_total(ad)
            sc.pp.log1p(ad)
        sc.tl.pca(ad)
        sc.pp.neighbors(ad, use_rep='X_pca')

    return compute_task1_metrics(
        ad, mc_idx, lk, bk, nk, save_path, ds_id, 'sure',
        n_components=n_components, k_neighbors=k_neighbors,
        affinity_type=affinity_type, graph_dir=graph_dir,
    )


def eval_sure_task2(ds_id):
    save_path = get_sure_model_dir(ds_id)
    ad, mc_ad = _load_sure(ds_id)

    bk = DATASETS[ds_id].get('batch_key', None)
    lk = DATASETS[ds_id]['label_key']

    sc.tl.pca(mc_ad)
    mc_ad.obsm['sure_mc_pca'] = mc_ad.obsm['X_pca']

    scalars = calc_task2_metrics(ad, mc_ad, lk, bk, ['sure_mc_pca'], 'sure', save_path)

    for metric_name, value in scalars.items():
        if value is not None:
            print(f"[sure task2] {metric_name}: {value:.4f}")

    metrics_path = os.path.join(save_path, 'metrics.json')
    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
    metrics.update({k: v for k, v in scalars.items() if v is not None})
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    return scalars


def eval_sure_task3(ds_id):
    nk = DATASETS[ds_id].get('niche_key')
    if not nk:
        print(f"[sure task3] no niche_key for '{ds_id}', skipped")
        return {}

    from interpretable_ssl.evaluation.de_helper import celltype_niche_dge
    from interpretable_ssl.evaluation.mc_metric_utils import calc_purity

    save_path = get_sure_model_dir(ds_id)
    ad, mc_ad = _load_sure(ds_id)
    lk = DATASETS[ds_id]['label_key']

    obs = ad.obs.copy()
    mc_idx = np.array([int(i.split('-')[-1]) for i in ad.obs['SEACell'].values])
    obs['_mc'] = mc_idx
    _, major_niche = calc_purity(obs, label_key=nk, mc_key='_mc',
                                 return_per_mc=True, return_major_label=True)
    mc_ad.obs[nk] = mc_ad.obs.index.map(
        lambda x: major_niche.get(str(int(x.split('-')[-1])))
    )

    summary, _ = celltype_niche_dge(ad, mc_ad, lk, nk, 'sure', save_path)
    ct_niche_rbo_avg = float(summary.values.mean())
    print(f"[sure task3] mean ct-niche RBO: {ct_niche_rbo_avg:.4f}")

    metrics_path = os.path.join(save_path, 'metrics.json')
    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
    metrics['ct_niche_rbo_avg'] = ct_niche_rbo_avg
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    return {'ct_niche_rbo_avg': ct_niche_rbo_avg}
