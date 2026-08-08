import os
import sys
import random
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from types import SimpleNamespace

from interpretable_ssl.datasets.dataset_configs import DATASETS
from interpretable_ssl.configs.paths import get_metaq_model_dir
from interpretable_ssl.evaluation.mc_metric_utils import compute_task1_metrics

_METAQ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baselines', 'MetaQ')


def _metaq_imports():
    if _METAQ_DIR not in sys.path:
        sys.path.insert(0, _METAQ_DIR)
    import torch
    from model import MetaQ as MetaQModel
    from engine import train_one_epoch, warm_one_epoch, inference as metaq_inference
    from data_utils import load_data as metaq_load_data
    return torch, MetaQModel, train_one_epoch, warm_one_epoch, metaq_inference, metaq_load_data


def train_metaq(ds_id, mode, data_type='RNA', batch_size=None, train_epochs=None):
    save_path = get_metaq_model_dir(ds_id)
    metaq_exists = os.path.exists(os.path.join(save_path, 'metaq_sc.h5ad'))

    if mode != 'train' and metaq_exists:
        print("MetaQ files found. Call eval_metaq_task1() for metrics.")
        return

    conf = DATASETS[ds_id]
    ad = sc.read_h5ad(conf['path'])
    bk = conf.get('batch_key', None)
    lk = conf['label_key']
    n_proto = conf['num_prototypes']
    batch_size = batch_size if batch_size is not None else conf.get('batch_size', 512)
    train_epochs = train_epochs if train_epochs is not None else conf.get('train_epochs', 300)
    print(f"Loaded {len(ad)} cells, target {n_proto} metacells, batch_size={batch_size}, train_epochs={train_epochs}")

    if 'counts' not in ad.layers:
        raise ValueError(
            f"adata.layers['counts'] not found for '{ds_id}'. "
            "MetaQ requires raw counts in layers['counts']."
        )
    ad.X = ad.layers['counts']

    import tempfile
    tmp_h5 = os.path.join(tempfile.mkdtemp(), f'{ds_id}_counts.h5ad')
    ad.write_h5ad(tmp_h5)

    torch, MetaQModel, train_one_epoch, warm_one_epoch, metaq_inference, metaq_load_data = _metaq_imports()

    args = SimpleNamespace(
        data_path=[tmp_h5],
        data_type=[data_type],
        metacell_num=n_proto,
        save_name=ds_id,
        type_key=lk or 'celltype',
        codebook_init='Random',
        train_epoch=train_epochs,
        batch_size=batch_size,
        converge_threshold=10,
        random_seed=1,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.random.manual_seed(args.random_seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.random_seed)

    device = torch.device(args.device)
    adata_list, dataloader_train, dataloader_eval, input_dims = metaq_load_data(args)

    net = MetaQModel(
        input_dims=input_dims,
        data_types=args.data_type,
        entry_num=args.metacell_num,
    ).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)

    print("======= Training Start =======")
    loss_rec_his = loss_vq_his = 1e7
    stable_epochs = 0

    for epoch in range(args.train_epoch):
        if epoch == 0:
            # codebook init: run one inference pass before any gradient steps
            embeds_init, _, _, _, _ = metaq_inference(
                model=net, data_types=args.data_type,
                data_loader=dataloader_eval, device=device,
            )
            net.quantizer.init_codebook(embeds_init, method=args.codebook_init)
            if len(adata_list) == 1:
                net.copy_decoder_q()
        else:
            loss_rec, loss_vq = train_one_epoch(
                model=net, data_types=args.data_type,
                dataloader=dataloader_train, optimizer=optimizer,
                epoch=epoch, device=device,
            )
            converge = (
                abs(loss_vq_his - loss_vq) <= 1e-5 and
                abs(loss_rec_his - loss_rec) <= 1e-5
            )
            if converge:
                stable_epochs += 1
                if stable_epochs >= args.converge_threshold:
                    print("Early stopping.")
                    break
            else:
                stable_epochs = 0
                loss_rec_his = loss_rec
                loss_vq_his = loss_vq

    print("======= Training Done =======")

    _, ids, _, _, _ = metaq_inference(
        model=net, data_types=args.data_type,
        data_loader=dataloader_eval, device=device,
    )
    ids = ids.astype(int)

    # Attach integer assignment to original adata
    ad.obs['metaq_id'] = ids

    # Build metacell adata from raw counts over all genes (consistent with SEACell format)
    X = ad.layers['counts']
    if sp.issparse(X):
        X = X.toarray()
    used_ids = np.unique(ids)
    data_meta = np.stack([X[ids == i].mean(axis=0) for i in used_ids])
    mc_ad = sc.AnnData(data_meta)
    mc_ad.obs_names = [str(i) for i in used_ids]
    mc_ad.var_names = ad.var_names

    # Majority vote for label and batch keys
    for key in ([lk] if lk else []) + ([bk] if bk else []):
        if key and key in ad.obs.columns:
            mc_ad.obs[key] = [
                ad.obs.loc[ad.obs['metaq_id'] == i, key].mode()[0]
                for i in used_ids
            ]

    os.makedirs(save_path, exist_ok=True)
    ad.write_h5ad(os.path.join(save_path, 'metaq_sc.h5ad'), compression="gzip")
    mc_ad.write_h5ad(os.path.join(save_path, 'metaq_mc.h5ad'), compression="gzip")
    print(f"Saved to {save_path}")

    try:
        os.remove(tmp_h5)
    except OSError:
        pass


def _save_metaq_umap_data(ds_id, ad, mc_ad):
    """Compute joint cell+metacell UMAP and save umap_cells.csv + umap_protos.csv +
    cell_assignments.csv + proto_vectors.npy."""
    from collections import Counter

    save_path = get_metaq_model_dir(ds_id)
    bk = DATASETS[ds_id].get('batch_key', None)
    lk = DATASETS[ds_id]['label_key']
    nk = DATASETS[ds_id].get('niche_key', None)

    mc_idx = ad.obs['metaq_id'].values.astype(int)
    used_ids = np.unique(mc_idx)
    cell_pca = ad.obsm['X_pca']

    # Metacell PCA = mean of assigned cells' PCA coordinates
    mc_pca = np.zeros((len(used_ids), cell_pca.shape[1]))
    for i, p in enumerate(used_ids):
        mask = mc_idx == p
        if mask.sum() > 0:
            mc_pca[i] = cell_pca[mask].mean(axis=0)

    # Joint UMAP: cells + metacell mean-PCA stacked
    combined = np.vstack([cell_pca, mc_pca])
    tmp_ad = sc.AnnData(combined)
    sc.pp.neighbors(tmp_ad, use_rep='X', n_neighbors=15, metric='cosine', random_state=42)
    sc.tl.umap(tmp_ad, random_state=42)
    z_umap = tmp_ad.obsm['X_umap'][:len(ad)]
    proto_umap = tmp_ad.obsm['X_umap'][len(ad):]

    # --- umap_cells.csv ---
    cells_df = pd.DataFrame({'cell_id': ad.obs_names, 'umap_1': z_umap[:, 0], 'umap_2': z_umap[:, 1], 'metacell_id': mc_idx})
    for col in [lk, bk]:
        if col and col in ad.obs.columns:
            cells_df[col] = ad.obs[col].values
    cells_df.to_csv(os.path.join(save_path, 'umap_cells.csv'), index=False)

    # --- cell_assignments.csv ---
    assign_df = pd.DataFrame({'cell_id': ad.obs_names, 'metacell_id': mc_idx})
    for col in [lk, bk]:
        if col and col in ad.obs.columns:
            assign_df[col] = ad.obs[col].values
    assign_df.to_csv(os.path.join(save_path, 'cell_assignments.csv'), index=False)

    # --- umap_protos.csv ---
    lk_vals = ad.obs[lk].values if lk and lk in ad.obs.columns else None
    nk_vals = ad.obs[nk].values if nk and nk in ad.obs.columns else None
    proto_rows = []
    for i, p in enumerate(used_ids):
        mask = mc_idx == p
        n = int(mask.sum())
        row = {'proto_id': p, 'umap_1': proto_umap[i, 0], 'umap_2': proto_umap[i, 1], 'n_cells': n}
        if lk_vals is not None:
            row[f'majority_{lk}'] = Counter(lk_vals[mask]).most_common(1)[0][0] if n > 0 else None
        if nk_vals is not None:
            row[f'majority_{nk}'] = Counter(nk_vals[mask]).most_common(1)[0][0] if n > 0 else None
        proto_rows.append(row)
    pd.DataFrame(proto_rows).to_csv(os.path.join(save_path, 'umap_protos.csv'), index=False)

    # --- proto_vectors.npy: PCA of metacell gene expression ---
    mc_tmp = mc_ad.copy()
    if mc_tmp.X.max() > 20:
        sc.pp.normalize_total(mc_tmp)
        sc.pp.log1p(mc_tmp)
    sc.tl.pca(mc_tmp)
    np.save(os.path.join(save_path, 'proto_vectors.npy'), mc_tmp.obsm['X_pca'])

    print(f"MetaQ UMAP data saved to {save_path}")


def eval_metaq_task1(ds_id, n_components=50, k_neighbors=50,
                     affinity_type='arbf', graph_dir='./graphs'):
    """Load saved MetaQ files and compute task-1 metacell quality metrics."""
    save_path = get_metaq_model_dir(ds_id)
    sc_file = os.path.join(save_path, 'metaq_sc.h5ad')
    if not os.path.exists(sc_file):
        raise FileNotFoundError(f"MetaQ files not found at {save_path}. Run train_metaq() first.")

    print(f"Loading MetaQ from {save_path} ...")
    ad = sc.read_h5ad(sc_file)
    mc_ad = sc.read_h5ad(os.path.join(save_path, 'metaq_mc.h5ad'))

    bk = DATASETS[ds_id].get('batch_key', None)
    lk = DATASETS[ds_id]['label_key']
    nk = DATASETS[ds_id].get('niche_key')
    mc_idx = ad.obs['metaq_id'].values.astype(int)

    # Ensure PCA + neighbors exist (needed for modularity and UMAP)
    if 'connectivities' not in ad.obsp:
        if sp.issparse(ad.X):
            max_val = ad.X.max()
        else:
            max_val = ad.X.max()
        if max_val > 20:
            sc.pp.normalize_total(ad)
            sc.pp.log1p(ad)
        sc.tl.pca(ad)
        sc.pp.neighbors(ad, use_rep='X_pca')

    result = compute_task1_metrics(
        ad, mc_idx, lk, bk, nk, save_path, ds_id, 'metaq',
        n_components=n_components, k_neighbors=k_neighbors,
        affinity_type=affinity_type, graph_dir=graph_dir,
    )
    _save_metaq_umap_data(ds_id, ad, mc_ad)
    return result


def eval_metaq_task2(ds_id):
    """Compute and save task-2 metrics (coverage, DGE consistency, scGraph) for MetaQ."""
    import json
    from interpretable_ssl.evaluation.mc_metric_utils import calc_task2_metrics

    save_path = get_metaq_model_dir(ds_id)
    if not os.path.exists(os.path.join(save_path, 'metaq_sc.h5ad')):
        raise FileNotFoundError(f"MetaQ files not found at {save_path}. Run train_metaq() first.")

    print(f"Loading MetaQ from {save_path} ...")
    ad = sc.read_h5ad(os.path.join(save_path, 'metaq_sc.h5ad'))
    mc_ad = sc.read_h5ad(os.path.join(save_path, 'metaq_mc.h5ad'))

    bk = DATASETS[ds_id].get('batch_key', None)
    lk = DATASETS[ds_id]['label_key']

    # Normalize mc_ad before PCA (X is mean raw counts)
    import scipy.sparse as sp_check
    if mc_ad.X.max() > 20 if not sp_check.issparse(mc_ad.X) else mc_ad.X.max() > 20:
        sc.pp.normalize_total(mc_ad)
        sc.pp.log1p(mc_ad)
    sc.tl.pca(mc_ad)
    mc_ad.obsm['metaq_mc_pca'] = mc_ad.obsm['X_pca']

    scalars = calc_task2_metrics(ad, mc_ad, lk, bk, ['metaq_mc_pca'], 'metaq', save_path)

    for metric_name, value in scalars.items():
        if value is not None:
            print(f"[metaq task2] {metric_name}: {value:.4f}")

    metrics_path = os.path.join(save_path, 'metrics.json')
    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
    metrics.update({k: v for k, v in scalars.items() if v is not None})
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    return scalars


def eval_metaq_task3(ds_id):
    """Compute and save task-3 spatial metrics (niche DGE consistency) for MetaQ.

    Only runs if dataset has a niche_key defined.
    """
    import json
    from interpretable_ssl.evaluation.de_helper import celltype_niche_dge
    from interpretable_ssl.evaluation.mc_metric_utils import calc_purity

    nk = DATASETS[ds_id].get('niche_key')
    if not nk:
        print(f"[metaq task3] no niche_key for '{ds_id}', skipped")
        return {}

    save_path = get_metaq_model_dir(ds_id)
    if not os.path.exists(os.path.join(save_path, 'metaq_sc.h5ad')):
        raise FileNotFoundError(f"MetaQ files not found at {save_path}. Run train_metaq() first.")

    print(f"Loading MetaQ from {save_path} ...")
    ad = sc.read_h5ad(os.path.join(save_path, 'metaq_sc.h5ad'))
    mc_ad = sc.read_h5ad(os.path.join(save_path, 'metaq_mc.h5ad'))
    lk = DATASETS[ds_id]['label_key']

    # Majority-vote niche label onto metacells
    mc_idx = ad.obs['metaq_id'].values.astype(int)
    obs = ad.obs.copy()
    obs['_mc'] = mc_idx
    _, major_niche = calc_purity(obs, label_key=nk, mc_key='_mc',
                                 return_per_mc=True, return_major_label=True)
    mc_ad.obs[nk] = mc_ad.obs.index.map(lambda x: major_niche.get(str(x)))

    summary, _ = celltype_niche_dge(ad, mc_ad, lk, nk, 'metaq', save_path)
    ct_niche_rbo_avg = float(summary.values.mean())
    std_ct_niche_rbo = float(summary.values.std(ddof=1)) if summary.values.size > 1 else 0.0
    print(f"[metaq task3] mean ct-niche RBO: {ct_niche_rbo_avg:.4f} ± {std_ct_niche_rbo:.4f}")

    metrics_path = os.path.join(save_path, 'metrics.json')
    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
    metrics['ct_niche_rbo_avg'] = ct_niche_rbo_avg
    metrics['std_ct_niche_rbo'] = std_ct_niche_rbo
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    return {'ct_niche_rbo_avg': ct_niche_rbo_avg, 'std_ct_niche_rbo': std_ct_niche_rbo}
