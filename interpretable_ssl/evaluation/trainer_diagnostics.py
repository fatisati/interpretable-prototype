"""
Post-hoc diagnostics computed directly from a live (trained) SCProtoTrainer.

Unlike the ds_id/keyword-driven figure and table helpers in paper_figures.py
and metric_helpers/, these take a trainer instance `t` straight out of
run_mc_task / find_metacells and read its in-memory model + on-disk dump
CSVs -- no reload-from-checkpoint step.
"""

import os

import numpy as np
import pandas as pd
import torch


def prototype_redundancy(t):
    """Mean/max pairwise cosine similarity between prototype vectors -- cheap
    embedding-space proxy for nassoc's off-diagonal redundancy term. NOT the
    exact batch-aware M_kj the loss itself optimizes (that needs the per-batch
    affinity graph W^(b)) -- treat this as a directional diagnostic, not a
    like-for-like recomputation of L_nassoc.
    """
    protos = t.model.get_prototypes().detach().cpu()
    protos = torch.nn.functional.normalize(protos, dim=1)
    sim = (protos @ protos.T).numpy()
    K = sim.shape[0]
    off_diag = sim[~np.eye(K, dtype=bool)]
    return {
        'proto_cosine_sim_mean': float(off_diag.mean()),
        'proto_cosine_sim_max': float(off_diag.max()),
    }


def active_prototype_count(t, min_cells=1):
    """Fraction of prototypes with >= min_cells assigned cells -- diagnostic
    for the usage loss's anti-collapse claim (dead prototypes = 0 assigned
    cells when the loss that prevents this is turned off).
    """
    K = t.model.get_prototypes().shape[0]
    counts = t.train_ds.adata.obs['metacell_id'].value_counts()
    n_active = int((counts >= min_cells).sum())
    return {'n_active_prototypes': n_active, 'K': int(K), 'active_frac': n_active / K}


def per_batch_variance(t):
    """Mean +/- std for purity/batch_entropy (size-weighted across metacells,
    matching eval_metacell_quality's own weighted_std formula) and modularity
    (unweighted across batches, matching Table 1's convention). Reads CSVs
    eval_metacell_quality already saves to disk for every run -- no retraining,
    no change to any existing loss/metric/pipeline code.
    """
    dump = t.get_dump_path()
    out = {}

    size_path = os.path.join(dump, 'size_per_mc.csv')
    size = pd.read_csv(size_path).set_index('metacell')['size'] if os.path.exists(size_path) else None

    for name, fname, col in [
        ('purity', 'purity_per_mc.csv', 'purity'),
        ('batch_entropy', 'batch_entropy_per_mc.csv', 'batch_entropy'),
    ]:
        path = os.path.join(dump, fname)
        if not os.path.exists(path) or size is None:
            continue
        s = pd.read_csv(path).set_index('metacell')[col]
        w = size.reindex(s.index).fillna(0)
        w_sum = w.sum()
        wmean = float((s * w).sum() / w_sum)
        wstd = float(np.sqrt(((s - wmean) ** 2 * w).sum() / w_sum))
        out[f'{name}_weighted_mean'] = wmean
        out[f'{name}_weighted_std'] = wstd

    mod_path = os.path.join(dump, 'modularity_per_batch.csv')
    if os.path.exists(mod_path):
        mod = pd.read_csv(mod_path)['modularity']
        out['modularity_mean'] = float(mod.mean())
        out['modularity_std'] = float(mod.std())

    return out


def format_mean_std(row, mean_col, std_col):
    """Render a `per_batch_variance` mean/std column pair as 'X.XXX ± Y.YYY'
    for display in a results table. Returns None if the mean column is
    missing or NaN for this row (e.g. niche_purity on a non-spatial dataset).
    """
    if mean_col not in row or pd.isna(row[mean_col]):
        return None
    return f"{row[mean_col]:.3f} ± {row[std_col]:.3f}"


def macro_celltype_purity(t, ds_id):
    """New metric, not in method.tex/results.tex: per-cell same-type fraction
    within its own metacell, meaned per cell type, then meaned unweighted
    across cell types -- gives rare types equal voice, unlike headline
    `purity` (cell-count weighted). Computed post-hoc from existing
    assignments only; does not touch any existing loss/metric/pipeline code.
    """
    from interpretable_ssl.datasets.dataset_configs import DATASETS

    lk = DATASETS[ds_id]['label_key']
    obs = t.train_ds.adata.obs[['metacell_id', lk]].copy()
    obs.columns = ['metacell_id', 'celltype']
    mc_type_counts = (obs.groupby(['metacell_id', 'celltype']).size()
                         .rename('n_same').reset_index())
    mc_size = obs.groupby('metacell_id').size().rename('mc_size')
    merged = obs.merge(mc_type_counts, on=['metacell_id', 'celltype'], how='left')
    merged = merged.merge(mc_size, on='metacell_id', how='left')
    merged['celltype_purity'] = merged['n_same'] / merged['mc_size']
    per_ct_mean = merged.groupby('celltype')['celltype_purity'].mean()
    return {'macro_celltype_purity': float(per_ct_mean.mean())}
