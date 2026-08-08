import json
import numpy as np
import pandas as pd
import scipy.sparse as sp

from interpretable_ssl.evaluation.mc_metric_utils import *
from interpretable_ssl.datasets.dataset_configs import *
from interpretable_ssl.configs.paths import get_seacell_model_dir


def get_seacell_path(ds_id, build_kernel_on="X_pca", num_prototypes=None):
    return get_seacell_model_dir(ds_id, build_kernel_on, num_prototypes=num_prototypes)



def _save_seacell_umap_data(ds_id, ad, mc_ad, build_kernel_on="X_pca", num_prototypes=None):
    """Compute joint cell+metacell UMAP and save umap_cells.csv + umap_protos.csv +
    cell_assignments.csv + proto_vectors.npy."""
    from collections import Counter

    save_path = get_seacell_path(ds_id, build_kernel_on, num_prototypes=num_prototypes)
    bk = DATASETS[ds_id].get('batch_key', None)
    lk = DATASETS[ds_id]['label_key']
    nk = DATASETS[ds_id].get('niche_key', None)

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
    for col in [lk, bk, nk]:
        if col and col in ad.obs.columns:
            cells_df[col] = ad.obs[col].values
    cells_df.to_csv(os.path.join(save_path, 'umap_cells.csv'), index=False)

    # --- cell_assignments.csv (all cells, for spatial figure join) ---
    assign_df = pd.DataFrame({'cell_id': ad.obs_names, 'metacell_id': mc_idx})
    for col in [lk, bk, nk]:
        if col and col in ad.obs.columns:
            assign_df[col] = ad.obs[col].values
    assign_df.to_csv(os.path.join(save_path, 'cell_assignments.csv'), index=False)

    # --- umap_protos.csv ---
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


def eval_seacell_task1(ds_id, build_kernel_on="X_pca", n_components=50, k_neighbors=50,
                       affinity_type='arbf', graph_dir='./graphs', num_prototypes=None):
    """Load saved SEACell files and compute task-1 metacell quality metrics."""
    from interpretable_ssl.evaluation.metric_helpers.embedding_metrics import load_seacell
    from interpretable_ssl.evaluation.mc_metric_utils import compute_task1_metrics

    save_path = get_seacell_path(ds_id, build_kernel_on, num_prototypes=num_prototypes)
    if not os.path.exists(os.path.join(save_path, 'seacell_sc.h5ad')):
        raise FileNotFoundError(f"SEACell files not found at {save_path}. Run train mode first.")

    print(f"Loading SEACell from {save_path} ...")
    ad, SEACell_ad = load_seacell(ds_id, build_kernel_on=build_kernel_on, num_prototypes=num_prototypes)

    bk = DATASETS[ds_id].get('batch_key', None)
    lk = DATASETS[ds_id]['label_key']
    nk = DATASETS[ds_id].get('niche_key')
    mc_idx = np.array([int(i.split('-')[-1]) for i in ad.obs['SEACell'].values])

    result = compute_task1_metrics(
        ad, mc_idx, lk, bk, nk, save_path, ds_id, 'seacell',
        n_components=n_components, k_neighbors=k_neighbors,
        affinity_type=affinity_type, graph_dir=graph_dir,
    )
    _save_seacell_umap_data(ds_id, ad, SEACell_ad, build_kernel_on, num_prototypes=num_prototypes)
    return result


def eval_seacell_task2(ds_id, build_kernel_on="X_pca", compute_dge=True, num_prototypes=None):
    """Compute and save task 2 metrics for SEACell metacells.

    Mirrors eval_seacell_quality() for task 1. Loads saved SEACell files,
    then computes coverage, DGE consistency, and scGraph.

    compute_dge: set False to skip DGE consistency (dge_*_avg come back None) --
    coverage and scGraph are still computed.

    Returns:
        dict with scalar summaries: coverage, dge_rbo_avg, dge_kendall_avg,
        dge_jaccard_avg, scgraph_corr_avg.
    """
    import scanpy as sc
    from interpretable_ssl.evaluation.metric_helpers.embedding_metrics import load_seacell

    save_path = get_seacell_path(ds_id, build_kernel_on, num_prototypes=num_prototypes)
    sc_file = os.path.join(save_path, 'seacell_sc.h5ad')
    if not os.path.exists(sc_file):
        raise FileNotFoundError(f"SEACell files not found at {save_path}. Run train mode first.")

    print(f"Loading SEACell from {save_path} ...")
    ad, mc_ad = load_seacell(ds_id, build_kernel_on=build_kernel_on, num_prototypes=num_prototypes)

    bk = DATASETS[ds_id].get('batch_key', None)
    lk = DATASETS[ds_id]['label_key']

    # PCA on mc_ad for scGraph
    sc.tl.pca(mc_ad)
    mc_ad.obsm['seacell_mc_pca'] = mc_ad.obsm['X_pca']

    scalars = calc_task2_metrics(ad, mc_ad, lk, bk, ['seacell_mc_pca'], 'seacell', save_path,
                                  compute_dge=compute_dge)

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
    std_ct_niche_rbo = float(summary.values.std(ddof=1)) if summary.values.size > 1 else 0.0
    print(f"[seacell task3] mean ct-niche RBO: {ct_niche_rbo_avg:.4f} ± {std_ct_niche_rbo:.4f}")

    metrics_path = os.path.join(save_path, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        metrics = {}
    metrics['ct_niche_rbo_avg'] = ct_niche_rbo_avg
    metrics['std_ct_niche_rbo'] = std_ct_niche_rbo
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    return {'ct_niche_rbo_avg': ct_niche_rbo_avg, 'std_ct_niche_rbo': std_ct_niche_rbo}


def load_dataset(ds_id):
    conf = DATASETS[ds_id]
    return (
        sc.read_h5ad(conf["path"]),
        conf.get('batch_key', None),
        conf["label_key"],
        conf["num_prototypes"],
    )


def train_seacell_spatial(
    ds_id,
    build_kernel_on='X_covet',
    affinity_type=None,
    n_SEACells=None,
    n_components=50,
    k_neighbors=50,
    graph_dir='./graphs',
    target_groups=None,
    celltype_key='celltypes',
    niche_key='niches_3D',
    covet_k=100,
    covet_n_pcs=25,
    load_seacell=False,
):
    """Run SEACell using COVET spatial embedding.

    When build_kernel_on contains 'covet', automatically computes COVET features
    (k=covet_k spatial neighbours, n_pcs=covet_n_pcs, alpha=1.0 covet-only, n_comps=auto)
    and stores them in ad.obsm[build_kernel_on] before building the kernel.

    If affinity_type is None: builds the SEACells kernel directly from build_kernel_on.
    If affinity_type is given: loads a precomputed affinity .pkl instead.

    Saves to: MODEL_DIR/{ds_id}/seacell_{build_kernel_on}/

    Args:
        ds_id:           dataset name in DATASETS (e.g. 's28nsc')
        build_kernel_on: ad.obsm key for kernel construction (default 'X_covet')
        affinity_type:   if set, load precomputed affinity .pkl instead of building from embedding
        n_SEACells:      number of metacells; defaults to DATASETS[ds_id]['num_prototypes']
        n_components:    must match save_affinity (default 50)
        k_neighbors:     must match save_affinity (default 50)
        graph_dir:       directory where affinity .pkl lives (default './graphs')
        target_groups:   list of 'celltype | niche' strings to evaluate.
                         Defaults to NSCLC_EVAL_GROUPS (all 12 biologically motivated pairs).
        celltype_key:    obs column for cell type labels
        niche_key:       obs column for niche labels
        covet_k:         spatial neighbours for COVET (default 100, sweep-optimal)
        covet_n_pcs:     PCA dims for COVET covariance (default 25, sweep-optimal)
        load_seacell:    if True, skip training and load from saved seacell_sc.h5ad.
                         Use this to recompute metrics without retraining.

    Returns:
        flat metrics dict
    """
    import scanpy as sc
    from interpretable_ssl.evaluation.spatial_immune_task import (
        compute_target_group_metrics, NSCLC_EVAL_GROUPS,
    )

    if target_groups is None:
        target_groups = NSCLC_EVAL_GROUPS

    conf = DATASETS[ds_id]
    save_dir = get_seacell_model_dir(ds_id, build_kernel_on)
    os.makedirs(save_dir, exist_ok=True)

    n_proto = n_SEACells or conf['num_prototypes']
    lk = conf['label_key']
    bk = conf.get('batch_key', None)

    # --- Load mode: skip training, read saved h5ad ---
    if load_seacell:
        sc_path = os.path.join(save_dir, 'seacell_sc.h5ad')
        if not os.path.exists(sc_path):
            raise FileNotFoundError(
                f"load_seacell=True but no saved model found at {sc_path}. "
                f"Run without load_seacell=True first."
            )
        print(f"Loading saved SEACell from {sc_path} ...")
        from interpretable_ssl.evaluation.metric_helpers.embedding_metrics import (
            load_seacell as _load_seacell,
        )
        ad, SEACell_ad = _load_seacell(ds_id, build_kernel_on=build_kernel_on)
    else:
        from interpretable_ssl.evaluation.metric_helpers.metacell_metrics import (
            compute_seacells, compute_seacells_from_affinity, agg_obs, save_seacell,
        )

        ad = sc.read_h5ad(conf['path'])
        if not conf.get('normalized', False):
            sc.pp.normalize_total(ad)
            sc.pp.log1p(ad)
        if 'X_pca' not in ad.obsm:
            sc.tl.pca(ad)

        if 'covet' in build_kernel_on.lower():
            from interpretable_ssl.augmenters.graph_generator import compute_covet_features
            print(f"Computing COVET: k={covet_k}, n_pcs={covet_n_pcs}, alpha=1.0, n_comps=auto ...")
            compute_covet_features(
                ad, k=covet_k, n_pcs=covet_n_pcs,
                alpha=1.0, n_comps=None, obsm_key=build_kernel_on,
            )

        if affinity_type is None:
            print(f"Building SEACells kernel from {build_kernel_on} ...")
            ad, SEACell_ad, model = compute_seacells(
                ad, n_SEACells=n_proto, build_kernel_on=build_kernel_on,
            )
        else:
            print(f"Loading precomputed affinity ({affinity_type}) ...")
            ad, SEACell_ad, model = compute_seacells_from_affinity(
                ad, n_SEACells=n_proto,
                ds_name=ds_id, affinity_type=affinity_type,
                n_components=n_components, k_neighbors=k_neighbors,
                graph_dir=graph_dir, build_kernel_on=build_kernel_on,
            )

        agg_obs(SEACell_ad, ad, lk)
        if bk is not None:
            agg_obs(SEACell_ad, ad, bk)

        save_seacell(ad, SEACell_ad, ds_id, build_kernel_on=build_kernel_on)
        print(f"Saved SEACell to {save_dir}")

    method_name = f'SEACell ({affinity_type or build_kernel_on})'
    metrics = compute_target_group_metrics(
        ad, mc_key='SEACell',
        target_groups=target_groups,
        celltype_key=celltype_key,
        niche_key=niche_key,
        method_name=method_name,
    )

    # --- Tumor niche evaluation (core vs surface cell type purity + niche purity) ---
    try:
        from interpretable_ssl.evaluation.spatial_immune_task import tumor_niche_metacell_eval
        tn_res = tumor_niche_metacell_eval(
            ad,
            mc_key='SEACell',
            celltype_key=celltype_key,
            niche_key=niche_key,
            plot=False,
            method_name=method_name,
        )
        metrics.update(tn_res['flat'])
        per_cell_df = tn_res.get('per_cell')
        if per_cell_df is not None and len(per_cell_df) > 0:
            csv_path = os.path.join(save_dir, 'tumor_niche_per_cell.csv')
            per_cell_df.to_csv(csv_path, index=False)
            print(f"[tumor_niche] per-cell CSV saved to {csv_path}")
    except Exception as e:
        import traceback
        print(f"Warning: tumor_niche_metacell_eval failed: {e}")
        traceback.print_exc()

    metrics['build_kernel_on'] = build_kernel_on
    metrics['affinity_type'] = affinity_type
    metrics['n_SEACells'] = n_proto

    metrics_path = os.path.join(save_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    return metrics


def train_seacell_own_affinity(
    ds_id, mode='train', affinity_type='mean_product', n_SEACells=None,
    k=50, per_space_sigma=True, n_waypoint_eigs=10, max_iter=100,
    graph_dir='./graphs',
):
    """Run SEACells' archetypal analysis on our own affinity graph (e.g.
    'mean_product'), with archetype seeding computed from that same affinity
    instead of SEACells' native palantir/X_pca waypoint init.

    See compute_seacells_own_affinity() in
    interpretable_ssl/evaluation/metric_helpers/metacell_metrics.py for what
    this skips (SEACells' own construct_kernel_matrix() and
    initialize_archetypes()) and why.

    Mirrors train_seacell()'s train/eval + load-if-exists pattern: skips
    (re)computing if a saved result already exists and mode != 'train'. The
    affinity itself is also loaded from graph_dir if already cached there
    (e.g. by another notebook/run using the same affinity_type), else built
    and saved via generate_affinity()/save_affinity().

    Saves to: MODEL_DIR/{ds_id}/seacell_{affinity_type}/

    Args:
        ds_id:            dataset key in DATASETS (e.g. 's28nsc').
        mode:             'train' to (re)compute, anything else to skip if a
                          saved result already exists (mirrors train_seacell).
        affinity_type:    generate_affinity() affinity_type tag (default
                          'mean_product').
        n_SEACells:       number of metacells; defaults to
                          DATASETS[ds_id]['num_prototypes'].
        k:                affinity graph kNN (default 50).
        per_space_sigma:  passed to generate_affinity for product affinities
                          (stricter per-space AND logic — default True).
        n_waypoint_eigs:  diffusion-map eigenvectors for archetype seeding.
        max_iter:         outer Frank-Wolfe iterations (model.fit).
        graph_dir:        directory for affinity .pkl cache (default './graphs').

    Returns:
        (ad, SEACell_ad, model) if computed, or None if skipped in eval mode.
    """
    import pickle
    import scanpy as sc
    from interpretable_ssl.evaluation.metric_helpers.metacell_metrics import (
        compute_seacells_own_affinity, agg_obs, save_seacell,
    )
    from interpretable_ssl.augmenters.graph_generator import generate_affinity, save_affinity
    from interpretable_ssl.configs.paths import get_affinity_path

    save_dir = get_seacell_model_dir(ds_id, affinity_type)
    seacell_exists = os.path.exists(os.path.join(save_dir, 'seacell_sc.h5ad'))

    if mode != 'train' and seacell_exists:
        print(f"eval mode, seacell file found at {save_dir}. "
              f"Call eval_seacell_task1/2/3(build_kernel_on='{affinity_type}') for metrics.")
        return None

    conf = DATASETS[ds_id]
    n_proto = n_SEACells or conf['num_prototypes']
    lk = conf['label_key']
    bk = conf.get('batch_key', None)

    os.makedirs(save_dir, exist_ok=True)
    ad, _, _, _ = load_dataset(ds_id)
    if ad.X.max() > 20:
        sc.pp.normalize_total(ad)
        sc.pp.log1p(ad)
    if 'X_pca' not in ad.obsm:
        sc.tl.pca(ad)

    aff_path = get_affinity_path(ds_id, len(ad), k_neighbors=k,
                                 affinity_type=affinity_type, graph_dir=graph_dir)
    if os.path.exists(aff_path):
        print(f"Loading existing affinity from {aff_path} ...")
        with open(aff_path, 'rb') as f:
            aff = pickle.load(f)
    else:
        print(f"No affinity found at {aff_path} — building '{affinity_type}' ...")
        aff = generate_affinity(ad, k=k, bk=bk, affinity_type=affinity_type,
                                per_space_sigma=per_space_sigma)
        aff.setdiag(0)
        aff.eliminate_zeros()
        save_affinity(aff, ds_id, len(ad), affinity_type=affinity_type,
                      k_neighbors=k, graph_dir=graph_dir)

    ad, SEACell_ad, model = compute_seacells_own_affinity(
        ad, n_SEACells=n_proto, aff=aff,
        n_waypoint_eigs=n_waypoint_eigs, max_iter=max_iter,
    )

    agg_obs(SEACell_ad, ad, lk)
    if bk is not None:
        agg_obs(SEACell_ad, ad, bk)

    save_seacell(ad, SEACell_ad, ds_id, build_kernel_on=affinity_type)
    print(f"Saved SEACell to {save_dir}")
    return ad, SEACell_ad, model


def train_seacell(ds_id, mode, k=50, build_kernel_on="X_pca", num_prototypes=None,
                   diffusion_affinity_type='arbf', diffusion_n_eigs=1024,
                   diffusion_t=0.5, diffusion_aff_k=50):
    """build_kernel_on='X_diffusion' runs SEACells' archetypal analysis
    directly on a rank-diffusion_n_eigs reconstruction of the
    'diffusion_affinity_type' affinity graph (default 'arbf' — the same
    PCA-based adaptive-RBF kernel SEACells(PCA) itself uses), injected via
    add_precomputed_kernel_matrix — NOT on a fresh kernel built from kNN
    distances in diffusion-coordinate space (that would be a different,
    uncontrolled kernel construction). See compute_seacells_diffusion() for
    the full reasoning. Isolates whether compacting the affinity graph to
    diffusion_n_eigs dimensions costs archetypal-analysis quality,
    independent of scProto's neural net / competing losses — see
    files/sim_recon_global_vs_local_compaction.md and
    compute_diffusion_embedding() for why diffusion_t=0.5 (not 0 or 1) is
    the setting that makes this a fair compaction test against SEACells(PCA).

    num_prototypes: None (default) uses the dataset config's own K, saved to the
    plain per-tag path exactly as before. Pass an int to override K and save to a
    '..._K{n}' path instead (get_seacell_model_dir) -- so a different K always gets
    its own directory rather than silently overwriting/reusing a differently-sized
    run under the same path.
    """
    save_path = get_seacell_path(ds_id, build_kernel_on, num_prototypes=num_prototypes)
    seacell_exists = os.path.exists(save_path + "/seacell_sc.h5ad")
    if mode == "train" or not (seacell_exists):
        from interpretable_ssl.evaluation.metric_helpers.metacell_metrics import (
            compute_seacells,
            compute_seacells_diffusion,
            agg_obs,
            save_seacell,
        )

        os.makedirs(save_path, exist_ok=True)
        # use the current dataset id, not a fixed string
        ad, bk, lk, n_proto = load_dataset(ds_id)
        if num_prototypes is not None:
            n_proto = num_prototypes
        print(len(ad))
        if ad.X.max() > 20:
            sc.pp.normalize_total(ad)
            sc.pp.log1p(ad)

        if build_kernel_on == "X_diffusion":
            from interpretable_ssl.augmenters.graph_generator import generate_affinity

            if "X_pca" not in ad.obsm:
                sc.tl.pca(ad)
            print(f"[diffusion] computing '{diffusion_affinity_type}' affinity (k={diffusion_aff_k}) ...")
            aff = generate_affinity(ad, diffusion_aff_k, bk, affinity_type=diffusion_affinity_type)
            ad, SEACell_ad, model = compute_seacells_diffusion(
                ad, n_proto, aff, n_eigs=diffusion_n_eigs, diffusion_t=diffusion_t,
                build_kernel_on=build_kernel_on,
            )
        else:
            if build_kernel_on == "X_pca" and "X_pca" not in ad.obsm:
                sc.tl.pca(ad)
            ad, SEACell_ad, model = compute_seacells(ad, n_proto, k=k, build_kernel_on=build_kernel_on)
        agg_obs(SEACell_ad, ad, lk)
        if bk is not None:
            agg_obs(SEACell_ad, ad, bk)
        save_seacell(ad, SEACell_ad, ds_id, build_kernel_on, num_prototypes=num_prototypes)
    else:
        print("eval mode, seacell file found. Call eval_seacell_quality() for metrics.")
