"""
Shared experiment task functions for metacell quality evaluation.

Usage in any notebook:
    from interpretable_ssl.experiments.tasks import get_trainer, run_mc_task
"""

import os
import json


# ---------------------------------------------------------------------------
# Lambda configs — pass one of these (or override individual keys) to run_mc_task
# ---------------------------------------------------------------------------

LAMBDA_PROTO_UMAP = dict(
    lambda_umap=1,
    lambda_swav=0,
    lambda_kl=0,
    lambda_recon=0,
    lambda_proto_recon=0.0,
    umap_similarity = 'proto'
)

LAMBDA_PARAM_UMAP = dict(
    lambda_umap=1,
    lambda_swav=0,
    lambda_kl=0,
    lambda_recon=0,
    lambda_proto_recon=0.0,
    umap_similarity='embedding',
)

LAMBDA_PROTO_UMAP_PRECON = dict(
    lambda_umap=1,
    lambda_swav=0,
    lambda_kl=0,
    lambda_recon=0,
    lambda_proto_recon=0.01,
    calibrate_eps=1,
    umap_proto_metric='dotp',
    prot_init='waypoint',
    lambda_nassoc=1,
    usage_norm_sim=0,
    proto_usage_mode='ema',
    lambda_proto_usage = 0.1,
    umap_similarity='proto',
    nassoc_agg = 'max'
)

LAMBDA_RECON_ONLY = dict(
    lambda_umap=0,
    lambda_swav=0,
    lambda_kl=0,
    lambda_recon=1,
    lambda_proto_recon=0.0,
)

LAMBDA_PROTO_RECON_ONLY = dict(
    lambda_umap=0,
    lambda_swav=0,
    lambda_kl=0,
    lambda_recon=0,
    lambda_proto_recon=1.0,
)

LAMBDA_PROTO_CTX_UMAP = dict(
    lambda_umap=1,
    lambda_swav=0,
    lambda_kl=0,
    lambda_recon=0,
    lambda_proto_recon=0.01,
    affinity_type='ctx_umap',
)


# ---------------------------------------------------------------------------
# Trainer factory
# ---------------------------------------------------------------------------

def get_trainer(**kwargs):
    """Create and set up an SCProtoTrainer with sensible defaults.

    Any kwarg overrides the defaults, including lambda values.
    """
    import importlib, sys
    from importlib import reload

    def _reload_interpretable_ssl():
        for m in list(sys.modules):
            if m.startswith("interpretable_ssl") and "dataset_configs" not in m:
                importlib.reload(sys.modules[m])

    import interpretable_ssl.configs.defaults
    reload(interpretable_ssl.configs.defaults)
    import interpretable_ssl.constants as constants
    reload(constants)
    _reload_interpretable_ssl()

    from interpretable_ssl.trainers.scproto import SCProtoTrainer

    t = SCProtoTrainer(
        debug=1,
        workers=0,
        umap_min_dist=0.5,
        umap_spread=1.0,
        umap_neg_rate=5,
        pretraining_epochs=0,
        **kwargs
    )
    print(t.get_model_name())
    t.setup()
    return t


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def find_metacells(
    ds_id,
    cvae_epochs,
    train_epochs,
    eval_freq,
    patience,
    batch_size=256,
    lambda_config=None,
    batch_key=None,
    label_key=None,
    niche_key=None,
    num_prototypes=None,
    affinity_type=None,
    result_save_path=None,
    min_delta=0.005,
    umap_steps_per_epoch=1000,
    load_pretrain=True,
    load_umap=False,
    freeze_batch_embedding=False,
    freeze_decoder=False,
    soft_metrics=False,
    trainer_kwargs=None,
):
    """Train and evaluate metacell quality for one dataset.

    Args:
        ds_id:                  Dataset ID (e.g. 'snsc') OR path to an .h5ad file.
        cvae_epochs:            Epochs for CVAE pre-training.
        train_epochs:           Max UMAP training epochs (early stopping applies).
        eval_freq:              Evaluate modularity every N epochs.
        patience:               Early stopping patience (epochs).
        batch_size:             Batch size.
        lambda_config:          Dict of lambda/training values. Defaults to
                                LAMBDA_PROTO_UMAP_PRECON.
        batch_key:              Column in adata.obs with batch labels. If None,
                                all cells are treated as one batch.
        label_key:              Column in adata.obs with cell-type labels.
                                Used only for evaluation metrics, not training.
        niche_key:              Column in adata.obs with niche annotations.
                                Optional — only needed for spatial niche metrics.
        num_prototypes:         Number of prototypes. Options:
                                  None            — use the dataset config default.
                                  int             — use this exact count.
                                  'lbat'          — floor(largest_batch_n / ratio) where
                                                    ratio = total_n / default_np from config.
        affinity_type:          Graph affinity type (e.g. 'arbf', 'ctx_umap').
                                Shortcut for passing via lambda_config.
        result_save_path:       If set, saves metrics as metrics.json here.
        min_delta:              Minimum modularity improvement to reset patience.
        umap_steps_per_epoch:   Gradient steps per epoch (caps dataset size).
        load_pretrain:          If True, reuse existing pretrain checkpoint if found.
        load_umap:              If True, skip training and load the UMAP checkpoint.
        freeze_batch_embedding: If True, freeze batch embedding before UMAP training.
        freeze_decoder:         If True, freeze decoder before UMAP training.
        trainer_kwargs:         Extra kwargs forwarded to get_trainer.

    Returns:
        (trainer, metrics) where metrics is a flat dict of evaluation scores.
    """
    lambda_config = dict(lambda_config) if lambda_config is not None else dict(LAMBDA_PROTO_UMAP_PRECON)

    # --- accept AnnData directly: write to temp file, treat as path ---
    import anndata
    if isinstance(ds_id, anndata.AnnData):
        import tempfile
        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, 'input.h5ad')
        ds_id.write_h5ad(tmp_path)
        ds_id = tmp_path

    # --- resolve ds_id: known string or h5ad path ---
    ds_id = _resolve_ds_id(ds_id, batch_key=batch_key, label_key=label_key, niche_key=niche_key, num_prototypes=num_prototypes)

    # Resolve num_prototypes (int / 'largest_batch' / None) to a concrete int or None
    resolved_num_prototypes = _resolve_num_prototypes(num_prototypes, ds_id)

    # For known datasets, fall back to the config's label_key
    if label_key is None:
        from interpretable_ssl.datasets.dataset_configs import DATASETS
        label_key = DATASETS.get(ds_id, {}).get('label_key')
    if label_key is None:
        raise ValueError("label_key is required — pass the adata.obs column name with cell-type labels.")

    if affinity_type is not None:
        lambda_config['affinity_type'] = affinity_type

    trainer_kwargs = trainer_kwargs or {}
    experiment_name = trainer_kwargs.pop('experiment_name', _infer_experiment_name(lambda_config))
    freeze_batch_embedding = int(lambda_config.pop('freeze_batch_embedding', freeze_batch_embedding))
    freeze_decoder = int(lambda_config.pop('freeze_decoder', freeze_decoder))
    # num_prototypes may be passed inside lambda_config; pop it and let it win over the direct param
    if 'num_prototypes' in lambda_config:
        num_prototypes = lambda_config.pop('num_prototypes')
        resolved_num_prototypes = _resolve_num_prototypes(num_prototypes, ds_id)
    t = get_trainer(
        experiment_name=experiment_name,
        cvae_epochs=cvae_epochs,
        affinity_type=lambda_config.get('affinity_type', 'arbf'),
        dataset_id=ds_id,
        l2norm=1,
        assignment_metric='dotp',
        batch_size=batch_size,
        umap_steps_per_epoch=umap_steps_per_epoch,
        umap_similarity=lambda_config.get('umap_similarity', 'proto'),
        freeze_batch_embedding=freeze_batch_embedding,
        freeze_decoder=freeze_decoder,
        **({} if resolved_num_prototypes is None else {'num_prototypes': resolved_num_prototypes}),
        **{k: v for k, v in lambda_config.items() if k not in ('umap_similarity', 'affinity_type')},
        **trainer_kwargs,
    )

    # --- Pretrain encoder ---
    if not load_umap:
        ckpt_path = os.path.join(t.get_pretrain_dump_path(), 'pretrain_checkpoint.pth')
        if load_pretrain and os.path.exists(ckpt_path):
            t.load_pretrain_checkpoint()
        else:
            t.pretrain_encoder()
            t.init_prototypes()
            t.save_pretrain_checkpoint()

        if freeze_batch_embedding and hasattr(t.model, 'freeze_batch_embedding'):
            t.model.freeze_batch_embedding()

        if freeze_decoder and hasattr(t.model, 'freeze_decoder'):
            t.model.freeze_decoder()

        # --- Train UMAP ---
        t.train_umap_edges(
            early_stop=True,
            early_stop_metric='modularity',
            eval_freq=eval_freq,
            patience=patience,
            max_epochs=train_epochs,
            min_delta=min_delta,
        )

    # --- Eval ---
    t.load_umap_checkpoint()
    res1 = t.eval_metacell_quality(soft_metrics=soft_metrics)
    res2 = t.eval_task2_metrics(soft_metrics=soft_metrics)
    res3 = t.eval_task3_metrics()

    metrics = {
        # Task 1
        'purity':             float(res1['purity'].mean())        if res1['purity'] is not None else None,
        'niche_purity':       float(res1['niche_purity'].mean())  if res1['niche_purity'] is not None else None,
        'batch_entropy':      float(res1['batch_entropy'].mean()) if res1['batch_entropy'] is not None else None,
        'modularity':         float(res1['modularity']['modularity']),
        # Task 2
        'coverage':           res2['coverage'],
        'dge_rbo_avg':        res2['dge_rbo_avg'],
        'dge_kendall_avg':    res2['dge_kendall_avg'],
        'dge_jaccard_avg':    res2['dge_jaccard_avg'],
        'scgraph_corr_avg':   res2['scgraph_corr_avg'],
        # Task 3 (spatial only)
        'ct_niche_rbo_avg':   res3.get('ct_niche_rbo_avg'),
    }

    # --- Attach metacell ID to original adata ---
    assignments, _ = t._get_assignments()
    t.train_ds.adata.obs['metacell_id'] = assignments

    # --- Aff-DC compactness: diffusion map on raw affinity graph ---
    try:
        from interpretable_ssl.evaluation.mc_metric_utils import compute_aff_dc_compactness
        aff = t.train_ds.aff_raw if hasattr(t.train_ds, 'aff_raw') else t.train_ds.aff
        batches_arr = t.train_ds.adata.obs[t.train_ds.batch_key].values
        # comp_df/counts_df shape: (n_metacells, n_batches)
        aff_comp_df, counts_df = compute_aff_dc_compactness(aff, assignments, batches_arr)
        # per-metacell compactness: weighted mean over batches by cell count
        # zero out counts where compactness is NaN (< 2 cells), then weighted sum / total cells
        valid_counts = counts_df.where(aff_comp_df.notna(), 0)
        per_mc_mean = (aff_comp_df.fillna(0) * valid_counts).sum(axis=1) / valid_counts.sum(axis=1)
        # per-batch mean: weighted by cell count across metacells
        per_batch_mean = (aff_comp_df.fillna(0) * valid_counts).sum(axis=0) / valid_counts.sum(axis=0)
        metrics['aff_compactness_per_batch'] = {str(b): float(v) for b, v in per_batch_mean.items()}
        metrics['aff_compactness_mean'] = float(per_mc_mean.mean())
        # save CSV: per-batch compactness columns + weighted_mean column for comparison
        csv_dir = result_save_path if result_save_path is not None else t.get_dump_path()
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = os.path.join(csv_dir, 'aff_dc_compactness.csv')
        out_df = aff_comp_df.copy()
        out_df['weighted_mean'] = per_mc_mean  # overall metacell compactness for comparison
        out_df.to_csv(csv_path)
        print(f"[aff_dc_compactness] mean={metrics['aff_compactness_mean']:.4f} | saved to {csv_path}")
    except Exception as e:
        import traceback
        print(f"Warning: aff_dc_compactness failed: {e}")
        traceback.print_exc()

    # --- Get metacell gene expression AnnData ---
    mc_adata = t.save_metacells()

    if result_save_path is not None:
        os.makedirs(result_save_path, exist_ok=True)
        out = os.path.join(result_save_path, 'metrics.json')
        with open(out, 'w') as f:
            json.dump({k: v for k, v in metrics.items() if v is not None}, f, indent=2)
        print(f"Metrics saved to {out}")
        mc_adata.write_h5ad(os.path.join(result_save_path, 'metacells.h5ad'))

    t.save_umap_data()

    return t, metrics, mc_adata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

run_mc_task = find_metacells  # backward-compatible alias


def _resolve_ds_id(ds_id, batch_key, label_key, niche_key, num_prototypes):
    """Return a valid DATASETS key, registering a custom h5ad if needed."""
    import scanpy as sc
    from pathlib import Path
    from interpretable_ssl.datasets.dataset_configs import DATASETS, register_dataset

    ds_id = str(ds_id)

    # Known dataset ID
    if ds_id in DATASETS:
        return ds_id

    # Must be a file path then
    if not os.path.isfile(ds_id):
        known = sorted(DATASETS.keys())
        raise ValueError(
            f"'{ds_id}' is not a known dataset ID and is not a file path.\n"
            f"Known IDs: {known}"
        )

    path = ds_id
    name = Path(path).stem

    adata = sc.read_h5ad(path)
    n_cells = len(adata)

    # Default batch_key: add a dummy column so scPoli doesn't fail
    if batch_key is None:
        batch_key = '_batch'
        adata.obs['_batch'] = 'batch_0'
        # save modified adata next to original so the path-based loader picks it up
        tmp_path = str(Path(path).parent / f"{name}_ssl_tmp.h5ad")
        adata.write_h5ad(tmp_path)
        path = tmp_path
        print(f"No batch_key given — treating all {n_cells} cells as one batch.")
    elif batch_key not in adata.obs.columns:
        raise ValueError(f"batch_key '{batch_key}' not found in adata.obs columns: {list(adata.obs.columns)}")

    if num_prototypes is None:
        num_prototypes = max(10, n_cells // 100)
        print(f"num_prototypes not set — using {num_prototypes} ({n_cells} cells // 100).")

    register_dataset(name, path, batch_key=batch_key, label_key=label_key, niche_key=niche_key, num_prototypes=num_prototypes)
    print(f"Registered dataset '{name}': {n_cells} cells, {num_prototypes} prototypes, batch_key='{batch_key}'.")
    return name


def _resolve_num_prototypes(num_prototypes, ds_id):
    """Resolve num_prototypes to a concrete int.

    Accepts:
        None            → return None (trainer uses dataset config default)
        int             → use as-is
        'largest_batch' → floor(largest_batch_n_cells / (total_n_cells / default_num_prototypes))
    """
    if num_prototypes is None:
        return None
    if isinstance(num_prototypes, int):
        return num_prototypes
    if num_prototypes == 'lbat':
        import math
        import anndata
        from interpretable_ssl.datasets.dataset_configs import DATASETS
        cfg = DATASETS[ds_id]
        default_np = cfg['num_prototypes']
        path = str(cfg['path'])
        batch_key = cfg.get('batch_key')
        adata = anndata.read_h5ad(path, backed='r')
        n_cells = adata.n_obs
        ratio = n_cells / default_np  # cells per prototype
        if batch_key and batch_key in adata.obs.columns:
            largest_batch_n = int(adata.obs[batch_key].value_counts().iloc[0])
        else:
            largest_batch_n = n_cells
        adata.file.close()
        resolved = max(1, math.floor(largest_batch_n / ratio))
        print(
            f"num_prototypes='lbat': total={n_cells}, default_np={default_np}, "
            f"ratio={ratio:.1f} cells/proto, largest_batch={largest_batch_n} → {resolved} prototypes"
        )
        return resolved
    raise ValueError(
        f"Unknown num_prototypes value '{num_prototypes}'. "
        "Use an int or 'lbat'."
    )


def _infer_experiment_name(lambda_config):
    if lambda_config.get('lambda_proto_recon', 0) > 0 and lambda_config.get('lambda_umap', 0) == 0 and lambda_config.get('lambda_recon', 0) == 0:
        return 'proto_recon_only'
    if lambda_config.get('lambda_recon', 0) > 0 and lambda_config.get('lambda_umap', 0) == 0:
        return 'recon_only'
    if lambda_config.get('lambda_umap', 0) > 0 and lambda_config.get('lambda_recon', 0) == 0:
        if lambda_config.get('umap_similarity') == 'embedding':
            return 'param_umap'
        return 'proto_umap'
    return 'custom'
