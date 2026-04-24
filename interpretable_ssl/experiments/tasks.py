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
    usage_norm_sim=4,
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
            if m.startswith("interpretable_ssl"):
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
        num_prototypes:         Number of prototypes. If None, defaults to
                                n_cells // 100.
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
    if label_key is None:
        raise ValueError("label_key is required — pass the adata.obs column name with cell-type labels.")

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

    if affinity_type is not None:
        lambda_config['affinity_type'] = affinity_type

    trainer_kwargs = trainer_kwargs or {}
    experiment_name = trainer_kwargs.pop('experiment_name', _infer_experiment_name(lambda_config))
    freeze_batch_embedding = int(lambda_config.pop('freeze_batch_embedding', freeze_batch_embedding))
    freeze_decoder = int(lambda_config.pop('freeze_decoder', freeze_decoder))
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
    res1 = t.eval_metacell_quality()
    res2 = t.eval_task2_metrics()
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

    # --- Get metacell gene expression AnnData ---
    mc_adata = t.save_metacells()

    if result_save_path is not None:
        os.makedirs(result_save_path, exist_ok=True)
        out = os.path.join(result_save_path, 'metrics.json')
        with open(out, 'w') as f:
            json.dump({k: v for k, v in metrics.items() if v is not None}, f, indent=2)
        print(f"Metrics saved to {out}")
        mc_adata.write_h5ad(os.path.join(result_save_path, 'metacells.h5ad'))

    return t, res, mc_adata


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
