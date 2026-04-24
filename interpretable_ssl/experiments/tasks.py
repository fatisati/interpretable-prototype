"""
Shared experiment task functions for metacell quality evaluation.

Usage in any notebook:
    from interpretable_ssl.experiments.tasks import get_trainer, run_mc_task
"""

import os


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
    import constants
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

def run_mc_task(
    ds_id,
    cvae_epochs,
    train_epochs,
    eval_freq,
    patience,
    batch_size=256,
    lambda_config=None,
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
        ds_id:                  Dataset ID (e.g. 'pancreas', 'pbmc-immune').
        cvae_epochs:            Epochs for CVAE pre-training.
        train_epochs:           Max UMAP training epochs (early stopping applies).
        eval_freq:              Evaluate modularity every N epochs.
        patience:               Early stopping patience (epochs).
        batch_size:             Batch size — must match the pretrain checkpoint you
                                want to reuse (pancreas=256, pbmc-immune=512).
        lambda_config:          Dict of lambda values. Defaults to LAMBDA_PROTO_UMAP.
                                Pass LAMBDA_RECON_ONLY or a custom dict to override.
        min_delta:              Minimum modularity improvement to reset patience.
        umap_steps_per_epoch:   Gradient steps per epoch (caps dataset size).
        load_pretrain:          If True, load existing pretrain checkpoint if available.
        load_umap:              If True, skip training and just load the UMAP checkpoint.
        freeze_batch_embedding: If True, freeze batch embedding params before UMAP training.
        trainer_kwargs:         Extra kwargs forwarded to get_trainer.

    Returns:
        (trainer, metrics) where metrics is a flat dict:
            {'purity': float, 'batch_entropy': float, 'modularity': float}
    """
    if lambda_config is None:
        lambda_config = LAMBDA_PROTO_UMAP

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
    return t, metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
