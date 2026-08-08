"""
Shared experiment task functions for metacell quality evaluation.

Usage in any notebook:
    from interpretable_ssl.experiments.tasks import get_trainer, run_mc_task
"""

import glob
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
    lambda_proto_usage=0.1,
    umap_similarity='proto',
    nassoc_agg='max',
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

    kwargs.setdefault('debug', 1)
    kwargs.setdefault('workers', 0)
    kwargs.setdefault('umap_min_dist', 0.5)
    kwargs.setdefault('umap_spread', 1.0)
    kwargs.setdefault('umap_neg_rate', 5)
    kwargs.setdefault('pretraining_epochs', 0)
    t = SCProtoTrainer(**kwargs)
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
    skip_eval=False,
    freeze_batch_embedding=False,
    freeze_decoder=False,
    soft_metrics=False,
    trainer_kwargs=None,
    target_groups=None,
    skip_metrics=None,
    covet_alpha=None,
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
        skip_eval:              If True (requires load_umap=True), return right after
                                loading the checkpoint -- skips eval_metacell_quality/
                                eval_task2/3, aff_dc_compactness, and save_metacells/
                                save_umap_data entirely. load_umap alone only skips
                                training; eval+save still run unconditionally without
                                this. Use when you only need the loaded trainer/model
                                (e.g. to encode cells) and don't want to re-run
                                evaluation or touch this run's saved output files.
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

    # covet_alpha: encode into affinity_type tag and expose to model naming
    if covet_alpha is not None:
        base_aff = affinity_type or lambda_config.get('affinity_type', 'covet')
        if base_aff in (None, 'covet') or base_aff.startswith('covet'):
            if covet_alpha == 1.0:
                affinity_type = 'covet'
            else:
                affinity_type = f'covet_a{int(round(covet_alpha * 10))}'
        trainer_kwargs = dict(trainer_kwargs or {})
        trainer_kwargs['covet_alpha'] = covet_alpha

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
    if skip_eval:
        # Caller only wants the loaded, trained model (e.g. to encode cells into its
        # latent) -- not a re-evaluation. Without this, load_umap=True still runs the
        # full eval+save pipeline below unconditionally on every call (that's what
        # load_umap actually skips is training, NOT eval/save), which is wasted
        # compute AND overwrites clusters.npz/metacells.h5ad in this run's own
        # directory via t.save_metacells()/t.save_umap_data() every time -- including
        # the canonical run directories whose numbers have been verified against the
        # published paper. Only valid alongside load_umap=True; there's no checkpoint
        # to skip evaluating yet on a fresh training run.
        if not load_umap:
            raise ValueError("skip_eval=True requires load_umap=True -- nothing has "
                              "been trained yet to skip evaluating.")
        return t, {}, None

    skip = set(skip_metrics or [])
    res1 = t.eval_metacell_quality(soft_metrics=soft_metrics) if 'task1' not in skip else {}
    res2 = t.eval_task2_metrics(soft_metrics=soft_metrics)    if 'task2' not in skip else {}
    res3 = t.eval_task3_metrics()                             if 'task3' not in skip else {}

    metrics = {
        'seed': getattr(t, 'seed', None),
        # Task 1
        'purity':           float(res1['purity'].mean())        if res1.get('purity') is not None else None,
        'niche_purity':     float(res1['niche_purity'].mean())  if res1.get('niche_purity') is not None else None,
        'batch_entropy':    float(res1['batch_entropy'].mean()) if res1.get('batch_entropy') is not None else None,
        'modularity':       float(res1['modularity']['modularity']) if res1.get('modularity') is not None else None,
        # Task 2
        'coverage':         res2.get('coverage'),
        'dge_rbo_avg':      res2.get('dge_rbo_avg'),
        'dge_kendall_avg':  res2.get('dge_kendall_avg'),
        'dge_jaccard_avg':  res2.get('dge_jaccard_avg'),
        'scgraph_corr_avg': res2.get('scgraph_corr_avg'),
        'scgraph_corr_std': res2.get('scgraph_corr_std'),
        # Task 3 (spatial only)
        'ct_niche_rbo_avg': res3.get('ct_niche_rbo_avg'),
    }

    # --- Attach metacell ID to original adata ---
    assignments, _ = t._get_assignments()
    t.train_ds.adata.obs['metacell_id'] = assignments

    # --- Aff-DC compactness: diffusion map on raw affinity graph ---
    if 'aff_compactness' not in skip:
        try:
            from interpretable_ssl.evaluation.mc_metric_utils import compute_aff_dc_compactness
            aff = t.train_ds.aff_raw if hasattr(t.train_ds, 'aff_raw') else t.train_ds.aff
            batches_arr = t.train_ds.adata.obs[t.train_ds.batch_key].values
            aff_comp_df, counts_df = compute_aff_dc_compactness(aff, assignments, batches_arr)
            valid_counts = counts_df.where(aff_comp_df.notna(), 0)
            per_mc_mean = (aff_comp_df.fillna(0) * valid_counts).sum(axis=1) / valid_counts.sum(axis=1)
            per_batch_mean = (aff_comp_df.fillna(0) * valid_counts).sum(axis=0) / valid_counts.sum(axis=0)
            metrics['aff_compactness_per_batch'] = {str(b): float(v) for b, v in per_batch_mean.items()}
            metrics['aff_compactness_mean'] = float(per_mc_mean.mean())
            csv_dir = result_save_path if result_save_path is not None else t.get_dump_path()
            os.makedirs(csv_dir, exist_ok=True)
            csv_path = os.path.join(csv_dir, 'aff_dc_compactness.csv')
            out_df = aff_comp_df.copy()
            out_df['weighted_mean'] = per_mc_mean
            out_df.to_csv(csv_path)
            print(f"[aff_dc_compactness] mean={metrics['aff_compactness_mean']:.4f} | saved to {csv_path}")
        except Exception as e:
            import traceback
            print(f"Warning: aff_dc_compactness failed: {e}")
            traceback.print_exc()

    # --- Group-level metacell quality (purity / coverage / homogeneity per target group) ---
    if target_groups is not None and niche_key is not None and 'group_metrics' not in skip:
        from interpretable_ssl.evaluation.spatial_immune_task import compute_target_group_metrics
        group_metrics = compute_target_group_metrics(
            t.train_ds.adata, mc_key='metacell_id',
            target_groups=target_groups,
            celltype_key=label_key, niche_key=niche_key,
            method_name='scProto',
        )
        metrics.update(group_metrics)

    # --- Tumor niche evaluation (core vs surface cell type purity + niche purity) ---
    if niche_key is not None and 'tumor_niche' not in skip:
        try:
            from interpretable_ssl.evaluation.spatial_immune_task import tumor_niche_metacell_eval
            tn_res = tumor_niche_metacell_eval(
                t.train_ds.adata,
                mc_key='metacell_id',
                celltype_key=label_key,
                niche_key=niche_key,
                plot=False,
                method_name='scProto',
            )
            metrics.update(tn_res['flat'])
            per_cell_df = tn_res.get('per_cell')
            if per_cell_df is not None and len(per_cell_df) > 0:
                csv_path = os.path.join(t.get_dump_path(), 'tumor_niche_per_cell.csv')
                os.makedirs(t.get_dump_path(), exist_ok=True)
                per_cell_df.to_csv(csv_path, index=False)
                print(f"[tumor_niche] per-cell CSV saved to {csv_path}")
        except Exception as e:
            import traceback
            print(f"Warning: tumor_niche_metacell_eval failed: {e}")
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


# ---------------------------------------------------------------------------
# Ablation-variant runner
#
# Shared by the component-ablation rebuttal notebooks: trains/reloads one
# variant of a lambda-config sweep through run_mc_task and (optionally)
# attaches prototype-level diagnostics on top of the usual metrics dict.
# ---------------------------------------------------------------------------

def _checkpoint_exists(ds_id, experiment_name):
    """Cheap existence check via directory-name prefix match -- experiment_name
    is always a literal prefix of the saved model dir name (model_name.py),
    so this avoids replicating the full abbreviation-based name-generation
    logic just to check whether a checkpoint is already on disk.
    """
    pattern = os.path.join(os.environ['MODEL_DIR'], ds_id, f'{experiment_name}_*', 'umap_checkpoint.pth')
    return len(glob.glob(pattern)) > 0


def run_ablation_variant(ds_id, tag, overrides, base_lambda_config, common_kwargs,
                          variant_display_names=None, load_umap=None,
                          experiment_prefix='ablation', extra_diagnostics=True):
    """One (dataset, variant) run: base_lambda_config with exactly one key
    overridden, trained/evaluated via the same run_mc_task pipeline the
    paper's own numbers come from. experiment_name='{experiment_prefix}_{tag}'
    keeps each variant's dump folder distinct and keyword-matchable in a
    results section.

    load_umap=None (default): auto-detect -- reload the existing checkpoint if
    one is already saved for this (ds_id, tag), else train fresh. Pass True/False
    explicitly to force one behavior regardless of what's on disk.

    extra_diagnostics=True: also attach prototype_redundancy /
    active_prototype_count (nassoc/usage-loss diagnostics) to the result dict.
    Set False for arms where these aren't meaningful (e.g. spatial ablations).
    """
    lambda_config = dict(base_lambda_config)
    lambda_config.update(overrides)
    experiment_name = f'{experiment_prefix}_{tag}'
    if load_umap is None:
        load_umap = _checkpoint_exists(ds_id, experiment_name)
    display = variant_display_names[tag] if variant_display_names else tag
    print(f"\n=== [{ds_id}] ablation variant: {tag} ({display}) ==="
          f"  [{'reloading existing checkpoint' if load_umap else 'training fresh'}]")

    t, res, mc_ad = run_mc_task(
        ds_id,
        lambda_config=lambda_config,
        trainer_kwargs={'experiment_name': experiment_name},
        load_umap=load_umap,
        **common_kwargs,
    )
    res = dict(res)
    if extra_diagnostics:
        from interpretable_ssl.evaluation.trainer_diagnostics import (
            prototype_redundancy, active_prototype_count,
        )
        res.update(prototype_redundancy(t))
        res.update(active_prototype_count(t))
    return t, res, mc_ad


def run_all_variants_for_dataset(ds_id, ablations, base_lambda_config, common_kwargs,
                                  variant_display_names=None, load_umap=None,
                                  experiment_prefix='ablation', extra_diagnostics=True):
    """Runs every entry of `ablations` for one dataset. Each variant is
    independent -- safe to re-run just one by calling run_ablation_variant
    directly if a single arm needs to change. load_umap=None (default):
    auto-detect per arm (see run_ablation_variant) -- pass True/False to force
    one behavior for every arm regardless of what's on disk.
    """
    trainers, results, mc_adatas = {}, {}, {}
    for tag, overrides in ablations.items():
        t, res, mc_ad = run_ablation_variant(
            ds_id, tag, overrides, base_lambda_config, common_kwargs,
            variant_display_names=variant_display_names, load_umap=load_umap,
            experiment_prefix=experiment_prefix, extra_diagnostics=extra_diagnostics,
        )
        trainers[tag], results[tag], mc_adatas[tag] = t, res, mc_ad
    return trainers, results, mc_adatas


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
