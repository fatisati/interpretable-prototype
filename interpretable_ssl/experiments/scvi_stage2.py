"""Rebuttal experiment: what does scProto's Stage-2 loss add, holding Stage 1 fixed?

One pretrained scVI encoder, two continuations:

    arm A  "scVI + clustering"   Leiden / SEACells on scVI's latent
    arm B  "scVI + scProto"      scProto Stage 2 keeps training that same encoder

Both arms come from the SAME scVI weights in the SAME process, are scored at the same
K, on the same affinity graph, with the same metric code -- so the difference between
them isolates the Stage-2 objective, with no scPoli anywhere in the picture. That is
the direct answer to (i) "the cross-batch alignment seems to come mainly from the
Stage-1 scPoli-style pretraining" and (ii) "the most important baseline is missing:
batch-correct first, then run an existing graph-based metacell method".

Everything is resume-safe: scVI weights, the Stage-1 model checkpoint, the Stage-2
checkpoint and both baselines' metrics are written to disk and reloaded if present, so
a Colab disconnect never costs more than the epochs since the last evaluation.

Typical use (see notebooks/scvi_stage2_pancreas.ipynb):

    from interpretable_ssl.experiments.scvi_stage2 import run_scvi_stage2_experiment
    res = run_scvi_stage2_experiment('pancreas', stage2_max_epochs=20, scvi_epochs=50)
"""

import json
import os

import numpy as np
import pandas as pd

from interpretable_ssl.configs.paths import get_dataset_model_dir
from interpretable_ssl.datasets.dataset_configs import DATASETS


# Tag used for arm A's embedding and for both baseline run directories:
#   {MODEL_DIR}/{ds}/seacell_X_scvidef  and  {MODEL_DIR}/{ds}/leiden_X_scvidef_K{K}
# Deliberately distinct from the existing 'X_scvi' (d=8, ZINB) and 'X_scvi_gauss'
# (d=8, Gaussian) baselines already on disk, so nothing collides or silently reuses
# another experiment's cached run.
SCVI_DEF_TAG = "X_scvidef"


# ---------------------------------------------------------------------------
# trainer factory
# ---------------------------------------------------------------------------


def get_scvi_proto_trainer(ds_id, scvi_epochs=50, scvi_n_latent=10,
                           scvi_gene_likelihood="zinb", batch_size=1024,
                           umap_steps_per_epoch=500, lambda_config=None,
                           experiment_name=None, **kwargs):
    """Build a `ScviProtoTrainer` with the paper's Stage-2 configuration.

    Defaults are scVI's own, not scProto's: `n_latent=10` and `gene_likelihood='zinb'`
    are what `scvi.model.SCVI` uses out of the box. That is the point of this
    experiment -- the reviewer's objection to the earlier comparison was that scVI had
    been pushed to a non-standard configuration (d=8, Gaussian likelihood) to match
    scProto, so here scVI keeps its own settings and scProto's Stage 2 adapts to them
    (the prototype layer is built at scVI's latent dimension).

    lambda_config defaults to LAMBDA_PROTO_UMAP_PRECON -- the exact Stage-2 objective
    the paper's headline numbers use, unchanged.
    """
    from interpretable_ssl.experiments.tasks import LAMBDA_PROTO_UMAP_PRECON
    from interpretable_ssl.trainers.scvi_proto import ScviProtoTrainer

    lambda_config = dict(lambda_config or LAMBDA_PROTO_UMAP_PRECON)
    affinity_type = lambda_config.pop("affinity_type", "arbf")
    umap_similarity = lambda_config.pop("umap_similarity", "proto")

    t = ScviProtoTrainer(
        dataset_id=ds_id,
        experiment_name=experiment_name or "scviproto",
        debug=1,
        workers=0,
        cvae_epochs=0,               # Stage 1 is scVI, not scPoli's cVAE
        pretraining_epochs=0,
        latent_dims=scvi_n_latent,
        scvi_epochs=scvi_epochs,
        scvi_gene_likelihood=scvi_gene_likelihood,
        l2norm=1,
        assignment_metric="dotp",
        affinity_type=affinity_type,
        umap_similarity=umap_similarity,
        batch_size=batch_size,
        umap_steps_per_epoch=umap_steps_per_epoch,
        umap_min_dist=0.5,
        umap_spread=1.0,
        umap_neg_rate=5,
        **lambda_config,
        **kwargs,
    )
    print(f"[scvi stage2] run dir: {t.get_dump_path()}")
    t.setup()
    return t


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------


def run_stage1(t, force_retrain=False):
    """Train or reload scVI and leave its weights in the Stage-2 model.

    Two levels of cache, checked in order: the Stage-1 model checkpoint (fastest --
    one torch.load), then scVI's own save directory (reloads scVI, no training). Only
    a complete miss trains.
    """
    ckpt = os.path.join(t.get_pretrain_dump_path(), "pretrain_checkpoint.pth")
    if os.path.exists(ckpt) and not force_retrain:
        t.load_pretrain_checkpoint()
        return t

    t.pretrain_encoder()
    t.save_pretrain_checkpoint()
    return t


# ---------------------------------------------------------------------------
# arm A -- scVI latent -> Leiden / SEACells
# ---------------------------------------------------------------------------


def get_or_build_latent_affinity(ds_id, ad, tag, use_cache=True):
    """The adaptive-RBF graph on `ad.obsm[tag]`, cached on disk.

    Building it takes minutes on these datasets and it is deterministic given the
    embedding, yet it was being rebuilt on every session -- including sessions where
    both clusterers hit their caches and never looked at the graph at all. Stored
    under the same `_cache_X_{tag}_aff.npz` convention the other batch-correction
    baselines use, so a resumed run reloads it in seconds.
    """
    import scipy.sparse as sp
    from interpretable_ssl.evaluation.batch_correct_baselines import (
        build_latent_affinity, _cache_paths,
    )

    method = tag[2:] if tag.startswith("X_") else tag
    _, aff_path = _cache_paths(ds_id, method)
    if use_cache and os.path.exists(aff_path):
        print(f"[{ds_id}] {tag}: reusing cached affinity graph at {aff_path}")
        return sp.load_npz(aff_path)

    aff = build_latent_affinity(ad, latent_key=tag)
    try:
        sp.save_npz(aff_path, sp.csr_matrix(aff))
        print(f"[{ds_id}] {tag}: cached affinity graph to {aff_path}")
    except Exception as e:  # noqa: BLE001 -- caching is an optimisation, never fatal
        print(f"[{ds_id}] {tag}: could not cache affinity ({type(e).__name__}: {e})")
    return aff


def run_scvi_clustering_baselines(t, z_scvi, tag=SCVI_DEF_TAG, K=None,
                                  run_seacells=True, run_leiden=True,
                                  skip_if_exists=True):
    """Leiden + SEACells on the pretrained scVI latent -- the reviewer's
    "batch-correct first, then run a graph-based metacell method" baseline.

    Reuses the exact functions the other batch-correction baselines in this rebuttal
    already go through (`batch_correct_baselines.run_seacells_on_latent` /
    `run_leiden_on_latent`), so these numbers are directly comparable to the Harmony /
    ComBat / BBKNN / scVI(Gaussian) tables: one adaptive-RBF affinity graph built on
    the embedding and shared by both clusterers, K matched to the prototype count, and
    modularity always rescored against the canonical ARBF-on-PCA graph.
    """
    from interpretable_ssl.evaluation.batch_correct_baselines import (
        run_leiden_on_latent, run_seacells_on_latent, rare_type_knn_purity,
    )

    ds_id = t.dataset_id
    ad = t.train_ds.adata
    lk = DATASETS[ds_id]["label_key"]
    K = K or t.num_prototypes

    if "X_pca" not in ad.obsm:
        # Only used for UMAP coordinates and as compute_task1_metrics' modularity
        # fallback (immediately overwritten by the canonical rescoring below) -- the
        # clustering itself runs on the scVI-latent affinity graph.
        import scanpy as sc
        sc.pp.pca(ad, n_comps=50)

    ad.obsm[tag] = z_scvi
    aff = get_or_build_latent_affinity(ds_id, ad, tag, use_cache=skip_if_exists)
    purity = rare_type_knn_purity(z_scvi, ad.obs[lk].values)
    print(f"[{ds_id}] {tag}: rare-type kNN purity = {purity}")

    seacells_res = None
    if run_seacells:
        seacells_res = run_seacells_on_latent(ds_id, ad, n_seacells=K, aff=aff, tag=tag,
                                              skip_if_exists=skip_if_exists)
    leiden_res = None
    if run_leiden:
        leiden_res = run_leiden_on_latent(ds_id, ad, n_target_clusters=K, aff=aff, tag=tag,
                                          skip_if_exists=skip_if_exists)

    return {"tag": tag, "K": K, "knn_purity": purity,
            "seacells": seacells_res, "leiden": leiden_res}


# ---------------------------------------------------------------------------
# arm B -- scProto Stage 2 on the same encoder
# ---------------------------------------------------------------------------


STAGE2_STATE_FILE = "stage2_state.json"


def _read_stage2_state(t):
    path = os.path.join(t.get_dump_path(), STAGE2_STATE_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _write_stage2_state(t, epochs_trained, max_epochs):
    state = {
        "epochs_trained": int(epochs_trained),
        "max_epochs_requested": int(max_epochs),
        # Early stopping ends a run before the cap; hitting the cap means the budget,
        # not convergence, was the binding constraint -- the only case where raising
        # max_epochs and re-running is meaningful.
        "hit_epoch_cap": bool(epochs_trained >= max_epochs),
    }
    with open(os.path.join(t.get_dump_path(), STAGE2_STATE_FILE), "w") as f:
        json.dump(state, f, indent=2)
    return state


def run_stage2(t, max_epochs=20, eval_freq=3, patience=6, min_delta=0.005,
               force_retrain=False):
    """Train (or resume) scProto's Stage 2 on the scVI encoder. Early stopping on
    modularity, the same criterion the paper's runs use.

    Re-running is idempotent by design -- the whole notebook is meant to be re-run
    after a disconnect, and silently continuing to train a run that already converged
    would change results you had already recorded. What happens on a second call:

      * no checkpoint            -> train from scratch
      * checkpoint, no state file -> a run from before this bookkeeping existed:
                                    reload it and STOP. Pass force_retrain=True to
                                    start over deliberately.
      * state says early-stopped -> converged; reload and stop.
      * state says it hit the cap and `max_epochs` is now higher -> continue for the
                                    difference only.
    """
    ckpt = os.path.join(t.get_dump_path(), "umap_checkpoint.pth")

    if os.path.exists(ckpt) and not force_retrain:
        state = _read_stage2_state(t)
        t.load_umap_checkpoint()

        if state is None:
            print(f"[scvi stage2] existing checkpoint found with no {STAGE2_STATE_FILE} "
                  f"(run predates this bookkeeping) -- reloading it as-is, not training "
                  f"further. Pass force_retrain=True to retrain from scratch.")
            return t

        done = state.get("epochs_trained", 0)
        if not state.get("hit_epoch_cap", False):
            print(f"[scvi stage2] previous run early-stopped after {done} epoch(s) "
                  f"(converged) -- reloading best checkpoint, no further training.")
            return t

        remaining = max_epochs - done
        if remaining <= 0:
            print(f"[scvi stage2] {done} epoch(s) already trained, max_epochs="
                  f"{max_epochs} -- nothing left to train.")
            return t

        print(f"[scvi stage2] previous run hit its {done}-epoch cap; continuing for "
              f"{remaining} more epoch(s)")
        t.continue_train_umap_edges(
            epochs=remaining, early_stop=True, early_stop_metric="modularity",
            eval_freq=eval_freq, patience=patience, max_epochs=remaining,
            min_delta=min_delta,
        )
        _write_stage2_state(t, t._umap_state.get("epoch", done + remaining), max_epochs)
        return t

    t.train_umap_edges(
        early_stop=True, early_stop_metric="modularity", eval_freq=eval_freq,
        patience=patience, max_epochs=max_epochs, min_delta=min_delta,
    )
    _write_stage2_state(t, t._umap_state.get("epoch", max_epochs), max_epochs)
    return t


def eval_stage2(t, skip_task2=False, skip_if_exists=True):
    """Evaluate the Stage-2 run with the same metric functions the paper's Table 1/2
    come from, writing metrics.json / metacells.h5ad / umap_cells.csv into the run
    directory so the rare-cell table and significance tests pick this run up exactly
    like any other method.

    Modularity is additionally rescored through `recompute_modularity_canonical` --
    the same call the Leiden/SEACells arms go through -- so all three arms are judged
    by one function on one graph, rather than each using its own path to a number that
    is meant to be compared.
    """
    from interpretable_ssl.evaluation.batch_correct_baselines import (
        recompute_modularity_canonical,
    )

    ds_id = t.dataset_id
    bk = DATASETS[ds_id].get("batch_key")
    save_path = t.get_dump_path()
    metrics_path = os.path.join(save_path, "metrics.json")

    # Already evaluated? Evaluation is deterministic given the checkpoint, and it is
    # not cheap (encode pass, task2, a UMAP in save_umap_data), so a completed one is
    # reloaded. 'stage1' is written only by this function, so its presence -- together
    # with the CSV the rare-cell table reads -- means a full eval finished here.
    if skip_if_exists and os.path.exists(metrics_path) and \
            os.path.exists(os.path.join(save_path, "umap_cells.csv")):
        try:
            existing = json.load(open(metrics_path))
        except (json.JSONDecodeError, OSError):
            existing = {}
        if existing.get("stage1") == "scvi":
            print(f"[scvi stage2] evaluation already complete -- reusing {metrics_path}")
            return existing

    # Score the BEST checkpoint, not whatever epoch training happened to stop on.
    # Early stopping keeps training past the best epoch until patience runs out, so
    # the in-memory model is the last epoch, not the best one -- `find_metacells` (the
    # path the paper's own numbers come from) calls load_umap_checkpoint() before
    # evaluating for exactly this reason. Without it the reported metrics silently
    # differ from how every other scProto run in the paper was scored.
    ckpt = os.path.join(save_path, "umap_checkpoint.pth")
    if os.path.exists(ckpt):
        t.load_umap_checkpoint()

    t.eval_metacell_quality()

    if not skip_task2:
        try:
            t.eval_task2_metrics()
        except Exception as e:  # noqa: BLE001 -- task2 is supporting evidence, not the test
            print(f"WARNING: task2 metrics failed ({type(e).__name__}: {e}) -- "
                  f"keeping task1 results, continuing.")

    assignments, _ = t._get_assignments()
    t.train_ds.adata.obs["metacell_id"] = assignments
    recompute_modularity_canonical(ds_id, t.train_ds.adata, assignments, bk, save_path,
                                   K=t.num_prototypes)

    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
    metrics.update({"latent_dim": int(t.latent_dims), "stage1": "scvi",
                    "stage1_epochs": int(t.scvi_epochs),
                    "gene_likelihood": t.scvi_gene_likelihood})
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[scvi stage2] metrics saved to {metrics_path}")

    t.save_metacells()
    t.save_umap_data()
    return metrics


# ---------------------------------------------------------------------------
# embedding-level diagnostic (no clustering involved)
# ---------------------------------------------------------------------------


def embedding_rare_affinity_purity(ds_id, ad, keys, lk=None, bk=None, ref_key=None,
                                   save=True, skip_if_exists=True):
    """Per-batch rare-cell affinity purity for each embedding in `keys`.

    Clustering-free: for every locally-rare-type cell, the fraction of its affinity
    mass that stays within its own cell type. Run on the scVI latent before and after
    Stage 2, this shows whether the Stage-2 loss changed the *latent space* itself --
    not just how it was partitioned -- which is the substantive part of the reviewer's
    "does Stage 2 do anything beyond clustering a pretrained latent" question.

    Returns a DataFrame with mean +/- std across batches and, when `ref_key` is given,
    a paired one-sided Wilcoxon (ref > other) with the win count.
    """
    from scipy.stats import wilcoxon
    from interpretable_ssl.evaluation.batch_correct_baselines import (
        rare_type_affinity_ratio_per_batch,
    )

    lk = lk or DATASETS[ds_id]["label_key"]
    bk = bk or DATASETS[ds_id].get("batch_key")
    path = os.path.join(get_dataset_model_dir(ds_id), f"scvi_stage2_affinity_purity_{ds_id}.json")

    # Each key here means building another adaptive-RBF graph, so a finished
    # diagnostic is reloaded rather than recomputed. The per-batch scores are what is
    # stored, so the table (means, stds, win counts, Wilcoxon) is rebuilt from them
    # exactly as if it had just been computed.
    cached = None
    if skip_if_exists and os.path.exists(path):
        try:
            with open(path) as f:
                cached = json.load(f)
        except (json.JSONDecodeError, OSError):
            cached = None
    if cached is not None and all(k in cached.get("per_batch_scores", {}) for k in keys):
        print(f"[{ds_id}] embedding-level rare affinity purity: reusing {path}")
        scores = {k: cached["per_batch_scores"][k] for k in keys}
        dims = {k: cached.get("embedding_dims", {}).get(k, ad.obsm[k].shape[1]) for k in keys}
    else:
        scores = {k: rare_type_affinity_ratio_per_batch(ad, k, lk, bk) for k in keys}
        dims = {k: int(ad.obsm[k].shape[1]) for k in keys}
        if save:
            with open(path, "w") as f:
                json.dump({"per_batch_scores": {k: list(map(float, v)) for k, v in scores.items()},
                           "embedding_dims": dims}, f, indent=2)
            print(f"[{ds_id}] embedding-level rare affinity purity saved to {path}")

    ref = np.array(scores[ref_key]) if ref_key and scores.get(ref_key) else None
    rows = []
    for k, vals in scores.items():
        arr = np.array(vals)
        row = {"embedding": k, "dim": dims.get(k), "n_batches": len(arr),
               "mean": round(float(arr.mean()), 4) if len(arr) else np.nan,
               "std": round(float(arr.std()), 4) if len(arr) else np.nan}
        if ref is not None and k != ref_key and len(arr) == len(ref) and len(arr):
            row["n_wins"] = int((ref > arr).sum())
            try:
                _, p = wilcoxon(ref, arr, alternative="greater")
                row["p_vs_ref"] = round(float(p), 4)
            except ValueError:
                pass
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# full experiment
# ---------------------------------------------------------------------------


def run_scvi_stage2_experiment(ds_id, scvi_epochs=50, scvi_n_latent=10,
                               scvi_gene_likelihood="zinb", stage2_max_epochs=20,
                               eval_freq=3, patience=6, batch_size=1024,
                               umap_steps_per_epoch=500, K=None,
                               run_seacells=True, run_leiden=True,
                               skip_if_exists=True, force_stage1=False,
                               force_stage2=False, skip_task2=False):
    """Both arms end-to-end for one dataset. Safe to re-run: every stage reloads.

    Returns a dict with the trainer, the two latents, the baseline results, the
    Stage-2 metrics and the embedding-level diagnostic table.
    """
    t = get_scvi_proto_trainer(
        ds_id,
        scvi_epochs=scvi_epochs,
        scvi_n_latent=scvi_n_latent,
        scvi_gene_likelihood=scvi_gene_likelihood,
        batch_size=batch_size,
        umap_steps_per_epoch=umap_steps_per_epoch,
        **({} if K is None else {"num_prototypes": K}),
    )

    print(f"\n=== [{ds_id}] Stage 1: scVI ===")
    run_stage1(t, force_retrain=force_stage1)
    z_scvi = t.get_latent(l2norm=False)
    print(f"[{ds_id}] scVI latent: {z_scvi.shape}")

    print(f"\n=== [{ds_id}] arm A: scVI latent -> Leiden / SEACells ===")
    baselines = run_scvi_clustering_baselines(
        t, z_scvi, K=K, run_seacells=run_seacells, run_leiden=run_leiden,
        skip_if_exists=skip_if_exists,
    )

    print(f"\n=== [{ds_id}] arm B: scProto Stage 2 on the same scVI encoder ===")
    run_stage2(t, max_epochs=stage2_max_epochs, eval_freq=eval_freq,
               patience=patience, force_retrain=force_stage2)
    stage2_metrics = eval_stage2(t, skip_task2=skip_task2, skip_if_exists=skip_if_exists)

    z_stage2 = t.get_latent()
    ad = t.train_ds.adata
    ad.obsm["X_scviproto"] = z_stage2
    purity_df = embedding_rare_affinity_purity(
        ds_id, ad, keys=[SCVI_DEF_TAG, "X_scviproto"], ref_key="X_scviproto",
        skip_if_exists=skip_if_exists,
    )

    print(f"\n[{ds_id}] done. Stage-2 run dir: {t.get_dump_path()}")
    print_dataset_summary(ds_id, t.get_dump_path())
    return {
        "trainer": t,
        "run_dir": t.get_dump_path(),
        "z_scvi": z_scvi,
        "z_stage2": z_stage2,
        "baselines": baselines,
        "stage2_metrics": stage2_metrics,
        "embedding_purity": purity_df,
    }


# The paper's own pancreas run. Only used to derive the shared, dataset-independent
# key below -- the lung/immune canonical runs normalise to the same string.
SCPROTO_CANONICAL_RUN = (
    "proto_umap_ds-panc_NP220_prtInit-wayp_aff-arbf_lprec0.01_usim-prot_cvae_e50_"
    "ecal1_lpu0.1_pum-ema_lna1_nagg-max_upm-dotp_v31"
)


def resolve_run_names(stage2_run_dir=None, tag=SCVI_DEF_TAG, include_canonical=True):
    """`model_keywords` for `rare_celltype_purity_table`, valid across ALL datasets.

    Keys are chosen so one dict resolves correctly for every dataset:

    * `seacell_{tag}` -- identical folder name everywhere, matches exactly.
    * `leiden_{tag}` -- deliberately WITHOUT the `_K{n}` suffix, since K differs per
      dataset (220 / 300 / 300) and `extract_model_key` does not strip it;
      `_resolve_run_dir` falls back to substring matching, which catches each
      dataset's own K. This is why the optional L2-norm control uses tag 'X_scvil2'
      rather than something starting with 'X_scvidef' -- a prefix-colliding tag would
      make this substring ambiguous and `_resolve_run_dir` would silently pick
      whichever sorts last.
    * the Stage-2 run -- passed through `extract_model_key`, which strips the
      dataset-specific `_ds-*`, `_NP*` and `_v*` tokens, leaving one key shared by all
      three datasets' runs.
    * the paper's own scProto runs (scPoli Stage 1), same normalisation, as a context
      row. Pass include_canonical=False to leave it out.
    """
    from interpretable_ssl.evaluation.metric_helpers.result_tables import extract_model_key

    names = {
        f"seacell_{tag}": "SEACells (scVI)",
        f"leiden_{tag}": "Leiden (scVI)",
    }
    if stage2_run_dir is not None:
        names[extract_model_key(os.path.basename(stage2_run_dir.rstrip("/")))] = (
            "scProto Stage 2 (scVI)"
        )
    if include_canonical:
        names[extract_model_key(SCPROTO_CANONICAL_RUN)] = "scProto (scPoli Stage 1)"
    return names


def collect_task1_table(ds_ids, model_keywords, dataset_display_names=None):
    """Purity / batch entropy / modularity / coverage for every (dataset, arm), read
    from each run's own metrics.json.

    Run directories are located with the same `_resolve_run_dir` the rare-cell table
    uses, so a row appearing here and a row appearing there always refer to the same
    folder -- rather than this table rebuilding folder names by hand and silently
    drifting from it.
    """
    from interpretable_ssl.evaluation.paper_figures import _resolve_run_dir

    dataset_display_names = dataset_display_names or {}
    rows = []
    for ds_id in ds_ids:
        for kw, name in model_keywords.items():
            run_dir = _resolve_run_dir(ds_id, kw, prefer_csv="metrics.json")
            if run_dir is None:
                print(f"  [task1] {name} / {ds_id}: no matching run dir")
                continue
            path = os.path.join(run_dir, "metrics.json")
            if not os.path.exists(path):
                print(f"  [task1] {name} / {ds_id}: metrics.json missing in {run_dir}")
                continue
            m = json.load(open(path))
            rows.append({
                "dataset": dataset_display_names.get(ds_id, ds_id),
                "method": name,
                "purity": m.get("mean_cell_type_purity"),
                "batch_entropy": m.get("mean_batch_entropy"),
                "modularity": m.get("mean_modularity_batch", m.get("modularity")),
                "modularity_std": m.get("std_modularity_batch"),
                "coverage": m.get("coverage"),
                "K": m.get("K_target"),
            })
    return pd.DataFrame(rows)


def print_dataset_summary(ds_id, stage2_run_dir):
    """Print this dataset's three-arm comparison the moment it finishes, instead of
    making the whole sweep complete before any number is visible. Reads each arm's
    metrics.json off disk, so it also works standalone for a dataset finished in an
    earlier session."""
    try:
        df = collect_task1_table([ds_id], resolve_run_names(stage2_run_dir))
        if df.empty:
            print(f"[{ds_id}] no arm metrics found yet")
            return
        cols = ["method", "modularity", "modularity_std", "purity", "batch_entropy", "coverage", "K"]
        print(f"\n----- [{ds_id}] arm comparison -----")
        print(df[[c for c in cols if c in df.columns]].to_string(index=False))
        print("-" * 40)
    except Exception as e:  # noqa: BLE001 -- a summary print must never fail a run
        print(f"[{ds_id}] could not print summary ({type(e).__name__}: {e})")


def run_all_datasets(ds_ids, **kwargs):
    """`run_scvi_stage2_experiment` for each dataset, one after another.

    A dataset that fails is reported and skipped rather than taking the whole sweep
    down with it -- with three datasets and a Colab session that can drop at any
    point, losing the two that already worked would be the expensive outcome. Every
    stage is reloaded from disk on a re-run, so simply calling this again after a
    failure resumes rather than repeats.
    """
    results = {}
    for ds_id in ds_ids:
        print(f"\n{'=' * 70}\n=== {ds_id} ===\n{'=' * 70}")
        try:
            results[ds_id] = run_scvi_stage2_experiment(ds_id, **kwargs)
        except Exception as e:  # noqa: BLE001 -- keep the other datasets' results
            import traceback
            print(f"\n!!! [{ds_id}] FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()
            results[ds_id] = None
    done = [d for d, r in results.items() if r is not None]
    print(f"\nfinished: {done}  failed: {[d for d in ds_ids if results.get(d) is None]}")
    return results
