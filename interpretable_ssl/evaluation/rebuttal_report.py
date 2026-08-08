"""
(sync touch)
Shared "results comparison" section for the batch-correct-then-cluster rebuttal
notebooks (Harmony / ComBat / BBKNN / ...). Each of those notebooks runs a
DIFFERENT correction method through the identical downstream pipeline
(`run_all_baselines_for_dataset` in batch_correct_baselines.py) and then wants the
identical set of comparison tables + significance tests against scProto -- this
module holds that shared second half, so adding a new correction-method notebook
doesn't mean re-copy-pasting ~10 near-identical cells again (that duplication is
exactly what caused the Leiden resolution-search bug to silently not propagate
across the old split scPoli/Harmony+scVI+BBKNN notebooks -- see
batch_correct_then_cluster_baselines.ipynb's own intro cell).

Import this AFTER nb_setup.py has run (needs load_task1_multi, show_table,
TASK1_METRICS, TASK2_METRICS, extract_model_key, rare_celltype_purity_table,
rare_metric_significance[_paired], graph_batch_significance[_paired] -- all already
pulled into the notebook namespace by nb_setup.py's `from ... import *`, and
re-imported explicitly here so this module also works standalone).
"""

import os

import pandas as pd
from IPython.display import display

from interpretable_ssl.evaluation.metric_helpers.result_tables import (
    load_task1_multi, show_table, TASK1_METRICS, TASK2_METRICS, extract_model_key,
)
from interpretable_ssl.evaluation.paper_figures import (
    rare_celltype_purity_table, rare_metric_significance, rare_metric_significance_paired,
    graph_batch_significance, graph_batch_significance_paired,
)


# scProto runs verified against the paper's published numbers via generate_tables.ipynb
# (the actual notebook used to build the paper's tables) -- bit-identical to the
# published numbers on multiple metrics simultaneously per dataset. One shared
# definition so every correction-method notebook compares against the exact same
# scProto reference.
SCPROTO_CANONICAL_RUNS = {
    'proto_umap_ds-panc_NP220_prtInit-wayp_aff-arbf_lprec0.01_usim-prot_cvae_e50_ecal1_lpu0.1_pum-ema_lna1_nagg-max_upm-dotp_v31',
    'proto_umap_ds-lung_prtInit-wayp_aff-arbf_lprec0.01_usim-prot_cvae_e50_ecal1_lpu0.1_pum-ema_lna1_nagg-max_upm-dotp_v31',
    'proto_umap_prtInit-wayp_aff-arbf_lprec0.01_usim-prot_cvae_e50_ecal1_lpu0.1_pum-ema_lna1_nagg-max_upm-dotp_v31',
}

SCPROTO_KEY = extract_model_key(next(iter(SCPROTO_CANONICAL_RUNS)))
assert all(extract_model_key(r) == SCPROTO_KEY for r in SCPROTO_CANONICAL_RUNS), (
    "SCPROTO_CANONICAL_RUNS entries no longer normalize to one shared key -- "
    "extract_model_key's stripping patterns changed; update the scProto matching logic."
)


# Methods whose on-disk folder tag is dimension-qualified (e.g. 'seacell_X_combat_d8')
# because they're corrected/embedded at scProto's own latent dimension instead of the
# default 50 -- see batch_correct_baselines.DIM_MATCHED_METHODS (the same set, kept as
# a separate copy here so this module doesn't need to import that one just for this).
DIM_MATCHED_METHODS = {'harmony', 'combat', 'bbknn'}


def build_model_keywords(correction_methods, method_display_names, extra_read_only=None,
                          matched_dim=8):
    """Build the {on-disk folder prefix: display name} dict every table/significance
    function in this module needs -- scProto + SEACells(PCA) always included, plus one
    {SEACells, Leiden} entry per method in correction_methods.

    matched_dim: dimension harmony/combat/bbknn are corrected/embedded at (matching
    scProto's own latent_dims, currently 8 for all three RNA-seq datasets here) --
    their on-disk folder tags encode this dimension (e.g. 'seacell_X_combat_d8'), so
    it must be passed here to build the right keyword for any of DIM_MATCHED_METHODS
    present in correction_methods, not derived automatically.

    extra_read_only: optional dict of additional {folder_prefix: display_name} entries
    to merge in as-is -- for pulling in a method this notebook does NOT compute itself,
    just reads whatever's already on disk from another notebook's run (e.g. Harmony's
    tags, when this notebook only runs ComBat/BBKNN itself). If that other notebook
    hasn't been run yet, these keys simply match nothing on disk -- not an error.
    """
    model_keywords = {SCPROTO_KEY: 'scProto', 'seacell': 'SEACells (PCA)'}
    for method in correction_methods:
        disp = method_display_names[method]
        if method in DIM_MATCHED_METHODS:
            model_keywords[f'seacell_X_{method}_d{matched_dim}'] = f'SEACells ({disp})'
            model_keywords[f'leiden_X_{method}_d{matched_dim}'] = f'Leiden ({disp})'
        else:
            model_keywords[f'seacell_X_{method}'] = f'SEACells ({disp})'
            model_keywords[f'leiden_X_{method}'] = f'Leiden ({disp})'
    if extra_read_only:
        model_keywords.update(extra_read_only)
    return model_keywords


def dim_matched_read_only_keywords(method, display_name, matched_dim=8):
    """Convenience `extra_read_only` value for pulling an already-computed
    dimension-matched run (harmony/combat/bbknn) from ANOTHER notebook into this
    notebook's comparison tables, without this notebook ever computing that method
    itself. matched_dim must match scProto's own latent_dims for the datasets in
    use -- a literal string match against the on-disk folder name, not computed
    dynamically.
    """
    return {
        f'seacell_X_{method}_d{matched_dim}': f'SEACells ({display_name})',
        f'leiden_X_{method}_d{matched_dim}': f'Leiden ({display_name})',
    }


def harmony_read_only_keywords(harmony_dim=8):
    """Convenience `extra_read_only` value for pulling in an already-computed Harmony
    run (from the Harmony notebook) into another notebook's comparison tables, without
    this notebook ever computing Harmony itself. Thin wrapper over
    dim_matched_read_only_keywords -- kept as its own name since existing notebooks
    already call it this way.
    """
    return dim_matched_read_only_keywords('harmony', 'Harmony', matched_dim=harmony_dim)


def _keep_and_rename_runs(df, model_keywords):
    """Filter load_task1_multi's output down to just the runs in model_keywords, and
    collapse each method's per-dataset '_K{n}' folder-name suffix (target K is
    scProto's num_prototypes and differs by dataset) into ONE shared display row per
    method -- otherwise Table 1/2 fragment into a separate row per distinct K value
    instead of one compact row per method spanning all datasets.

    If more than one on-disk run for the same dataset maps to the same display name
    (e.g. a stale leftover run from an earlier num_prototypes value), the extra row is
    dropped with a WARNING rather than silently picked or crashing show_table's
    unstack() on the duplicate index.
    """
    runs = df.index.get_level_values('run')
    datasets = df.index.get_level_values('dataset')
    stripped = runs.str.replace(r'_K\d+$', '', regex=True)
    keep_mask = stripped.isin(model_keywords)

    kept_runs, kept_datasets = runs[keep_mask], datasets[keep_mask]
    kept_display = stripped[keep_mask].map(model_keywords)

    out = df[keep_mask].copy()
    out.index = pd.MultiIndex.from_arrays([kept_datasets, kept_display], names=['dataset', 'run'])

    dupe_mask = out.index.duplicated(keep='first')
    if dupe_mask.any():
        stale = list(zip(kept_datasets[dupe_mask], kept_runs[dupe_mask], kept_display[dupe_mask]))
        print(f"WARNING: dropped {dupe_mask.sum()} duplicate (dataset, method) row(s) -- "
              f"likely a stale run at an old K value still on disk. "
              f"(dataset, on-disk folder, display name): {stale}")
        out = out[~dupe_mask]
    return out


RARE_CELL_METRICS = [
    'batch_rare_coverage_mean', 'batch_rare_recall_macro_mean',
    'batch_rare_precision_macro_mean', 'batch_rare_homogeneity_mean',
    'batch_rare_cross_batch_homog_mean', 'batch_rare_f1_macro_mean',
]
RARE_CELL_SIG_METRICS = (
    '_batch_rare_f1_macro_per_batch',
    '_batch_rare_homogeneity_per_batch',
    '_batch_rare_cross_batch_homog_per_batch',
)


def _display_pivot(sig_df, label):
    for metric_name in sig_df['metric'].unique():
        print(f"=== {metric_name}: scProto vs. each same-K baseline, {label}, "
              f"Bonferroni-corrected per dataset ===")
        sub = sig_df[sig_df['metric'] == metric_name].copy()
        sub['cell'] = sub.apply(
            lambda r: f"{r['median']:.3f} (K={r['k']}, n={r['n']}) [ref]" if r['method'] == 'scProto'
            else f"{r['median']:.3f} (K={r['k']}, n={r['n']}, wins={r.get('n_wins', '?')}/{r['n']})  "
                 f"{r.get('sig', '?')}  p_adj={r.get('p_adj', float('nan')):.3g}",
            axis=1,
        )
        display(sub.pivot(index='method', columns='dataset', values='cell'))


def render_full_comparison_report(rna_seq_datasets, dataset_display_names, model_keywords,
                                   ref_name='scProto'):
    """Runs + displays the full shared comparison section: Table 1 (modularity/batch
    entropy/purity), Table 2 (coverage/scGraph), the rare-cell-type table, and all
    four significance tests (unpaired + paired, for both the rare-cell table and
    Table 1) -- the exact same set of tables every batch-correct-then-cluster
    notebook (Harmony, ComBat, BBKNN, ...) wants, parameterized only by which methods
    are in model_keywords.

    Returns a dict of the underlying DataFrames (task1, task2, rare, and the four
    significance tables) for any further inspection beyond the printed/displayed
    tables.
    """
    results = {}

    df_task1 = load_task1_multi(rna_seq_datasets, metrics=TASK1_METRICS)
    df_task1 = _keep_and_rename_runs(df_task1, model_keywords)
    print("=== Table 1: community structure / batch integration ===")
    show_table(df_task1, metrics=TASK1_METRICS, dataset_display_names=dataset_display_names)
    results['task1'] = df_task1

    df_task2 = load_task1_multi(rna_seq_datasets, metrics=TASK2_METRICS)
    df_task2 = _keep_and_rename_runs(df_task2, model_keywords)
    print("\n=== Table 2: metacell representation quality ===")
    show_table(df_task2, metrics=TASK2_METRICS, dataset_display_names=dataset_display_names)
    results['task2'] = df_task2

    df_rare = rare_celltype_purity_table(rna_seq_datasets, model_keywords=model_keywords, verbose=True)
    dupe_mask = df_rare.index.duplicated(keep='first')
    if dupe_mask.any():
        print(f"WARNING: dropped {dupe_mask.sum()} duplicate row(s) from df_rare: "
              f"{df_rare.index[dupe_mask].tolist()}")
        df_rare = df_rare[~dupe_mask]
    print("\n=== Rare-cell-type table (key hypothesis test) ===")
    show_table(df_rare, metrics=RARE_CELL_METRICS, dataset_display_names=dataset_display_names)
    results['rare'] = df_rare

    print("\n=== UNPAIRED rare-cell significance (Mann-Whitney U) ===")
    df_sig = rare_metric_significance(
        df_rare, ref_name=ref_name, metrics=RARE_CELL_SIG_METRICS,
        dataset_display_names=dataset_display_names,
    )
    display(df_sig)
    results['rare_sig_unpaired'] = df_sig

    print("\n=== PAIRED rare-cell significance (Wilcoxon signed-rank, recommended) ===")
    df_sig_paired = rare_metric_significance_paired(
        df_rare, ref_name=ref_name, metrics=RARE_CELL_SIG_METRICS,
        dataset_display_names=dataset_display_names,
    )
    _display_pivot(df_sig_paired, "PAIRED one-sided Wilcoxon signed-rank (scProto > other)")
    results['rare_sig_paired'] = df_sig_paired

    print("\n=== UNPAIRED Table 1 significance (Mann-Whitney U) ===")
    sig_df_table1 = graph_batch_significance(
        rna_seq_datasets, model_keywords, ref_name=ref_name,
        dataset_display_names=dataset_display_names,
    )
    _display_pivot(sig_df_table1, "one-sided Mann-Whitney U (scProto > other)")
    results['table1_sig_unpaired'] = sig_df_table1

    print("\n=== PAIRED Table 1 significance (modularity only, recommended for that metric) ===")
    sig_df_table1_paired = graph_batch_significance_paired(
        rna_seq_datasets, model_keywords, ref_name=ref_name,
        dataset_display_names=dataset_display_names,
    )
    _display_pivot(sig_df_table1_paired, "PAIRED one-sided Wilcoxon signed-rank (scProto > other)")
    results['table1_sig_paired'] = sig_df_table1_paired

    return results


def per_celltype_breakdown(ds_id, model_keywords, cell_types):
    """Per-INDIVIDUAL-cell-type homogeneity/precision/recall/F1 -- NOT macro-averaged
    across all locally-rare types the way rare_celltype_purity_table's
    batch_rare_*_mean columns are. Reads the same already-saved umap_cells.csv /
    umap_protos.csv each run already wrote (no recompute, no retraining).

    Why this exists: the aggregate rare-cell metrics macro-average across every
    locally-rare type within a batch, then across batches. If a method handles one
    rare type well and another badly -- exactly what "ComBat applies one shift per
    gene per whole batch, not per cell type" predicts -- that unevenness averages
    out and can look identical to a method that's uniformly mediocre across both.
    This function reports each requested type separately so that unevenness (or
    its absence) is directly visible, and so the SPREAD across types (not just the
    mean) can be compared between methods.

    cell_types: explicit list of cell-type label strings to report on -- pass the
    specific types you have a mechanistic reason to check (e.g. a rare type
    confined almost entirely to one batch, the sharpest case for the "ComBat
    confounds batch artifact with real rare biology" hypothesis), not an
    auto-detected "all rare types" list -- this is a targeted diagnostic, not a
    re-derivation of the aggregate metric.

    Formulas mirror paper_figures._rare_table_one exactly, just computed globally
    per type instead of per-batch-then-macro-averaged:
        homogeneity(ct) = mean over every cell of type ct of the fraction of its
            OWN metacell that shares its label (any batch).
        precision(ct)   = mean purity of metacell(s) whose majority label is ct.
        recall(ct)      = fraction of ct's cells that landed in a metacell whose
            majority label is ct.
        f1(ct)          = harmonic mean of precision and recall.

    Returns a tidy DataFrame: one row per (dataset, method, cell_type), columns
    n_cells, homogeneity, precision, recall, f1. A missing (dataset, method) run
    or a cell_type with zero cells in that run is included with NaN metrics
    (not silently dropped), so gaps are visible rather than invisible.
    """
    from interpretable_ssl.evaluation.paper_figures import _resolve_run_dir

    ds_ids = [ds_id] if isinstance(ds_id, str) else list(ds_id)
    rows = []

    for did in ds_ids:
        for keyword, name in model_keywords.items():
            run_dir = _resolve_run_dir(did, keyword, prefer_csv='umap_cells.csv')
            if run_dir is None:
                continue
            cells_path = os.path.join(run_dir, 'umap_cells.csv')
            if not os.path.exists(cells_path):
                continue
            cells_df = pd.read_csv(cells_path, usecols=lambda c: c not in ('umap_1', 'umap_2'))

            label_key = None
            protos_path = os.path.join(run_dir, 'umap_protos.csv')
            if os.path.exists(protos_path):
                protos_df = pd.read_csv(protos_path)
                for c in protos_df.columns:
                    if c.startswith('majority_'):
                        candidate = c[len('majority_'):]
                        if candidate in cells_df.columns:
                            label_key = candidate
                            break
            if label_key is None:
                non_umap = [c for c in cells_df.columns if c not in ('umap_1', 'umap_2', 'metacell_id')]
                label_key = non_umap[0] if non_umap else None
            if label_key is None:
                continue

            mc_sizes = cells_df.groupby('metacell_id').size()
            mc_label_counts = cells_df.groupby(['metacell_id', label_key]).size()
            mc_label_frac = mc_label_counts / mc_sizes
            mc_majority = mc_label_counts.groupby(level='metacell_id').idxmax().map(lambda x: x[1])
            mc_purity_series = mc_label_frac.reindex(
                pd.MultiIndex.from_arrays([mc_majority.index, mc_majority.values])
            ).fillna(0.0)
            mc_purity_series.index = mc_majority.index
            mc_purity_by_ct = mc_purity_series.groupby(mc_majority).mean().to_dict()

            for ct in cell_types:
                ct_cells = cells_df[cells_df[label_key] == ct]
                if ct_cells.empty:
                    rows.append({
                        'dataset': did, 'method': name, 'cell_type': ct,
                        'n_cells': 0, 'homogeneity': float('nan'),
                        'precision': float('nan'), 'recall': float('nan'), 'f1': float('nan'),
                    })
                    continue

                lookup = pd.MultiIndex.from_arrays([ct_cells['metacell_id'], ct_cells[label_key]])
                homog_scores = mc_label_frac.reindex(lookup).fillna(0.0)
                homogeneity = float(homog_scores.mean())

                recall = float((ct_cells['metacell_id'].map(mc_majority) == ct).mean())
                precision = float(mc_purity_by_ct.get(ct, 0.0))
                denom = precision + recall
                f1 = 2 * precision * recall / denom if denom > 0 else 0.0

                rows.append({
                    'dataset': did, 'method': name, 'cell_type': ct,
                    'n_cells': int(len(ct_cells)), 'homogeneity': round(homogeneity, 3),
                    'precision': round(precision, 3), 'recall': round(recall, 3),
                    'f1': round(f1, 3),
                })

    return pd.DataFrame(rows)


def render_realized_k_check(rna_seq_datasets, datasets_config, method):
    """Sanity check that method's downstream SEACells/Leiden actually landed at
    (approximately) scProto's own num_prototypes for each dataset, per the paper's
    own protocol ("All baselines are configured to produce the same number of
    metacells K as scProto"). Reads directly from each run's saved outputs -- no
    recompute. Generic over any method whose tag is the plain 'X_{method}' pattern
    (i.e. NOT Harmony's dimension-qualified 'X_harmony_d{n}' -- check that one
    manually, or pass its literal tag string in place of `method` here).
    """
    from interpretable_ssl.evaluation.batch_correct_baselines import get_realized_seacell_count

    target_k = {ds: datasets_config[ds]['num_prototypes'] for ds in rna_seq_datasets}

    df_k = load_task1_multi(rna_seq_datasets, metrics=['n_clusters', 'resolution'])
    if df_k.empty:
        print("No runs found yet under MODEL_DIR for these datasets -- run the "
              "'Run: ...' cells above first.")
        df_leiden_k = pd.DataFrame()
    else:
        is_this_method = df_k.index.get_level_values('run').str.startswith(f'leiden_X_{method}')
        df_leiden_k = df_k[is_this_method].copy()
        if df_leiden_k.empty:
            print(f"No leiden_X_{method} run found yet for any dataset.")
        else:
            df_leiden_k['target_k'] = [target_k[ds] for ds, _run in df_leiden_k.index]
            df_leiden_k['matches_target'] = df_leiden_k['n_clusters'] == df_leiden_k['target_k']
            display(df_leiden_k)

    seacell_k_rows = []
    for ds_id in rna_seq_datasets:
        n_actual = get_realized_seacell_count(ds_id, f'X_{method}')
        k = target_k[ds_id]
        seacell_k_rows.append({
            'dataset': ds_id, 'method': method,
            'n_actual': n_actual, 'target_k': k,
            'matches_target': (n_actual is not None and abs(n_actual - k) <= 0.05 * k),
        })
    df_seacell_k = pd.DataFrame(seacell_k_rows).set_index(['dataset', 'method'])
    if df_seacell_k['n_actual'].isna().all():
        print(f"No seacell_X_{method} runs found on disk yet for these datasets.")
    else:
        display(df_seacell_k)

    return df_leiden_k, df_seacell_k
