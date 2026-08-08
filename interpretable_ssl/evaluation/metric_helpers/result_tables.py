"""
result_tables.py — load, clean, and display results tables for Task 1 and Task 2.

Typical usage in a notebook:
    from interpretable_ssl.evaluation.metric_helpers.result_tables import (
        load_task1_multi, show_table, TASK1_METRICS, TASK2_METRICS, TASK3_METRICS
    )

    df = load_task1_multi(['pancreas', 'pbmc-immune'])
    show_table(df, dataset_display_names={'pancreas': 'Pancreas', 'pbmc-immune': 'Immune'})
"""

import json
import os
import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

TASK1_METRICS = [
    'weighted_mean_cell_type_purity', 'weighted_std_cell_type_purity',
    # 'soft_weighted_mean_cell_type_purity', 'soft_weighted_std_cell_type_purity',
    'weighted_mean_batch_entropy', 'weighted_std_batch_entropy',
    # 'soft_weighted_mean_batch_entropy', 'soft_weighted_std_batch_entropy',
    'mean_modularity_batch', 'std_modularity_batch', 'coverage',
]
TASK2_METRICS = ['coverage', 'dge_rbo_avg', 'dge_kendall_avg', 'dge_jaccard_avg', 'scgraph_corr_avg', 'modularity']
TASK3_METRICS = ['mean_cell_type_purity', 'std_cell_type_purity', 'mean_niche_purity', 'std_niche_purity', 'ct_niche_rbo_avg', 'std_ct_niche_rbo', 'n_unused_protos', 'unused_proto_ratio']

# Pairs of (mean_col, std_col) to merge into "X ± Y" display
MEAN_STD_PAIRS = {
    'mean_cell_type_purity':               'std_cell_type_purity',
    'weighted_mean_cell_type_purity':      'weighted_std_cell_type_purity',
    'soft_weighted_mean_cell_type_purity': 'soft_weighted_std_cell_type_purity',
    'mean_batch_entropy':                  'std_batch_entropy',
    'weighted_mean_batch_entropy':         'weighted_std_batch_entropy',
    'soft_weighted_mean_batch_entropy':    'soft_weighted_std_batch_entropy',
    'mean_niche_purity':     'std_niche_purity',
    'ct_niche_rbo_avg':      'std_ct_niche_rbo',
    'mean_modularity_batch':  'std_modularity_batch',
    'batch_rare_coverage_mean':      'batch_rare_coverage_std',
    'batch_rare_repr_macro_mean':    'batch_rare_repr_macro_std',
    'batch_rare_repr_micro_mean':    'batch_rare_repr_micro_std',
    'batch_rare_homogeneity_mean':   'batch_rare_homogeneity_std',
    'batch_rare_cross_batch_homog_mean': 'batch_rare_cross_batch_homog_std',
    'batch_rare_purity_mean':        'batch_rare_purity_std',
    'batch_rare_f1_macro_mean':      'batch_rare_f1_macro_std',
}

METRIC_DISPLAY_NAMES = {
    'modularity':            'Modularity',
    'mean_batch_entropy':          'Batch Entropy',
    'std_batch_entropy':           'Batch Entropy Std',
    'weighted_mean_batch_entropy': 'Batch Entropy (W)',
    'weighted_std_batch_entropy':  'Batch Entropy (W) Std',
    'soft_weighted_mean_batch_entropy': 'Batch Entropy (SW)',
    'soft_weighted_std_batch_entropy':  'Batch Entropy (SW) Std',
    'mean_cell_type_purity':          'CT Purity',
    'weighted_mean_cell_type_purity': 'CT Purity (W)',
    'weighted_std_cell_type_purity':  'CT Purity (W) Std',
    'soft_weighted_mean_cell_type_purity': 'CT Purity (SW)',
    'soft_weighted_std_cell_type_purity':  'CT Purity (SW) Std',
    'mean_niche_purity':     'Niche Purity',
    'std_niche_purity':      'Niche Purity Std',
    'n_unused_protos':       'Unused Protos',
    'unused_proto_ratio':    'Unused Ratio',
    'coverage':              'Coverage',
    'soft_coverage':         'Coverage (S)',
    'dge_rbo_avg':           'DGE RBO',
    'soft_dge_rbo_avg':      'DGE RBO (S)',
    'dge_kendall_avg':       'DGE Kendall',
    'dge_jaccard_avg':       'DGE Jaccard',
    'scgraph_corr_avg':      'scGraph Corr',
    'soft_scgraph_corr_avg': 'scGraph Corr (S)',
    'ct_niche_rbo_avg':      'CT-Niche RBO',
    'std_ct_niche_rbo':      'CT-Niche RBO Std',
    'mean_modularity_batch': 'Modularity/batch',
    'std_modularity_batch':  'Modularity/batch Std',
    'batch_rare_coverage_mean':       'Batch-Rare Coverage',
    'batch_rare_coverage_std':        'Batch-Rare Coverage Std',
    'batch_rare_repr_macro_mean':     'Batch-Rare Repr (macro)',
    'batch_rare_repr_macro_std':      'Batch-Rare Repr (macro) Std',
    'batch_rare_repr_micro_mean':     'Batch-Rare Repr (micro)',
    'batch_rare_repr_micro_std':      'Batch-Rare Repr (micro) Std',
    'batch_rare_homogeneity_mean':    'Batch-Rare Homogeneity',
    'batch_rare_homogeneity_std':     'Batch-Rare Homogeneity Std',
    'batch_rare_cross_batch_homog_mean': 'Cross-Batch Rare Homogeneity',
    'batch_rare_cross_batch_homog_std':  'Cross-Batch Rare Homogeneity Std',
    'batch_rare_purity_mean':         'Batch-Rare Purity',
    'batch_rare_purity_std':          'Batch-Rare Purity Std',
    'batch_rare_f1_macro_mean':       'Batch-Rare F1 (macro)',
    'batch_rare_f1_macro_std':        'Batch-Rare F1 (macro) Std',
}

# Direction for each metric key: 'up' (higher is better) or 'down' (lower is better)
METRIC_DIRECTION = {
    'coverage':                              'up',
    'soft_coverage':                         'up',
    'modularity':                            'up',
    'mean_cell_type_purity':                 'up',
    'weighted_mean_cell_type_purity':        'up',
    'soft_weighted_mean_cell_type_purity':   'up',
    'mean_batch_entropy':                    'up',
    'weighted_mean_batch_entropy':           'up',
    'soft_weighted_mean_batch_entropy':      'up',
    'mean_niche_purity':                     'up',
    'ct_niche_rbo_avg':                      'up',
    'mean_modularity_batch':                 'up',
    'dge_rbo_avg':                           'up',
    'dge_kendall_avg':                       'up',
    'dge_jaccard_avg':                       'up',
    'scgraph_corr_avg':                      'up',
    'batch_rare_coverage_mean':              'up',
    'batch_rare_repr_macro_mean':            'up',
    'batch_rare_repr_micro_mean':            'up',
    'batch_rare_homogeneity_mean':           'up',
    'batch_rare_cross_batch_homog_mean':     'up',
    'batch_rare_purity_mean':                'up',
    'batch_rare_f1_macro_mean':              'up',
    'n_unused_protos':                       'down',
    'unused_proto_ratio':                    'down',
}

# Map display names -> direction (built from METRIC_DIRECTION + METRIC_DISPLAY_NAMES)
_DISPLAY_DIRECTION = {
    METRIC_DISPLAY_NAMES.get(k, k): v for k, v in METRIC_DIRECTION.items()
}

# Patterns to strip from folder names — all use _ as delimiter (not \b)
# because _ is a word character and \b doesn't work as expected here.
_DATASET_SPECIFIC_PATTERNS = [
    r'_v\d+$',       # version suffix:    _v6, _v7
    r'_[Bb][Ss]\d+', # batch size:        _bs256, _BS256, _bs512
    r'_NP\d+',       # num prototypes:    _NP50, _NP150
    r'_ep\d+',       # epochs:            _ep100
    r'_cvae_e\d+',   # cvae epochs:       _cvae_e100
    r'_ds-[^_]+',    # dataset id token:  _ds-pancreas, _ds-pbmc
]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_run_metrics(ds_id):
    """Scan MODEL_DIR/<ds_id>/ and return raw metrics dicts keyed by folder name.

    Reads metrics.json when present; falls back to metrics_json inside clusters.npz.

    Returns:
        dict: {folder_name: metrics_dict}
    """
    from interpretable_ssl.configs.paths import get_dataset_model_dir
    base_dir = get_dataset_model_dir(ds_id)

    results = {}
    for entry in sorted(os.listdir(base_dir)):
        run_dir = os.path.join(base_dir, entry)
        if not os.path.isdir(run_dir):
            continue

        json_path = os.path.join(run_dir, 'metrics.json')
        npz_path  = os.path.join(run_dir, 'clusters.npz')

        if os.path.exists(json_path):
            with open(json_path) as f:
                m = json.load(f)
        elif os.path.exists(npz_path):
            data = np.load(npz_path, allow_pickle=True)
            m = json.loads(str(data['metrics_json']))
        else:
            continue

        # --- Backfill stds from the per-mc CSVs ---
        # metrics.json is the source of truth: eval now writes every std scalar
        # directly (see mc_metric_utils.save_metacell_metrics). These CSV reads only
        # fill keys metrics.json doesn't already have, so runs saved before that
        # change still show their "± std" columns, and a run whose CSVs are missing
        # no longer silently loses the std column. Note the two directions differ:
        # a key already present in metrics.json is never overwritten from a CSV.
        _csv_backfill = [
            # (csv filename, {metric key: 'mean'|'std'})
            ('purity_per_mc.csv',             {'std_cell_type_purity': 'std'}),
            ('batch_entropy_per_mc.csv',      {'std_batch_entropy': 'std'}),
            ('niche_purity_per_mc.csv',       {'mean_niche_purity': 'mean',
                                               'std_niche_purity': 'std'}),
            ('ct_niche_rbo.csv',              {'ct_niche_rbo_avg': 'mean',
                                               'std_ct_niche_rbo': 'std'}),
            ('modularity_per_batch.csv',      {'mean_modularity_batch': 'mean',
                                               'std_modularity_batch': 'std'}),
            ('soft_purity_per_mc.csv',        {'soft_std_cell_type_purity': 'std'}),
            ('soft_batch_entropy_per_mc.csv', {'soft_std_batch_entropy': 'std'}),
        ]
        for fname, key_ops in _csv_backfill:
            missing = {k: op for k, op in key_ops.items() if m.get(k) is None}
            if not missing:
                continue  # metrics.json already has these — don't re-read the CSV
            csv_path = os.path.join(run_dir, fname)
            if not os.path.exists(csv_path):
                continue
            s = pd.read_csv(csv_path, index_col=0).squeeze()
            for k, op in missing.items():
                m[k] = float(s.mean() if op == 'mean' else s.std())

        results[entry] = m

    return results


def load_task1_df(ds_id, metrics=None):
    """Load scalar metrics for all runs of one dataset.

    Row labels are cleaned via extract_model_key so that dataset-specific
    info (batch size, num prototypes, version, etc.) is stripped, leaving
    only the model config name (e.g. 'recon_only', 'proto_umap').

    Args:
        ds_id:   dataset id (subfolder under MODEL_DIR).
        metrics: list of metric keys to keep. None = all scalars.

    Returns:
        DataFrame with model config names as rows and metrics as columns.
    """
    raw = load_run_metrics(ds_id)
    if not raw:
        print(f'No runs found for dataset "{ds_id}"')
        return pd.DataFrame()

    rows = {}
    for folder, m in raw.items():
        key = extract_model_key(folder, ds_id=ds_id)

        if metrics is not None:
            rows[key] = {k: m.get(k) for k in metrics}
        else:
            rows[key] = {k: v for k, v in m.items()
                         if not k.startswith('_info/') and isinstance(v, (int, float))}

    df = pd.DataFrame(rows).T
    df.index.name = 'run'
    return df


def load_task1_multi(ds_ids, metrics=None):
    """Load metrics for multiple datasets into one (dataset, run) MultiIndex DataFrame.

    Args:
        ds_ids:  list of dataset ids.
        metrics: metric keys to keep. None = all scalars.

    Returns:
        DataFrame with (dataset, run) MultiIndex rows.
    """
    dfs = []
    for ds in ds_ids:
        df = load_task1_df(ds, metrics)
        if df.empty:
            continue
        df.index = pd.MultiIndex.from_tuples(
            [(ds, run) for run in df.index],
            names=['dataset', 'run'],
        )
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs)


# ---------------------------------------------------------------------------
# Per-metacell purity aggregation
# ---------------------------------------------------------------------------

def load_purity_stats(ds_ids):
    """Load purity_per_mc.csv from each run and compute mean ± std.

    Args:
        ds_ids: list of dataset ids.

    Returns:
        DataFrame with (dataset, run) MultiIndex, columns: [ct_purity_mean, ct_purity_std]
    """
    from interpretable_ssl.configs.paths import get_dataset_model_dir, get_seacell_model_dir

    data = {}  # (ds, run) -> {mean, std} — last entry wins on duplicate keys
    for ds in ds_ids:
        # SCProto runs
        base_dir = get_dataset_model_dir(ds)
        if os.path.isdir(base_dir):
            for entry in sorted(os.listdir(base_dir)):
                csv_path = os.path.join(base_dir, entry, 'purity_per_mc.csv')
                if os.path.exists(csv_path):
                    s = pd.read_csv(csv_path, index_col=0).squeeze()
                    key = extract_model_key(entry, ds_id=ds)
                    data[(ds, key)] = {'ct_purity_mean': s.mean(), 'ct_purity_std': s.std()}

        # SEACells run
        sc_dir = get_seacell_model_dir(ds)
        csv_path = os.path.join(sc_dir, 'purity_per_mc.csv')
        if os.path.exists(csv_path):
            s = pd.read_csv(csv_path, index_col=0).squeeze()
            data[(ds, 'seacell')] = {'ct_purity_mean': s.mean(), 'ct_purity_std': s.std()}

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data).T
    df.index = pd.MultiIndex.from_tuples(df.index, names=['dataset', 'run'])
    return df


def show_purity_table(ds_ids, dataset_display_names=None):
    """Display mean ± std cell-type purity across runs and datasets.

    Args:
        ds_ids: list of dataset ids.
        dataset_display_names: dict mapping ds_id -> display name.
    """
    from IPython.display import display

    df = load_purity_stats(ds_ids)
    if df.empty:
        print("No purity_per_mc.csv found.")
        return

    # Format as "mean ± std" strings
    formatted = df.apply(
        lambda row: f"{row['ct_purity_mean']:.3f} ± {row['ct_purity_std']:.3f}", axis=1
    ).unstack(level='dataset')

    if dataset_display_names:
        formatted = formatted.rename(columns=dataset_display_names)

    formatted.index.name = 'Method'
    display(formatted)
    return formatted


# ---------------------------------------------------------------------------
# Extract model key
# ---------------------------------------------------------------------------

def extract_model_key(folder_name, ds_id=None):
    """Strip dataset-specific tokens from a run folder name.

    The folder name encodes both model config (experiment_name, lambda values)
    and dataset-specific info (batch size, num prototypes, dataset id, version).
    This function keeps only the config part so the same experiment aligns
    as the same row across datasets.

    Examples:
        'recon_only_bs256_NP50_v7'   -> 'recon_only'
        'proto_umap_bs512_NP150_v7'  -> 'proto_umap'
        'seacells_K8'                -> 'seacells_K8'    (unchanged)
        'spectral_K16'               -> 'spectral_K16'   (unchanged)

    Args:
        folder_name: run folder name (not full path).
        ds_id:       dataset id — stripped if it appears as a leading prefix.

    Returns:
        Cleaned model config string.
    """
    name = folder_name

    if ds_id and name.startswith(ds_id + '_'):
        name = name[len(ds_id) + 1:]

    for pat in _DATASET_SPECIFIC_PATTERNS:
        name = re.sub(pat, '', name)

    name = re.sub(r'_+', '_', name).strip('_')
    return name or folder_name


# ---------------------------------------------------------------------------
# Clean (legacy helper, kept for compatibility)
# ---------------------------------------------------------------------------

def clean_run_names(df, level='run'):
    """Strip the longest common prefix from run names.

    Useful as a fallback when extract_model_key doesn't fully clean names.
    Works on both flat and MultiIndex DataFrames.
    """
    if isinstance(df.index, pd.MultiIndex):
        pos = df.index.names.index(level)
        names = list(df.index.get_level_values(pos))
        short = _strip_common_prefix(names)
        mapping = dict(zip(names, short))
        new_labels = [mapping[n] for n in names]
        df = df.copy()
        df.index = df.index.set_codes(
            [list(sorted(set(short))).index(l) for l in new_labels],
            level=pos,
        )
        df.index = df.index.set_levels(sorted(set(short)), level=pos)
        return df
    else:
        names = list(df.index)
        short = _strip_common_prefix(names)
        df = df.copy()
        df.index = pd.Index(short, name=df.index.name)
        return df


def filter_metrics(df, keys):
    """Keep only the specified metric columns (missing ones silently dropped)."""
    keep = [k for k in keys if k in df.columns]
    return df[keep]


# ---------------------------------------------------------------------------
# Reshape + Show
# ---------------------------------------------------------------------------

def pivot_table(df, metrics=None, dataset_display_names=None):
    """Reshape (dataset, run) MultiIndex df into rows=methods, cols=(Dataset, Metric)."""
    if metrics is not None:
        df = filter_metrics(df, metrics)

    df = df.apply(pd.to_numeric, errors='coerce')
    pivoted = df.unstack(level='dataset')

    datasets = df.index.get_level_values('dataset').unique()
    metric_cols = df.columns.tolist()

    data = {}
    tuples = []
    for ds in datasets:
        ds_label = (dataset_display_names or {}).get(ds, ds)
        for m in metric_cols:
            m_label = METRIC_DISPLAY_NAMES.get(m, m)
            tuples.append((ds_label, m_label))
            data[(ds_label, m_label)] = pivoted[(m, ds)]

    result = pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(tuples, names=['Dataset', 'Metric']))
    result.index.name = 'Method'
    return result


def show_table(df, metrics=None, dataset_display_names=None, exclude_keywords=None,
               include_keywords=None, latex=False, latex_caption=None, latex_label=None,
               return_dict=False):
    """Display results: rows = methods, cols = (Dataset x Metric).

    Best value bolded, second-best underlined per column.
    Vertical separator between dataset groups, dataset name centered above metrics.
    Methods with no metrics at all (all NaN) are excluded from the table and
    printed by name below it.

    Args:
        df:                    MultiIndex DataFrame from load_task1_multi.
        metrics:               metric keys to show. None = all.
        dataset_display_names: dict mapping ds_id -> display name.
        exclude_keywords:      list of strings; any row whose name contains one
                               of these keywords (case-insensitive) is hidden.
        include_keywords:      list of strings; when given, only rows whose name
                               contains at least one keyword are kept.
        latex:                 if True, print a LaTeX table instead of the styled display.
        latex_caption:         optional caption string for the LaTeX table.
        latex_label:           optional label string (e.g. 'tab:rare_cells').
        return_dict:           if True, skip display and return a dict
                               {dataset_name: DataFrame} — one per dataset, with
                               methods as rows and metrics as columns (mean±std merged).

    Displays the styled table (or prints LaTeX) and prints any excluded model names.
    Returns None normally; returns dict when return_dict=True.
    """
    from IPython.display import display
    from interpretable_ssl.evaluation.metric_helpers.embedding_tables import highlight_max_second

    # Expand metrics to include _std companions for any _mean/_median key requested,
    # so that the merge step below has both halves of the pair available.
    if metrics is not None:
        all_df_cols = set(df.columns)
        expanded = list(metrics)
        for m in list(metrics):
            for mean_sfx, std_sfx in (('_mean', '_std'), ('_median', '_std')):
                if m.endswith(mean_sfx):
                    companion = m[:-len(mean_sfx)] + std_sfx
                    if companion in all_df_cols and companion not in expanded:
                        expanded.append(companion)
            if m in MEAN_STD_PAIRS:
                companion = MEAN_STD_PAIRS[m]
                if companion in all_df_cols and companion not in expanded:
                    expanded.append(companion)
        metrics = expanded

    pivoted = pivot_table(df, metrics=metrics, dataset_display_names=dataset_display_names)

    if include_keywords:
        kw_lower = [kw.lower() for kw in include_keywords]
        mask = pivoted.index.map(lambda name: any(kw in name.lower() for kw in kw_lower))
        pivoted = pivoted.loc[mask]

    if exclude_keywords:
        kw_lower = [kw.lower() for kw in exclude_keywords]
        mask = pivoted.index.map(lambda name: any(kw in name.lower() for kw in kw_lower))
        pivoted = pivoted.loc[~mask]

    # Build mean/std pairs: start from registered MEAN_STD_PAIRS, then auto-detect
    # any remaining columns that follow the _mean/_std or _median/_std convention.
    all_metric_keys = list(pivoted.columns.get_level_values('Metric').unique())
    label_to_key    = {METRIC_DISPLAY_NAMES.get(k, k): k for k in all_metric_keys}

    auto_pairs = dict(MEAN_STD_PAIRS)
    for key in label_to_key.values():
        if key in auto_pairs:
            continue
        for mean_sfx, std_sfx in (('_mean', '_std'), ('_median', '_std')):
            if key.endswith(mean_sfx):
                candidate_std = key[:-len(mean_sfx)] + std_sfx
                if candidate_std in label_to_key.values() and key not in auto_pairs:
                    auto_pairs[key] = candidate_std

    # Merge mean ± std pairs into single string columns
    missing_std = []  # (dataset, metric) pairs shown as a bare mean, no std available
    for mean_col, std_col in auto_pairs.items():
        mean_label = METRIC_DISPLAY_NAMES.get(mean_col, mean_col)
        std_label  = METRIC_DISPLAY_NAMES.get(std_col, std_col)
        for ds in pivoted.columns.get_level_values('Dataset').unique():
            if (ds, mean_label) not in pivoted.columns:
                continue
            if (ds, std_label) not in pivoted.columns:
                # Mean is present but its std isn't. Previously this branch did
                # nothing, so the column rendered as a bare mean and the missing
                # "± std" was indistinguishable from a metric that never had one.
                missing_std.append(f"{ds}/{mean_label}")
                continue
            pivoted[(ds, mean_label)] = pivoted.apply(
                lambda r, d=ds, ml=mean_label, sl=std_label:
                    f"{r[(d, ml)]:.3f} ± {r[(d, sl)]:.3f}"
                    if pd.notna(r[(d, ml)]) and pd.notna(r[(d, sl)]) else '-',
                axis=1
            )
            pivoted = pivoted.drop(columns=[(ds, std_label)])

    if missing_std:
        print(f"WARNING: no std found for {len(missing_std)} column(s), shown as bare "
              f"means: {', '.join(sorted(missing_std))}\n"
              f"         Re-run eval for those runs to write the std into metrics.json.")

    # Rows where every cell is NaN have no metrics — exclude them.
    all_nan_mask = pivoted.isna().all(axis=1)
    excluded = pivoted.index[all_nan_mask].tolist()
    pivoted = pivoted.loc[~all_nan_mask]

    if excluded:
        print("No metrics available (excluded from table):", ", ".join(excluded))

    if latex:
        print(_pivot_to_latex(pivoted, caption=latex_caption, label=latex_label))
        return

    if return_dict:
        datasets = pivoted.columns.get_level_values('Dataset').unique()
        return {
            ds: pivoted[ds].copy()
            for ds in datasets
        }

    n_metrics = len(pivoted.columns.get_level_values('Metric').unique())
    datasets  = pivoted.columns.get_level_values('Dataset').unique()

    col_styles = []
    for i in range(len(datasets)):
        first = i * n_metrics
        col_styles += [
            {'selector': f'th.col_heading.level1.col{first}', 'props': 'border-left: 2px solid #555;'},
            {'selector': f'td.col{first}',                    'props': 'border-left: 2px solid #555;'},
        ]
    col_styles += [
        {'selector': 'th.col_heading.level0', 'props': 'text-align: center; font-weight: bold; border-bottom: 1px solid #aaa;'},
        {'selector': 'th.col_heading.level1', 'props': 'text-align: center;'},
        {'selector': 'th.row_heading',        'props': 'text-align: left;'},
    ]

    def _fmt(v):
        if isinstance(v, str):
            return v  # already formatted (e.g. "mean ± std")
        try:
            return f"{v:.2f}"
        except (TypeError, ValueError):
            return '-'

    styler = (
        pivoted.style
        .apply(highlight_max_second, axis=0)
        .format(_fmt, na_rep='-')
        .set_table_styles(col_styles)
    )
    display(styler)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _pivot_to_latex(pivoted, caption=None, label=None):
    """Render a pivoted (Dataset x Metric) DataFrame as a LaTeX table string."""
    datasets = list(pivoted.columns.get_level_values('Dataset').unique())
    metrics  = list(pivoted.columns.get_level_values('Metric').unique())
    n_m = len(metrics)

    col_fmt = 'l' + '|'.join(['c' * n_m] * len(datasets))

    lines = [
        r'\begin{table*}[t]',
        r'\centering',
    ]
    if caption:
        lines.append(r'\caption{' + caption + r'}')
    if label:
        lines.append(r'\label{' + label + r'}')
    lines += [
        r'\resizebox{\textwidth}{!}{%',
        r'\begin{tabular}{' + col_fmt + r'}',
        r'\toprule',
    ]

    # Dataset multi-column header
    ds_header = [' '] + [
        r'\multicolumn{' + str(n_m) + r'}{c}{' + ds + r'}' for ds in datasets
    ]
    lines.append(' & '.join(ds_header) + r' \\')

    # cmidrule under each dataset group
    cmidrules = []
    for i in range(len(datasets)):
        start = i * n_m + 2
        end   = start + n_m - 1
        cmidrules.append(r'\cmidrule(lr){' + str(start) + '-' + str(end) + r'}')
    lines.append(' '.join(cmidrules))

    # Metric sub-header with direction arrows
    metric_parts = ['Method']
    for _ds in datasets:
        for m in metrics:
            direction = _DISPLAY_DIRECTION.get(m)
            if direction == 'up':
                arrow = r' $\uparrow$'
            elif direction == 'down':
                arrow = r' $\downarrow$'
            else:
                arrow = ''
            metric_parts.append(m + arrow)
    lines.append(' & '.join(metric_parts) + r' \\')
    lines.append(r'\midrule')

    # Data rows
    for method in pivoted.index:
        row_parts = [method]
        for ds in datasets:
            for m in metrics:
                val = pivoted.loc[method, (ds, m)]
                if isinstance(val, str):
                    row_parts.append('$' + val.replace('±', r'\pm') + '$')
                elif pd.isna(val):
                    row_parts.append('---')
                else:
                    row_parts.append(f'{float(val):.3f}')
        lines.append(' & '.join(row_parts) + r' \\')

    lines += [
        r'\bottomrule',
        r'\end{tabular}}',
        r'\end{table*}',
    ]
    return '\n'.join(lines)


def _strip_common_prefix(names):
    if len(names) <= 1:
        return list(names)
    prefix = os.path.commonprefix(names)
    for sep in ('_', '-'):
        idx = prefix.rfind(sep)
        if idx > 0:
            prefix = prefix[:idx + 1]
            break
    return [n[len(prefix):] or n for n in names]
