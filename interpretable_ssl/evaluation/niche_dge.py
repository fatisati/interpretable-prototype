"""Niche-program recovery (DGE) — library version of the plan1 rebuttal notebook.

What this measures: do a method's metacells, *within one cell type*, reproduce the
niche-vs-rest transcriptional differences that are visible in the real single cells?

    ground truth   niche-X-vs-rest Wilcoxon DE per (cell type, niche), real single cells
    branch 1       Pearson r / Kendall tau between metacell-level logFC and the
                   single-cell logFC, over the ground-truth gene list. No significance
                   gate  (MetaQ's continuous recovery metric)  -> PRIMARY
    branch 2       size-weighted Welch t-test at metacell level; TPR = fraction of the
                   ground-truth gene list that also clears p<0.05 there
                   (SuperCell's recipe)                          -> SECONDARY

Everything is cached to CSV and reloaded when present, so a Colab disconnect costs one
step, not the run. Pass force=True to recompute.

Provenance: the pipeline is a direct port of
`neurips_manuscript/rebuttle/notebooks/plan1_niche_recovery_eval.ipynb` so numbers stay
comparable with the results already reported in the rebuttal. Two things it deliberately
keeps from that notebook:

  - pseudobulk is rebuilt from each run's real per-cell assignment against the dataset's
    raw `counts` layer, NOT from any saved `metacells.h5ad`. scProto's saved metacells are
    decoder reconstructions, so they are not a real-cell aggregation and are not
    comparable to SEACells' pseudobulk.
  - metacell labelling is plain majority vote for every method, via
    `niche_recovery.mc_majority_label`, so no method gets a labelling advantage.

Known caveat, carried over: this validates *recovery* of programs already visible in the
single cells. It is not on its own a *discovery* claim. `discovery_candidates` below is
the non-circular version — pairs whose genes are invisible to pooled cell-type DE.
"""

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr, kendalltau

from interpretable_ssl.evaluation.niche_recovery import mc_majority_label

# defaults — kept identical to plan1_niche_recovery_eval.ipynb
MIN_POS_CELLS = 5      # min cells in the niche-positive group for a ground-truth pair
MIN_CTRL_CELLS = 20    # min cells in the control group
MIN_POS_MC = 2         # min metacells in the positive group for a scored pair
MIN_CTRL_MC = 2
MIN_GENES = 5          # skip a pair with fewer usable ground-truth genes than this
GT_MAX_GENES = 100     # cap the ground-truth gene list per pair, for stability
PSEUDOCOUNT = 1.0      # on the CPM scale, MetaCell1's lfp idea
EXCLUDE_NICHES = ('Excluded',)


# ---------------------------------------------------------------------------
# per-cell metacell labels
# ---------------------------------------------------------------------------

def labels_from_run(ds_id, keyword, obs_names):
    """Hard per-cell metacell id for an already-saved run.

    Reuses paper_figures._load_cell_assignments, which resolves the run directory from a
    keyword and falls back from cell_assignments.csv to umap_cells.csv. Returns a Series
    indexed like obs_names, or None if the run has neither file.
    """
    from interpretable_ssl.evaluation.paper_figures import _load_cell_assignments
    return _load_cell_assignments(ds_id, keyword, obs_names)


def labels_from_trainer(trainer, adata=None):
    """Hard per-cell metacell id for a freshly trained model (argmax of soft assignment)."""
    from interpretable_ssl.evaluation.collapse_diagnostics import (
        labels_from_trainer as _lft,
    )
    return _lft(trainer, adata)


# ---------------------------------------------------------------------------
# Setup A — ground truth from real single cells
# ---------------------------------------------------------------------------

def compute_ground_truth(adata, ct_key, niche_key,
                         min_pos=MIN_POS_CELLS, min_ctrl=MIN_CTRL_CELLS,
                         exclude_niches=EXCLUDE_NICHES):
    """niche-vs-rest Wilcoxon DE within each cell type, on real single cells.

    Returns dict[(cell_type, niche)] -> full ranked DataFrame
    (names, logfoldchanges, pvals, pvals_adj), sorted by |logFC| descending.
    Runs on log-normalized `adata.X` (the same test family niche_recovery.py uses).
    """
    exclude = set(exclude_niches)
    gt = {}
    for ct in adata.obs[ct_key].unique():
        ct_ad = adata[adata.obs[ct_key] == ct].copy()
        niches = [n for n in ct_ad.obs[niche_key].dropna().unique() if n not in exclude]

        for niche in niches:
            pos = (ct_ad.obs[niche_key] == niche).to_numpy()
            if int(pos.sum()) < min_pos or int((~pos).sum()) < min_ctrl:
                continue

            tmp = ct_ad.copy()
            tmp.obs['_group'] = np.where(pos, 'pos', 'ctrl')
            sc.tl.rank_genes_groups(
                tmp, '_group', groups=['pos'], reference='ctrl', method='wilcoxon',
            )
            df = sc.get.rank_genes_groups_df(tmp, group='pos')
            df = df.reindex(df['logfoldchanges'].abs().sort_values(ascending=False).index)
            gt[(ct, niche)] = df.reset_index(drop=True)

    return gt


def save_ground_truth(gt, path):
    """Flatten to one long CSV: cell_type, niche, names, logfoldchanges, pvals, pvals_adj."""
    rows = []
    for (ct, niche), df in gt.items():
        d = df.copy()
        d.insert(0, 'niche', niche)
        d.insert(0, 'cell_type', ct)
        rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    return out


def load_ground_truth(path):
    """Inverse of save_ground_truth."""
    long = pd.read_csv(path)
    return {
        (ct, niche): d.drop(columns=['cell_type', 'niche']).reset_index(drop=True)
        for (ct, niche), d in long.groupby(['cell_type', 'niche'])
    }


def get_ground_truth(adata, path, ct_key, niche_key, force=False, **kwargs):
    """Load-if-exists wrapper around compute_ground_truth. Prints where it came from."""
    if os.path.exists(path) and not force:
        gt = load_ground_truth(path)
        print(f'ground truth: loaded from {path}  ({len(gt)} (cell type, niche) pairs)')
        return gt

    print('ground truth: computing from single cells ...')
    gt = compute_ground_truth(adata, ct_key, niche_key, **kwargs)
    save_ground_truth(gt, path)
    print(f'ground truth: computed and saved to {path}  ({len(gt)} pairs)')
    return gt


# ---------------------------------------------------------------------------
# Setup B / C — metacell labels and raw-count pseudobulk
# ---------------------------------------------------------------------------

def majority_label_metacells(labels, adata, ct_key, niche_key):
    """One row per metacell: majority cell type and niche, their purities, and size.

    `labels` = per-cell metacell id, indexed like adata.obs_names.
    """
    lab = pd.Series(labels).reindex(adata.obs_names)
    df = pd.DataFrame({
        'metacell_id': lab,
        ct_key: adata.obs[ct_key].reindex(lab.index),
        niche_key: adata.obs[niche_key].reindex(lab.index),
    }).dropna(subset=['metacell_id'])

    fake_ad = SimpleNamespace(obs=df)
    ct_label, ct_purity = mc_majority_label(fake_ad, 'metacell_id', ct_key)
    niche_label, niche_purity = mc_majority_label(fake_ad, 'metacell_id', niche_key)
    size = df.groupby('metacell_id').size().reindex(ct_label.index)

    return pd.DataFrame({
        'cell_type': ct_label, 'cell_type_purity': ct_purity,
        'niche': niche_label, 'niche_purity': niche_purity,
        'n_cells': size,
    })


def build_pseudobulk(labels, adata, raw_layer='counts'):
    """Sum real raw counts per metacell over its real member cells -> (metacells x genes)."""
    lab = pd.Series(labels).reindex(adata.obs_names).dropna()
    raw = adata.layers[raw_layer] if raw_layer in adata.layers else adata.X
    raw = raw.toarray() if hasattr(raw, 'toarray') else np.asarray(raw)
    raw_df = pd.DataFrame(raw, index=adata.obs_names, columns=adata.var_names)
    cells = lab.index.intersection(raw_df.index)
    return raw_df.loc[cells].groupby(lab.loc[cells]).sum()


def normalize_cpm(pb_df):
    lib = pb_df.sum(axis=1)
    return pb_df.div(lib.replace(0, np.nan), axis=0) * 1e6


def pseudobulk_spot_check(pseudobulk_df, mc_labels_df):
    """Total pseudobulk counts should scale ~linearly with member-cell count if raw-count
    aggregation held. A weak correlation means pre-normalized values were averaged instead.
    """
    size = mc_labels_df['n_cells'].reindex(pseudobulk_df.index)
    return float(np.corrcoef(size, pseudobulk_df.sum(axis=1))[0, 1])


# ---------------------------------------------------------------------------
# Branch 1 / Branch 2
# ---------------------------------------------------------------------------

def _pair_groups(cpm, mc_labels_df, ct, niche):
    """Positive / control metacell ids for one (cell type, niche) pair, or None."""
    mcs = mc_labels_df[mc_labels_df['cell_type'] == ct]
    pos = mcs.index[mcs['niche'] == niche].intersection(cpm.index)
    ctrl = mcs.index[mcs['niche'] != niche].intersection(cpm.index)
    if len(pos) < MIN_POS_MC or len(ctrl) < MIN_CTRL_MC:
        return None
    return pos, ctrl


def branch1_recovery(pseudobulk_df, mc_labels_df, ground_truth,
                     max_genes=GT_MAX_GENES, pseudocount=PSEUDOCOUNT):
    """Continuous recovery: correlation of metacell-level logFC with single-cell logFC."""
    cpm = normalize_cpm(pseudobulk_df)
    rows = []
    for (ct, niche), gt_df in ground_truth.items():
        gt_sig = gt_df[gt_df['pvals_adj'] < 0.05].head(max_genes)
        genes = [g for g in gt_sig['names'] if g in cpm.columns]
        if len(genes) < MIN_GENES:
            continue

        groups = _pair_groups(cpm, mc_labels_df, ct, niche)
        if groups is None:
            continue
        pos_mc, ctrl_mc = groups

        mean_pos = cpm.loc[pos_mc, genes].mean(axis=0)
        mean_ctrl = cpm.loc[ctrl_mc, genes].mean(axis=0)
        mc_logfc = np.log2((mean_pos + pseudocount) / (mean_ctrl + pseudocount))
        sc_logfc = gt_sig.set_index('names').loc[genes, 'logfoldchanges']

        rows.append({
            'cell_type': ct, 'niche': niche, 'n_genes': len(genes),
            'n_pos_mc': len(pos_mc), 'n_ctrl_mc': len(ctrl_mc),
            'pearson_r': pearsonr(mc_logfc, sc_logfc)[0],
            'kendall_tau': kendalltau(mc_logfc, sc_logfc)[0],
        })
    return pd.DataFrame(rows)


def _weighted_ttest(a, wa, b, wb):
    """Two-sample Welch t-test with per-sample weights (metacell size)."""
    from statsmodels.stats.weightstats import DescrStatsW, CompareMeans
    da = DescrStatsW(a, weights=wa, ddof=0)
    db = DescrStatsW(b, weights=wb, ddof=0)
    return CompareMeans(da, db).ttest_ind(usevar='unequal')[1]


def branch2_recovery(pseudobulk_df, mc_labels_df, ground_truth, alpha=0.05):
    """Discrete recovery: TPR of the ground-truth gene list under a size-weighted t-test."""
    cpm = normalize_cpm(pseudobulk_df)
    rows = []
    for (ct, niche), gt_df in ground_truth.items():
        genes = [g for g in gt_df.loc[gt_df['pvals_adj'] < alpha, 'names'] if g in cpm.columns]
        if len(genes) < MIN_GENES:
            continue

        groups = _pair_groups(cpm, mc_labels_df, ct, niche)
        if groups is None:
            continue
        pos_mc, ctrl_mc = groups

        w_pos = mc_labels_df.loc[pos_mc, 'n_cells'].to_numpy()
        w_ctrl = mc_labels_df.loc[ctrl_mc, 'n_cells'].to_numpy()
        n_sig = sum(
            _weighted_ttest(cpm.loc[pos_mc, g].to_numpy(), w_pos,
                            cpm.loc[ctrl_mc, g].to_numpy(), w_ctrl) < alpha
            for g in genes
        )
        rows.append({
            'cell_type': ct, 'niche': niche, 'n_genes': len(genes),
            'n_pos_mc': len(pos_mc), 'n_ctrl_mc': len(ctrl_mc),
            'tpr': n_sig / len(genes),
        })
    return pd.DataFrame(rows)


def macro_average(branch_df, value_cols):
    """Average across niches within a cell type, then across cell types — the same
    convention the manuscript already uses for purity."""
    if branch_df.empty:
        return pd.Series({c: np.nan for c in value_cols})
    return branch_df.groupby('cell_type')[value_cols].mean().mean()


def macro_std(branch_df, value_cols):
    """Std across cell types of the per-cell-type means — reported next to every mean."""
    if branch_df.empty:
        return pd.Series({f'{c}_std': np.nan for c in value_cols})
    per_ct = branch_df.groupby('cell_type')[value_cols].mean()
    return per_ct.std().rename(lambda c: f'{c}_std')


# ---------------------------------------------------------------------------
# per-arm driver, with caching
# ---------------------------------------------------------------------------

def _slug(name):
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in str(name))


def run_niche_dge(name, labels, adata, ground_truth, out_dir,
                  ct_key, niche_key, raw_layer='counts', force=False):
    """Full per-arm pipeline: labels -> metacell table -> pseudobulk -> both branches.

    Writes `<name>_per_pair.csv` and `<name>_summary.csv` into out_dir and reloads them on
    a later run unless force=True. Returns dict with keys:
        per_pair  one row per (cell type, niche) — both branches merged
        summary   one row: macro-averaged pearson_r / kendall_tau / tpr, each with a std
        mc_labels per-metacell table (recomputed, cheap)
    """
    os.makedirs(out_dir, exist_ok=True)
    slug = _slug(name)
    pair_path = os.path.join(out_dir, f'{slug}_per_pair.csv')
    summ_path = os.path.join(out_dir, f'{slug}_summary.csv')

    mc_labels = majority_label_metacells(labels, adata, ct_key, niche_key)

    if os.path.exists(pair_path) and os.path.exists(summ_path) and not force:
        print(f'[{name}] loaded from {pair_path}')
        return {
            'per_pair': pd.read_csv(pair_path),
            'summary': pd.read_csv(summ_path, index_col=0),
            'mc_labels': mc_labels,
        }

    print(f'[{name}] computing — {len(mc_labels)} metacells, '
          f'median size {mc_labels["n_cells"].median():.0f} cells')

    pb = build_pseudobulk(labels, adata, raw_layer=raw_layer)
    print(f'[{name}] pseudobulk spot check corr(total counts, n_cells) = '
          f'{pseudobulk_spot_check(pb, mc_labels):.3f}   (want ~1.0)')

    b1 = branch1_recovery(pb, mc_labels, ground_truth)
    b2 = branch2_recovery(pb, mc_labels, ground_truth)

    key = ['cell_type', 'niche']
    if b1.empty and b2.empty:
        per_pair = pd.DataFrame(columns=key + ['pearson_r', 'kendall_tau', 'tpr'])
    elif b1.empty:
        per_pair = b2
    elif b2.empty:
        per_pair = b1
    else:
        per_pair = b1.merge(b2[key + ['tpr']], on=key, how='outer')

    # attach the pos-group size stats the size-floor diagnostic needs
    size_stats = (
        mc_labels.groupby(['cell_type', 'niche'])['n_cells']
        .agg(median_pos_mc_size='median', min_pos_mc_size='min')
        .reset_index()
    )
    pur_stats = (
        mc_labels.groupby(['cell_type', 'niche'])['niche_purity']
        .mean().rename('mean_niche_purity').reset_index()
    )
    per_pair = per_pair.merge(size_stats, on=key, how='left')
    per_pair = per_pair.merge(pur_stats, on=key, how='left')

    summary = pd.concat([
        macro_average(b1, ['pearson_r', 'kendall_tau']),
        macro_std(b1, ['pearson_r', 'kendall_tau']),
        macro_average(b2, ['tpr']),
        macro_std(b2, ['tpr']),
    ])
    summary['n_pairs'] = len(per_pair)
    summary['n_metacells'] = len(mc_labels)
    summary['median_mc_size'] = mc_labels['n_cells'].median()
    summary = summary.to_frame(name).T

    per_pair.to_csv(pair_path, index=False)
    summary.to_csv(summ_path)
    print(f'[{name}] computed and saved to {pair_path}')

    return {'per_pair': per_pair, 'summary': summary, 'mc_labels': mc_labels}


# ---------------------------------------------------------------------------
# comparison across arms
# ---------------------------------------------------------------------------

def summary_table(results):
    """Stack every arm's summary row into one table. results = dict[name] -> run_niche_dge out."""
    return pd.concat([r['summary'] for r in results.values()])


def compare_arms(results, reference, metric='pearson_r'):
    """Paired per-(cell type, niche) comparison of every arm against `reference`.

    Returns one row per pair, with `<metric>_<arm>` columns and a `<metric>_gap_<arm>`
    column = arm minus reference. Positive gap = the arm beats the reference on that pair.
    """
    key = ['cell_type', 'niche']
    cols = key + [metric, 'tpr', 'mean_niche_purity', 'median_pos_mc_size']
    cols = [c for c in dict.fromkeys(cols)]

    def _take(name):
        df = results[name]['per_pair']
        keep = [c for c in cols if c in df.columns]
        return df[keep].rename(columns={c: f'{c}_{name}' for c in keep if c not in key})

    out = _take(reference)
    for name in results:
        if name == reference:
            continue
        out = out.merge(_take(name), on=key, how='outer')
        if f'{metric}_{name}' in out.columns and f'{metric}_{reference}' in out.columns:
            out[f'{metric}_gap_{name}'] = (
                out[f'{metric}_{name}'] - out[f'{metric}_{reference}']
            )
    return out


def size_floor_sweep(results, size_arm, floors=(0, 5, 10, 20, 30, 50),
                     metrics=('pearson_r', 'tpr')):
    """Does an arm's disadvantage shrink once its own metacells are well-populated?

    Restricts the paired comparison to (cell type, niche) pairs where `size_arm`'s
    niche-positive metacells clear an increasing median size floor, and reports every arm's
    mean metric on that shrinking subset. A gap that closes as the floor rises means "too
    few real cells per metacell for a stable pseudobulk logFC", not "no signal".
    """
    key = ['cell_type', 'niche']
    base = results[size_arm]['per_pair'][key + ['median_pos_mc_size']]

    rows = []
    for floor in floors:
        keep = base.loc[base['median_pos_mc_size'] >= floor, key]
        row = {'floor': floor, 'n_pairs': len(keep)}
        for name, r in results.items():
            sub = r['per_pair'].merge(keep, on=key, how='inner')
            for m in metrics:
                row[f'{name}_{m}'] = sub[m].mean() if m in sub.columns else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# R4 — the non-circular discovery check
# ---------------------------------------------------------------------------

def pooled_celltype_de(adata, ct_key, path=None, force=False, reference='rest'):
    """Standard cell-type DE with niche ignored — the analysis this paper argues is blind
    to niche programs. Used as the negative control for `discovery_candidates`.

    Returns dict[cell_type] -> ranked DataFrame.
    """
    if path and os.path.exists(path) and not force:
        long = pd.read_csv(path)
        print(f'pooled cell-type DE: loaded from {path}')
        return {ct: d.drop(columns=['cell_type']).reset_index(drop=True)
                for ct, d in long.groupby('cell_type')}

    print('pooled cell-type DE: computing ...')
    tmp = adata.copy()
    sc.tl.rank_genes_groups(tmp, ct_key, method='wilcoxon', reference=reference)
    out = {}
    for ct in tmp.obs[ct_key].unique():
        try:
            out[ct] = sc.get.rank_genes_groups_df(tmp, group=str(ct))
        except (KeyError, ValueError):
            continue

    if path:
        rows = []
        for ct, d in out.items():
            d = d.copy()
            d.insert(0, 'cell_type', ct)
            rows.append(d)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.concat(rows, ignore_index=True).to_csv(path, index=False)
        print(f'pooled cell-type DE: saved to {path}')
    return out


def discovery_candidates(ground_truth, pooled_de, results, arm,
                         alpha=0.05, max_genes=GT_MAX_GENES, top_n=20):
    """Genes that are niche programs but NOT visible to standard pooled cell-type DE.

    For each (cell type, niche) pair, returns the ground-truth genes that are
    significant at the niche level (pvals_adj < alpha) and NOT significant in the pooled
    cell-type DE for that same cell type — i.e. the programs the paper claims existing
    analysis averages away. `results[arm]['per_pair']` supplies the recovery score, so the
    output is ranked by how well the arm actually recovers each candidate pair.

    This is the R4 evidence. Recovery of genes already visible in single cells is not.
    """
    per_pair = results[arm]['per_pair'].set_index(['cell_type', 'niche'])

    rows = []
    for (ct, niche), gt_df in ground_truth.items():
        niche_genes = gt_df.loc[gt_df['pvals_adj'] < alpha, 'names'].head(max_genes)
        if len(niche_genes) < MIN_GENES:
            continue

        pooled = pooled_de.get(ct)
        if pooled is None:
            continue
        pooled_sig = set(pooled.loc[pooled['pvals_adj'] < alpha, 'names'])

        hidden = [g for g in niche_genes if g not in pooled_sig]
        if not hidden:
            continue

        score = per_pair['pearson_r'].get((ct, niche), np.nan)
        rows.append({
            'cell_type': ct, 'niche': niche,
            'n_niche_genes': len(niche_genes),
            'n_hidden_from_pooled': len(hidden),
            'frac_hidden': len(hidden) / len(niche_genes),
            f'pearson_r_{arm}': score,
            'hidden_genes': ', '.join(hidden[:top_n]),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(
        [f'pearson_r_{arm}', 'n_hidden_from_pooled'], ascending=False,
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

_ACCENT = '#3B82F6'
_NEUTRAL = ['#9CA3AF', '#6B7280', '#D1D5DB', '#4B5563']


def _colors(names, highlight):
    out, i = {}, 0
    for n in names:
        if n == highlight:
            out[n] = _ACCENT
        else:
            out[n] = _NEUTRAL[i % len(_NEUTRAL)]
            i += 1
    return out


def plot_dge_summary(summary_df, highlight=None, out_path=None, title=None):
    """Bar per metric, one bar per arm, error bars from the matching *_std column."""
    import matplotlib.pyplot as plt

    metrics = [m for m in ('pearson_r', 'kendall_tau', 'tpr') if m in summary_df.columns]
    colors = _colors(summary_df.index, highlight)

    fig, axes = plt.subplots(1, len(metrics), figsize=(3.7 * len(metrics), 3.5))
    axes = np.atleast_1d(axes)
    titles = {'pearson_r': 'Branch 1: Pearson r',
              'kendall_tau': 'Branch 1: Kendall tau',
              'tpr': 'Branch 2: TPR'}

    for ax, m in zip(axes, metrics):
        vals = summary_df[m]
        err = summary_df.get(f'{m}_std')
        ax.bar(vals.index, vals.values, width=0.55,
               color=[colors[n] for n in vals.index],
               yerr=None if err is None else err.values,
               capsize=3, ecolor='#374151')
        ax.set_title(titles.get(m, m), fontsize=10)
        ax.tick_params(axis='x', labelrotation=20, labelsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        for i, v in enumerate(vals.values):
            if np.isfinite(v):
                ax.text(i, v, f'{v:.2f}', ha='center',
                        va='bottom' if v >= 0 else 'top', fontsize=8)

    if title:
        fig.suptitle(title, y=1.03, fontsize=11)
    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, bbox_inches='tight', dpi=200)
        print(f'saved {out_path}')
    return fig


def plot_size_floor_sweep(sweep, arms, highlight=None,
                          metrics=('pearson_r', 'tpr'), out_path=None, title=None):
    """One panel per metric: every arm's mean vs. the minimum metacell-size floor."""
    import matplotlib.pyplot as plt

    colors = _colors(arms, highlight)
    titles = {'pearson_r': 'Branch 1: Pearson r', 'tpr': 'Branch 2: TPR'}

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.7 * len(metrics), 3.5))
    axes = np.atleast_1d(axes)

    for ax, m in zip(axes, metrics):
        for name in arms:
            col = f'{name}_{m}'
            if col not in sweep.columns:
                continue
            ax.plot(sweep['floor'], sweep[col], marker='o',
                    color=colors[name], label=name)
        for _, row in sweep.iterrows():
            first = f'{arms[0]}_{m}'
            if first in sweep.columns and np.isfinite(row[first]):
                ax.annotate(f"n={row['n_pairs']:.0f}", (row['floor'], row[first]),
                            fontsize=7, textcoords='offset points',
                            xytext=(0, 6), ha='center')
        ax.set_xlabel('min median metacell size (cells)', fontsize=9)
        ax.set_title(titles.get(m, m), fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)

    fig.suptitle(title or 'Does the gap close once metacells are well-populated?',
                 y=1.03, fontsize=11)
    fig.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, bbox_inches='tight', dpi=200)
        print(f'saved {out_path}')
    return fig
