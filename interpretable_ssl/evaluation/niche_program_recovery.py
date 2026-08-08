"""niche_program_recovery.py — plan1.md's within-cell-type niche transcriptional-program
recovery evaluation (Setup A/B/C + Branch 1/2), as reusable functions.

Ported from neurips_manuscript/rebuttle/notebooks/plan1_niche_recovery_eval.ipynb, where
this logic first lived inline. That notebook is left as-is (already run, already validated);
this module is for new scoped experiments (e.g. the Fibroblast-only affinity ablation) that
need the same evaluation without duplicating the code.

Usage in a notebook:
    from interpretable_ssl.evaluation.niche_program_recovery import (
        compute_ground_truth, save_ground_truth, load_ground_truth,
        majority_label_metacells, build_pseudobulk, normalize_cpm,
        branch1_recovery, branch2_recovery, macro_average,
    )
"""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr, kendalltau
from statsmodels.stats.weightstats import DescrStatsW, CompareMeans

from interpretable_ssl.evaluation.niche_recovery import mc_majority_label

EXCLUDE_NICHES = {'Excluded'}


# ---------------------------------------------------------------------------
# Setup A — ground truth: niche-vs-rest DE within cell type, real single cells
# ---------------------------------------------------------------------------

def compute_ground_truth(adata, ct_key, niche_key, min_pos=5, min_ctrl=20,
                          exclude_niches=EXCLUDE_NICHES):
    """Niche-X-vs-rest DE within each cell type, real single cells, true labels.

    Returns dict[(cell_type, niche)] -> full ranked gene DataFrame
    (names, logfoldchanges, pvals, pvals_adj), sorted by |logfoldchanges| desc.
    """
    gt = {}
    for ct in adata.obs[ct_key].unique():
        ct_ad = adata[adata.obs[ct_key] == ct].copy()
        niches = [n for n in ct_ad.obs[niche_key].dropna().unique() if n not in exclude_niches]

        for niche in niches:
            pos = (ct_ad.obs[niche_key] == niche).to_numpy()
            npos, nctrl = int(pos.sum()), int((~pos).sum())
            if npos < min_pos or nctrl < min_ctrl:
                continue

            tmp = ct_ad.copy()
            tmp.obs['_group'] = np.where(pos, 'pos', 'ctrl')
            sc.tl.rank_genes_groups(
                tmp, '_group', groups=['pos'], reference='ctrl', method='wilcoxon',
            )
            df = sc.get.rank_genes_groups_df(tmp, group='pos')
            df = df.reindex(df['logfoldchanges'].abs().sort_values(ascending=False).index)
            df = df.reset_index(drop=True)
            gt[(ct, niche)] = df

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
    df = pd.read_csv(path)
    return {
        (ct, niche): sub.drop(columns=['cell_type', 'niche']).reset_index(drop=True)
        for (ct, niche), sub in df.groupby(['cell_type', 'niche'])
    }


# ---------------------------------------------------------------------------
# Setup B — majority-vote metacell labeling
# ---------------------------------------------------------------------------

def majority_label_metacells(cell_assign_df, ct_key, niche_key):
    """Plain majority vote (default option in plan1.md Setup B) — reuses
    niche_recovery.mc_majority_label as-is (it only needs a `.obs`-like frame, so a
    SimpleNamespace stands in for an AnnData). Requires cell_assign_df to already have
    a 'metacell_id' column and the niche_key column joined in.
    """
    fake_ad = SimpleNamespace(obs=cell_assign_df)
    ct_label, ct_purity = mc_majority_label(fake_ad, 'metacell_id', ct_key)
    niche_label, niche_purity = mc_majority_label(fake_ad, 'metacell_id', niche_key)
    size = cell_assign_df.groupby('metacell_id').size().reindex(ct_label.index)
    return pd.DataFrame({
        'cell_type': ct_label, 'cell_type_purity': ct_purity,
        'niche': niche_label, 'niche_purity': niche_purity,
        'n_cells': size,
    })


def load_cell_assignments(run_dir, adata, niche_key):
    """Real per-cell hard metacell assignment (already saved by training) + real niche
    label joined in (cell_assignments.csv itself only carries cell type, not niche)."""
    df = pd.read_csv(os.path.join(run_dir, 'cell_assignments.csv')).set_index('cell_id')
    df[niche_key] = adata.obs[niche_key].reindex(df.index)
    return df


# ---------------------------------------------------------------------------
# Soft-label variant of Setup B/C -- same Branch 1/2 functions below consume
# whatever mc_labels_df/pseudobulk_df they're given, hard- or soft-derived,
# without caring which; only how those two frames get built differs here.
# ---------------------------------------------------------------------------

def load_soft_assignments(run_dir):
    """Load a (n_cells x n_protos) soft assignment matrix + the cell_id order it's
    aligned to, as saved by save_soft_assignments (batch_correct_baselines.py) --
    the shared format both scProto's and SEACells' soft-assignment saves use, so
    this one loader works for either."""
    import scipy.sparse as sp

    S = sp.load_npz(os.path.join(run_dir, 'soft_assignments.npz')).tocsc()
    cell_ids = np.load(os.path.join(run_dir, 'soft_assignments_cell_ids.npy'), allow_pickle=True)
    return S, cell_ids


def soft_label_metacells(S, cell_ids, adata, ct_key, niche_key):
    """Soft analogue of majority_label_metacells: each metacell (column of S) gets
    the soft-weighted-majority cell_type/niche across ALL cells (weighted by that
    column's S values, not just cells that would hard-argmax into it), and its
    'n_cells' is the column's effective size (sum of weights, generally fractional)
    rather than a hard count -- branch2_recovery's weighted_ttest already treats
    'n_cells' purely as a sample weight, so this plugs in without any changes there.
    """
    obs = adata.obs.loc[cell_ids]

    def _weighted_mode_and_purity(labels):
        cats = pd.Categorical(labels)
        onehot = np.eye(len(cats.categories))[cats.codes]  # (N, L)
        w = np.asarray(S.T @ onehot)                        # (K, L)
        eff_size = w.sum(axis=1)
        best = w.argmax(axis=1)
        label = np.array(cats.categories)[best]
        purity = np.divide(w.max(axis=1), eff_size,
                            out=np.zeros_like(eff_size), where=eff_size > 0)
        return label, purity, eff_size

    ct_label, ct_purity, eff_size = _weighted_mode_and_purity(obs[ct_key].to_numpy())
    niche_label, niche_purity, _ = _weighted_mode_and_purity(obs[niche_key].to_numpy())

    mc_ids = [str(k) for k in range(S.shape[1])]
    return pd.DataFrame({
        'cell_type': ct_label, 'cell_type_purity': ct_purity,
        'niche': niche_label, 'niche_purity': niche_purity,
        'n_cells': eff_size,
    }, index=mc_ids).rename_axis('metacell_id')


def build_soft_pseudobulk(S, cell_ids, adata, raw_layer='counts'):
    """Soft analogue of build_pseudobulk: pseudobulk_k = sum_i S[i,k] * raw_counts[i]
    (a weighted sum instead of a hard groupby-sum over exactly-assigned members)."""
    raw = adata[cell_ids].layers[raw_layer]
    raw = raw.toarray() if hasattr(raw, 'toarray') else np.asarray(raw)
    pb = np.asarray(S.T @ raw)  # (K, n_genes)
    mc_ids = [str(k) for k in range(S.shape[1])]
    return pd.DataFrame(pb, index=mc_ids, columns=adata.var_names).rename_axis('metacell_id')


def build_decoded_pseudobulk(decoded, var_names):
    """scProto-only third pseudobulk variant: each metacell's expression profile
    comes from the model's OWN decoder reconstruction of that prototype
    (trainer.decode_prototypes()'s [K, n_genes] array) instead of any aggregate
    of real cells' counts. Tests a stronger form of "niche-correlated
    transcriptional programs" than build_soft_pseudobulk: does the model's own
    generative understanding of a prototype encode the right marker genes,
    independent of which real cells happened to carry soft weight toward it.

    `decoded` is on the decoder's reconstruction-target scale (log1p-normalized,
    same as adata.X) -- expm1'd back to normalized-count-like scale here so
    downstream normalize_cpm/pseudocount/log2-ratio machinery in
    branch1_recovery/branch2_recovery treats it the same as a real pseudobulk,
    without a silent scale mismatch. Pair with soft_label_metacells (not
    majority_label_metacells) for the metacell labels -- decoded profiles have
    no real per-cell membership to hard-argmax over.
    """
    mc_ids = [str(k) for k in range(decoded.shape[0])]
    return pd.DataFrame(np.expm1(decoded), index=mc_ids, columns=var_names).rename_axis('metacell_id')


def gini(sizes):
    """Gini coefficient of a metacell-size distribution (0 = all equal, ->1 = a few
    metacells hold almost all the cells). Works on hard counts or soft effective
    sizes (fractional) equally."""
    x = np.sort(np.asarray(sizes, dtype=float))
    n = len(x)
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * x) / (n * np.sum(x))) - (n + 1) / n


def effective_n_metacells(sizes):
    """Inverse Simpson index on cell-count (or soft effective-size) shares --
    collapses toward 1 if one giant metacell dominates, regardless of how many
    other near-empty ones exist on paper."""
    p = np.asarray(sizes, dtype=float)
    p = p / p.sum()
    return 1.0 / np.sum(p ** 2)


def topk_share(sizes, k=5):
    """Fraction of all cells (or soft mass) captured by just the k largest metacells."""
    sizes = pd.Series(sizes)
    return sizes.sort_values(ascending=False).head(k).sum() / sizes.sum()


def size_concentration_summary(sizes, name=None):
    """One-row summary combining the three size-concentration metrics above --
    hard counts (mc_labels_df['n_cells']) or soft effective sizes (soft_label_
    metacells' 'n_cells') both work, so this is the single place to check whether
    a run has collapsed regardless of which labeling was used to build it."""
    sizes = np.asarray(sizes, dtype=float)
    return {
        'run': name,
        'n_metacells_used': int((sizes > 1e-6).sum()),
        'median_size': float(np.median(sizes)),
        'gini': gini(sizes),
        'effective_n_metacells': effective_n_metacells(sizes),
        'top5_share_of_cells': topk_share(sizes, 5),
    }


def per_pair_diagnostics(mc_labels_df, branch1_df, branch2_df):
    """One row per (cell type, niche): the pos-group's mean niche purity, its size stats
    (n metacells, median real cells per metacell), and both branches' metrics. Ported from
    plan1_niche_recovery_eval.ipynb's inline diagnostic (kept identical) so any notebook
    can reuse the exact same breakdown instead of re-deriving it."""
    rows = []
    for (ct, niche), grp in mc_labels_df.groupby(['cell_type', 'niche']):
        rows.append({
            'cell_type': ct, 'niche': niche,
            'n_pos_mc': len(grp),
            'median_pos_mc_size': grp['n_cells'].median(),
            'min_pos_mc_size': grp['n_cells'].min(),
            'mean_niche_purity': grp['niche_purity'].mean(),
        })
    # Explicit columns so zero (cell_type, niche) groups in mc_labels_df still
    # returns the right schema for the merges below (same fix as branch1/2_recovery).
    pur_cols = ['cell_type', 'niche', 'n_pos_mc', 'median_pos_mc_size',
                'min_pos_mc_size', 'mean_niche_purity']
    pur = pd.DataFrame(rows, columns=pur_cols)
    out = pur.merge(branch1_df[['cell_type', 'niche', 'pearson_r', 'kendall_tau']],
                     on=['cell_type', 'niche'], how='left')
    out = out.merge(branch2_df[['cell_type', 'niche', 'tpr']],
                     on=['cell_type', 'niche'], how='left')
    return out


# ---------------------------------------------------------------------------
# Setup C — raw-count pseudobulk
# ---------------------------------------------------------------------------

def build_pseudobulk(cell_assign_df, adata, raw_layer='counts'):
    """Sum real raw counts per metacell over its real member cells. Built from
    cell_assignments.csv against the raw counts layer for every method identically —
    NOT from a method's own saved aggregate .h5ad (e.g. scProto's metacells.h5ad is
    decoder-reconstructed, not a real-cell aggregation; see plan1_niche_recovery_eval.ipynb
    for the full reasoning)."""
    raw = adata.layers[raw_layer]
    raw = raw.toarray() if hasattr(raw, 'toarray') else np.asarray(raw)
    raw_df = pd.DataFrame(raw, index=adata.obs_names, columns=adata.var_names)
    cell_ids = cell_assign_df.index.intersection(raw_df.index)
    grouped = raw_df.loc[cell_ids].groupby(cell_assign_df.loc[cell_ids, 'metacell_id']).sum()
    return grouped


def normalize_cpm(pb_df):
    lib = pb_df.sum(axis=1)
    return pb_df.div(lib.replace(0, np.nan), axis=0) * 1e6


# ---------------------------------------------------------------------------
# Branch 1 — MetaQ-style continuous recovery (primary)
# ---------------------------------------------------------------------------

def branch1_recovery(pseudobulk_df, mc_labels_df, ground_truth,
                      pseudocount=1.0, gt_max_genes=100, min_pos_mc=2, min_ctrl_mc=2):
    """Pearson r / Kendall tau of metacell-level vs. single-cell-level logFC, restricted
    to genes already significant in the ground truth (no significance gate on the
    metacell side)."""
    cpm = normalize_cpm(pseudobulk_df)
    rows = []
    for (ct, niche), gt_df in ground_truth.items():
        gt_sig = gt_df[gt_df['pvals_adj'] < 0.05].head(gt_max_genes)
        genes = [g for g in gt_sig['names'] if g in cpm.columns]
        if len(genes) < 5:
            continue

        mcs = mc_labels_df[mc_labels_df['cell_type'] == ct]
        pos_mc = mcs.index[mcs['niche'] == niche].intersection(cpm.index)
        ctrl_mc = mcs.index[mcs['niche'] != niche].intersection(cpm.index)
        if len(pos_mc) < min_pos_mc or len(ctrl_mc) < min_ctrl_mc:
            continue

        mean_pos = cpm.loc[pos_mc, genes].mean(axis=0)
        mean_ctrl = cpm.loc[ctrl_mc, genes].mean(axis=0)
        mc_logfc = np.log2((mean_pos + pseudocount) / (mean_ctrl + pseudocount))
        sc_logfc = gt_sig.set_index('names').loc[genes, 'logfoldchanges']

        r, _ = pearsonr(mc_logfc, sc_logfc)
        tau, _ = kendalltau(mc_logfc, sc_logfc)

        rows.append({
            'cell_type': ct, 'niche': niche, 'n_genes': len(genes),
            'n_pos_mc': len(pos_mc), 'n_ctrl_mc': len(ctrl_mc),
            'pearson_r': r, 'kendall_tau': tau,
        })
    # Explicit columns so a zero-pairs-passed run still returns the right schema
    # (an empty `rows` list would otherwise produce a 0x0 DataFrame with no
    # column names at all, which crashes any downstream [['cell_type', ...]]
    # selection or merge instead of just reporting 0 pairs tested).
    cols = ['cell_type', 'niche', 'n_genes', 'n_pos_mc', 'n_ctrl_mc', 'pearson_r', 'kendall_tau']
    return pd.DataFrame(rows, columns=cols)


def top_genes_for_pair(pseudobulk_df, mc_labels_df, ground_truth, cell_type, niche,
                        pseudocount=1.0, gt_max_genes=100, top_n=20):
    """Gene-level detail for ONE (cell_type, niche) pair -- same computation
    branch1_recovery does internally, but returns the actual per-gene logFC
    comparison instead of collapsing it to a single pearson_r/kendall_tau.
    For pulling out literature-searchable gene names behind a strong result.

    Returns a DataFrame (gene, mc_logfc, sc_logfc, same_direction), sorted by
    |mc_logfc| descending -- the genes this run's metacells say are most
    strongly up/down in this (cell_type, niche), restricted to genes already
    significant in the ground truth. `same_direction` flags whether the
    metacell-level and single-cell-level logFC agree in sign -- a quick sanity
    check before searching literature for a gene that's actually numerically
    ambiguous.
    """
    cpm = normalize_cpm(pseudobulk_df)
    gt_df = ground_truth[(cell_type, niche)]
    gt_sig = gt_df[gt_df['pvals_adj'] < 0.05].head(gt_max_genes)
    genes = [g for g in gt_sig['names'] if g in cpm.columns]

    mcs = mc_labels_df[mc_labels_df['cell_type'] == cell_type]
    pos_mc = mcs.index[mcs['niche'] == niche].intersection(cpm.index)
    ctrl_mc = mcs.index[mcs['niche'] != niche].intersection(cpm.index)

    mean_pos = cpm.loc[pos_mc, genes].mean(axis=0)
    mean_ctrl = cpm.loc[ctrl_mc, genes].mean(axis=0)
    mc_logfc = np.log2((mean_pos + pseudocount) / (mean_ctrl + pseudocount))
    sc_logfc = gt_sig.set_index('names').loc[genes, 'logfoldchanges']

    out = pd.DataFrame({
        'gene': genes,
        'mc_logfc': mc_logfc.values,
        'sc_logfc': sc_logfc.values,
    })
    out['same_direction'] = np.sign(out['mc_logfc']) == np.sign(out['sc_logfc'])
    return out.reindex(out['mc_logfc'].abs().sort_values(ascending=False).index).head(top_n)


# ---------------------------------------------------------------------------
# Branch 2 — SuperCell-style discrete recovery (secondary)
# ---------------------------------------------------------------------------

def weighted_ttest(a, wa, b, wb):
    """Two-sample Welch t-test with per-sample weights (metacell size)."""
    da = DescrStatsW(a, weights=wa, ddof=0)
    db = DescrStatsW(b, weights=wb, ddof=0)
    _, p, _ = CompareMeans(da, db).ttest_ind(usevar='unequal')
    return p


def branch2_recovery(pseudobulk_df, mc_labels_df, ground_truth, min_pos_mc=2, min_ctrl_mc=2):
    """Sample-weighted (by metacell size) t-test TPR on genes already significant at
    single-cell level (no logFC filter, no ranking step — SuperCell's own recipe)."""
    cpm = normalize_cpm(pseudobulk_df)
    rows = []
    for (ct, niche), gt_df in ground_truth.items():
        genes = [g for g in gt_df.loc[gt_df['pvals_adj'] < 0.05, 'names'] if g in cpm.columns]
        if len(genes) < 5:
            continue

        mcs = mc_labels_df[mc_labels_df['cell_type'] == ct]
        pos_mc = mcs.index[mcs['niche'] == niche].intersection(cpm.index)
        ctrl_mc = mcs.index[mcs['niche'] != niche].intersection(cpm.index)
        if len(pos_mc) < min_pos_mc or len(ctrl_mc) < min_ctrl_mc:
            continue

        w_pos = mc_labels_df.loc[pos_mc, 'n_cells'].to_numpy()
        w_ctrl = mc_labels_df.loc[ctrl_mc, 'n_cells'].to_numpy()

        n_sig = sum(
            weighted_ttest(cpm.loc[pos_mc, g].to_numpy(), w_pos,
                            cpm.loc[ctrl_mc, g].to_numpy(), w_ctrl) < 0.05
            for g in genes
        )
        rows.append({
            'cell_type': ct, 'niche': niche, 'n_genes': len(genes),
            'n_pos_mc': len(pos_mc), 'n_ctrl_mc': len(ctrl_mc),
            'tpr': n_sig / len(genes),
        })
    # Explicit columns so a zero-pairs-passed run still returns the right schema
    # (see branch1_recovery's identical fix for why this matters).
    cols = ['cell_type', 'niche', 'n_genes', 'n_pos_mc', 'n_ctrl_mc', 'tpr']
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def macro_average(branch_df, value_cols):
    """Macro-average across niches within a cell type, then across cell types (same
    averaging convention this manuscript already uses for purity)."""
    if branch_df.empty:
        return pd.Series({c: np.nan for c in value_cols})
    per_ct = branch_df.groupby('cell_type')[value_cols].mean()
    return per_ct.mean()


def coverage_penalized_average(per_pair_long, ground_truth, value_cols=('pearson_r', 'kendall_tau', 'tpr')):
    """Union/full-ground-truth alternative to macro_average and matched_macro_average:
    a pair a run never manages to clear the Branch 1/2 gate for (min_pos_mc>=2,
    min_ctrl_mc>=2) -- whether because no metacell was ever majority/soft-labeled
    with that (cell_type, niche) at all, or because it was labeled but the gate
    still failed -- is NOT dropped from that run's average. It's scored 0 on
    every metric (no correlation / no rank agreement / no genes recovered), the
    natural "recovered nothing here" floor for these three metrics, directly
    penalizing coverage gaps in the same units as accuracy.

    Starts from the FULL ground_truth pair list (every real niche program that
    exists, e.g. 146/150) -- but a pair where EVERY compared run scores 0 (none
    could clear the gate at all) is then dropped before averaging. Such a pair
    carries no information for comparing methods (everyone fails identically)
    and would only dilute every run's average by the same fixed amount, so
    keeping it in doesn't help distinguish methods -- it's not a case any
    method can be faulted for relative to the others. A pair stays in as long
    as AT LEAST ONE compared run tested it; any OTHER run that failed on a kept
    pair still scores 0 on it, which is the actual penalty this function exists
    to apply. This makes the result -- and its `n_pairs_used` denominator --
    depend on which runs are passed in together (unlike a fixed 146/150), by
    design: it's answering "of the niche programs at least one of these
    methods could find, how much does each one actually deliver."

    This is the number to cite for "does method X recover niche programs," as
    opposed to matched_macro_average (fairest head-to-head accuracy restricted
    to pairs every run clears) or plain macro_average (a run's own
    self-selected, not directly comparable, achievable-subset average).

    `per_pair_long`: long-format DataFrame with a 'run' column plus 'cell_type',
    'niche', and value_cols -- e.g. per_pair_eval as already built by
    report_run/report_all's per-pair output (concatenated across runs).
    """
    combined = coverage_penalized_detail(per_pair_long, ground_truth, value_cols)
    value_cols = list(value_cols)

    pair_tested_any = combined.groupby(['cell_type', 'niche'])['tested'].any()
    kept = pair_tested_any[pair_tested_any].index
    n_dropped = int((~pair_tested_any).sum())

    combined = combined.set_index(['cell_type', 'niche'])
    combined = combined.loc[combined.index.isin(kept)].reset_index()

    per_ct = combined.groupby(['run', 'cell_type'])[value_cols].mean()
    result = per_ct.groupby('run').mean()
    result['n_pairs_tested'] = combined.groupby('run')['tested'].sum()
    result['n_pairs_used'] = len(kept)
    result['n_pairs_dropped_untestable_by_all'] = n_dropped
    return result


def coverage_penalized_detail(per_pair_long, ground_truth, value_cols=('pearson_r', 'kendall_tau', 'tpr')):
    """Un-aggregated companion to coverage_penalized_average: one row per
    (run, cell_type, niche) for EVERY pair in ground_truth (all 146/150), not
    just the ones a run happened to clear the Branch 1/2 gate for. Untested
    pairs get value_cols filled to 0.0 and `tested=False`, so the exact same
    (cell_type, niche) grid is visible per run before any averaging -- which
    real pairs a run actually recovers vs. which get the 0-penalty, at a
    glance, rather than only seeing the collapsed mean.
    """
    value_cols = list(value_cols)
    all_pairs = pd.DataFrame(list(ground_truth.keys()), columns=['cell_type', 'niche'])
    runs = per_pair_long['run'].unique()

    rows = []
    for run in runs:
        sub = per_pair_long[per_pair_long['run'] == run]
        full = all_pairs.merge(sub[['cell_type', 'niche'] + value_cols],
                                on=['cell_type', 'niche'], how='left')
        full['tested'] = full[value_cols[0]].notna()
        full[value_cols] = full[value_cols].fillna(0.0)
        full['run'] = run
        rows.append(full)
    return pd.concat(rows, ignore_index=True)


def matched_macro_average(per_pair_df, value_cols=('pearson_r', 'kendall_tau', 'tpr')):
    """Same two-level averaging convention as macro_average (mean within cell_type,
    then mean across cell_types), computed per run over whatever rows are passed in.

    Intended for a caller-supplied subset restricted to the intersection of pairs
    every compared run actually clears the Branch 1/2 gate on (`per_pair_df` needs a
    'run' column) -- matching plan1_niche_recovery_eval.ipynb's own "paired over the
    same N pairs" convention for head-to-head claims, instead of letting each run's
    average be computed over its own self-selected, differently-sized subset."""
    value_cols = list(value_cols)
    per_ct = per_pair_df.groupby(['run', 'cell_type'])[value_cols].mean()
    return per_ct.groupby('run').mean()


def report_run(name, run_dir, adata, ground_truth, ct_key, niche_key):
    """Hard + soft (if available) Branch 1/2 summary, size-concentration, and
    per-pair breakdown for one already-trained run -- reads cell_assignments.csv
    and soft_assignments.npz directly (no re-encoding/retraining). The single
    place every scProto-vs-baseline comparison notebook should call instead of
    re-deriving this scoring loop inline."""
    summary_rows = {}
    size_rows = []
    per_pair_rows = []

    # ---- hard ----
    cell_assign = load_cell_assignments(run_dir, adata, niche_key)
    mc_labels = majority_label_metacells(cell_assign, ct_key, niche_key)
    pseudobulk = build_pseudobulk(cell_assign, adata)
    b1 = branch1_recovery(pseudobulk, mc_labels, ground_truth)
    b2 = branch2_recovery(pseudobulk, mc_labels, ground_truth)
    row = macro_average(b1, ['pearson_r', 'kendall_tau'])
    row['tpr'] = macro_average(b2, ['tpr'])['tpr']
    row['n_pairs_tested'] = len(b1)
    summary_rows[f'{name} (hard)'] = row
    size_rows.append(size_concentration_summary(mc_labels['n_cells'], name=f'{name} (hard)'))
    pp = per_pair_diagnostics(mc_labels, b1, b2)
    pp['run'] = f'{name} (hard)'
    per_pair_rows.append(pp)

    # ---- soft (skipped if this run predates soft_assignments.npz saving) ----
    npz_path = os.path.join(run_dir, 'soft_assignments.npz')
    if os.path.exists(npz_path):
        S, cell_ids = load_soft_assignments(run_dir)
        mc_labels_soft = soft_label_metacells(S, cell_ids, adata, ct_key, niche_key)
        pseudobulk_soft = build_soft_pseudobulk(S, cell_ids, adata)
        b1s = branch1_recovery(pseudobulk_soft, mc_labels_soft, ground_truth)
        b2s = branch2_recovery(pseudobulk_soft, mc_labels_soft, ground_truth)
        row_s = macro_average(b1s, ['pearson_r', 'kendall_tau'])
        row_s['tpr'] = macro_average(b2s, ['tpr'])['tpr']
        row_s['n_pairs_tested'] = len(b1s)
        summary_rows[f'{name} (soft)'] = row_s
        size_rows.append(size_concentration_summary(mc_labels_soft['n_cells'], name=f'{name} (soft)'))
        pps = per_pair_diagnostics(mc_labels_soft, b1s, b2s)
        pps['run'] = f'{name} (soft)'
        per_pair_rows.append(pps)
    else:
        print(f'{name}: no soft_assignments.npz found, soft scoring skipped')

    summary_df = pd.DataFrame(summary_rows).T
    size_df = pd.DataFrame(size_rows).set_index('run')
    per_pair_df = pd.concat(per_pair_rows, ignore_index=True)
    return summary_df, size_df, per_pair_df


def report_all(run_dirs, missing, adata, ground_truth, ct_key, niche_key):
    """report_run for every run in run_dirs not in missing, concatenated into
    one summary / size / per-pair table each."""
    summaries, sizes, pairs = [], [], []
    for name, run_dir in run_dirs.items():
        if name in missing:
            continue
        s, z, p = report_run(name, run_dir, adata, ground_truth, ct_key, niche_key)
        summaries.append(s)
        sizes.append(z)
        pairs.append(p)
        print(f'{name}: done')
    return pd.concat(summaries), pd.concat(sizes), pd.concat(pairs, ignore_index=True)


def report_run_decoded(name, t, adata, ground_truth, ct_key, niche_key, batch_mode='mean'):
    """scProto-only third labeling mode: SOFT assignment for labeling (which
    niche/celltype each prototype represents, via soft_label_metacells) but the
    metacell's gene-expression profile is the model's OWN decoder output for that
    prototype (build_decoded_pseudobulk), not a real-cell aggregate. Needs the
    live trainer `t` (decode_prototypes/get_dump_path), not just a saved run_dir
    -- SEACells has no decoder, so this has no baseline-method equivalent.

    Returns the same (summary_df, size_df, per_pair_df) shape as report_run/
    report_all so it can be pd.concat'd alongside their output, but only ever
    produces one row (soft-labels, decoded-profile) -- there is no hard-vs-soft
    split here, since decoded profiles have no real per-cell membership to
    hard-argmax over.
    """
    run_dir = t.get_dump_path()
    npz_path = os.path.join(run_dir, 'soft_assignments.npz')
    if not os.path.exists(npz_path):
        print(f'{name}: no soft_assignments.npz found, decoded-profile scoring skipped')
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    S, cell_ids = load_soft_assignments(run_dir)
    mc_labels_soft = soft_label_metacells(S, cell_ids, adata, ct_key, niche_key)

    decoded = t.decode_prototypes(batch_mode=batch_mode)
    pseudobulk_decoded = build_decoded_pseudobulk(decoded, adata.var_names)

    label = f'{name} (soft-labels, decoded-profile)'
    b1 = branch1_recovery(pseudobulk_decoded, mc_labels_soft, ground_truth)
    b2 = branch2_recovery(pseudobulk_decoded, mc_labels_soft, ground_truth)
    row = macro_average(b1, ['pearson_r', 'kendall_tau'])
    row['tpr'] = macro_average(b2, ['tpr'])['tpr']
    row['n_pairs_tested'] = len(b1)

    summary_df = pd.DataFrame({label: row}).T
    size_df = pd.DataFrame(
        [size_concentration_summary(mc_labels_soft['n_cells'], name=label)]
    ).set_index('run')
    pp = per_pair_diagnostics(mc_labels_soft, b1, b2)
    pp['run'] = label

    print(f'{name}: done (decoded-profile)')
    return summary_df, size_df, pp
