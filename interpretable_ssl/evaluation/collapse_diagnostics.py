"""Metacell collapse / hierarchy diagnostics.

The failure this module detects: training converges to a few very large metacells plus a
long tail of near-empty ones, with nothing in between, and hierarchical substructure
(sub-states within a cell type) is never resolved.

Purity alone cannot detect it -- a singleton metacell is 100% pure by definition, so a
collapsed partition scores *well* on mean purity. Every number here is therefore either
size-aware or reported as a macro/micro pair.

Two quantities do all the work:

    size[p]      = number of cells assigned to metacell p
    tightness[p] = edges_inside[p] / possible_pairs[p]
                 = the average affinity between two randomly chosen members of p
                 = the induced-subgraph edge density
                   (this is the same quantity the loss optimizes under
                    nassoc_diag_norm='pair'; see files/notes/nassoc_redesign.md)

Note on what is and isn't pathological in biological data: population abundance should
show up as the NUMBER of metacells, not their SIZE. So a long tail of *small* metacells is
expected (rare cell types genuinely have few cells), while a *huge* metacell is always
suspect -- on a kNN affinity graph tightness[p] <= k/(size[p]-1), so a 2700-cell metacell
is arithmetically incapable of being a tight community. `size_vs_tightness` separates the
three regimes: small+tight = real rare population, small+loose = dead slot,
large+loose = collapsed giant.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# core quantities
# ---------------------------------------------------------------------------

def metacell_sizes(labels):
    """Cells per metacell. `labels` = per-cell metacell id (array / Series)."""
    return pd.Series(labels).value_counts().sort_index()


def metacell_tightness(affinity, labels, min_size=2):
    """tightness[p] = edges_inside[p] / possible_pairs[p], i.e. the mean affinity between
    two members of p. Metacells smaller than `min_size` have no pairs and get NaN.

    affinity : (n_cells, n_cells) sparse or dense, symmetric, aligned to `labels`.
    """
    labels = np.asarray(pd.Series(labels).values)
    A = sp.csr_matrix(affinity) if not sp.issparse(affinity) else affinity.tocsr()
    A = sp.triu(A, k=1).tocoo()          # each undirected edge exactly once, no self-loops

    lab_i, lab_j = labels[A.row], labels[A.col]
    inside = lab_i == lab_j
    edges_inside = (pd.Series(A.data[inside])
                    .groupby(pd.Series(lab_i[inside]))
                    .sum())

    sizes = metacell_sizes(labels)
    possible_pairs = 0.5 * sizes * (sizes - 1)

    tight = edges_inside.reindex(sizes.index).fillna(0.0) / possible_pairs
    tight[sizes < min_size] = np.nan
    return tight


def macro_micro_gap(values_per_metacell, sizes):
    """The one trick used for both tightness and purity.

    macro = each METACELL counts equally  -> inflated by fragmentation (singletons are
                                             trivially tight / trivially pure)
    micro = each CELL counts equally      -> inflated by collapse (a giant metacell of the
                                             majority class scores well for free)
    gap   = macro - micro                 -> ~0 healthy; >>0 means the BIG metacells are
                                             the bad ones, which is the collapse signature
    """
    v = pd.Series(values_per_metacell).astype(float)
    s = pd.Series(sizes).astype(float).reindex(v.index)
    ok = v.notna() & s.notna()
    v, s = v[ok], s[ok]
    macro = float(v.mean())
    micro = float((v * s).sum() / s.sum()) if s.sum() > 0 else np.nan
    return macro, micro, macro - micro


# ---------------------------------------------------------------------------
# shape of the size distribution
# ---------------------------------------------------------------------------

def effective_n_metacells(sizes):
    """exp(entropy of the size distribution) -- "perplexity". If 200 metacells exist but 12
    hold nearly all the cells, this reads ~12. Divide by len(sizes) for the fraction."""
    p = np.asarray(sizes, dtype=float)
    p = p[p > 0]
    if p.sum() == 0:
        return np.nan
    p = p / p.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def dip_test(sizes):
    """Hartigan's dip test for unimodality, run on log10(size).

    Tests the specific 'cliff' shape: a few huge metacells, many near-empty ones, and a GAP
    with nothing in between. Biological abundance is continuous, so a genuine gap is an
    optimization artifact, not biology.

    Returns (dip_statistic, p_value). SMALL p = significantly NOT one hump = the cliff.
    HEALTHY p > 0.05, COLLAPSED p < 0.01. The statistic is the usable trend line when the
    number of metacells is small and the p-value is noisy.

    Requires `pip install diptest`; returns (nan, nan) if unavailable.
    """
    try:
        import diptest as _diptest
    except ImportError:
        return np.nan, np.nan
    x = np.asarray(sizes, dtype=float)
    x = np.log10(x[x > 0])
    if len(x) < 4:
        return np.nan, np.nan
    dip, pval = _diptest.diptest(x)
    return float(dip), float(pval)


# ---------------------------------------------------------------------------
# one-row summary
# ---------------------------------------------------------------------------

def collapse_summary(labels, affinity=None, name=None, purity=None):
    """One row per run. `affinity` optional -- without it the tightness columns are NaN.
    `purity` optional: a per-metacell purity Series (e.g. joint cell-type x niche purity),
    indexed like the metacell ids, to get its macro/micro gap in the same row.
    """
    sizes = metacell_sizes(labels)
    n_cells = float(sizes.sum())
    dip, dip_p = dip_test(sizes)

    row = {
        'run': name,
        # fraction of ALL cells sitting in the single biggest metacell.
        # No target size involved. Watch the trend: rising = collapsing.
        'largest_share': float(sizes.max() / n_cells),
        # Hartigan dip on log10(sizes): SMALL p = not one hump = the collapse cliff
        'dip_stat': dip,
        'dip_p': dip_p,
    }

    if affinity is not None:
        tight = metacell_tightness(affinity, labels)
        t_macro, t_micro, t_gap = macro_micro_gap(tight, sizes)
        ok = tight.notna()
        row.update({
            # macro and micro are computed but not reported -- the gap is what carries the
            # signal. >>0 means the big metacells are the loose ones = indicts a giant.
            'tightness_gap': t_gap,
            # EXPECTED ~ -0.2 (on a kNN graph tightness falls as ~k/size, unavoidable).
            # < -0.6 means the giants are pathological, not merely large.
            'size_tightness_corr': float(
                pd.Series(sizes[ok]).corr(tight[ok], method='spearman')) if ok.sum() > 2 else np.nan,
        })

    if purity is not None:
        # purity_macro is computed but NOT reported: it rises with fragmentation (a
        # singleton metacell is 100% pure), so it is not comparable across runs with
        # different metacell counts and is easy to misread. purity_micro (each CELL
        # counts equally) and the gap are what get reported.
        p_macro, p_micro, p_gap = macro_micro_gap(purity, sizes)
        row.update({'purity_micro': p_micro, 'purity_gap': p_gap})

    return row


def compare_runs(runs, affinity=None):
    """runs: {name: labels} or {name: (labels, affinity)}. Returns a comparison DataFrame,
    one row per run, in the order given."""
    rows = []
    for name, val in runs.items():
        if isinstance(val, tuple):
            labels, aff = val
        else:
            labels, aff = val, affinity
        rows.append(collapse_summary(labels, affinity=aff, name=name))
    return pd.DataFrame(rows).set_index('run')


# ---------------------------------------------------------------------------
# getting labels out of a run
# ---------------------------------------------------------------------------

def labels_from_trainer(trainer, adata=None):
    """Hard metacell id per cell = argmax of the soft prototype assignment.

    Same path the rebuttal notebooks use to build soft assignments
    (encode -> model.prototypes -> softmax / trainer.epsilon), just taken to argmax.
    Returns a pd.Series indexed by cell name so it aligns with adata.obs.
    """
    import torch
    import torch.nn.functional as F

    ad = adata if adata is not None else trainer.train_ds.adata
    model = trainer.model
    model.eval()
    with torch.no_grad():
        z = trainer.encode_adata(ad, model, z_idx=1).to(trainer.device)
        logits = model.prototypes(z)
        lab = F.softmax(logits / trainer.epsilon, dim=1).argmax(1).cpu().numpy()
    return pd.Series(lab, index=ad.obs_names.to_numpy(), name='metacell_id')


def joint_purity_per_metacell(labels, adata, ct_key, niche_key):
    """Per-metacell purity of the JOINT (cell type | niche) label -- the headline label-based
    metric, because neither collapse mode can game it (collapse hurts niche purity,
    fragmentation is exposed by the macro/micro gap that `collapse_summary` reports).
    """
    lab = pd.Series(labels)
    joint = (adata.obs[ct_key].astype(str).values + '|'
             + adata.obs[niche_key].astype(str).values)
    tab = pd.crosstab(lab.values, joint)
    return (tab.div(tab.sum(1), axis=0).max(1)).sort_index()


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def plot_collapse(labels, affinity=None, name='', axes=None, figsize=(11, 4)):
    """Two panels:

    1. sorted metacell sizes, log y   -- collapse looks like a CLIFF, healthy like a PLATEAU
    2. size vs tightness scatter      -- small+tight  = real rare population
                                         small+loose  = dead slot
                                         large+loose  = collapsed giant
    """
    import matplotlib.pyplot as plt

    sizes = metacell_sizes(labels)
    tight = metacell_tightness(affinity, labels) if affinity is not None else None
    n_panels = 2 if tight is not None else 1

    if axes is None:
        _, axes = plt.subplots(1, n_panels, figsize=figsize)
    axes = np.atleast_1d(axes)

    ax = axes[0]
    ax.plot(np.arange(len(sizes)), np.sort(sizes.values)[::-1], marker='.', lw=1)
    ax.set_yscale('log')
    ax.set_xlabel('metacell (sorted)'); ax.set_ylabel('cells')
    ax.set_title(f'{name} size distribution\nlargest={sizes.max():.0f} '
                 f'({sizes.max()/sizes.sum():.1%} of cells)')

    if tight is not None:
        ax = axes[1]
        ok = tight.notna()
        ax.scatter(sizes[ok], tight[ok], s=14, alpha=0.6)
        ax.set_xscale('log')
        ax.set_xlabel('size (cells, log)'); ax.set_ylabel('tightness')
        rho = pd.Series(sizes[ok]).corr(tight[ok], method='spearman') if ok.sum() > 2 else np.nan
        t_macro, t_micro, t_gap = macro_micro_gap(tight, sizes)
        ax.set_title(f'{name} size vs tightness\nspearman={rho:+.2f}, gap={t_gap:+.4f}')

    return axes
