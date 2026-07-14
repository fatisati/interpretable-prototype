"""
spatial_immune_task.py — validation of tumor-immune interface task for NSCLC spatial data.

Usage in Colab:
    from interpretable_ssl.evaluation.spatial_immune_task import analyze_tumor_immune_interface
    results = analyze_tumor_immune_interface(ad)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.stats import mannwhitneyu

# ---------------------------------------------------------------------------
# Biologically motivated (celltype, niche) target groups for NSCLC s28 data.
# Each entry has a short biological rationale drawn from the NSCLC 3D ST paper
# (Pentimalli et al.). These are the groups where spatial context matters most
# and where scProto is expected to have an advantage over expression-only or
# affinity-only baselines.
# ---------------------------------------------------------------------------
# Tier 1: clear hypothesis, strong paper support — always evaluate these.
# Tier 2: good story, secondary — evaluate if n_cells >= min_cells threshold.
NSCLC_TARGET_GROUPS = [
    # --- Tier 1 ---
    {
        'celltype': 'Tumor cells',
        'niche':    'Tumor surface',
        'tier':     1,
        'story':    (
            'Tumor cells at the immune boundary. '
            'Paper: CDH1, EFNA1, AREG ligands enriched; higher fibroblast, macrophage '
            'and cytotoxic T cell counts than core. '
            'Natural contrast to Tumor core — largest spatially-defined tumor subpopulation.'
        ),
    },
    {
        'celltype': 'Tumor cells',
        'niche':    'Tumor core',
        'tier':     1,
        'story':    (
            'Immune-excluded tumor cells. '
            'Paper: highest cellular density, lowest diversity — dominated by tumor cells. '
            'Paired contrast to Tumor surface: same cell type, opposite spatial context. '
            'A method that only uses gene expression cannot split core from surface.'
        ),
    },
    {
        'celltype': 'Tumor cells',
        'niche':    'DC islands',
        'tier':     1,
        'story':    (
            'Tumor cells at the immune interface with DCs. '
            'Paper: MIF→CD44 immunosuppression, PD-L1 upregulation, CCL19 enriched. '
            'Missed entirely by 2D neighborhoods (51% reassigned to Tumor surface in 2D). '
            'Rare but highest biological significance for immune escape.'
        ),
    },
    {
        'celltype': 'Macrophages',
        'niche':    'Macrophage islands',
        'tier':     1,
        'story':    (
            'Macrophages in their dedicated niche. '
            'Paper: CCL3 (MIP-1α) macrophage recruitment signal enriched here. '
            'Macrophages are paradigmatically context-dependent — their polarization '
            'is shaped by the surrounding neighborhood more than any other cell type.'
        ),
    },
    {
        'celltype': 'Macrophages',
        'niche':    'Tumor surface',
        'tier':     1,
        'story':    (
            'Macrophages at the tumor-stromal boundary. '
            'Distinct activation state from macrophages in immune niches. '
            'Large group — provides stable metrics while testing spatial context discrimination. '
            'SEACell-PCA groups all macrophages together; COVET-archetype mixes with tumor/fibroblasts.'
        ),
    },
    {
        'celltype': 'Fibroblasts',
        'niche':    'Desmoplastic stroma',
        'tier':     1,
        'story':    (
            'Cancer-associated fibroblasts in the desmoplastic compartment. '
            'Paper: IGF1 and FGF7 pro-fibrotic ligands enriched; highest fibroblast density; '
            'collagen-receptor interactions drive ECM remodeling. '
            'Strong spatial identity — largest and cleanest fibroblast subpopulation.'
        ),
    },
    {
        'celltype': 'T cells',
        'niche':    'T cell aggregates',
        'tier':     1,
        'story':    (
            'T cells in lymphoid-like aggregates. '
            'Paper: CCL5 (RANTES) chemoattractant enriched, regulating T cell recruitment. '
            'T cell aggregates in 3D had lowest 2D concordance (46%) — strong 3D-specific structure. '
            'Tests whether COVET captures lymphoid aggregate context.'
        ),
    },
    # --- Tier 2 ---
    {
        'celltype': 'Macrophages',
        'niche':    'DC islands',
        'tier':     2,
        'story':    (
            'Macrophages co-located with DCs. '
            'Paper: CCL19 marks both DC and T cell niches, regulating CCR7+ DC/T cell homing. '
            'Tests macrophage spatial subtype resolution beyond just cell-type identity.'
        ),
    },
    {
        'celltype': 'T cells',
        'niche':    'Tumor surface',
        'tier':     2,
        'story':    (
            'Cytotoxic T cells at the tumor boundary. '
            'Distinct activation context from T cells in lymphoid aggregates further from tumor. '
            'Paper: tumor surface interlocks with immune niches — these T cells are actively engaged.'
        ),
    },
    {
        'celltype': 'Fibroblasts',
        'niche':    'Vascular stroma',
        'tier':     2,
        'story':    (
            'Fibroblasts in the vascular niche. '
            'Paper: PDGFB vascular support interactions enriched; '
            'vascular stroma distinguished from desmoplastic by pericyte and endothelial density. '
            'Contrast to desmoplastic fibroblasts: same cell type, different ECM context.'
        ),
    },
    {
        'celltype': 'Tumor cells',
        'niche':    'Macrophage islands',
        'tier':     2,
        'story':    (
            'Tumor cells co-located with macrophages. '
            'Paper: CCL3 myeloid suppression axis — different from DC-island immune escape. '
            'Smaller group; tests whether scProto can resolve rare spatial tumor subtypes.'
        ),
    },
    {
        'celltype': 'T cells',
        'niche':    'DC islands',
        'tier':     2,
        'story':    (
            'T cells homing to DC niches. '
            'Paper: CCL19 regulates CCR7+ T cell homing to DC niches. '
            'Rare group but biologically meaningful: DCs activate and retain T cells via CCL19.'
        ),
    },
]

# Flat string list derived from NSCLC_TARGET_GROUPS — pass directly to run_mc_task target_groups.
# Groups with no cells in a given section are silently skipped by compute_target_group_metrics.
NSCLC_EVAL_GROUPS = [f"{g['celltype']} | {g['niche']}" for g in NSCLC_TARGET_GROUPS]

# Immune-education genes: tumor cells at immune interface upregulate these
IMMUNE_EDUCATION_GENES = [
    'CD274',   # PD-L1 — canonical immune checkpoint, IFN-γ induced
    'IDO1',    # immunosuppression, tryptophan depletion
    'CXCL9',   # IFN-γ response chemokine, T cell recruitment
    'HLA-A',   # antigen presentation upregulated by IFN-γ
    'MIF',     # tumour sends to recruit DCs/macrophages (paper finding)
    'LGALS9',  # Galectin-9, Tim-3 ligand, immunosuppression
    'CXCL10',  # IFN-γ response chemokine
]


def analyze_tumor_immune_interface(
    ad,
    celltype_key='celltypes',
    niche_key='niches_3D',
    tumor_celltype='Tumor cells',
    surface_niche='Tumor surface',
    core_niche='Tumor core',
    genes=None,
    plot=True,
    figsize=(12, 4),
):
    """Validate that tumour cells at the immune interface differ from immune-excluded core cells.

    Filters tumour cells by niche label (surface vs core), compares expression of
    known immune-education genes, and returns a summary DataFrame.

    Args:
        ad:              AnnData — full spatial dataset (e.g. NSCLC_3D_section_28.h5ad)
        celltype_key:    obs column with cell type labels
        niche_key:       obs column with niche labels (niches_3D or niches_2D)
        tumor_celltype:  value in celltype_key that identifies tumour cells
        surface_niche:   niche label for immune-interface tumour cells
        core_niche:      niche label for immune-excluded tumour cells
        genes:           list of genes to compare; defaults to IMMUNE_EDUCATION_GENES
        plot:            if True, show violin/bar plots
        figsize:         figure size

    Returns:
        dict with keys:
            'stats'   — DataFrame: gene, mean_surface, mean_core, log2fc, pval, significant
            'surface' — AnnData subset: tumour cells in surface niche
            'core'    — AnnData subset: tumour cells in core niche
    """
    if genes is None:
        genes = IMMUNE_EDUCATION_GENES

    # --- filter tumour cells ---
    tumor = ad[ad.obs[celltype_key] == tumor_celltype].copy()
    print(f"Total tumour cells: {len(tumor)}")

    valid_niches = tumor.obs[niche_key].unique().tolist()
    if surface_niche not in valid_niches:
        raise ValueError(f"'{surface_niche}' not found in {niche_key}. Available: {valid_niches}")
    if core_niche not in valid_niches:
        raise ValueError(f"'{core_niche}' not found in {niche_key}. Available: {valid_niches}")

    surface = tumor[tumor.obs[niche_key] == surface_niche]
    core    = tumor[tumor.obs[niche_key] == core_niche]
    print(f"  {surface_niche}: {len(surface)} cells")
    print(f"  {core_niche}:    {len(core)} cells")

    # --- get expression matrix (dense) ---
    def _dense(adata, g):
        if g not in adata.var_names:
            return None
        idx = adata.var_names.get_loc(g)
        x = adata.X[:, idx]
        return np.asarray(x.todense()).ravel() if sp.issparse(x) else np.asarray(x).ravel()

    # --- compute stats per gene ---
    rows = []
    for g in genes:
        s_vals = _dense(surface, g)
        c_vals = _dense(core, g)
        if s_vals is None:
            print(f"  [skip] {g} not in var_names")
            continue

        mean_s = float(s_vals.mean())
        mean_c = float(c_vals.mean())
        log2fc = float(np.log2((mean_s + 1e-8) / (mean_c + 1e-8)))

        _, pval = mannwhitneyu(s_vals, c_vals, alternative='two-sided')

        rows.append({
            'gene':         g,
            'mean_surface': mean_s,
            'mean_core':    mean_c,
            'log2fc':       log2fc,
            'pval':         pval,
            'significant':  pval < 0.05 and abs(log2fc) > 0.5,
        })

    stats = pd.DataFrame(rows).set_index('gene')
    stats = stats.sort_values('log2fc', ascending=False)

    print("\n--- Gene expression: surface vs core tumour cells ---")
    print(stats[['mean_surface', 'mean_core', 'log2fc', 'pval', 'significant']].to_string())

    # --- plot ---
    if plot:
        _plot_stats(stats, surface_niche, core_niche, figsize)

    return {
        'stats':   stats,
        'surface': surface,
        'core':    core,
    }


def select_best_slide(
    ad,
    section_key='section',
    celltype_key='celltypes',
    niche_key='niches_3D',
    tumor_celltype='Tumor cells',
    surface_niche='Tumor surface',
    core_niche='Tumor core',
    genes=None,
    min_surface_cells=100,
):
    """Score all slides and rank by strength of tumor-immune interface signal.

    For each slide, computes mean log2FC of upregulated immune-education genes
    (surface > core). Returns a ranked DataFrame so you can pick the best slide
    for downstream analysis.

    Args:
        ad:               AnnData with all slides
        section_key:      obs column that identifies slides/sections
        min_surface_cells: skip slides with fewer surface tumor cells than this

    Returns:
        pd.DataFrame ranked by score, columns:
            section, n_tumor, n_surface, n_core, mean_log2fc,
            n_sig_genes, top_gene, top_log2fc
    """
    if genes is None:
        genes = IMMUNE_EDUCATION_GENES

    sections = ad.obs[section_key].unique().tolist()
    rows = []

    for sec in sections:
        sub = ad[ad.obs[section_key] == sec]
        tumor = sub[sub.obs[celltype_key] == tumor_celltype]

        niches = tumor.obs[niche_key].unique().tolist()
        if surface_niche not in niches or core_niche not in niches:
            continue

        surface = tumor[tumor.obs[niche_key] == surface_niche]
        core    = tumor[tumor.obs[niche_key] == core_niche]

        if len(surface) < min_surface_cells:
            continue

        def _dense(adata, g):
            if g not in adata.var_names:
                return None
            idx = adata.var_names.get_loc(g)
            x = adata.X[:, idx]
            return np.asarray(x.todense()).ravel() if sp.issparse(x) else np.asarray(x).ravel()

        gene_rows = []
        for g in genes:
            s_vals = _dense(surface, g)
            c_vals = _dense(core, g)
            if s_vals is None:
                continue
            mean_s = float(s_vals.mean())
            mean_c = float(c_vals.mean())
            log2fc = float(np.log2((mean_s + 1e-8) / (mean_c + 1e-8)))
            _, pval = mannwhitneyu(s_vals, c_vals, alternative='two-sided')
            gene_rows.append({'gene': g, 'log2fc': log2fc, 'pval': pval})

        if not gene_rows:
            continue

        gdf = pd.DataFrame(gene_rows)
        upregulated = gdf[gdf['log2fc'] > 0]
        mean_log2fc = float(upregulated['log2fc'].mean()) if len(upregulated) else 0.0
        n_sig = int(((gdf['pval'] < 0.05) & (gdf['log2fc'].abs() > 0.5)).sum())
        top = gdf.loc[gdf['log2fc'].idxmax()]

        rows.append({
            'section':     sec,
            'n_tumor':     len(tumor),
            'n_surface':   len(surface),
            'n_core':      len(core),
            'mean_log2fc': mean_log2fc,
            'n_sig_genes': n_sig,
            'top_gene':    top['gene'],
            'top_log2fc':  round(top['log2fc'], 3),
        })

    result = pd.DataFrame(rows).sort_values('mean_log2fc', ascending=False).reset_index(drop=True)
    print(result.to_string(index=False))
    return result


def check_intermediate_states(
    K,
    obs,
    celltype_key='celltypes',
    niche_key='niches_3D',
    groups=None,
    min_cells=20,
    plot=True,
    figsize=(10, 7),
    top_n=15,
    annotate_top=10,
    highlight=None,
):
    """Check which (celltype, niche) groups are dense internally but cross-connected outward.

    Top-right of the scatter = intermediate dense states = scProto advantage over SEACell.

    Args:
        K:             scipy sparse or dense affinity matrix (n_cells × n_cells)
        obs:           pd.DataFrame with per-cell metadata (same row order as K)
        celltype_key:  column in obs for cell type labels
        niche_key:     column in obs for niche labels
        groups:        list of (celltype, niche) tuples to evaluate; if None, all combos
        min_cells:     skip groups smaller than this
        plot:          if True, show scatter
        figsize:       figure size
        top_n:         print only the top N groups by density (None = all)
        annotate_top:  annotate only the top N points in the scatter (None = all)
        highlight:     list of group strings to mark in red, e.g. ['Tumor cells | DC islands']

    Returns:
        pd.DataFrame with columns: group, n_cells, density, cross_frac  (full, not truncated)
    """
    if sp.issparse(K):
        K = K.tocsr()

    labels = obs[celltype_key].astype(str) + ' | ' + obs[niche_key].astype(str)
    unique_groups = labels.unique().tolist()

    if groups is not None:
        keep = {f"{ct} | {niche}" for ct, niche in groups}
        unique_groups = [g for g in unique_groups if g in keep]

    rows = []
    for g in unique_groups:
        idx = np.where(labels == g)[0]
        if len(idx) < min_cells:
            continue

        if sp.issparse(K):
            K_in  = K[idx][:, idx]
            K_all = K[idx]
            w_in  = float(K_in.sum())
            w_all = float(K_all.sum())
        else:
            K_in  = K[np.ix_(idx, idx)]
            K_all = K[idx]
            w_in  = float(K_in.sum())
            w_all = float(K_all.sum())

        n = len(idx)
        density    = w_in / max(n * (n - 1), 1)
        cross_frac = (w_all - w_in) / max(w_all, 1e-12)

        rows.append({
            'group':      g,
            'n_cells':    n,
            'density':    density,
            'cross_frac': cross_frac,
        })

    result = pd.DataFrame(rows).sort_values('density', ascending=False).reset_index(drop=True)

    display_df = result.head(top_n) if top_n is not None else result
    fmt = display_df.copy()
    fmt['density']    = fmt['density'].map('{:.4f}'.format)
    fmt['cross_frac'] = fmt['cross_frac'].map('{:.3f}'.format)
    total = len(result)
    shown = len(display_df)
    print(f"\n--- Intermediate states ({niche_key}): top {shown} of {total} groups by density ---")
    print(fmt[['group', 'n_cells', 'density', 'cross_frac']].to_string(index=True))
    if shown < total:
        print(f"  ... {total - shown} more groups not shown (increase top_n to see all)")

    if plot and len(result):
        _plot_intermediate_scatter(result, figsize, annotate_top=annotate_top,
                                   highlight=highlight, niche_key=niche_key)

    return result


def _plot_intermediate_scatter(df, figsize, annotate_top=10, highlight=None, niche_key=''):
    fig, ax = plt.subplots(figsize=figsize)

    highlight_set = set(highlight or [])
    colors = ['#d62728' if g in highlight_set else '#1f77b4' for g in df['group']]

    ax.scatter(df['cross_frac'], df['density'],
               s=df['n_cells'] / df['n_cells'].max() * 300 + 20,
               c=colors, alpha=0.7, edgecolors='k', linewidths=0.4)

    # annotate only top points by density to avoid clutter
    to_annotate = df.head(annotate_top) if annotate_top is not None else df
    for _, row in to_annotate.iterrows():
        ax.annotate(row['group'], (row['cross_frac'], row['density']),
                    fontsize=7, ha='left', va='bottom',
                    xytext=(4, 3), textcoords='offset points')

    # always annotate highlighted groups even if outside top_n
    for _, row in df[df['group'].isin(highlight_set)].iterrows():
        ax.annotate(row['group'], (row['cross_frac'], row['density']),
                    fontsize=8, ha='left', va='bottom', color='#d62728', fontweight='bold',
                    xytext=(4, 3), textcoords='offset points')

    ax.axvline(df['cross_frac'].median(), color='grey', linewidth=0.8, linestyle='--')
    ax.axhline(df['density'].median(),    color='grey', linewidth=0.8, linestyle='--')

    ax.set_xlabel('Cross-fraction  (edges leaving the group)')
    ax.set_ylabel('Density  (mean within-group edge weight)')
    title = f'Intermediate dense states  [{niche_key}]' if niche_key else 'Intermediate dense states'
    ax.set_title(title + '\n(top-right = scProto advantage)')
    plt.tight_layout()
    plt.show()


def _plot_stats(stats, surface_label, core_label, figsize):
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # left: log2FC bar chart
    ax = axes[0]
    colors = ['#d62728' if v > 0 else '#1f77b4' for v in stats['log2fc']]
    ax.barh(stats.index, stats['log2fc'], color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.axvline(0.5,  color='grey', linewidth=0.6, linestyle='--')
    ax.axvline(-0.5, color='grey', linewidth=0.6, linestyle='--')
    ax.set_xlabel('log2 FC (surface / core)')
    ax.set_title(f'Immune-education genes\n{surface_label} vs {core_label}')
    for i, (gene, row) in enumerate(stats.iterrows()):
        if row['significant']:
            ax.text(row['log2fc'] + 0.02 * np.sign(row['log2fc']),
                    i, '*', va='center', fontsize=10, color='black')

    # right: mean expression heatmap-style
    ax = axes[1]
    x = np.arange(len(stats))
    w = 0.35
    ax.bar(x - w/2, stats['mean_surface'], w, label=surface_label, color='#d62728', alpha=0.8)
    ax.bar(x + w/2, stats['mean_core'],    w, label=core_label,    color='#1f77b4', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(stats.index, rotation=45, ha='right')
    ax.set_ylabel('Mean expression')
    ax.set_title('Mean expression per group')
    ax.legend()

    plt.tight_layout()
    plt.show()


def eval_group_metacell_quality(
    ad,
    target_group,
    celltype_key='celltypes',
    niche_key='niches_3D',
    mc_key='SEACell',
    method_name='model',
):
    """Evaluate how well a target (celltype, niche) group is captured as dedicated metacell(s).

    Three metrics:
      - Purity:       for each target cell, fraction of its metacell that is also target; averaged.
      - Coverage:     fraction of target cells whose metacell's dominant label IS the target.
      - Homogeneity:  fraction of target cells in the single most-assigned metacell.

    Args:
        ad:           AnnData with ad.obs[mc_key] assignments already set.
        target_group: group string, e.g. 'Tumor cells | DC islands'.
        celltype_key: obs column for cell type.
        niche_key:    obs column for niche.
        mc_key:       obs column with metacell assignments.
        method_name:  label for printed output.

    Returns:
        dict with keys: purity, coverage, homogeneity, n_target, mc_distribution.
    """
    obs = ad.obs.copy()
    obs['_group'] = obs[celltype_key].astype(str) + ' | ' + obs[niche_key].astype(str)

    target_mask = obs['_group'] == target_group
    n_target = int(target_mask.sum())
    if n_target == 0:
        print(f"[{method_name}] '{target_group}' not found.")
        return None

    # fraction of target label within each metacell
    mc_target_frac = obs.groupby(mc_key)['_group'].apply(lambda x: (x == target_group).mean())

    # purity: per target cell, fraction of its metacell that is also target — then avg
    purity = float(obs.loc[target_mask, mc_key].map(mc_target_frac).mean())

    # coverage: fraction of target cells in a metacell where target is the dominant label
    mc_dominant = obs.groupby(mc_key)['_group'].agg(lambda x: x.value_counts().idxmax())
    dedicated_mcs = mc_dominant[mc_dominant == target_group].index
    coverage = float(obs.loc[target_mask, mc_key].isin(dedicated_mcs).mean())

    # homogeneity: fraction of target cells in the single top metacell
    mc_dist = obs.loc[target_mask, mc_key].value_counts(normalize=True)
    homogeneity = float(mc_dist.iloc[0]) if len(mc_dist) else 0.0

    print(f"\n--- [{method_name}] '{target_group}' ---")
    print(f"  n_target     : {n_target}")
    print(f"  purity       : {purity:.3f}  (avg target fraction within each cell's metacell)")
    print(f"  coverage     : {coverage:.3f}  (fraction in a metacell dominated by target)")
    print(f"  homogeneity  : {homogeneity:.3f}  (fraction in top-1 metacell)")
    print(f"  dedicated MCs: {list(dedicated_mcs)}")
    print(f"  top MC dist  :\n{mc_dist.head(5).to_string()}")

    return {
        'purity':       purity,
        'coverage':     coverage,
        'homogeneity':  homogeneity,
        'n_target':     n_target,
        'mc_distribution': mc_dist,
    }


def compute_target_group_metrics(ad, mc_key, target_groups, celltype_key='celltypes',
                                 niche_key='niches_3D', method_name='model'):
    """Run eval_group_metacell_quality for each group and return a flat metrics dict.

    Suitable for merging directly into a metrics.json. Keys are slugified group names,
    e.g. 'tumor_cells_dc_islands_purity', 'tumor_cells_dc_islands_coverage', etc.

    Args:
        ad:            AnnData with ad.obs[mc_key] assignments set.
        mc_key:        obs column with metacell assignments (int or str).
        target_groups: list of 'celltype | niche' strings to evaluate.
        celltype_key:  obs column for cell type.
        niche_key:     obs column for niche.
        method_name:   label for printed output.

    Returns:
        flat dict ready to merge into metrics.json
    """
    out = {}
    for grp in target_groups:
        res = eval_group_metacell_quality(
            ad,
            target_group=grp,
            celltype_key=celltype_key,
            niche_key=niche_key,
            mc_key=mc_key,
            method_name=f'{method_name} | {grp}',
        )
        if res is None:
            continue
        slug = grp.lower().replace(' ', '_').replace('|', '').replace('__', '_').strip('_')
        out[f'{slug}_purity']      = res['purity']
        out[f'{slug}_coverage']    = res['coverage']
        out[f'{slug}_homogeneity'] = res['homogeneity']
    return out


def audit_target_groups(
    ad,
    aff,
    target_groups=None,
    celltype_key='celltypes',
    niche_key='niches_3D',
    covet_key='X_covet',
    k_nn=15,
    min_cells=50,
    max_tier=2,
    verbose=True,
):
    """Score biologically motivated (celltype, niche) target groups on two axes:

    1. COVET feature quality  — does X_covet separate this niche within the cell type?
    2. Affinity connectivity  — does the affinity wire same-niche cells together,
                                and how much does it mix cell types?

    These together tell you:
      - Is the affinity good enough for this task? (covet_purity, niche_spec)
      - Does scProto have an advantage over baselines? (scproto_score)
        scProto wins when COVET has signal (covet_purity high) AND archetypal-COVET
        would mix cell types (aff_cross_ct high).

    Args:
        ad:            AnnData with obsm[covet_key] and obs[celltype_key/niche_key].
        aff:           sparse affinity matrix (N x N), same cell order as ad.
        target_groups: list of dicts with keys 'celltype', 'niche', 'story'.
                       Defaults to NSCLC_TARGET_GROUPS.
        celltype_key:  obs column for cell type labels.
        niche_key:     obs column for niche labels (e.g. 'niches_3D' or 'niches_2D').
        covet_key:     obsm key for COVET features.
        k_nn:          neighbours for kNN purity in COVET space.
        min_cells:     skip groups smaller than this.
        max_tier:      include groups up to this tier (1=tier1 only, 2=all).

    Returns:
        pd.DataFrame sorted by scproto_score descending, with columns:
            tier, celltype, niche, n_cells,
            covet_purity   — kNN label purity in COVET space (within same cell type)
            aff_same       — mean fraction of affinity weight to same-niche same-ct
            aff_diff_niche — mean fraction to diff-niche same-ct
            aff_cross_ct   — mean fraction to other cell types
            niche_spec     — aff_same / (aff_same + aff_diff_niche): affinity niche selectivity
            scproto_score  — covet_purity * aff_cross_ct: higher = better scProto task
            story          — biological rationale
    """
    from sklearn.neighbors import NearestNeighbors

    if target_groups is None:
        target_groups = [g for g in NSCLC_TARGET_GROUPS if g.get('tier', 1) <= max_tier]

    X      = ad.obsm[covet_key].astype(np.float32)
    ct_all = ad.obs[celltype_key].values
    nk_all = ad.obs[niche_key].values
    aff_csr = aff.tocsr()

    rows_out = []
    for grp in target_groups:
        ct, niche = grp['celltype'], grp['niche']
        story = grp.get('story', '')

        group_mask = (ct_all == ct) & (nk_all == niche)
        ct_mask    = ct_all == ct
        n = int(group_mask.sum())

        if n < min_cells:
            print(f"[skip] '{ct} | {niche}': only {n} cells < {min_cells}")
            continue

        # --- 1. kNN purity in COVET space (within same cell type) ---
        ct_idx      = np.where(ct_mask)[0]
        ct_niches   = nk_all[ct_idx]
        tgt_local   = np.where(ct_niches == niche)[0]

        k = min(k_nn, len(ct_idx) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine').fit(X[ct_idx])
        _, nbr = nn.kneighbors(X[ct_idx][tgt_local])
        covet_purity = float((ct_niches[nbr[:, 1:]] == niche).mean())

        # --- 2. Affinity connectivity fractions ---
        group_idx = np.where(group_mask)[0]
        aff_rows  = aff_csr[group_idx]
        total_w   = np.array(aff_rows.sum(axis=1)).ravel().clip(1e-12)

        f_same  = float((np.array(aff_rows[:,  group_mask].sum(axis=1)).ravel() / total_w).mean())
        f_diff  = float((np.array(aff_rows[:,  ct_mask & ~group_mask].sum(axis=1)).ravel() / total_w).mean())
        f_cross = float((np.array(aff_rows[:, ~ct_mask].sum(axis=1)).ravel() / total_w).mean())

        niche_spec    = f_same / (f_same + f_diff + 1e-9)
        scproto_score = covet_purity * f_cross

        rows_out.append({
            'tier':          grp.get('tier', 1),
            'celltype':      ct,
            'niche':         niche,
            'n_cells':       n,
            'covet_purity':  round(covet_purity, 3),
            'aff_same':      round(f_same,  3),
            'aff_diff_niche':round(f_diff,  3),
            'aff_cross_ct':  round(f_cross, 3),
            'niche_spec':    round(niche_spec, 3),
            'scproto_score': round(scproto_score, 3),
            'story':         story,
        })

    df = (pd.DataFrame(rows_out)
            .sort_values('scproto_score', ascending=False)
            .reset_index(drop=True))

    if verbose:
        _print_audit(df)
    return df


def _print_audit(df):
    """Pretty-print the audit table with bio stories below."""
    cols = ['tier', 'celltype', 'niche', 'n_cells', 'covet_purity',
            'aff_same', 'aff_diff_niche', 'aff_cross_ct', 'niche_spec', 'scproto_score']
    print(df[cols].to_string(index=True))
    print()
    print("Bio stories:")
    for i, row in df.iterrows():
        print(f"  [{i}] {row['celltype']} | {row['niche']}")
        print(f"       {row['story']}")
        print()


# Biological mapping between 3D and 2D niche labels based on Pentimalli et al.
# quality:
#   'clean'          — same biological entity, same label name, high expected concordance
#   'partial'        — same biological entity but 2D misses boundary cells (lower concordance)
#   'no_2d_equiv'    — 3D-specific discovery with no 2D equivalent; cells split across other 2D niches
NICHE_3D_TO_2D = {
    'Tumor core':               {'2d_equiv': 'Tumor core',               'quality': 'clean'},
    'Tumor surface':            {'2d_equiv': 'Tumor surface',            'quality': 'clean'},
    'Desmoplastic stroma':      {'2d_equiv': 'Desmoplastic stroma',      'quality': 'clean'},
    'Vascular stroma':          {'2d_equiv': 'Vascular stroma',          'quality': 'clean'},
    'Macrophage islands':       {'2d_equiv': 'Macrophage islands',       'quality': 'clean'},
    'Airways':                  {'2d_equiv': 'Airways',                  'quality': 'clean'},
    'Alveolar spaces':          {'2d_equiv': 'Alveolar spaces',          'quality': 'clean'},
    'Smooth muscle structures': {'2d_equiv': 'Smooth muscle structures', 'quality': 'clean'},
    'Excluded':                 {'2d_equiv': 'Excluded',                 'quality': 'clean'},
    # Same entity but 2D misses boundary cells that form 3D-spatial continuity
    # Paper: 35.3% of 3D T cell aggregate cells reassigned to Desmoplastic stroma in 2D
    'T cell aggregates':        {'2d_equiv': 'T cell aggregates',        'quality': 'partial',
                                 'note': '46.2% concordance; 35.3% → Desmoplastic stroma in 2D'},
    # No 2D equivalent — unique 3D discovery validated by IF staining
    # Paper: 51.2% → Tumor surface, 23.6% → T cell aggregates in 2D
    'DC islands':               {'2d_equiv': None,                       'quality': 'no_2d_equiv',
                                 'note': '51.2% → Tumor surface, 23.6% → T cell aggregates in 2D'},
}


def niche_label_concordance(
    ad,
    celltype_key='celltypes',
    niche_2d_key='niches_2D',
    niche_3d_key='niches_3D',
    min_cells=50,
):
    """Compare 2D and 3D niche labels per (celltype, niche_3D) group.

    Handles the fact that 2D and 3D label sets are NOT identical — some niches
    (e.g. DC islands) only exist in 3D and are always discordant in 2D.

    For each (celltype, niche_3D) group reports:
      - concordance: fraction of cells where niche_2D == niche_3D
      - n_cells: group size
      - is_shared: whether the 3D niche label also exists in the 2D label set
      - top_2d_label: the most common 2D reassignment (informative for 3D-only niches)

    Also prints a summary of 3D-only labels and where they land in 2D.

    Args:
        ad:            AnnData with obs columns for celltypes, 2D and 3D niches.
        celltype_key:  obs column for cell type labels.
        niche_2d_key:  obs column for 2D niche labels.
        niche_3d_key:  obs column for 3D niche labels.
        min_cells:     skip groups smaller than this.

    Returns:
        pd.DataFrame sorted by concordance descending, with columns:
            celltype, niche_3d, n_cells, concordance, is_shared, top_2d_label, top_2d_frac
    """
    obs = ad.obs[[celltype_key, niche_2d_key, niche_3d_key]].copy()
    obs.columns = ['celltype', 'niche_2d', 'niche_3d']
    # cast to str to avoid Categorical comparison errors (different category sets)
    obs['niche_2d'] = obs['niche_2d'].astype(str)
    obs['niche_3d'] = obs['niche_3d'].astype(str)

    labels_2d = set(obs['niche_2d'].unique())
    labels_3d = set(obs['niche_3d'].unique())

    # --- print label overview with biological mapping ---
    print("3D → 2D niche mapping (from Pentimalli et al.):")
    print(f"  {'3D label':<28} {'quality':<16} {'2D equivalent / note'}")
    print(f"  {'-'*75}")
    for niche_3d in sorted(labels_3d):
        info = NICHE_3D_TO_2D.get(niche_3d, {'2d_equiv': '?', 'quality': 'unknown'})
        note = info.get('note', info.get('2d_equiv', '—'))
        print(f"  {niche_3d:<28} {info['quality']:<16} {note}")

    unrecognised = labels_3d - set(NICHE_3D_TO_2D)
    if unrecognised:
        print(f"\n  [!] Unrecognised 3D labels (not in mapping): {sorted(unrecognised)}")

    # --- where do 3D-only niche cells land in 2D? (verify against paper) ---
    no_equiv = [n for n, v in NICHE_3D_TO_2D.items()
                if v['quality'] == 'no_2d_equiv' and n in labels_3d]
    if no_equiv:
        print(f"\nActual 2D reassignment for 3D-only niches (should match paper):")
        for niche in no_equiv:
            sub = obs[obs['niche_3d'] == niche]
            dist = sub['niche_2d'].value_counts(normalize=True).head(3)
            print(f"  {niche}: " + ", ".join(f"{v} ({p:.1%})" for v, p in dist.items()))

    # --- per (celltype, niche_3D) concordance ---
    rows = []
    for (ct, niche_3d), grp in obs.groupby(['celltype', 'niche_3d']):
        if len(grp) < min_cells:
            continue
        info = NICHE_3D_TO_2D.get(niche_3d, {'2d_equiv': None, 'quality': 'unknown'})
        quality   = info['quality']
        bio_note  = info.get('note', '')
        has_equiv = info['2d_equiv'] is not None

        # concordance: only meaningful when a 2D equivalent exists
        concordance = float((grp['niche_2d'] == grp['niche_3d']).mean()) if has_equiv else 0.0

        top_2d = grp['niche_2d'].value_counts()
        rows.append({
            'celltype':     ct,
            'niche_3d':     niche_3d,
            'n_cells':      len(grp),
            'quality':      quality,
            'concordance':  round(concordance, 3),
            'top_2d_label': top_2d.index[0],
            'top_2d_frac':  round(top_2d.iloc[0] / len(grp), 3),
            'bio_note':     bio_note,
        })

    df = (pd.DataFrame(rows)
            .sort_values(['quality', 'concordance'], ascending=[True, False])
            .reset_index(drop=True))

    print(f"\n{'='*60}")
    print("Concordance per (celltype, niche_3D):")
    cols = ['celltype', 'niche_3d', 'n_cells', 'quality', 'concordance', 'top_2d_label', 'top_2d_frac']
    print(df[cols].to_string(index=False))

    print(f"\nReliable GT groups (clean/partial quality AND concordance >= 0.7):")
    reliable = df[(df['quality'].isin(['clean', 'partial'])) & (df['concordance'] >= 0.7)]
    print(reliable[['celltype', 'niche_3d', 'n_cells', 'quality', 'concordance']].to_string(index=False))

    return df


def sweep_covet_params(
    ad,
    settings=None,
    affinity_k=50,
    target_groups=None,
    celltype_key='celltypes',
    niche_key='niches_3D',
    min_cells=50,
    max_tier=2,
):
    """Sweep COVET (k, n_pcs, n_comps) settings and compare affinity quality.

    Caches spatial kNN by k and covariance matrix by (k, n_pcs) so each expensive
    computation runs only once. Only the cheap PCA-compaction step reruns per n_comps.

    Args:
        ad:           AnnData with obsm['X_pca'] and obsm['spatial'].
        settings:     list of (k, n_pcs, n_comps) tuples. n_comps=None means auto.
                      Defaults to [(50,15,None),(50,15,20),(50,15,10),
                                   (70,20,None),(70,20,20),(100,25,None)].
                      Skips any where k < 3*n_pcs (unstable covariance).
        affinity_k:   k for build_seacell_kernel (default 50).
        target_groups, celltype_key, niche_key, min_cells, max_tier:
                      passed through to audit_target_groups.

    Returns:
        pivot  — pd.DataFrame: rows=groups, cols=settings, values=scproto_score
        detail — dict mapping setting label → full audit DataFrame
    """
    from interpretable_ssl.augmenters.graph_generator import (
        _covet_spatial_knn, _covet_cov_flat, _covet_apply, build_seacell_kernel,
    )

    if settings is None:
        settings = [
            (50, 15, None), (50, 15, 20), (50, 15, 10),
            (70, 20, None), (70, 20, 20), (100, 25, None),
        ]

    # normalise to 3-tuples
    settings = [(s[0], s[1], s[2] if len(s) > 2 else None) for s in settings]

    knn_cache = {}    # k -> I
    cov_cache = {}    # (k, n_pcs) -> cov_flat

    detail = {}
    score_rows = []

    for k, n_pcs, n_comps in settings:
        if k < 3 * n_pcs:
            print(f"[skip] k={k}, n_pcs={n_pcs}: ratio {k/n_pcs:.1f} < 3 (unstable covariance)")
            continue

        nc_label = f'nc={n_comps}' if n_comps is not None else 'nc=auto'
        label = f'k={k}_npc={n_pcs}_{nc_label}'

        # --- cached spatial kNN (expensive: faiss over N cells) ---
        if k not in knn_cache:
            print(f"  [knn]  computing spatial kNN k={k} ...", end=' ', flush=True)
            knn_cache[k] = _covet_spatial_knn(ad, k)
            print("done")
        I = knn_cache[k]

        # --- cached covariance (expensive: einsum N×k×p×p) ---
        if (k, n_pcs) not in cov_cache:
            print(f"  [cov]  computing covariance (k={k}, n_pcs={n_pcs}) ...", end=' ', flush=True)
            cov_cache[(k, n_pcs)] = _covet_cov_flat(ad, I, n_pcs)
            print("done")
        cov_flat = cov_cache[(k, n_pcs)]

        # --- cheap: PCA-compaction + affinity ---
        print(f"  [run]  {label} ...", end=' ', flush=True)
        _covet_apply(ad, cov_flat, n_comps=n_comps, alpha=1.0, obsm_key='_X_covet_sweep')

        X = ad.obsm['_X_covet_sweep'].astype('float32')
        aff = build_seacell_kernel(X, X, k=affinity_k, graph_mode='knn')
        aff.setdiag(0)
        aff.eliminate_zeros()

        df = audit_target_groups(
            ad, aff,
            target_groups=target_groups,
            celltype_key=celltype_key,
            niche_key=niche_key,
            covet_key='_X_covet_sweep',
            min_cells=min_cells,
            max_tier=max_tier,
            verbose=False,
        )
        detail[label] = df
        print("done")

        for _, row in df.iterrows():
            score_rows.append({
                'group':        f"{row['celltype']} | {row['niche']}",
                'setting':      label,
                'scproto_score': row['scproto_score'],
                'covet_purity':  row['covet_purity'],
                'niche_spec':    row['niche_spec'],
            })

    # clean up temp obsm key
    if '_X_covet_sweep' in ad.obsm:
        del ad.obsm['_X_covet_sweep']

    if not score_rows:
        print("No valid settings ran.")
        return pd.DataFrame(), {}

    long = pd.DataFrame(score_rows)

    pivot = (long
             .pivot_table(index='group', columns='setting', values='scproto_score')
             .round(3))
    pivot['mean'] = pivot.mean(axis=1).round(3)
    pivot = pivot.sort_values('mean', ascending=False)

    purity_pivot = (long
                    .pivot_table(index='group', columns='setting', values='covet_purity')
                    .round(3))
    purity_pivot['mean'] = purity_pivot.mean(axis=1).round(3)
    purity_pivot = purity_pivot.reindex(pivot.index)

    print(f"\n{'='*60}")
    print("scproto_score pivot (higher = better task for scProto):")
    print(pivot.to_string())
    print(f"\ncovet_purity pivot (higher = COVET separates niche better):")
    print(purity_pivot.to_string())

    return pivot, detail


def tumor_niche_metacell_eval(
    ad,
    mc_key='metacell_id',
    celltype_key='celltypes',
    niche_key='niches_3D',
    tumor_celltype='Tumor cells',
    target_niches=('Tumor core', 'Tumor surface'),
    pseudotime_key='tumor_pseudotime_rank',
    method_name='model',
    plot=True,
    figsize=(14, 4),
):
    """Evaluate metacell quality for tumor cells split by niche (core vs surface).

    Three metrics, each computed per tumor cell:
      1. celltype_purity : fraction of its metacell that are tumor cells.
                           Drops for surface tumor cells in SEACell because COVET affinity
                           links surface tumor to adjacent stromal cells.
      2. niche_purity    : of the tumor cells in its metacell, fraction from the SAME niche.
                           Tests whether core and surface tumor cells are separated within
                           the tumor compartment.
      3. pseudotime_spread: (per metacell) std of tumor_pseudotime_rank within metacell.
                            Lower = EMT gradient is respected; cells at the same EMT stage
                            are grouped together.

    Args:
        ad:              AnnData with obs[mc_key], obs[celltype_key], obs[niche_key].
        mc_key:          obs column with metacell assignments (int or 'SEACell-0' format).
        celltype_key:    obs column for cell type.
        niche_key:       obs column for niche labels.
        tumor_celltype:  value identifying tumor cells in celltype_key.
        target_niches:   niches to include in the per-cell DataFrame and plots.
        pseudotime_key:  obs column with tumor pseudotime rank; skipped if missing.
        method_name:     label for printed output and plot titles.
        plot:            if True, show distribution plots.
        figsize:         figure size.

    Returns:
        dict with keys:
            'per_cell'  — pd.DataFrame: per tumor cell (celltype_purity, niche_purity).
            'flat'      — dict: scalar means ready to merge into metrics.json.
            'pt_spread' — pd.Series: per-metacell pseudotime std (None if key missing).
    """
    obs = ad.obs.copy()
    obs[mc_key] = obs[mc_key].astype(str)

    tumor_mask = obs[celltype_key].astype(str) == str(tumor_celltype)
    tumor_obs = obs[tumor_mask].copy()

    if len(tumor_obs) == 0:
        print(f"[tumor_niche_metacell_eval] No cells matching '{tumor_celltype}' in {celltype_key}.")
        return {'per_cell': pd.DataFrame(), 'flat': {}, 'pt_spread': None}

    # --- per-metacell counts ---
    mc_total      = obs.groupby(mc_key).size()
    mc_tumor_cnt  = tumor_obs.groupby(mc_key).size().reindex(mc_total.index, fill_value=0)

    # celltype_purity: tumor cells in metacell / total cells in metacell
    mc_tumor_frac = (mc_tumor_cnt / mc_total).fillna(0)
    tumor_obs['celltype_purity'] = tumor_obs[mc_key].map(mc_tumor_frac).astype(float)

    # niche_purity: same-niche tumor cells in metacell / all tumor cells in metacell
    # stack to a (mc, niche) → count series for an efficient lookup
    mc_niche_tumor = (
        tumor_obs.groupby([mc_key, niche_key])
        .size()
        .rename('same_niche_cnt')
    )
    mc_niche_tumor_df = mc_niche_tumor.reset_index()

    tumor_obs = tumor_obs.reset_index(drop=False).rename(columns={'index': '_cell_id'})
    tumor_obs = tumor_obs.merge(
        mc_niche_tumor_df, on=[mc_key, niche_key], how='left'
    )
    tumor_obs['same_niche_cnt'] = tumor_obs['same_niche_cnt'].fillna(0)

    # denominator: total cells in metacell (same as celltype_purity)
    # → niche_purity is always <= celltype_purity
    tumor_obs['_mc_total_cnt'] = tumor_obs[mc_key].map(mc_total)
    tumor_obs['niche_purity'] = (
        tumor_obs['same_niche_cnt'] / tumor_obs['_mc_total_cnt'].clip(lower=1)
    ).astype(float)

    # filter to target niches
    target_niches_str = [str(n) for n in target_niches]
    tumor_obs['_niche_str'] = tumor_obs[niche_key].astype(str)
    per_cell = (
        tumor_obs[tumor_obs['_niche_str'].isin(target_niches_str)]
        [['_cell_id', '_niche_str', 'celltype_purity', 'niche_purity']]
        .rename(columns={'_cell_id': 'cell_id', '_niche_str': 'niche'})
        .reset_index(drop=True)
    )

    # --- pseudotime spread: std of pseudotime per metacell (tumor cells only) ---
    pt_spread = None
    if pseudotime_key in obs.columns:
        tumor_obs['_pt'] = pd.to_numeric(tumor_obs[pseudotime_key], errors='coerce')
        pt_by_mc = tumor_obs.groupby(mc_key)['_pt']
        _spread = pt_by_mc.std()
        _counts = pt_by_mc.count()
        pt_spread = _spread[_counts >= 2].dropna()

    # --- scalar summaries ---
    flat = {}
    for niche in target_niches_str:
        slug = niche.lower().replace(' ', '_')
        sub = per_cell[per_cell['niche'] == niche]
        if len(sub) == 0:
            continue
        flat[f'tumor_{slug}_celltype_purity'] = round(float(sub['celltype_purity'].mean()), 4)
        flat[f'tumor_{slug}_niche_purity']    = round(float(sub['niche_purity'].mean()), 4)
    if pt_spread is not None and len(pt_spread) > 0:
        flat['tumor_pseudotime_spread_mean'] = round(float(pt_spread.mean()), 4)

    # --- print ---
    print(f"\n--- Tumor niche metacell quality [{method_name}] ---")
    for niche in target_niches_str:
        slug = niche.lower().replace(' ', '_')
        cp = flat.get(f'tumor_{slug}_celltype_purity')
        np_ = flat.get(f'tumor_{slug}_niche_purity')
        if cp is not None:
            print(f"  {niche:30s}: celltype_purity={cp:.3f}  niche_purity={np_:.3f}")
    if 'tumor_pseudotime_spread_mean' in flat:
        print(f"  {'pseudotime_spread_mean':30s}: {flat['tumor_pseudotime_spread_mean']:.3f}  (lower=better)")

    if plot and len(per_cell) > 0:
        _plot_tumor_niche_distributions(per_cell, pt_spread, target_niches_str, method_name, figsize)

    return {'per_cell': per_cell, 'flat': flat, 'pt_spread': pt_spread}


def _plot_tumor_niche_distributions(per_cell, pt_spread, target_niches, method_name, figsize):
    n_panels = 3 if pt_spread is not None and len(pt_spread) > 0 else 2
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)

    palette = {n: c for n, c in zip(target_niches, ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e'])}

    def _violin_or_hist(ax, metric, title, xlabel):
        try:
            import seaborn as sns
            sns.violinplot(
                data=per_cell, x='niche', y=metric, order=target_niches,
                palette=palette, inner='quartile', ax=ax, cut=0,
            )
        except ImportError:
            for niche in target_niches:
                vals = per_cell.loc[per_cell['niche'] == niche, metric]
                ax.hist(vals, bins=30, alpha=0.6, label=niche, color=palette[niche])
            ax.legend(fontsize=8)
        ax.set_title(f'{title}\n[{method_name}]', fontsize=10)
        ax.set_xlabel('')
        ax.set_ylabel(xlabel)
        ax.set_ylim(0, 1.02)
        ax.axhline(1.0, color='grey', linewidth=0.6, linestyle='--')
        ax.tick_params(axis='x', labelrotation=20)

    _violin_or_hist(axes[0], 'celltype_purity',
                    'Cell type purity\n(tumor cells in metacell)',
                    'Fraction of metacell = Tumor')
    _violin_or_hist(axes[1], 'niche_purity',
                    'Within-niche purity\n(same-niche tumor in metacell)',
                    'Fraction of tumor members = same niche')

    if n_panels == 3:
        ax = axes[2]
        ax.boxplot(pt_spread.values, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='#aec7e8'), medianprops=dict(color='navy'))
        ax.set_title(f'Pseudotime spread\n[{method_name}]', fontsize=10)
        ax.set_ylabel('Std of pseudotime rank within metacell\n(lower = better)')
        ax.set_xticks([1])
        ax.set_xticklabels(['tumor metacells'])

    plt.tight_layout()
    plt.show()


def all_celltype_niche_purity(
    obs,
    celltype_key='celltypes',
    niche_key='niches_2D',
    mc_key='metacell_id',
    min_cells=20,
    min_same_type_count=3,
):
    """Per-cell celltype purity and niche purity (joint metric) for every cell type.

    Generalizes tumor_niche_metacell_eval's core idea beyond tumor cells. For each
    cell i (true celltype c, true niche n) in metacell m(i):
        celltype_purity(i)           = |{j in m(i): celltype_j == c}|                 / |m(i)|
        niche_purity(i)              = |{j in m(i): celltype_j == c AND niche_j == n}| / |m(i)|
        niche_given_celltype_purity(i) = niche_purity(i) / celltype_purity(i)
                                        = |{j in m(i): celltype_j == c AND niche_j == n}|
                                          / |{j in m(i): celltype_j == c}|

    niche_purity uses the same denominator as celltype_purity (not the same-celltype
    subset), so niche_purity <= celltype_purity always: a metacell can only score well
    on niche purity if it also groups cell types well. This avoids the failure mode where
    conditioning the niche score on "same celltype neighbours only" would let a method
    with poor celltype grouping (few/no same-type neighbours) look artificially niche-pure.

    niche_given_celltype_purity is that same-celltype-only conditional ratio, provided as
    a derived column for exactly the cases where you want to read it alongside
    celltype_purity (celltype_purity(i) * niche_given_celltype_purity(i) == niche_purity(i)
    by construction, per cell — this only holds approximately once you aggregate median/mean
    across cells, since neither operation distributes over multiplication). It's always
    well-defined since celltype_purity(i) > 0 (cell i is always in its own metacell and
    matches its own type), but it's a ratio over the *same-type count specifically*
    (_same_ct_cnt), which can be small even when the metacell itself is large or
    celltype_purity looks fine (e.g. a 2-cell metacell that's 100% one cell type still only
    has 2 same-type cells to compute this ratio from) — a handful of same-type neighbours
    can push it to a trivial 0 or 1. min_same_type_count masks (sets to NaN) cells whose
    _same_ct_cnt falls below that threshold, so those coin-flip values don't quietly bias
    the aggregate; celltype_purity and niche_purity are left alone since they aren't
    exposed to this same fragility.

    Args:
        obs:          DataFrame with columns [mc_key, celltype_key, niche_key] — either
                      ad.obs directly, or a plain per-cell DataFrame (e.g. loaded from
                      cell_assignments.csv and joined with niche labels).
        celltype_key: column with true cell type labels.
        niche_key:    column with true niche labels.
        mc_key:       column with metacell/prototype assignment (int or str).
        min_cells:    drop cell types with fewer than this many cells total.
        min_same_type_count: mask niche_given_celltype_purity to NaN for cells whose
                      metacell has fewer than this many same-type cells (default 3).
                      Pass None to disable masking.

    Returns:
        pd.DataFrame with columns: celltype, niche, celltype_purity, niche_purity,
        niche_given_celltype_purity (one row per cell, filtered to cell types with
        >= min_cells cells).
    """
    for col in (mc_key, celltype_key, niche_key):
        if col not in obs.columns:
            raise ValueError(f"'{col}' not found in obs columns: {list(obs.columns)}")

    obs = obs[[mc_key, celltype_key, niche_key]].copy()

    mc_total = obs.groupby(mc_key).size().rename('_mc_total')
    ct_cnt = obs.groupby([mc_key, celltype_key]).size().rename('_same_ct_cnt')
    ctn_cnt = obs.groupby([mc_key, celltype_key, niche_key]).size().rename('_same_ctn_cnt')

    obs = obs.join(mc_total, on=mc_key)
    obs = obs.join(ct_cnt, on=[mc_key, celltype_key])
    obs = obs.join(ctn_cnt, on=[mc_key, celltype_key, niche_key])

    obs['celltype_purity'] = obs['_same_ct_cnt'] / obs['_mc_total']
    obs['niche_purity'] = obs['_same_ctn_cnt'] / obs['_mc_total']
    obs['niche_given_celltype_purity'] = obs['_same_ctn_cnt'] / obs['_same_ct_cnt']
    if min_same_type_count is not None:
        obs.loc[obs['_same_ct_cnt'] < min_same_type_count, 'niche_given_celltype_purity'] = np.nan

    keep_cts = obs[celltype_key].value_counts()
    keep_cts = keep_cts[keep_cts >= min_cells].index
    obs = obs[obs[celltype_key].isin(keep_cts)]

    return (
        obs[[celltype_key, niche_key, 'celltype_purity', 'niche_purity', 'niche_given_celltype_purity']]
        .rename(columns={celltype_key: 'celltype', niche_key: 'niche'})
        .reset_index(drop=True)
    )


def celltype_purity_table(per_cell_by_model, metric='celltype_purity', stat='median_iqr'):
    """Aggregate all_celltype_niche_purity output across models into a model x celltype table.

    Args:
        per_cell_by_model: dict {model_name: per_cell DataFrame from all_celltype_niche_purity}.
        metric:            'celltype_purity' or 'niche_purity'.
        stat:              'median_iqr' (default, recommended) or 'mean_std'.
                           Per-cell purity is often bimodal (some cells land in a small
                           near-pure metacell, others in a large mixed one) — mean_std can
                           show a std larger than the mean, which is a symptom of forcing a
                           "center ± symmetric spread" summary onto a two-cluster distribution,
                           not a computation error. median_iqr (median, Q1, Q3) is robust to
                           that and doesn't assume symmetry.

    Returns:
        stat='mean_std':   (mean_df, std_df)
        stat='median_iqr': (median_df, q25_df, q75_df)
        All indexed by model name, columns = cell type.
    """
    if stat not in ('mean_std', 'median_iqr'):
        raise ValueError(f"stat must be 'mean_std' or 'median_iqr', got {stat!r}")

    if stat == 'mean_std':
        means, stds = {}, {}
        for name, df in per_cell_by_model.items():
            g = df.groupby('celltype')[metric]
            means[name] = g.mean()
            stds[name] = g.std()
        return pd.DataFrame(means).T, pd.DataFrame(stds).T

    medians, q25s, q75s = {}, {}, {}
    for name, df in per_cell_by_model.items():
        g = df.groupby('celltype')[metric]
        medians[name] = g.median()
        q25s[name] = g.quantile(0.25)
        q75s[name] = g.quantile(0.75)
    return pd.DataFrame(medians).T, pd.DataFrame(q25s).T, pd.DataFrame(q75s).T


def metacell_composite_confusion(
    adata,
    mc_key='metacell_id',
    celltype_key='celltypes',
    niche_2d_key='niches_2D',
    niche_3d_key='niches_3D',
    sep=' | ',
    figsize=None,
    normalize='true',
    cmap='Blues',
    annot_threshold=600,
):
    """Confusion matrix between true per-cell composite labels and majority-voted metacell labels.

    For each metacell, majority-vote celltype, niche_2D, niche_3D **separately**, then
    concatenate to form the predicted composite label for every cell in that metacell.
    True label for each cell = "{celltype} | {niche_2D} | {niche_3D}".

    Args:
        adata:            AnnData with obs containing mc_key, celltype_key, niche_2d_key, niche_3d_key.
                          Typically t.train_ds.adata after run_mc_task.
        mc_key:           obs column with metacell assignment.
        celltype_key:     obs column for cell type.
        niche_2d_key:     obs column for 2D niche.
        niche_3d_key:     obs column for 3D niche.
        sep:              separator string for composite labels.
        figsize:          figure size; auto-computed if None.
        normalize:        'true' = row-normalize (fraction of true-label cells going to each pred),
                          'pred' = col-normalize, None = raw counts.
        cmap:             matplotlib colormap for the heatmap.
        annot_threshold:  annotate cells with numbers only if n_true * n_pred <= this.

    Returns:
        dict with keys:
            'cm_df'    — pd.DataFrame: confusion matrix (true × predicted), possibly normalized.
            'obs_df'   — pd.DataFrame: per-cell true/predicted labels + metacell id.
            'mc_labels' — pd.DataFrame: per-metacell majority-voted labels and predicted composite.
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    obs = adata.obs[[mc_key, celltype_key, niche_2d_key, niche_3d_key]].copy()
    obs.columns = ['mc', 'celltype', 'niche_2d', 'niche_3d']
    obs = obs.astype(str)

    def _majority(s):
        return s.value_counts().idxmax()

    mc_labels = obs.groupby('mc').agg(
        mc_celltype=('celltype', _majority),
        mc_niche_2d=('niche_2d', _majority),
        mc_niche_3d=('niche_3d', _majority),
    )
    mc_labels['predicted'] = (
        mc_labels['mc_celltype'] + sep +
        mc_labels['mc_niche_2d'] + sep +
        mc_labels['mc_niche_3d']
    )

    obs['predicted'] = obs['mc'].map(mc_labels['predicted'])
    obs['true'] = obs['celltype'] + sep + obs['niche_2d'] + sep + obs['niche_3d']

    cm_raw = pd.crosstab(obs['true'], obs['predicted'])

    if normalize == 'true':
        cm_df = cm_raw.div(cm_raw.sum(axis=1), axis=0)
        fmt, cbar_label = '.2f', 'Fraction of true-label cells'
    elif normalize == 'pred':
        cm_df = cm_raw.div(cm_raw.sum(axis=0), axis=1)
        fmt, cbar_label = '.2f', 'Fraction of predicted-label cells'
    else:
        cm_df = cm_raw.astype(float)
        fmt, cbar_label = 'd', 'Count'

    n_true = len(cm_df.index)
    n_pred = len(cm_df.columns)
    if figsize is None:
        figsize = (max(10, n_pred * 0.6 + 4), max(8, n_true * 0.4 + 3))

    fig, ax = plt.subplots(figsize=figsize)

    try:
        import seaborn as sns
        annot = (n_true * n_pred) <= annot_threshold
        sns.heatmap(
            cm_df, ax=ax, cmap=cmap,
            annot=annot, fmt=fmt,
            linewidths=0.3, linecolor='lightgray',
            cbar_kws={'label': cbar_label},
        )
    except ImportError:
        im = ax.imshow(cm_df.values, aspect='auto', cmap=cmap)
        ax.set_xticks(range(n_pred))
        ax.set_xticklabels(cm_df.columns, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(n_true))
        ax.set_yticklabels(cm_df.index, fontsize=8)
        fig.colorbar(im, ax=ax, label=cbar_label)

    norm_tag = {'true': ' (row-norm)', 'pred': ' (col-norm)', None: ''}[normalize]
    ax.set_xlabel('Predicted label  (majority vote per metacell)', fontsize=10)
    ax.set_ylabel('True label  (per cell)', fontsize=10)
    ax.set_title(f'Metacell composite confusion matrix{norm_tag}', fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.show()

    print(f"\nTrue labels : {n_true}  |  Predicted labels : {n_pred}  |  Cells : {len(obs)}")
    print(f"Metacells   : {len(mc_labels)}")

    # print top misclassified true labels (off-diagonal mass)
    diag_mask = np.array([
        cm_df.columns.get_loc(t) if t in cm_df.columns else -1
        for t in cm_df.index
    ])
    diag_vals = np.array([
        cm_df.iloc[i, diag_mask[i]] if diag_mask[i] >= 0 else 0.0
        for i in range(n_true)
    ])
    diag_series = pd.Series(diag_vals, index=cm_df.index)
    print("\nOn-diagonal fraction per true label (sorted ascending = worst first):")
    print(diag_series.sort_values().head(15).round(3).to_string())

    return {
        'cm_df':    cm_df,
        'obs_df':   obs,
        'mc_labels': mc_labels,
    }
