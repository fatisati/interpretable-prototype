"""
paper_figures.py — publication-ready figure functions.

Typical usage:
    from interpretable_ssl.evaluation.paper_figures import fig_purity_entropy

    fig = fig_purity_entropy(
        ds_id='pancreas',
        model_keywords={'proto_umap': 'SCProto', 'seacell': 'SEACells'},
    )
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba, PowerNorm, TwoSlopeNorm


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_series(path):
    if os.path.exists(path):
        s = pd.read_csv(path, index_col=0).squeeze()
        s.index = s.index.astype(str)
        return s
    return None


def _pack(run_dir, purity_s, entropy_s):
    size_s = _read_series(os.path.join(run_dir, 'size_per_mc.csv'))
    ref = purity_s.index if purity_s is not None else (entropy_s.index if entropy_s is not None else None)
    return {
        'purity':  purity_s.values if purity_s is not None else None,
        'entropy': entropy_s.values if entropy_s is not None else None,
        'size':    size_s.reindex(ref).values.astype(float) if (size_s is not None and ref is not None) else None,
    }


# Cache subdirectory listings per base_dir to avoid repeated Drive stat calls
# when the same dataset is queried for multiple model keywords.
_subdir_cache: dict = {}


def _list_subdirs(base_dir):
    """Return sorted subdirectory names under base_dir, cached per base_dir."""
    if base_dir not in _subdir_cache:
        _subdir_cache[base_dir] = sorted(
            e for e in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, e))
        )
    return _subdir_cache[base_dir]


def _resolve_run_dir(ds_id, keyword, prefer_csv='purity_per_mc.csv'):
    """Return the run directory for ds_id/keyword.

    For keyword='seacell' returns get_seacell_model_dir(ds_id) directly.
    Otherwise scans get_dataset_model_dir(ds_id) for subdirectories whose
    extracted model key contains keyword, then picks the last alphabetically
    among those that contain prefer_csv (falling back to last overall).

    Args:
        prefer_csv: filename to prefer when multiple runs match.
                    Pass None to always take the last alphabetically.
    """
    from interpretable_ssl.configs.paths import get_dataset_model_dir, get_seacell_model_dir
    from interpretable_ssl.evaluation.metric_helpers.result_tables import extract_model_key

    if keyword == 'seacell':
        return get_seacell_model_dir(ds_id)

    base_dir = get_dataset_model_dir(ds_id)
    if not os.path.isdir(base_dir):
        return None

    matched = [
        e for e in _list_subdirs(base_dir)
        if (e == keyword or keyword in extract_model_key(e, ds_id=ds_id))
    ]
    if not matched:
        return None

    if prefer_csv:
        with_csv = [e for e in matched if os.path.exists(os.path.join(base_dir, e, prefer_csv))]
        chosen = (with_csv if with_csv else matched)[-1]
    else:
        chosen = matched[-1]

    if len(matched) > 1:
        print(f"  [fig] '{keyword}' matched {len(matched)} runs — using '{chosen}'")

    return os.path.join(base_dir, chosen)


def _load_per_mc(ds_id, keyword):
    """Return dict {purity, entropy, size} for the last run matching keyword."""
    _empty = {'purity': None, 'entropy': None, 'size': None}

    run_dir = _resolve_run_dir(ds_id, keyword, prefer_csv='purity_per_mc.csv')
    if run_dir is None:
        return _empty

    return _pack(
        run_dir,
        _read_series(os.path.join(run_dir, 'purity_per_mc.csv')),
        _read_series(os.path.join(run_dir, 'batch_entropy_per_mc.csv')),
    )


def _size_scatter_kwargs(color, sizes, encoding, flat_alpha,
                         alpha_range=(0.15, 0.85), size_range=(8, 80)):
    """Return kwargs for ax.scatter based on size encoding mode.

    encoding: 'alpha' | 'size' | 'both'
    """
    sz_norm = (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-9)
    kwargs = {}

    if encoding in ('alpha', 'both'):
        alphas = alpha_range[0] + (alpha_range[1] - alpha_range[0]) * sz_norm
        base_rgb = to_rgba(color)[:3]
        kwargs['c'] = np.column_stack([np.tile(base_rgb, (len(sizes), 1)), alphas])
    else:
        kwargs['color'] = color
        kwargs['alpha'] = flat_alpha

    if encoding in ('size', 'both'):
        kwargs['s'] = size_range[0] + (size_range[1] - size_range[0]) * sz_norm
    # 'alpha' mode keeps caller's scatter_s (set outside)

    return kwargs


def _violin_panel(ax, names, arrays, sizes, colors, ylabel, point_alpha=0.4, point_s=8, swap_positions=False):
    """
    sizes: list of size arrays (same order as arrays), or list of Nones.
    When sizes are provided, the violin uses a size-weighted distribution
    (np.repeat) so it matches the weighted_mean metric. Scatter points
    show the raw per-metacell values (one dot per metacell, sized by count).
    """
    valid = [
        (n, a, sz, c)
        for n, a, sz, c in zip(names, arrays, sizes, colors)
        if a is not None
    ]
    if swap_positions:
        valid = valid[::-1]
    if not valid:
        return
    vnames, varrays, vsizes, vcolors = zip(*valid)

    # Build weighted distributions for violin shape
    violin_data = []
    for arr, sz in zip(varrays, vsizes):
        if sz is not None and len(sz) == len(arr):
            counts = np.round(sz / sz.sum() * 1000).astype(int).clip(min=1)
            violin_data.append(np.repeat(arr, counts))
        else:
            violin_data.append(arr)

    parts = ax.violinplot(violin_data, positions=range(1, len(vnames) + 1),
                          showmedians=True, showextrema=False)
    for body, c in zip(parts['bodies'], vcolors):
        body.set_facecolor(c)
        body.set_alpha(0.6)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.5)

    # Scatter: raw per-metacell points, dot size encodes metacell size
    rng = np.random.default_rng(42)
    for i, (arr, sz, c) in enumerate(zip(varrays, vsizes, vcolors), start=1):
        jitter = rng.uniform(-0.08, 0.08, size=len(arr))
        if sz is not None and len(sz) == len(arr):
            sz_norm = (sz - sz.min()) / (sz.max() - sz.min() + 1e-9)
            pt_sizes = 4 + 30 * sz_norm
        else:
            pt_sizes = point_s
        ax.scatter(i + jitter, arr, color=c, alpha=point_alpha, s=pt_sizes, zorder=3)

    ax.set_xticks(range(1, len(vnames) + 1))
    ax.set_xticklabels(vnames, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel(ylabel)


def _box_panel(ax, names, arrays, sizes, colors, ylabel, point_alpha=0.4, swap_positions=False):
    """Box plots with jittered scatter overlay, one box per model.

    sizes: list of size arrays or Nones. When provided, box uses size-weighted
    distribution (np.repeat); scatter dot size encodes metacell size.
    """
    valid = [
        (n, a, sz, c)
        for n, a, sz, c in zip(names, arrays, sizes, colors)
        if a is not None
    ]
    if swap_positions:
        valid = valid[::-1]
    if not valid:
        return
    vnames, varrays, vsizes, vcolors = zip(*valid)

    rng = np.random.default_rng(42)
    for i, (name, arr, sz, color) in enumerate(zip(vnames, varrays, vsizes, vcolors), start=1):
        if sz is not None and len(sz) == len(arr):
            counts = np.round(sz / sz.sum() * 1000).astype(int).clip(min=1)
            box_data = np.repeat(arr, counts)
        else:
            box_data = arr

        bp = ax.boxplot(box_data, positions=[i], widths=0.5,
                        patch_artist=True, manage_ticks=False,
                        boxprops=dict(facecolor=color, alpha=0.7),
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(color=color, linewidth=1.2),
                        capprops=dict(color=color, linewidth=1.2),
                        flierprops=dict(marker='o', markersize=2,
                                        markerfacecolor=color, alpha=0.4,
                                        linestyle='none'))

        jitter = rng.uniform(-0.12, 0.12, size=len(arr))
        if sz is not None and len(sz) == len(arr):
            sz_norm = (sz - sz.min()) / (sz.max() - sz.min() + 1e-9)
            pt_sizes = 4 + 30 * sz_norm
        else:
            pt_sizes = 8
        ax.scatter(i + jitter, arr, color=color, alpha=point_alpha,
                   s=pt_sizes, zorder=3, linewidths=0)

    ax.set_xticks(range(1, len(vnames) + 1))
    ax.set_xticklabels(vnames, rotation=20, ha='right', fontsize=9)
    ax.set_xlim(0.5, len(vnames) + 0.5)
    ax.set_ylabel(ylabel)


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def debug_mc_alignment(ds_id, model_keywords):
    """Print purity vs size index alignment for each keyword.

    Same keyword resolution as fig_purity_entropy — use identical arguments
    to check what will actually be loaded.

    Usage:
        debug_mc_alignment('pancreas', {'proto_umap': 'SCProto', 'seacell': 'SEACells'})
    """
    for keyword, display_name in model_keywords.items():
        print(f"\n{'='*60}")
        print(f"  {display_name}  (keyword='{keyword}')")
        print(f"{'='*60}")

        run_dir = _resolve_run_dir(ds_id, keyword, prefer_csv='purity_per_mc.csv')
        if run_dir is None:
            print("  ERROR: no matching run directory found")
            continue
        print(f"  run_dir: {run_dir}")

        # purity
        purity_path = os.path.join(run_dir, 'purity_per_mc.csv')
        purity_s = _read_series(purity_path)
        if purity_s is None:
            print("  purity_per_mc.csv: NOT FOUND")
        else:
            print(f"  purity_per_mc.csv: {len(purity_s)} metacells")
            print(f"    index[:10] = {purity_s.index[:10].tolist()}")

        # sizes from size_per_mc.csv
        size_s = _read_series(os.path.join(run_dir, 'size_per_mc.csv'))
        if size_s is None:
            print("  size_per_mc.csv: NOT FOUND (re-run eval to generate it)")
            continue

        print(f"  size_per_mc.csv: {len(size_s)} metacells")
        print(f"    index[:10] = {size_s.index[:10].tolist()}")

        if purity_s is not None:
            only_purity = set(purity_s.index) - set(size_s.index)
            only_sizes  = set(size_s.index)  - set(purity_s.index)
            print(f"  in purity but not sizes : {sorted(only_purity) if only_purity else '(none)'}")
            print(f"  in sizes  but not purity: {sorted(only_sizes)  if only_sizes  else '(none)'}")

            aligned = size_s.reindex(purity_s.index)
            df = pd.DataFrame({'purity': purity_s.values, 'size': aligned.values},
                              index=purity_s.index)
            print(f"\n  First 10 rows after alignment:")
            print(df.head(10).to_string())


# ---------------------------------------------------------------------------
# Public figure functions
# ---------------------------------------------------------------------------

def _draw_purity_entropy_row(
    axes, model_data, name_to_color,
    ds_id, scatter_alpha, scatter_s, size_encoding,
    purity_thresh, entropy_thresh, threshold_use_count,
    show_threshold_panel, is_first_row, swap_positions=False,
):
    """Draw one row of purity/entropy panels for a single dataset onto *axes*."""
    names  = list(model_data.keys())
    colors = [name_to_color[n] for n in names]

    # Panel A: scatter purity vs entropy
    ax = axes[0]
    for i, name in enumerate(names):
        p, e, sz = model_data[name]['purity'], model_data[name]['entropy'], model_data[name]['size']
        if p is None or e is None or len(p) != len(e):
            continue
        if sz is not None and len(sz) == len(p):
            kw = _size_scatter_kwargs(colors[i], sz, size_encoding, scatter_alpha)
            if 's' not in kw:
                kw['s'] = scatter_s
            ax.scatter(e, p, **kw, label=name)
        else:
            ax.scatter(e, p, color=colors[i], alpha=scatter_alpha, s=scatter_s, label=name)
    ax.set_xlabel('Batch Entropy')
    ax.set_ylabel(f'{ds_id}\nCell Type Purity')
    if is_first_row:
        ax.set_title('Purity vs Batch Entropy\n(darker = larger metacell)')
        ax.legend(markerscale=1.0, fontsize=9, handletextpad=0.4, labelspacing=0.3)

    # Panel B: purity violin
    _violin_panel(
        axes[1], names,
        [model_data[n]['purity'] for n in names],
        [model_data[n]['size']   for n in names],
        colors, 'Cell Type Purity', swap_positions=swap_positions,
    )
    if is_first_row:
        axes[1].set_title('Cell Type Purity\n(violin: size-weighted)')

    # Panel C: entropy violin
    _violin_panel(
        axes[2], names,
        [model_data[n]['entropy'] for n in names],
        [model_data[n]['size']    for n in names],
        colors, 'Batch Entropy', swap_positions=swap_positions,
    )
    if is_first_row:
        axes[2].set_title('Batch Entropy\n(violin: size-weighted)')

    # Panel D: threshold bar chart (optional)
    if show_threshold_panel:
        ax = axes[3]
        bar_width = 0.6
        for i, (name, color) in enumerate(zip(names, colors)):
            p, e = model_data[name]['purity'], model_data[name]['entropy']
            if p is not None and e is not None and len(p) == len(e):
                mask = (p > purity_thresh) & (e > entropy_thresh)
                val  = int(mask.sum()) if threshold_use_count else float(mask.mean())
            else:
                val = 0
            ax.bar(i, val, width=bar_width, color=color, alpha=0.85)
            label_text = str(int(val)) if threshold_use_count else f'{val:.2f}'
            ax.text(i, val + (0.01 if not threshold_use_count else max(1, val * 0.02)),
                    label_text, ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
        ax.set_ylabel('# metacells' if threshold_use_count else 'Ratio of metacells')
        if not threshold_use_count:
            ax.set_ylim(0, 1.12)
        if is_first_row:
            ax.set_title(f'Purity > {purity_thresh}\nand Entropy > {entropy_thresh}')


def fig_purity_entropy(
    ds_id,
    model_keywords,
    save_path=None,
    figsize=(18, 4),
    palette=None,
    scatter_alpha=0.5,
    scatter_s=12,
    size_encoding='both',
    purity_thresh=0.9,
    entropy_thresh=0.5,
    threshold_use_count=False,
    show_threshold_panel=True,
    swap_positions=False,
):
    """4-panel figure per dataset: scatter + purity violin + entropy violin + threshold bar.

    ds_id can be a single dataset id string or a list of dataset ids. When a list
    is given, one row of panels is drawn per dataset; column titles appear on the
    first row only and each row is labelled by its dataset id on the y-axis.

    Metacell size (from size_per_mc.csv) is encoded in the scatter via size_encoding.
    Falls back to flat alpha + fixed dot size if size_per_mc.csv is absent.

    Args:
        ds_id:               dataset id string or list of dataset id strings.
        model_keywords:      dict {keyword: display_name}. Keyword matched as substring
                             of the extracted model key. Use 'seacell' for SEACells.
        save_path:           optional path to save (e.g. 'fig2.pdf'). dpi=300.
        figsize:             (width, height) per row; total height scales with n datasets.
        palette:             list of hex/named colors. Defaults to tab10.
        scatter_alpha:       flat alpha used as fallback when size info is unavailable,
                             and as the fixed alpha for size_encoding='size'.
        scatter_s:           fixed dot size used as fallback and for size_encoding='alpha'.
        size_encoding:       how metacell size is shown in the scatter.
                             'alpha' — larger = darker (fixed dot size)
                             'size'  — larger = bigger dot (fixed alpha)
                             'both'  — larger = bigger + darker  [default]
        purity_thresh:       purity threshold for panel D (default 0.9).
        entropy_thresh:      entropy threshold for panel D (default 0.5).
        threshold_use_count: if True, panel D shows absolute count instead of ratio.
        show_threshold_panel: if False, panel D is omitted (3-panel layout).

    Returns:
        matplotlib Figure
    """
    if palette is None:
        palette = list(plt.cm.tab10.colors)

    ds_ids = [ds_id] if isinstance(ds_id, str) else list(ds_id)

    # stable colour assignment so models keep the same colour across all rows
    all_names     = list(model_keywords.values())
    name_to_color = {n: palette[i % len(palette)] for i, n in enumerate(all_names)}

    n_panels = 4 if show_threshold_panel else 3
    panel_w, panel_h = figsize
    fig, axes = plt.subplots(
        len(ds_ids), n_panels,
        figsize=(panel_w, panel_h * len(ds_ids)),
        squeeze=False,
    )

    for row, did in enumerate(ds_ids):
        model_data = {}
        for keyword, display_name in model_keywords.items():
            d = _load_per_mc(did, keyword)
            if d['purity'] is None and d['entropy'] is None:
                print(f"Warning: no per-mc CSVs found for keyword='{keyword}', ds='{did}'")
                continue
            model_data[display_name] = d

        if not model_data:
            print(f"No data loaded for ds='{did}' — check keyword spelling and MODEL_DIR.")
            for ax in axes[row]:
                ax.set_visible(False)
            continue

        _draw_purity_entropy_row(
            axes[row], model_data, name_to_color,
            did, scatter_alpha, scatter_s, size_encoding,
            purity_thresh, entropy_thresh, threshold_use_count,
            show_threshold_panel, is_first_row=(row == 0),
            swap_positions=swap_positions,
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


# ---------------------------------------------------------------------------
# UMAP figure
# ---------------------------------------------------------------------------

def _load_umap(ds_id, keyword):
    """Return dict {cells_df, protos_df, extra_cols} or None if files absent."""
    run_dir = _resolve_run_dir(ds_id, keyword, prefer_csv='umap_cells.csv')
    if run_dir is None:
        return None
    cells_path = os.path.join(run_dir, 'umap_cells.csv')
    protos_path = os.path.join(run_dir, 'umap_protos.csv')
    if not os.path.exists(cells_path):
        print(f"  [fig_umap] umap_cells.csv not found in {run_dir} — run save_umap_data() first")
        return None
    cells_df = pd.read_csv(cells_path)
    protos_df = pd.read_csv(protos_path) if os.path.exists(protos_path) else None
    fixed = {'umap_1', 'umap_2', 'metacell_id'}
    extra_cols = [c for c in cells_df.columns if c not in fixed]
    return {'cells': cells_df, 'protos': protos_df, 'extra_cols': extra_cols}


def _umap_panel(ax, cells_df, protos_df, color_col, title,
                point_s=10, point_alpha=0.5, show_protos=True,
                proto_size_scale=0.5):
    """Draw one UMAP scatter on ax. Returns cat_to_color for shared legend."""
    if color_col not in cells_df.columns:
        ax.set_visible(False)
        return {}

    categories = sorted(cells_df[color_col].dropna().unique(), key=str)
    cmap = plt.cm.get_cmap('tab20', len(categories))
    cat_to_color = {c: cmap(i) for i, c in enumerate(categories)}

    for cat in categories:
        mask = cells_df[color_col] == cat
        ax.scatter(cells_df.loc[mask, 'umap_1'], cells_df.loc[mask, 'umap_2'],
                   c=[cat_to_color[cat]], s=point_s, alpha=point_alpha,
                   linewidths=0, rasterized=True)

    if show_protos and protos_df is not None:
        majority_col = f'majority_{color_col}'
        if majority_col not in protos_df.columns and color_col in cells_df.columns and 'metacell_id' in cells_df.columns:
            from collections import Counter
            maj_map = (
                cells_df.groupby('metacell_id')[color_col]
                .agg(lambda x: Counter(x).most_common(1)[0][0])
            )
        else:
            maj_map = None

        for _, row in protos_df.iterrows():
            if majority_col in protos_df.columns:
                maj = row[majority_col]
            elif maj_map is not None:
                maj = maj_map.get(row['proto_id'])
            else:
                maj = None
            c = cat_to_color.get(maj, 'white') if maj is not None else 'white'
            size = proto_size_scale * max(50, min(300, int(row['n_cells']) // 2))
            ax.scatter(row['umap_1'], row['umap_2'], c=[c],
                       edgecolors='black', linewidths=1, s=size, zorder=10)

    ax.set_title(title, fontsize=9)
    ax.set_xlabel('UMAP 1', fontsize=8)
    ax.set_ylabel('UMAP 2', fontsize=8)
    ax.tick_params(labelsize=7)
    return cat_to_color


def fig_umap(
    ds_id,
    model_keywords,
    celltype_key=None,
    batch_key=None,
    save_path=None,
    figsize=(5, 4),
    show_protos=True,
    proto_size_scale=0.5,
    point_s=10,
    point_alpha=0.5,
    celltype_no_proto=False,
):
    """UMAP figure: 2 columns (cell type + batch) per model row, one row per model.

    Args:
        ds_id:              dataset id (e.g. 'pancreas').
        model_keywords:     dict {keyword: display_name}, same as fig_purity_entropy.
        celltype_key:       obs column for cell type (auto-detected from CSV if None).
        batch_key:          obs column for batch (auto-detected from CSV if None).
        save_path:          optional save path (dpi=300).
        figsize:            (w, h) per subplot panel.
        show_protos:        overlay prototypes on all panels.
        celltype_no_proto:  if True, add an extra column: cell type coloring, no prototypes.
        point_s:            marker size for cells.
        point_alpha:        alpha for cells.

    Returns:
        matplotlib Figure
    """
    model_data = {}
    for keyword, display_name in model_keywords.items():
        d = _load_umap(ds_id, keyword)
        if d is None:
            continue
        extra = d['extra_cols']
        ck = celltype_key if celltype_key else (extra[0] if len(extra) > 0 else None)
        bk = batch_key if batch_key else (extra[1] if len(extra) > 1 else None)
        model_data[display_name] = {**d, 'celltype_key': ck, 'batch_key': bk}

    if not model_data:
        print("No UMAP data found — check keywords and run save_umap_data() first.")
        return None

    names = list(model_data.keys())
    n_rows = len(names)
    n_cols = 3 if celltype_no_proto else 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows),
                             squeeze=False)

    col_cat_colors = [{}] * n_cols
    for row, name in enumerate(names):
        d = model_data[name]
        cells_df, protos_df = d['cells'], d['protos']
        ck, bk = d['celltype_key'], d['batch_key']

        c2c_ck = _umap_panel(axes[row, 0], cells_df, protos_df, ck,
                              title=f'{name} — {ck}',
                              point_s=point_s, point_alpha=point_alpha,
                              show_protos=show_protos,
                              proto_size_scale=proto_size_scale)
        c2c_bk = _umap_panel(axes[row, 1], cells_df, protos_df, bk,
                              title=f'{name} — {bk}',
                              point_s=point_s, point_alpha=point_alpha,
                              show_protos=show_protos,
                              proto_size_scale=proto_size_scale)
        if celltype_no_proto:
            _umap_panel(axes[row, 2], cells_df, protos_df, ck,
                        title=f'{name} — {ck} (cells)',
                        point_s=point_s, point_alpha=point_alpha,
                        show_protos=False)
        if row == 0:
            col_cat_colors = [c2c_ck, c2c_bk] + ([c2c_ck] if celltype_no_proto else [])

    # One shared legend per column, placed below the bottom panel
    from matplotlib.patches import Patch
    for col, cat_colors in enumerate(col_cat_colors):
        if not cat_colors:
            continue
        handles = [Patch(facecolor=cat_colors[c], label=str(c)) for c in sorted(cat_colors, key=str)]
        axes[-1, col].legend(
            handles=handles, ncol=2, fontsize=7, frameon=True,
            loc='upper center', bbox_to_anchor=(0.5, -0.18),
            borderaxespad=0,
        )

    fig.suptitle(ds_id, fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


# ---------------------------------------------------------------------------
# Per-model purity summary (CT purity + niche purity, one box per model)
# ---------------------------------------------------------------------------

def fig_purity_niche_summary(
    ds_id,
    model_keywords,
    save_path=None,
    figsize=(8, 4),
    palette=None,
    size_weighted=False,
    swap_positions=False,
    plot_type='box',
):
    """2-panel figure: cell-type purity violin | niche purity violin, one per model.

    Args:
        ds_id:          dataset id string or list of dataset id strings.
        model_keywords: dict {keyword: display_name}.
        save_path:      optional save path (dpi=300).
        figsize:        (width, height) per row; total height scales with n datasets.
        palette:        list of colours; defaults to tab10.
        size_weighted:  if True, violin shape is size-weighted.
        swap_positions: reverse model order.

    Returns:
        matplotlib Figure
    """
    if palette is None:
        palette = list(plt.cm.tab10.colors)

    ds_ids = [ds_id] if isinstance(ds_id, str) else list(ds_id)
    all_names     = list(model_keywords.values())
    name_to_color = {n: palette[i % len(palette)] for i, n in enumerate(all_names)}

    panel_w, panel_h = figsize
    fig, axes = plt.subplots(
        len(ds_ids), 2,
        figsize=(panel_w, panel_h * len(ds_ids)),
        squeeze=False,
    )

    for row, did in enumerate(ds_ids):
        model_data = {}
        for keyword, display_name in model_keywords.items():
            df = _load_niche_purity(did, keyword)
            if df is None:
                print(f"Warning: no data for keyword='{keyword}', ds='{did}'")
                continue
            model_data[display_name] = df

        if not model_data:
            for ax in axes[row]:
                ax.set_visible(False)
            continue

        names  = list(model_data.keys())
        colors = [name_to_color[n] for n in names]

        for metric, col, ylabel in [
            ('purity',       0, 'Cell Type Purity'),
            ('niche_purity', 1, 'Niche Purity'),
        ]:
            arrays = []
            sizes  = []
            for name in names:
                df = model_data[name]
                valid = df.dropna(subset=[metric])
                arrays.append(valid[metric].values if len(valid) else None)
                sz = valid['size'].values.astype(float) if len(valid) else None
                sizes.append(sz if (size_weighted and sz is not None) else None)

            _panel = _violin_panel if plot_type == 'violin' else _box_panel
            _panel(
                axes[row, col], names, arrays, sizes, colors,
                ylabel=f'{did}\n{ylabel}' if col == 0 else ylabel,
                swap_positions=swap_positions,
            )
            if row == 0:
                axes[row, col].set_title(
                    f'{ylabel}\n({"size-weighted" if size_weighted else "unweighted"})'
                )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# ---------------------------------------------------------------------------
# Per-cell-type purity comparison figure
# ---------------------------------------------------------------------------

def _draw_significance_brackets(ax, names, arrays, stat_ref, swap_positions):
    """Draw Mann-Whitney U significance brackets vs stat_ref on ax."""
    from scipy.stats import mannwhitneyu

    valid = [(n, a) for n, a in zip(names, arrays) if a is not None]
    if swap_positions:
        valid = valid[::-1]
    vnames = [n for n, _ in valid]
    varrays = [a for _, a in valid]

    if stat_ref not in vnames:
        return

    ref_pos = vnames.index(stat_ref) + 1   # 1-based x position
    ref_arr = varrays[vnames.index(stat_ref)]

    ymax = ax.get_ylim()[1]
    step = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.08
    bracket_height = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])

    level = 0
    for i, (name, arr) in enumerate(zip(vnames, varrays)):
        if name == stat_ref:
            continue
        try:
            _, p = mannwhitneyu(ref_arr, arr, alternative='greater')
        except Exception:
            continue

        if p < 0.001:
            stars = '***'
        elif p < 0.01:
            stars = '**'
        elif p < 0.05:
            stars = '*'
        else:
            continue   # skip ns

        other_pos = i + 1
        y = ymax + step * (level + 1)
        x1, x2 = min(ref_pos, other_pos), max(ref_pos, other_pos)

        ax.plot([x1, x1, x2, x2],
                [y, y + bracket_height, y + bracket_height, y],
                color='black', linewidth=0.8)
        ax.text((x1 + x2) / 2, y + bracket_height, stars,
                ha='center', va='bottom', fontsize=8)

        ax.set_ylim(ax.get_ylim()[0], y + bracket_height * 3)
        level += 1


def fig_purity_by_celltype(
    ds_id,
    model_keywords,
    cell_types,
    save_path=None,
    figsize=(4, 4),
    palette=None,
    size_weighted=False,
    swap_positions=False,
    plot_type='box',
    stat_ref=None,
    scatter_row=False,
    scatter_highlight=None,
    scatter_size_weighted=False,
):
    """2-row figure filtered to specific cell types.

    Row 1: niche purity box plots — one column per cell type, one box per model.
    Row 2: cell type purity box plots — same layout.

    Args:
        ds_id:          dataset id string.
        model_keywords: dict {keyword: display_name}.
        cell_types:     list of cell type names to include (must match majority label).
        save_path:      optional save path (dpi=300).
        figsize:        (width, height) per subplot panel.
        palette:        list of colours; defaults to tab10.
        size_weighted:  if True, box shape is size-weighted.
        swap_positions: reverse model order within each panel.

    Returns:
        matplotlib Figure
    """
    if palette is None:
        palette = list(plt.cm.tab10.colors)

    all_names     = list(model_keywords.values())
    name_to_color = {n: palette[i % len(palette)] for i, n in enumerate(all_names)}

    model_data = {}
    for keyword, display_name in model_keywords.items():
        df = _load_niche_purity(ds_id, keyword)
        if df is None:
            print(f"Warning: no data for keyword='{keyword}', ds='{ds_id}'")
            continue
        model_data[display_name] = df

    if not model_data:
        print(f"No data loaded for ds='{ds_id}'")
        return None

    names  = list(model_data.keys())
    colors = [name_to_color[n] for n in names]
    n_cts  = len(cell_types)

    pw, ph = figsize
    n_fig_rows = 3 if scatter_row else 2
    box_row_offset = 1 if scatter_row else 0
    fig, axes = plt.subplots(n_fig_rows, n_cts, figsize=(pw * n_cts, ph * n_fig_rows), squeeze=False)

    # --- scatter row ---
    if scatter_row:
        from matplotlib.patches import Patch
        scatter_legend_handles = []
        for col, ct in enumerate(cell_types):
            ax = axes[0, col]
            handles_this = []
            for name in names:
                color = name_to_color[name]
                df = model_data[name]
                sub = df[df['cell_type'] == ct].dropna(subset=['purity', 'niche_purity'])
                if len(sub) == 0:
                    continue
                x, y = sub['purity'].values, sub['niche_purity'].values
                if scatter_size_weighted:
                    w = sub['size'].values.astype(float)
                    w = w / w.sum()
                    cx = float(np.average(x, weights=w))
                    cy = float(np.average(y, weights=w))
                    n_eff = 1.0 / float(np.sum(w ** 2))
                    xe = float(np.sqrt(np.sum(w * (x - cx) ** 2) * n_eff / (n_eff - 1) / n_eff)) if n_eff > 1 else 0.0
                    ye = float(np.sqrt(np.sum(w * (y - cy) ** 2) * n_eff / (n_eff - 1) / n_eff)) if n_eff > 1 else 0.0
                else:
                    n = len(x)
                    cx, cy = float(x.mean()), float(y.mean())
                    xe = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
                    ye = float(y.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
                is_hl = (scatter_highlight is not None and name == scatter_highlight)
                ax.errorbar(cx, cy, xerr=xe, yerr=ye,
                            fmt='*' if is_hl else 'o',
                            color=color,
                            markersize=13 if is_hl else 7,
                            markeredgecolor='black',
                            markeredgewidth=1.5 if is_hl else 0.6,
                            elinewidth=2.0 if is_hl else 1.2,
                            capsize=3, capthick=1.2, zorder=6 if is_hl else 5)
                handles_this.append(Patch(facecolor=color, label=name))
            ax.set_xlim(0, 1.05)
            ax.set_ylim(0, 1.05)
            ax.set_title(ct, fontsize=9, fontweight='bold')
            ax.set_xlabel('CT Purity', fontsize=8)
            ax.set_ylabel('Niche Purity' if col == 0 else '', fontsize=8)
            ax.tick_params(labelsize=7)
            if col == 0:
                scatter_legend_handles = handles_this
        if scatter_legend_handles:
            leg = axes[0, 0].legend(handles=scatter_legend_handles, fontsize=8,
                                    handletextpad=0.4, labelspacing=0.3, framealpha=0.8)
            if scatter_highlight is not None:
                for text in leg.get_texts():
                    if text.get_text() == scatter_highlight:
                        text.set_fontweight('bold')

    # --- box/violin rows ---
    for col, ct in enumerate(cell_types):
        for row, (metric, ylabel) in enumerate([
            ('niche_purity', 'Niche Purity'),
            ('purity',       'Cell Type Purity'),
        ]):
            arrays, sizes = [], []
            for name in names:
                df = model_data[name]
                sub = df[df['cell_type'] == ct].dropna(subset=[metric])
                arrays.append(sub[metric].values if len(sub) else None)
                sz = sub['size'].values.astype(float) if len(sub) else None
                sizes.append(sz if size_weighted else None)

            ax = axes[row + box_row_offset, col]
            _panel = _violin_panel if plot_type == 'violin' else _box_panel
            _panel(ax, names, arrays, sizes, colors,
                   ylabel=ylabel if col == 0 else '',
                   swap_positions=swap_positions,
                   point_alpha=0)
            if stat_ref is not None:
                _draw_significance_brackets(ax, names, arrays, stat_ref, swap_positions)
            if row == 0 and not scatter_row:
                ax.set_title(ct, fontsize=9, fontweight='bold')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# ---------------------------------------------------------------------------
# Per-cell-type niche vs CT purity scatter figure
# ---------------------------------------------------------------------------

def fig_scatter_by_celltype(
    ds_id,
    model_keywords,
    cell_types,
    save_path=None,
    figsize=(4, 4),
    palette=None,
    scatter_alpha=0.5,
    scatter_s=18,
    scatter_mode='scatter',
    show_centroid=True,
    centroid_arrows=None,
    size_weighted=True,
    group_by='cell_type',
    highlight=None,
    n_cols=None,
    n_rows=None,
):
    """One scatter panel per cell type: x=cell-type purity, y=niche purity, color=model.

    Each model's dots are plotted with low alpha; a filled centroid marker shows
    the mean position.

    Args:
        ds_id:            dataset id string.
        model_keywords:   dict {keyword: display_name}.
        cell_types:       list of cell type names (one column per type).
        save_path:        optional save path (dpi=300).
        figsize:          (width, height) per panel.
        palette:          list of colours; defaults to tab10.
        scatter_alpha:    dot transparency (default 0.3).
        scatter_s:        dot size (default 8).
        scatter_mode:     'scatter', 'kde', or 'mean_se' (no scatter; one errorbar marker
                          per model showing mean ± 1 SE in both x and y).
        show_centroid:    if True, overlay a large centroid marker per model.
        size_weighted:    if True (default), weight mean/SE in 'mean_se' mode by metacell size.

    Returns:
        matplotlib Figure
    """
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch
    from scipy.stats import gaussian_kde

    if palette is None:
        palette = list(plt.cm.tab10.colors)

    all_names     = list(model_keywords.values())
    name_to_color = {n: palette[i % len(palette)] for i, n in enumerate(all_names)}

    model_data = {}
    for keyword, display_name in model_keywords.items():
        df = _load_niche_purity(ds_id, keyword)
        if df is None:
            print(f"Warning: no data for keyword='{keyword}', ds='{ds_id}'")
            continue
        model_data[display_name] = df

    if not model_data:
        print(f"No data loaded for ds='{ds_id}'")
        return None

    names = list(model_data.keys())

    if cell_types is None:
        seen = []
        for df in model_data.values():
            for v in df[group_by].dropna().unique():
                if v not in seen:
                    seen.append(v)
        cell_types = sorted(seen)

    n_cts = len(cell_types)

    # resolve grid dimensions
    if n_cols is not None and n_rows is None:
        ncols = n_cols
        nrows = int(np.ceil(n_cts / ncols))
    elif n_rows is not None and n_cols is None:
        nrows = n_rows
        ncols = int(np.ceil(n_cts / nrows))
    elif n_rows is not None and n_cols is not None:
        nrows, ncols = n_rows, n_cols
    else:
        nrows, ncols = 1, n_cts

    pw, ph = figsize
    fig, axes = plt.subplots(nrows, ncols, figsize=(pw * ncols, ph * nrows), squeeze=False)

    legend_handles = []

    for idx, ct in enumerate(cell_types):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        handles_this = []
        centroids = {}
        for name in names:
            color = name_to_color[name]
            df = model_data[name]
            sub = df[df[group_by] == ct].dropna(subset=['purity', 'niche_purity'])
            if len(sub) == 0:
                continue
            x, y = sub['purity'].values, sub['niche_purity'].values

            if scatter_mode == 'mean_se':
                if size_weighted:
                    w = sub['size'].values.astype(float)
                    w = w / w.sum()
                    cx = float(np.average(x, weights=w))
                    cy = float(np.average(y, weights=w))
                    n_eff = 1.0 / float(np.sum(w ** 2))
                    xe = float(np.sqrt(np.sum(w * (x - cx) ** 2) * n_eff / (n_eff - 1) / n_eff)) if n_eff > 1 else 0.0
                    ye = float(np.sqrt(np.sum(w * (y - cy) ** 2) * n_eff / (n_eff - 1) / n_eff)) if n_eff > 1 else 0.0
                else:
                    n = len(x)
                    cx, cy = float(x.mean()), float(y.mean())
                    xe = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
                    ye = float(y.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
                is_hl = (highlight is not None and name == highlight)
                ax.errorbar(cx, cy, xerr=xe, yerr=ye,
                            fmt='*' if is_hl else 'o',
                            color=color,
                            markersize=13 if is_hl else 7,
                            markeredgecolor='black',
                            markeredgewidth=1.5 if is_hl else 0.6,
                            elinewidth=2.0 if is_hl else 1.2,
                            capsize=3, capthick=1.2, zorder=6 if is_hl else 5)
                centroids[name] = (cx, cy)
                handles_this.append(Patch(facecolor=color, label=name))
            elif scatter_mode == 'kde' and len(sub) >= 4:
                try:
                    kernel = gaussian_kde(np.vstack([x, y]))
                    xlo, xhi = max(0, x.min() - 0.02), min(1, x.max() + 0.02)
                    ylo, yhi = max(0, y.min() - 0.02), min(1, y.max() + 0.02)
                    xx, yy = np.mgrid[xlo:xhi:80j, ylo:yhi:80j]
                    z = kernel(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                    rgba = to_rgba(color)
                    cmap = LinearSegmentedColormap.from_list('', [(*rgba[:3], 0), (*rgba[:3], 0.6)])
                    ax.contourf(xx, yy, z, levels=6, cmap=cmap, zorder=2)
                    ax.contour(xx, yy, z, levels=3, colors=[color], linewidths=0.8, alpha=0.9, zorder=2)
                except Exception:
                    ax.scatter(x, y, color=color, alpha=scatter_alpha, s=scatter_s,
                               linewidths=0, zorder=2)
                handles_this.append(Patch(facecolor=color, label=name))
            else:
                ax.scatter(x, y, color=color, alpha=scatter_alpha, s=scatter_s,
                           linewidths=0, zorder=2)
                handles_this.append(Patch(facecolor=color, label=name))

            if scatter_mode != 'mean_se':
                cx, cy = float(x.mean()), float(y.mean())
                centroids[name] = (cx, cy)
                if show_centroid:
                    ax.scatter(cx, cy, color=color, s=90, marker='*',
                               edgecolors='black', linewidths=0.8, zorder=5)

        # arrows between centroid pairs
        if centroid_arrows:
            for src, dst in centroid_arrows:
                if src not in centroids:
                    print(f"[arrow] '{src}' not in centroids for '{ct}'. Available: {list(centroids)}")
                    continue
                if dst not in centroids:
                    print(f"[arrow] '{dst}' not in centroids for '{ct}'. Available: {list(centroids)}")
                    continue
                x0, y0 = centroids[src]
                x1, y1 = centroids[dst]
                ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(
                                arrowstyle='->', color='black',
                                lw=2.0, shrinkA=3, shrinkB=3,
                                mutation_scale=15,
                            ))

        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)

        ax.set_title(ct, fontsize=9, fontweight='bold')
        ax.set_xlabel('Cell Type Purity', fontsize=8)
        ax.set_ylabel('Niche Purity' if col == 0 else '', fontsize=8)
        ax.tick_params(labelsize=7)

        if idx == 0:
            legend_handles = handles_this

    # hide unused panels
    for idx in range(n_cts, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    if legend_handles:
        leg = axes[0, 0].legend(handles=legend_handles, fontsize=8,
                                handletextpad=0.4, labelspacing=0.3, framealpha=0.8)
        if highlight is not None:
            for text in leg.get_texts():
                if text.get_text() == highlight:
                    text.set_fontweight('bold')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# ---------------------------------------------------------------------------
# Niche purity binned by cell-type purity figure
# ---------------------------------------------------------------------------

_DEFAULT_PURITY_BINS = [0.0, 0.5, 0.7, 0.85, 0.95, 1.01]


def _binned_niche_panel(ax, names, model_data, colors, purity_bins, size_weighted,
                        swap_positions=False, is_first_row=True):
    """Grouped bar chart: x = cell-type purity bin, y = mean niche purity per model."""
    bin_edges = np.array(purity_bins)
    n_bins = len(bin_edges) - 1
    bin_labels = [f'{bin_edges[k]:.2f}–{bin_edges[k+1]:.2f}' for k in range(n_bins)]

    n_models = len(names)
    group_width = 0.8
    bar_width = group_width / n_models
    rng = np.random.default_rng(42)

    for color_i, name in enumerate(names):
        pos_i = (n_models - 1 - color_i) if swap_positions else color_i
        df = model_data.get(name)
        if df is None:
            continue
        valid = df.dropna(subset=['purity', 'niche_purity', 'size'])
        color = colors[color_i]

        means, errs, xs = [], [], []
        for j in range(n_bins):
            lo, hi = bin_edges[j], bin_edges[j + 1]
            mask = (valid['purity'] >= lo) & (valid['purity'] < hi)
            sub = valid[mask]
            if len(sub) == 0:
                continue
            vals = sub['niche_purity'].values
            sizes = sub['size'].values.astype(float)
            if size_weighted:
                mean_val = np.average(vals, weights=sizes)
                # weighted std
                err_val = np.sqrt(np.average((vals - mean_val) ** 2, weights=sizes))
            else:
                mean_val = vals.mean()
                err_val = vals.std()
            x = j + (pos_i - (n_models - 1) / 2) * bar_width
            means.append(mean_val)
            errs.append(err_val)
            xs.append(x)

            # jittered scatter overlay
            jitter = rng.uniform(-bar_width * 0.3, bar_width * 0.3, size=len(vals))
            sz_norm = (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-9)
            ax.scatter(x + jitter, vals, color=color,
                       s=3 + 15 * sz_norm, alpha=0.35, linewidths=0, zorder=3)

        if xs:
            ax.bar(xs, means, width=bar_width * 0.85, color=color, alpha=0.75,
                   label=name, zorder=2)
            ax.errorbar(xs, means, yerr=errs, fmt='none', color='black',
                        linewidth=1, capsize=3, zorder=4)

    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(bin_labels, rotation=25, ha='right', fontsize=8)
    ax.set_xlabel('Cell Type Purity Bin', fontsize=9)
    ax.set_ylabel('Niche Purity', fontsize=9)
    ax.set_ylim(bottom=0)
    if is_first_row:
        ax.set_title('Niche Purity by Cell Type Purity Bin\n(bar = mean ± std, dots = metacells)')
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=8)


def _draw_niche_binned_row(
    axes, model_data, name_to_color, ds_id,
    purity_bins, size_weighted, is_first_row, swap_positions=False,
):
    """Draw one dataset row: binned bar | niche purity boxplot | CT purity boxplot."""
    names  = list(model_data.keys())
    colors = [name_to_color[n] for n in names]

    _binned_niche_panel(
        axes[0], names, model_data, colors, purity_bins, size_weighted,
        swap_positions=swap_positions, is_first_row=is_first_row,
    )
    axes[0].set_ylabel(f'{ds_id}\nNiche Purity', fontsize=9)

    ct_order = _compute_ct_order(model_data, names)

    has_ct_labels = any(
        df is not None and 'cell_type' in df.columns and not df['cell_type'].isna().all()
        for df in model_data.values()
    )
    if not has_ct_labels:
        for ax in axes[1:]:
            ax.text(0.5, 0.5, 'cell type labels unavailable\n(run save_umap_data() first)',
                    ha='center', va='center', transform=ax.transAxes, fontsize=8, color='gray')
        return

    _grouped_box_panel(
        axes[1], names, model_data, colors, ct_order,
        metric='niche_purity', ylabel='Niche Purity',
        swap_positions=swap_positions, size_weighted=size_weighted,
    )
    if is_first_row:
        axes[1].set_title('Niche Purity per Cell Type\n(sorted by SCProto advantage)')
        handles, labels = axes[1].get_legend_handles_labels()
        if handles:
            axes[1].legend(handles, labels, fontsize=8)

    _grouped_box_panel(
        axes[2], names, model_data, colors, ct_order,
        metric='purity', ylabel='Cell Type Purity',
        swap_positions=swap_positions, size_weighted=size_weighted,
    )
    if is_first_row:
        axes[2].set_title('Cell Type Purity per Cell Type\n(same order as panel 2)')


def fig_niche_purity_binned(
    ds_id,
    model_keywords,
    save_path=None,
    figsize=(18, 4),
    palette=None,
    purity_bins=None,
    size_weighted=False,
    swap_positions=False,
):
    """3-panel figure per dataset: binned bar | niche purity boxplot | CT purity boxplot.

    Panel 1: metacells binned by cell-type purity; bars show mean niche purity per bin
             per model (± std), with jittered metacell dots overlaid.
    Panel 2: grouped box plots of niche purity per cell type, sorted by first-model advantage.
    Panel 3: cell type purity per cell type in the same order.

    Args:
        ds_id:          dataset id string or list of dataset id strings.
        model_keywords: dict {keyword: display_name}. First entry is the reference model.
        save_path:      optional path to save (dpi=300).
        figsize:        (width, height) per row.
        palette:        list of colours; defaults to tab10.
        purity_bins:    bin edges for cell-type purity (default [0, 0.5, 0.7, 0.85, 0.95, 1.01]).
        size_weighted:  if True, weight means and box distributions by metacell size.
        swap_positions: reverse model order within groups.

    Returns:
        matplotlib Figure
    """
    if palette is None:
        palette = list(plt.cm.tab10.colors)
    if purity_bins is None:
        purity_bins = _DEFAULT_PURITY_BINS

    ds_ids = [ds_id] if isinstance(ds_id, str) else list(ds_id)
    all_names     = list(model_keywords.values())
    name_to_color = {n: palette[i % len(palette)] for i, n in enumerate(all_names)}

    panel_w, panel_h = figsize
    fig, axes = plt.subplots(
        len(ds_ids), 3,
        figsize=(panel_w, panel_h * len(ds_ids)),
        squeeze=False,
    )

    for row, did in enumerate(ds_ids):
        model_data = {}
        for keyword, display_name in model_keywords.items():
            df = _load_niche_purity(did, keyword)
            if df is None:
                print(f"Warning: no data for keyword='{keyword}', ds='{did}'")
                continue
            model_data[display_name] = df

        if not model_data:
            print(f"No data loaded for ds='{did}'")
            for ax in axes[row]:
                ax.set_visible(False)
            continue

        _draw_niche_binned_row(
            axes[row], model_data, name_to_color, did,
            purity_bins, size_weighted,
            is_first_row=(row == 0),
            swap_positions=swap_positions,
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


# ---------------------------------------------------------------------------
# Cell-type × niche heatmap figure
# ---------------------------------------------------------------------------

def _pivot_ct_niche(df, metric, size_weighted):
    """Pivot per-metacell df to (cell_type × niche) matrices of mean and std.

    Returns (mean_pivot, std_pivot). Both are None if no valid data.
    """
    valid = df.dropna(subset=['cell_type', 'niche', metric, 'size'])
    if valid.empty:
        return None, None

    if size_weighted:
        def _wmean(g):
            v = g[metric].values
            w = g['size'].values.astype(float)
            m = ~np.isnan(v)
            return np.average(v[m], weights=w[m]) if m.any() else np.nan

        def _wstd(g):
            v = g[metric].values
            w = g['size'].values.astype(float)
            m = ~np.isnan(v)
            if m.sum() < 2:
                return np.nan
            mu = np.average(v[m], weights=w[m])
            return np.sqrt(np.average((v[m] - mu) ** 2, weights=w[m]))

        grp = valid.groupby(['cell_type', 'niche'])
        mean_pivot = grp.apply(_wmean).unstack('niche')
        std_pivot  = grp.apply(_wstd).unstack('niche')
    else:
        grp = valid.groupby(['cell_type', 'niche'])[metric]
        mean_pivot = grp.mean().unstack('niche')
        std_pivot  = grp.std().unstack('niche')

    return mean_pivot, std_pivot


def _draw_heatmap_ax(ax, matrix, title, cmap, vmin, vmax, annot, fmt, fontsize=8,
                     show_xticklabels=True, show_yticklabels=True,
                     show_xlabel=True, show_ylabel=True, std_matrix=None,
                     annot_matrix=None, xlabel='Niche', ylabel_label='Cell Type',
                     norm=None):
    """Draw a single cell_type × niche heatmap on ax. Returns the AxesImage.

    Cells with no ground-truth samples (NaN in matrix) render as plain white,
    distinct from a measured-but-near-zero value.

    norm: optional matplotlib Normalize instance (e.g. PowerNorm to spread out low values
    that would otherwise all look like the same pale shade, or TwoSlopeNorm for a diverging
    diff heatmap centered at 0). Overrides the plain linear vmin/vmax scaling when given;
    vmin/vmax are still used for the white/black annotation-text threshold.
    """
    if matrix is None or matrix.empty:
        ax.text(0.5, 0.5, 'no data\n(need niche labels in umap_protos.csv)',
                ha='center', va='center', transform=ax.transAxes, fontsize=8, color='gray')
        return None

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad('white')

    if norm is not None:
        im = ax.imshow(matrix.values, aspect='auto', cmap=cmap_obj,
                       norm=norm, interpolation='nearest')
    else:
        im = ax.imshow(matrix.values, aspect='auto', cmap=cmap_obj,
                       vmin=vmin, vmax=vmax, interpolation='nearest')

    ax.set_xticks(range(matrix.shape[1]))
    if show_xticklabels:
        ax.set_xticklabels(matrix.columns, rotation=35, ha='right', fontsize=fontsize)
    else:
        ax.set_xticklabels([])

    ax.set_yticks(range(matrix.shape[0]))
    if show_yticklabels:
        ax.set_yticklabels(matrix.index, fontsize=fontsize)
    else:
        ax.set_yticklabels([])

    if show_xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if show_ylabel:
        ax.set_ylabel(ylabel_label, fontsize=fontsize)

    if title:
        ax.set_title(title, fontsize=9)

    if annot:
        thresh = vmin + (vmax - vmin) * 0.6
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                val = matrix.values[r, c]
                if np.isnan(val):
                    continue
                is_high = norm(val) > 0.6 if norm is not None else val > thresh
                txt_color = 'white' if is_high else 'black'
                if annot_matrix is not None:
                    text = annot_matrix.values[r, c]
                    if not text:
                        continue
                elif std_matrix is not None:
                    std_val = std_matrix.values[r, c]
                    text = f'{val:{fmt}}\n±{std_val:{fmt}}' if not np.isnan(std_val) else f'{val:{fmt}}'
                else:
                    text = f'{val:{fmt}}'
                ax.text(c, r, text, ha='center', va='center',
                        fontsize=max(fontsize - 2, 5), color=txt_color,
                        linespacing=1.2)
    return im


def fig_ct_niche_heatmap(
    ds_id,
    model_keywords,
    save_path=None,
    figsize=(5, 4),
    cmap_niche='YlOrRd',
    cmap_ct='YlGnBu',
    size_weighted=False,
    annot=False,
    fmt='.2f',
    cell_types=None,
    transpose=False,
):
    """2-row heatmap: row 1 = niche purity, row 2 = cell type purity.

    One column per model. Each heatmap: rows = niches, columns = cell types,
    color = mean (or size-weighted mean) purity. Color scale shared across
    models within each row for direct comparison.

    Requires save_umap_data() to have been called so that umap_protos.csv
    contains majority_{cell_type_key} and majority_{niche_key} columns.

    Args:
        ds_id:          single dataset id string.
        model_keywords: dict {keyword: display_name}.
        save_path:      optional save path (dpi=300).
        figsize:        (w, h) per heatmap panel; total figure scales accordingly.
        cmap_niche:     colormap for niche purity row (default 'YlOrRd').
        cmap_ct:        colormap for cell type purity row (default 'YlGnBu').
        size_weighted:  if True, weight purity values by metacell size.
        annot:          annotate each cell with its numeric value.
        fmt:            format string for annotations (default '.2f').

    Returns:
        matplotlib Figure
    """
    model_data = {}
    for keyword, display_name in model_keywords.items():
        df = _load_niche_purity(ds_id, keyword)
        if df is None:
            print(f"Warning: no data for keyword='{keyword}', ds='{ds_id}'")
            continue
        model_data[display_name] = df

    if not model_data:
        print(f"No data loaded for ds='{ds_id}'")
        return None

    names = list(model_data.keys())
    n_models = len(names)

    niche_pivots = {n: _pivot_ct_niche(model_data[n], 'niche_purity', size_weighted) for n in names}
    ct_pivots    = {n: _pivot_ct_niche(model_data[n], 'purity',       size_weighted) for n in names}

    # Unpack (mean, std) tuples
    niche_mean  = {n: v[0] for n, v in niche_pivots.items()}
    niche_std   = {n: v[1] for n, v in niche_pivots.items()}
    ct_mean     = {n: v[0] for n, v in ct_pivots.items()}
    ct_std      = {n: v[1] for n, v in ct_pivots.items()}

    # Unified ordering across models (rows=cell_types, cols=niches)
    all_cts = sorted({
        ct for p in [*niche_mean.values(), *ct_mean.values()]
        if p is not None for ct in p.index
    })
    all_niches = sorted({
        ni for p in [*niche_mean.values(), *ct_mean.values()]
        if p is not None for ni in p.columns
    })

    if cell_types is not None:
        all_cts = [ct for ct in all_cts if ct in cell_types]

    def _reindex(p):
        if p is None:
            return None
        p = p.reindex(index=all_cts, columns=all_niches)
        return p.T if transpose else p

    niche_mean = {n: _reindex(p) for n, p in niche_mean.items()}
    niche_std  = {n: _reindex(p) for n, p in niche_std.items()}
    ct_mean    = {n: _reindex(p) for n, p in ct_mean.items()}
    ct_std     = {n: _reindex(p) for n, p in ct_std.items()}

    def _vrange(means):
        arrays = [p.values.ravel() for p in means.values() if p is not None]
        if not arrays:
            return (0.0, 1.0)
        vals = np.concatenate(arrays)
        vals = vals[~np.isnan(vals)]
        return (float(vals.min()), float(vals.max())) if len(vals) else (0.0, 1.0)

    niche_vmin, niche_vmax = _vrange(niche_mean)
    ct_vmin,    ct_vmax    = _vrange(ct_mean)

    pw, ph = figsize
    fig, axes = plt.subplots(2, n_models, figsize=(pw * n_models, ph * 2), squeeze=False)

    row_label  = 'Niche'     if transpose else 'Cell Type'
    col_label  = 'Cell Type' if transpose else 'Niche'

    for col, name in enumerate(names):
        is_left = (col == 0)

        _draw_heatmap_ax(
            axes[0, col], niche_mean[name],
            title=name, cmap=cmap_niche,
            vmin=niche_vmin, vmax=niche_vmax, annot=annot, fmt=fmt,
            show_xticklabels=False, show_yticklabels=is_left,
            show_xlabel=False, show_ylabel=False,
            std_matrix=niche_std[name] if annot else None,
            xlabel=col_label, ylabel_label=row_label,
        )
        _draw_heatmap_ax(
            axes[1, col], ct_mean[name],
            title='', cmap=cmap_ct,
            vmin=ct_vmin, vmax=ct_vmax, annot=annot, fmt=fmt,
            show_xticklabels=True, show_yticklabels=is_left,
            show_xlabel=is_left, show_ylabel=False,
            std_matrix=ct_std[name] if annot else None,
            xlabel=col_label, ylabel_label=row_label,
        )

    # One row label per row, placed to the left of the leftmost heatmap
    for row_ax, label in [(axes[0, 0], 'Niche Purity'), (axes[1, 0], 'Cell Type Purity')]:
        row_ax.set_ylabel(label, fontsize=10, fontweight='bold')

    for row, (cmap, vmin, vmax, label) in enumerate([
        (cmap_niche, niche_vmin, niche_vmax, 'Niche Purity'),
        (cmap_ct,    ct_vmin,    ct_vmax,    'Cell Type Purity'),
    ]):  # noqa
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        fig.colorbar(sm, ax=axes[row, -1], fraction=0.046, pad=0.04, label=label)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


# ---------------------------------------------------------------------------
# Niche purity figure
# ---------------------------------------------------------------------------

def _load_niche_purity(ds_id, keyword):
    """Load per-metacell DataFrame: purity, niche_purity, size, cell_type, niche."""
    run_dir = _resolve_run_dir(ds_id, keyword, prefer_csv='niche_purity_per_mc.csv')
    if run_dir is None:
        return None

    purity_s = _read_series(os.path.join(run_dir, 'purity_per_mc.csv'))
    if purity_s is None:
        return None

    df = pd.DataFrame({'purity': purity_s})

    niche_s = _read_series(os.path.join(run_dir, 'niche_purity_per_mc.csv'))
    df['niche_purity'] = niche_s.reindex(df.index) if niche_s is not None else np.nan

    size_s = _read_series(os.path.join(run_dir, 'size_per_mc.csv'))
    df['size'] = size_s.reindex(df.index).values.astype(float) if size_s is not None else 1.0

    df['cell_type'] = None
    df['niche'] = None
    protos_path = os.path.join(run_dir, 'umap_protos.csv')
    if os.path.exists(protos_path):
        protos_df = pd.read_csv(protos_path)
        protos_df.index = protos_df['proto_id'].astype(str)
        maj_cols = [c for c in protos_df.columns if c.startswith('majority_')]
        if len(maj_cols) >= 1:
            df['cell_type'] = protos_df[maj_cols[0]].reindex(df.index).values
        if len(maj_cols) >= 2:
            df['niche'] = protos_df[maj_cols[1]].reindex(df.index).values

    return df


def _aggregate_by_ct_niche(df):
    """Group by (cell_type, niche) — or just cell_type if niche is absent — and average."""
    has_niche = 'niche' in df.columns and not df['niche'].isna().all()
    group_cols = ['cell_type', 'niche'] if has_niche else ['cell_type']
    valid = df.dropna(subset=group_cols)
    if valid.empty:
        return df
    agg = (
        valid.groupby(group_cols, dropna=True)
        .agg(
            purity=('purity', 'mean'),
            niche_purity=('niche_purity', 'mean'),
            size=('size', 'sum'),
        )
        .reset_index()
    )
    # keep cell_type/niche columns consistent with non-aggregated path
    if 'cell_type' not in agg.columns:
        agg['cell_type'] = None
    if 'niche' not in agg.columns:
        agg['niche'] = None
    return agg


def _compute_ct_order(model_data, names):
    """Sort cell types by first-model niche purity advantage over mean of other models."""
    if not names:
        return []

    ref_df = model_data.get(names[0])
    if ref_df is None or ref_df['niche_purity'].isna().all():
        # fall back to CT purity sort
        for name in names:
            df = model_data.get(name)
            if df is not None and 'cell_type' in df.columns:
                m = df.dropna(subset=['purity', 'cell_type']).groupby('cell_type')['purity'].median()
                return m.sort_values(ascending=False).index.tolist()
        return []

    ref_medians = (
        ref_df.dropna(subset=['niche_purity', 'cell_type'])
        .groupby('cell_type')['niche_purity']
        .median()
    )

    if len(names) == 1:
        return ref_medians.sort_values(ascending=False).index.tolist()

    other_series = []
    for name in names[1:]:
        df = model_data.get(name)
        if df is not None:
            m = df.dropna(subset=['niche_purity', 'cell_type']).groupby('cell_type')['niche_purity'].median()
            other_series.append(m)

    if not other_series:
        return ref_medians.sort_values(ascending=False).index.tolist()

    mean_other = pd.concat(other_series, axis=1).mean(axis=1)
    all_cts = ref_medians.index.union(mean_other.index)
    delta = (ref_medians.reindex(all_cts).fillna(0) - mean_other.reindex(all_cts).fillna(0))
    return delta.sort_values(ascending=False).index.tolist()


def _grouped_box_panel(ax, names, model_data, colors, ct_order, metric, ylabel,
                       swap_positions=False, size_weighted=False):
    """Grouped box plots: x=cell type (ordered), hue=model, y=metric."""
    if not ct_order:
        return

    n_models = len(names)
    group_width = 0.8
    box_width = group_width / n_models
    rng = np.random.default_rng(42)

    for color_i, name in enumerate(names):
        pos_i = (n_models - 1 - color_i) if swap_positions else color_i
        df = model_data.get(name)
        if df is None:
            continue
        color = colors[color_i]
        labeled = False

        for j, ct in enumerate(ct_order):
            sub = df[df['cell_type'] == ct].dropna(subset=[metric, 'size'])
            if len(sub) < 2:
                continue
            vals = sub[metric].values
            sizes = sub['size'].values.astype(float)
            x = j + (pos_i - (n_models - 1) / 2) * box_width

            if size_weighted:
                counts = np.round(sizes / sizes.sum() * 500).astype(int).clip(min=1)
                weighted_vals = np.repeat(vals, counts)
            else:
                weighted_vals = vals

            bp = ax.boxplot(
                weighted_vals, positions=[x], widths=box_width * 0.85,
                patch_artist=True, manage_ticks=False,
                boxprops=dict(facecolor=color, alpha=0.7),
                medianprops=dict(color='black', linewidth=1.5),
                whiskerprops=dict(color=color),
                capprops=dict(color=color),
                flierprops=dict(marker='o', markersize=2,
                               markerfacecolor=color, alpha=0.4, linestyle='none'),
            )
            if not labeled:
                bp['boxes'][0].set_label(name)
                labeled = True

            sz_norm = (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-9)
            jitter = rng.uniform(-box_width * 0.25, box_width * 0.25, size=len(vals))
            ax.scatter(x + jitter, vals, color=color,
                       s=4 + 20 * sz_norm, alpha=0.5, linewidths=0, zorder=3)

    ax.set_xticks(range(len(ct_order)))
    ax.set_xticklabels(ct_order, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.5, len(ct_order) - 0.5)


def _draw_niche_purity_row(
    axes, model_data, name_to_color, ds_id,
    scatter_alpha, scatter_s, size_encoding,
    is_first_row, swap_positions=False, aggregate=False, size_weighted=False,
    scatter_mode='scatter',
):
    """Draw one dataset row: scatter/kde | niche purity boxplot | CT purity boxplot."""
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.stats import gaussian_kde

    names  = list(model_data.keys())
    colors = [name_to_color[n] for n in names]

    # Panel 1: scatter or KDE, x=CT purity, y=niche purity
    ax = axes[0]
    legend_handles = []
    for i, name in enumerate(names):
        df = model_data[name]
        if df is None:
            continue
        plot_df = _aggregate_by_ct_niche(df) if aggregate else df
        valid = plot_df.dropna(subset=['purity', 'niche_purity'])
        if len(valid) == 0:
            continue
        x, y = valid['purity'].values, valid['niche_purity'].values
        color = colors[i]

        if scatter_mode == 'kde' and len(valid) >= 4:
            try:
                kernel = gaussian_kde(np.vstack([x, y]))
                xlo, xhi = x.min() - 0.02, x.max() + 0.02
                ylo, yhi = y.min() - 0.02, y.max() + 0.02
                xx, yy = np.mgrid[xlo:xhi:80j, ylo:yhi:80j]
                z = kernel(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                rgba = to_rgba(color)
                cmap = LinearSegmentedColormap.from_list('', [(*rgba[:3], 0), (*rgba[:3], 0.7)])
                ax.contourf(xx, yy, z, levels=6, cmap=cmap)
                ax.contour(xx, yy, z, levels=3, colors=[color], linewidths=0.8, alpha=0.9)
            except Exception:
                ax.scatter(x, y, color=color, alpha=scatter_alpha, s=scatter_s)
            from matplotlib.patches import Patch
            legend_handles.append(Patch(facecolor=color, label=name))
        else:
            sz = valid['size'].values
            kw = _size_scatter_kwargs(color, sz, size_encoding, scatter_alpha) if size_encoding else {}
            if 's' not in kw:
                kw['s'] = scatter_s
            if 'color' not in kw and 'c' not in kw:
                kw['color'] = color
                kw.setdefault('alpha', scatter_alpha)
            sc = ax.scatter(x, y, **kw, label=name)
            legend_handles.append(sc)

    ax.set_xlabel('Cell Type Purity')
    ax.set_ylabel(f'{ds_id}\nNiche Purity')
    if is_first_row:
        title = 'Niche vs Cell Type Purity (KDE)' if scatter_mode == 'kde' else 'Niche vs Cell Type Purity'
        ax.set_title(title)
        if legend_handles:
            ax.legend(handles=legend_handles, fontsize=9, handletextpad=0.4, labelspacing=0.3)

    # Cell type order: first-model niche purity advantage, descending
    ct_order = _compute_ct_order(model_data, names)

    has_ct_labels = any(
        df is not None and 'cell_type' in df.columns and not df['cell_type'].isna().all()
        for df in model_data.values()
    )
    if not has_ct_labels:
        for ax in axes[1:]:
            ax.text(0.5, 0.5, 'cell type labels unavailable\n(run save_umap_data() first)',
                    ha='center', va='center', transform=ax.transAxes, fontsize=8, color='gray')
        return

    # Panel 2: niche purity per cell type
    _grouped_box_panel(
        axes[1], names, model_data, colors, ct_order,
        metric='niche_purity', ylabel='Niche Purity',
        swap_positions=swap_positions, size_weighted=size_weighted,
    )
    if is_first_row:
        axes[1].set_title('Niche Purity per Cell Type\n(sorted by SCProto advantage)')
        handles, labels = axes[1].get_legend_handles_labels()
        if handles:
            axes[1].legend(handles, labels, fontsize=8)

    # Panel 3: CT purity per cell type — same order to allow direct comparison
    _grouped_box_panel(
        axes[2], names, model_data, colors, ct_order,
        metric='purity', ylabel='Cell Type Purity',
        swap_positions=swap_positions, size_weighted=size_weighted,
    )
    if is_first_row:
        axes[2].set_title('Cell Type Purity per Cell Type\n(same order as panel 2)')


def fig_niche_purity(
    ds_id,
    model_keywords,
    save_path=None,
    figsize=(18, 4),
    palette=None,
    scatter_alpha=0.5,
    scatter_s=12,
    size_encoding='both',
    swap_positions=False,
    aggregate=False,
    size_weighted=False,
    scatter_mode='scatter',
):
    """3-panel figure per dataset: scatter | niche purity boxplot | CT purity boxplot.

    Panel 1: per-metacell scatter, x=cell type purity, y=niche purity, color=model.
             With aggregate=True, one dot per (cell_type, niche) group (mean values).
    Panel 2: grouped box plots of niche purity per cell type (models as hue), cell types
             sorted left-to-right by first-model niche purity advantage over others.
    Panel 3: cell type purity per cell type in same order — shows CT purity is comparable.

    Args:
        ds_id:          dataset id string or list of dataset id strings. One row per dataset.
        model_keywords: dict {keyword: display_name}. First entry is the reference model
                        used for sorting cell types (highest niche advantage on the left).
        save_path:      optional path to save figure (dpi=300).
        figsize:        (width, height) per row; total height scales with n datasets.
        palette:        list of hex/named colours; defaults to tab10.
        scatter_alpha:  fallback alpha when size info unavailable.
        scatter_s:      fallback dot size when size info unavailable.
        size_encoding:  how metacell size is encoded in the scatter.
                        'alpha' | 'size' | 'both' | None (no encoding).
        swap_positions: reverse model order within each cell-type group in box plots.
        aggregate:      if True, panel 1 shows one dot per (cell_type × niche) group
                        instead of one dot per metacell (uses mean purity values).

    Returns:
        matplotlib Figure
    """
    if palette is None:
        palette = list(plt.cm.tab10.colors)

    ds_ids = [ds_id] if isinstance(ds_id, str) else list(ds_id)
    all_names     = list(model_keywords.values())
    name_to_color = {n: palette[i % len(palette)] for i, n in enumerate(all_names)}

    panel_w, panel_h = figsize
    fig, axes = plt.subplots(
        len(ds_ids), 3,
        figsize=(panel_w, panel_h * len(ds_ids)),
        squeeze=False,
    )

    for row, did in enumerate(ds_ids):
        model_data = {}
        for keyword, display_name in model_keywords.items():
            df = _load_niche_purity(did, keyword)
            if df is None:
                print(f"Warning: no data found for keyword='{keyword}', ds='{did}'")
                continue
            model_data[display_name] = df

        if not model_data:
            print(f"No data loaded for ds='{did}' — check keyword spelling and MODEL_DIR.")
            for ax in axes[row]:
                ax.set_visible(False)
            continue

        _draw_niche_purity_row(
            axes[row], model_data, name_to_color, did,
            scatter_alpha, scatter_s, size_encoding,
            is_first_row=(row == 0),
            swap_positions=swap_positions,
            aggregate=aggregate,
            size_weighted=size_weighted,
            scatter_mode=scatter_mode,
        )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


# ---------------------------------------------------------------------------
# Purity vs cell-type frequency figure
# ---------------------------------------------------------------------------

def _load_cell_purity_ctfreq(ds_id, keyword, verbose=True):
    """Per-cell homogeneity loader.

    For each cell returns:
        majority_label : the cell's own label
        ct_freq        : frequency of that label within the cell's own batch
                         (falls back to global frequency if no batch column is detected)
        homogeneity    : fraction of cells in this cell's metacell (all batches) with same label
        n_cells        : metacell size (for reference only, not used for weighting)

    Requires umap_cells.csv (save_umap_data).
    """
    run_dir = _resolve_run_dir(ds_id, keyword, prefer_csv='umap_cells.csv')
    if run_dir is None:
        if verbose:
            print(f"  no run dir: keyword='{keyword}', ds='{ds_id}'")
        return None, None

    cells_path = os.path.join(run_dir, 'umap_cells.csv')
    if not os.path.exists(cells_path):
        if verbose:
            print(f"  umap_cells.csv missing: {run_dir}")
        return None, None
    cells_df = pd.read_csv(cells_path)

    # detect label column
    label_key = None
    protos_path = os.path.join(run_dir, 'umap_protos.csv')
    if os.path.exists(protos_path):
        for c in pd.read_csv(protos_path).columns:
            if c.startswith('majority_'):
                candidate = c[len('majority_'):]
                if candidate in cells_df.columns:
                    label_key = candidate
                    break
    if label_key is None:
        non_umap = [c for c in cells_df.columns
                    if c not in ('umap_1', 'umap_2', 'metacell_id')]
        label_key = non_umap[0] if non_umap else None
    if label_key is None:
        return None, None

    # detect batch column (same logic as _rare_table_one)
    _non_data = {'umap_1', 'umap_2', 'metacell_id', 'cell_id', label_key}
    batch_key = next((c for c in cells_df.columns if c not in _non_data), None)
    if batch_key is not None and cells_df[batch_key].nunique() > len(cells_df) * 0.5:
        batch_key = None

    # per-batch CT frequency for each cell; fall back to global if no batch column
    if batch_key is not None:
        batch_sizes = cells_df.groupby(batch_key).size()
        batch_ct_counts = cells_df.groupby([batch_key, label_key]).size()
        batch_ct_freq = (batch_ct_counts / batch_sizes).rename('ct_freq')
        ct_freq_vals = (
            cells_df.set_index([batch_key, label_key])
            .index.map(batch_ct_freq.to_dict())
        )
        if verbose:
            print(f"  ct_freq: per-batch (batch_key='{batch_key}')")
    else:
        global_freq  = cells_df[label_key].value_counts(normalize=True)
        ct_freq_vals = cells_df[label_key].map(global_freq).values
        if verbose:
            print(f"  ct_freq: global (no batch column detected)")

    mc_sizes      = cells_df.groupby('metacell_id').size()
    mc_label_frac = cells_df.groupby(['metacell_id', label_key]).size() / mc_sizes

    lookup_idx  = pd.MultiIndex.from_arrays(
        [cells_df['metacell_id'], cells_df[label_key]]
    )
    hom_vals    = mc_label_frac.reindex(lookup_idx).fillna(0.0).values
    mc_size_vals = mc_sizes.reindex(cells_df['metacell_id']).values

    result = pd.DataFrame({
        'majority_label': cells_df[label_key].values,
        'ct_freq':        ct_freq_vals,
        'homogeneity':    hom_vals,
        'n_cells':        mc_size_vals,
    })
    return result.dropna(subset=['majority_label', 'ct_freq']), label_key


def _load_purity_ctfreq(ds_id, keyword, verbose=True):
    """Load purity, majority label, and global cell-type frequency for one model.

    Requires both purity_per_mc.csv and umap_protos.csv / umap_cells.csv to
    exist in the same run directory.  Uses the same run-dir resolution as
    _load_umap (prefers directories that contain umap_cells.csv).

    Returns:
        (merged_df, label_key)  where merged_df has columns
            proto_id (int), purity, majority_label, n_cells, ct_freq
        or (None, None) if required files are missing.
    """
    run_dir = _resolve_run_dir(ds_id, keyword, prefer_csv='umap_cells.csv')
    if run_dir is None:
        if verbose:
            print(f"  no run dir found for keyword='{keyword}', ds='{ds_id}'")
        return None, None

    if verbose:
        print(f"\n  keyword='{keyword}'  →  {run_dir}")

    # --- purity_per_mc.csv (index = str proto_id, active metacells only) ---
    purity_s = _read_series(os.path.join(run_dir, 'purity_per_mc.csv'))
    if purity_s is None:
        if verbose:
            print("    purity_per_mc.csv: NOT FOUND — skipping")
        return None, None
    if verbose:
        print(f"    purity_per_mc.csv : {len(purity_s)} active metacells")
        print(f"      index[:5]        = {purity_s.index[:5].tolist()}")

    # --- umap_protos.csv (proto_id int, majority_{label_key}, n_cells) ---
    protos_path = os.path.join(run_dir, 'umap_protos.csv')
    if not os.path.exists(protos_path):
        if verbose:
            print("    umap_protos.csv  : NOT FOUND — run t.save_umap_data() first")
        return None, None
    protos_df = pd.read_csv(protos_path)
    protos_df.index = protos_df['proto_id'].astype(str)

    if verbose:
        n_dead = int((protos_df['n_cells'] == 0).sum())
        if verbose:
            print(f"    umap_protos.csv  : {len(protos_df)} total prototypes, {n_dead} dead (n_cells==0)")

    # --- detect majority label column ---
    maj_cols = [c for c in protos_df.columns if c.startswith('majority_')]
    if not maj_cols:
        if verbose:
            print("    no majority_* column in umap_protos.csv — skipping")
        return None, None
    maj_col  = maj_cols[0]
    label_key = maj_col[len('majority_'):]
    if verbose:
        print(f"    majority col     : '{maj_col}'  →  label_key='{label_key}'")

    # --- index alignment audit ---
    purity_ids = set(purity_s.index)
    protos_ids = set(protos_df.index)
    dead_ids   = protos_ids - purity_ids   # dead protos: exist in protos, absent from purity
    orphan_ids = purity_ids - protos_ids   # should never happen

    if verbose:
        print(f"    index alignment  :")
        print(f"      active (purity)  : {len(purity_ids)}")
        print(f"      dead protos      : {len(dead_ids)}  ← absent from purity_per_mc (expected)")
        if orphan_ids:
            print(f"      WARNING orphan IDs in purity but not protos: {sorted(orphan_ids)[:10]}")
        else:
            print(f"      orphan IDs       : (none) ✓")

    # --- merge: left join from purity so dead protos are automatically excluded ---
    merged = pd.DataFrame({'purity': purity_s})
    merged = merged.join(protos_df[[maj_col, 'n_cells']], how='left')
    merged = merged.rename(columns={maj_col: 'majority_label'})
    merged['proto_id'] = merged.index.astype(int)

    n_missing = int(merged['majority_label'].isna().sum())
    if verbose and n_missing:
        print(f"    WARNING: {n_missing} active metacells have no majority label after join")
    elif verbose:
        print(f"    majority label join: all {len(merged)} active metacells matched ✓")

    # --- umap_cells.csv → global cell-type frequency ---
    cells_path = os.path.join(run_dir, 'umap_cells.csv')
    if not os.path.exists(cells_path):
        if verbose:
            print("    umap_cells.csv   : NOT FOUND — run t.save_umap_data() first")
        return None, None
    cells_df = pd.read_csv(cells_path)
    if label_key not in cells_df.columns:
        if verbose:
            print(f"    WARNING: '{label_key}' not in umap_cells.csv columns: {cells_df.columns.tolist()}")
        return None, None

    ct_freq = cells_df[label_key].value_counts(normalize=True)
    if verbose:
        print(f"    umap_cells.csv   : {len(cells_df)} cells, {len(ct_freq)} cell types")
        print(f"      top 5: { {k: round(v,3) for k,v in ct_freq.head(5).items()} }")

    merged['ct_freq'] = merged['majority_label'].map(ct_freq)

    n_unmatched_freq = int(merged['ct_freq'].isna().sum())
    if verbose and n_unmatched_freq:
        print(f"    WARNING: {n_unmatched_freq} metacells have no ct_freq (label not in cells CSV)")
    elif verbose:
        print(f"    ct_freq mapping  : all {len(merged)} active metacells matched ✓")

    if verbose:
        print(f"\n    Preview (first 8 rows):")
        cols = ['proto_id', 'purity', 'majority_label', 'n_cells', 'ct_freq']
        print(merged[cols].head(8).to_string(index=False))

    return merged.dropna(subset=['majority_label', 'ct_freq']), label_key


def _draw_purity_ctfreq_ax(
    ax, model_data, palette,
    scatter_alpha, scatter_s, show_diagonal,
    n_bins, plot_type, show_mean=False, swap_positions=False, cell_level=False,
    max_scatter_per_bin=None,
):
    """Draw a single purity/homogeneity-vs-ctfreq panel onto *ax*."""
    y_col = 'homogeneity' if cell_level else 'purity'

    if n_bins is None:
        for i, (name, df) in enumerate(model_data.items()):
            ax.scatter(df['ct_freq'], df[y_col],
                       color=palette[i % len(palette)],
                       s=scatter_s, alpha=scatter_alpha,
                       linewidths=0, rasterized=True, label=name)

        if show_diagonal:
            all_dfs = list(model_data.values())
            lim = max(
                max(df['ct_freq'].max() for df in all_dfs),
                max(df[y_col].max()     for df in all_dfs),
            ) * 1.05
            ax.plot([0, lim], [0, lim], color='black', linewidth=0.8,
                    linestyle='--', alpha=0.5, label='y = x (random)')
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)

        ax.set_xlabel('Cell-type frequency (global)', fontsize=9)

    else:
        # one frequency per unique cell type (from single-cell data), not per metacell
        all_freqs = (
            pd.concat([df.drop_duplicates('majority_label')[['majority_label', 'ct_freq']]
                       for df in model_data.values()])
            .drop_duplicates('majority_label')['ct_freq']
        )
        bin_edges = np.quantile(all_freqs.dropna(), np.linspace(0, 1, n_bins + 1))
        bin_edges = np.unique(bin_edges)
        actual_bins = len(bin_edges) - 1

        bin_labels = [
            f'{bin_edges[k]:.2f}–{bin_edges[k+1]:.2f}'
            for k in range(actual_bins)
        ]

        names    = list(model_data.keys())
        n_models = len(names)
        group_width = 0.8
        box_width   = group_width / n_models

        rng = np.random.default_rng(42)
        for color_i, name in enumerate(names):
            pos_i = (n_models - 1 - color_i) if swap_positions else color_i
            df = model_data[name].copy()
            df['bin'] = pd.cut(df['ct_freq'], bins=bin_edges, labels=bin_labels,
                               include_lowest=True)
            color = palette[color_i % len(palette)]
            labeled = False
            for j, bl in enumerate(bin_labels):
                sub = df[df['bin'] == bl].dropna(subset=[y_col, 'n_cells'])
                if len(sub) < 2:
                    continue
                vals  = sub[y_col].values
                sizes = sub['n_cells'].values.astype(float)
                x = j + (pos_i - (n_models - 1) / 2) * box_width

                if cell_level:
                    weighted_vals = vals  # each row is already a cell — no weighting
                else:
                    counts = np.round(sizes / sizes.sum() * 1000).astype(int).clip(min=1)
                    weighted_vals = np.repeat(vals, counts)

                if plot_type == 'box':
                    bp = ax.boxplot(weighted_vals, positions=[x], widths=box_width * 0.85,
                                    patch_artist=True, manage_ticks=False,
                                    boxprops=dict(facecolor=color, alpha=0.7),
                                    medianprops=dict(color='black', linewidth=1.5),
                                    whiskerprops=dict(color=color),
                                    capprops=dict(color=color),
                                    flierprops=dict(marker='o', markersize=2,
                                                    markerfacecolor=color, alpha=0.4,
                                                    linestyle='none'))
                    if not labeled:
                        bp['boxes'][0].set_label(name)
                        labeled = True
                else:
                    parts = ax.violinplot(weighted_vals, positions=[x], widths=box_width * 0.85,
                                          showmedians=True, showextrema=False)
                    for body in parts['bodies']:
                        body.set_facecolor(color)
                        body.set_alpha(0.6)
                        if not labeled:
                            body.set_label(name)
                            labeled = True
                    parts['cmedians'].set_color('black')
                    parts['cmedians'].set_linewidth(1.5)
                    if show_mean:
                        wmean = np.average(vals, weights=sizes)
                        ax.scatter(x, wmean, marker='D', s=18, color='white',
                                   edgecolors='black', linewidths=1, zorder=6)

                if max_scatter_per_bin != 0:
                    scatter_idx = np.arange(len(vals))
                    if max_scatter_per_bin is not None and len(vals) > max_scatter_per_bin:
                        scatter_idx = rng.choice(len(vals), max_scatter_per_bin, replace=False)
                    sv = vals[scatter_idx]
                    sz = sizes[scatter_idx]
                    sz_norm = (sz - sz.min()) / (sz.max() - sz.min() + 1e-9)
                    pt_sizes = 4 + 30 * sz_norm
                    jitter = rng.uniform(-box_width * 0.25, box_width * 0.25, size=len(sv))
                    ax.scatter(x + jitter, sv, color=color, s=pt_sizes,
                               alpha=0.5, linewidths=0, zorder=3)

        ax.set_xticks(range(actual_bins))
        ax.set_xticklabels(bin_labels, rotation=25, ha='right', fontsize=8)
        ax.set_xlabel('Cell-type frequency bin (quantile)', fontsize=9)
        ax.set_xlim(-0.5, actual_bins - 0.5)

    ax.set_ylabel('Homogeneity' if cell_level else 'Metacell purity', fontsize=9)
    ax.tick_params(labelsize=8)


def fig_purity_vs_ctfreq(
    ds_id,
    model_keywords,
    save_path=None,
    figsize=None,
    palette=None,
    scatter_alpha=0.7,
    scatter_s=30,
    show_diagonal=True,
    n_bins=None,
    plot_type='violin',
    show_mean=False,
    swap_positions=False,
    unit='metacell',
    max_scatter_per_bin=None,
    verbose=True,
):
    """Purity or homogeneity (y) vs global cell-type frequency (x).

    unit='metacell': y = metacell purity (fraction of cells matching majority label).
    unit='cell':     y = homogeneity (fraction of same-CT cells in each cell's metacell).
                     Use this to reproduce the paper's rare-cell homogeneity figure.

    ds_id can be a single dataset id string or a list of dataset ids.
    When a list is provided, one panel per dataset is drawn side by side with
    a shared legend.

    n_bins=None  — continuous scatter, diagonal y=x reference line available.
    n_bins=int   — quantile-bin ct_freq into n_bins groups; grouped violin or
                   box plot per bin (controlled by plot_type).

    Requires save_umap_data() (produces umap_cells.csv).
    unit='metacell' additionally requires eval_metacell_quality() (purity_per_mc.csv).

    Args:
        ds_id:          dataset id string or list of dataset id strings.
        model_keywords: dict {keyword: display_name}, same as fig_purity_entropy.
        save_path:      optional file path to save (dpi=300).
        figsize:        figure size tuple; defaults to (6, 5) per panel.
        palette:        list of colours (one per model); defaults to tab10.
        scatter_alpha:  dot transparency (scatter mode only).
        scatter_s:      dot size (scatter mode only).
        show_diagonal:  draw the y=x reference line (scatter mode only).
        n_bins:         if set, bin ct_freq into this many quantile bins.
        plot_type:      'violin' or 'box' — used when n_bins is set.
        unit:           'cell' (homogeneity, default for paper) or 'metacell' (purity).
        verbose:        print alignment log while loading.

    Returns:
        matplotlib Figure
    """
    if palette is None:
        palette = list(plt.cm.tab10.colors)

    ds_ids = [ds_id] if isinstance(ds_id, str) else list(ds_id)
    panel_w, panel_h = (figsize if figsize is not None else (6, 5))
    fig, axes = plt.subplots(
        1, len(ds_ids),
        figsize=(panel_w * len(ds_ids), panel_h),
        squeeze=False,
    )
    axes = axes[0]  # shape: (n_datasets,)

    for ax, did in zip(axes, ds_ids):
        model_data = {}
        _loader = _load_cell_purity_ctfreq if unit == 'cell' else _load_purity_ctfreq
        for keyword, display_name in model_keywords.items():
            df, lk = _loader(did, keyword, verbose=verbose)
            if df is None:
                if verbose:
                    print(f"  skipping '{display_name}' for ds='{did}' — missing files")
                continue
            model_data[display_name] = df

        if not model_data:
            if verbose:
                print(f"No data loaded for ds='{did}' — check keywords and ensure save_umap_data() was called.")
            ax.set_visible(False)
            continue

        _draw_purity_ctfreq_ax(
            ax, model_data, palette,
            scatter_alpha, scatter_s, show_diagonal,
            n_bins, plot_type, show_mean=show_mean,
            swap_positions=swap_positions,
            cell_level=(unit == 'cell'),
            max_scatter_per_bin=max_scatter_per_bin,
        )
        ax.set_title(did, fontweight='bold', fontsize=11)
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(ylo, yhi + (yhi - ylo) * 0.08)

        # only keep y-label on leftmost panel
        if ax is not axes[0]:
            ax.set_ylabel('')

    # shared legend from the last visible axes that has handles
    legend_ax = next(
        (ax for ax in reversed(axes) if ax.get_visible() and ax.get_legend_handles_labels()[0]),
        None,
    )
    if legend_ax is not None:
        handles, labels = legend_ax.get_legend_handles_labels()
        if swap_positions:
            handles, labels = handles[::-1], labels[::-1]
        fig.legend(handles, labels, loc='lower center',
                   ncol=len(labels), fontsize=9,
                   handletextpad=0.4, labelspacing=0.3,
                   bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


# ---------------------------------------------------------------------------
# Rare cell-type purity table
# ---------------------------------------------------------------------------

def _weighted_quantile(values, weights, q):
    """Weighted quantile: q in [0, 1]."""
    values  = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order   = np.argsort(values)
    values, weights = values[order], weights[order]
    cum_w   = np.cumsum(weights) - weights / 2  # midpoint convention
    cum_w  /= cum_w[-1]
    return float(np.interp(q, cum_w, values))


def _get_rare_cts(cells_df, label_key, batch_key, rare_mode, rare_quantile, rare_thresh):
    """Return the set of rare cell-type labels given a rarity mode.

    rare_mode='overall'     : CT global freq < thresh (Q25 of global freqs).
    rare_mode='all_batches' : CT per-batch freq < thresh in every batch.
                              thresh = Q25 of per-batch freqs (pooled across batches).
                              Requires batch_key in cells_df.
    rare_mode='median_batch': median per-batch CT freq < thresh.
                              thresh = Q25 of those median freqs.
                              Requires batch_key in cells_df.
    """
    if rare_mode == 'overall' or batch_key is None or batch_key not in cells_df.columns:
        agg_freq = cells_df[label_key].value_counts(normalize=True)
        thresh   = rare_thresh if rare_thresh is not None else agg_freq.quantile(rare_quantile)
        return set(agg_freq[agg_freq < thresh].index), thresh, agg_freq

    # per-batch CT frequencies: index=CT, columns=batch
    batch_freq = (
        cells_df.groupby([batch_key, label_key])
        .size()
        .groupby(level=batch_key, group_keys=False)
        .apply(lambda s: s / s.sum())
        .unstack(level=batch_key, fill_value=0.0)
    )

    if rare_mode == 'all_batches':
        all_vals = batch_freq.values.flatten()
        thresh   = rare_thresh if rare_thresh is not None else pd.Series(all_vals).quantile(rare_quantile)
        agg_freq = batch_freq.min(axis=1)
        rare_cts = set(batch_freq.index[(batch_freq < thresh).all(axis=1)])
    elif rare_mode in ('mean_batch', 'median_batch'):
        agg_freq = batch_freq.mean(axis=1) if rare_mode == 'mean_batch' else batch_freq.median(axis=1)
        thresh   = rare_thresh if rare_thresh is not None else agg_freq.quantile(rare_quantile)
        rare_cts = set(agg_freq[agg_freq < thresh].index)
    elif rare_mode == 'outlier':
        # per-batch mean freq, then flag CTs more than 1 std below the mean
        # falls back to Q25 if no CTs qualify (uniform distribution)
        agg_freq = (
            cells_df.groupby([batch_key, label_key]).size()
            .groupby(level=batch_key, group_keys=False).apply(lambda s: s / s.sum())
            .unstack(level=batch_key, fill_value=0.0).mean(axis=1)
        ) if batch_key and batch_key in cells_df.columns else cells_df[label_key].value_counts(normalize=True)
        if rare_thresh is not None:
            thresh   = rare_thresh
            rare_cts = set(agg_freq[agg_freq < thresh].index)
        else:
            thresh   = float(agg_freq.mean() - agg_freq.std())
            rare_cts = set(agg_freq[agg_freq < thresh].index)
            if not rare_cts:
                thresh   = float(agg_freq.quantile(rare_quantile))
                rare_cts = set(agg_freq[agg_freq < thresh].index)
                print(f"  [outlier] no outlier CTs found — falling back to Q{int(rare_quantile*100)} (thresh={thresh:.4f})")
    else:
        raise ValueError(f"rare_mode must be 'overall', 'all_batches', 'mean_batch', 'median_batch', or 'outlier', got '{rare_mode}'")

    return rare_cts, thresh, agg_freq


def _rare_table_one(did, keyword, display_name, rare_quantile, verbose, quiet=False):
    """Compute batch-level rare-CT metrics for a single (dataset, model) pair.

    Returns ((did, display_name), metrics_dict, verbose_lines) where
    verbose_lines is a list of strings to print after all workers finish.
    """
    import time
    tag = f"[{display_name}|{did}]"
    _log = (lambda *a, **kw: None) if quiet else print

    vlines = []

    _log(f"  {tag} resolving run dir ...", flush=True)
    t0 = time.time()
    run_dir = _resolve_run_dir(did, keyword, prefer_csv='umap_cells.csv')
    _log(f"  {tag} run dir resolved ({time.time()-t0:.1f}s)", flush=True)
    if run_dir is None:
        vlines.append(f"  [rare_table] skipping '{display_name}' ds='{did}': no matching run dir")
        return (did, display_name), None, vlines

    cells_path = os.path.join(run_dir, 'umap_cells.csv')
    if not os.path.exists(cells_path):
        vlines.append(f"  [rare_table] skipping '{display_name}' ds='{did}': umap_cells.csv missing — run save_umap_data() first")
        return (did, display_name), None, vlines

    _log(f"  {tag} reading umap_cells.csv ...", flush=True)
    t0 = time.time()
    cells_df = pd.read_csv(cells_path, usecols=lambda c: c not in ('umap_1', 'umap_2'))
    _log(f"  {tag} umap_cells.csv loaded ({len(cells_df)} rows, {time.time()-t0:.1f}s)", flush=True)

    # detect label column via majority_* in umap_protos.csv
    label_key = None
    protos_path = os.path.join(run_dir, 'umap_protos.csv')
    if os.path.exists(protos_path):
        _log(f"  {tag} reading umap_protos.csv ...", flush=True)
        t0 = time.time()
        protos_df = pd.read_csv(protos_path)
        _log(f"  {tag} umap_protos.csv loaded ({time.time()-t0:.1f}s)", flush=True)
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
        vlines.append(f"  [rare_table] cannot detect label column: {run_dir}")
        return (did, display_name), None, vlines

    _non_data_cols = {'umap_1', 'umap_2', 'metacell_id', 'cell_id', label_key}
    batch_key = next(
        (c for c in cells_df.columns if c not in _non_data_cols),
        None,
    )
    # safety net: reject if still too many unique values to be a real batch
    if batch_key is not None:
        n_uniq = cells_df[batch_key].nunique()
        sample = sorted(cells_df[batch_key].dropna().unique())[:5]
        _log(f"  {tag} batch='{batch_key}' | {n_uniq} unique values, e.g. {sample}", flush=True)
        if n_uniq > len(cells_df) * 0.5:
            _log(f"  {tag} ignoring '{batch_key}' as batch (too many unique values)", flush=True)
            batch_key = None
    else:
        _log(f"  {tag} no batch column found", flush=True)
    _log(f"  {tag} label='{label_key}', batch='{batch_key}' | "
         f"computing metacell label fractions ...", flush=True)
    t0 = time.time()
    mc_sizes        = cells_df.groupby('metacell_id').size()
    mc_label_counts = cells_df.groupby(['metacell_id', label_key]).size()
    mc_label_frac   = mc_label_counts / mc_sizes
    _log(f"  {tag} label fractions done ({time.time()-t0:.1f}s)", flush=True)

    mc_majority = mc_label_counts.groupby(level='metacell_id').idxmax().map(lambda x: x[1])

    # purity of each metacell at its majority label (used for F1 precision and batch purity)
    mc_purity_series = mc_label_frac.reindex(
        pd.MultiIndex.from_arrays([mc_majority.index, mc_majority.values])
    ).fillna(0.0)
    mc_purity_series.index = mc_majority.index
    # avg purity of metacells dedicated to each CT (F1 precision term)
    mc_purity_by_ct = mc_purity_series.groupby(mc_majority).mean().to_dict()

    # --- batch-local rare CT metrics ---
    # For each batch: find locally-rare CTs (outlier rule on within-batch freq, Q25 fallback),
    # then compute coverage, repr (macro/micro), purity, and macro F1.
    # All scores are aggregated as mean ± std across batches.
    bl_coverage_per_batch    = []
    bl_macro_per_batch       = []
    bl_micro_per_batch       = []
    bl_homogeneity_per_batch = []
    bl_purity_per_batch      = []
    bl_f1_per_batch          = []
    if batch_key is not None and batch_key in cells_df.columns:
        for _, batch_df in cells_df.groupby(batch_key):
            ct_freq = batch_df[label_key].value_counts(normalize=True)
            bl_thresh = float(ct_freq.mean() - ct_freq.std())
            bl_rare   = set(ct_freq[ct_freq < bl_thresh].index)
            if not bl_rare:
                bl_rare = set(ct_freq[ct_freq < ct_freq.quantile(rare_quantile)].index)
            if not bl_rare:
                continue
            bl_cells  = batch_df[batch_df[label_key].isin(bl_rare)]
            if bl_cells.empty:
                continue

            # coverage: fraction of locally-rare CTs where at least one cell of that type
            # in THIS batch is assigned to a metacell dedicated to it
            n_covered = sum(
                (batch_df.loc[batch_df[label_key] == ct, 'metacell_id']
                 .map(mc_majority) == ct).any()
                for ct in bl_rare
            )
            bl_coverage_per_batch.append(n_covered / len(bl_rare))

            # repr scores / homogeneity
            # For each rare-CT cell: fraction of same-CT cells in its metacell (all batches).
            lookup = pd.MultiIndex.from_arrays([bl_cells['metacell_id'], bl_cells[label_key]])
            scores = mc_label_frac.reindex(lookup).fillna(0.0)
            scores.index = bl_cells.index
            bl_macro_per_batch.append(scores.groupby(bl_cells[label_key]).mean().mean())
            per_cell_score = float(scores.mean())
            bl_micro_per_batch.append(per_cell_score)
            bl_homogeneity_per_batch.append(per_cell_score)

            # purity: size-weighted purity of metacells whose majority is a locally-rare CT
            bl_rare_mc_mask  = mc_majority.isin(bl_rare)
            bl_rare_mc_pur   = mc_purity_series[bl_rare_mc_mask].values
            bl_rare_mc_sizes = mc_sizes[bl_rare_mc_mask].values.astype(float)
            if len(bl_rare_mc_pur) > 0:
                w_bl = bl_rare_mc_sizes / bl_rare_mc_sizes.sum()
                bl_purity_per_batch.append(float((bl_rare_mc_pur * w_bl).sum()))

            # F1: precision = global avg purity of ct's dedicated metacells
            #     recall    = fraction of ct-cells in THIS batch in a dedicated metacell
            f1_vals = []
            for ct in bl_rare:
                ct_cells_b = batch_df[batch_df[label_key] == ct]
                if ct_cells_b.empty:
                    continue
                recall_c    = float((ct_cells_b['metacell_id'].map(mc_majority) == ct).mean())
                precision_c = mc_purity_by_ct.get(ct, 0.0)
                denom = precision_c + recall_c
                f1_vals.append(2 * precision_c * recall_c / denom if denom > 0 else 0.0)
            if f1_vals:
                bl_f1_per_batch.append(float(np.mean(f1_vals)))

    def _mean_std(vals):
        if not vals:
            return float('nan'), float('nan')
        a = np.array(vals)
        return round(float(a.mean()), 2), round(float(a.std()), 2)

    bl_coverage_mean,     bl_coverage_std     = _mean_std(bl_coverage_per_batch)
    bl_macro_mean,        bl_macro_std        = _mean_std(bl_macro_per_batch)
    bl_micro_mean,        bl_micro_std        = _mean_std(bl_micro_per_batch)
    bl_homogeneity_mean,  bl_homogeneity_std  = _mean_std(bl_homogeneity_per_batch)
    bl_purity_mean,       bl_purity_std       = _mean_std(bl_purity_per_batch)
    bl_f1_mean,           bl_f1_std           = _mean_std(bl_f1_per_batch)

    if verbose:
        vlines.append(
            f"  [{display_name}] ds={did}: "
            f"coverage={bl_coverage_mean:.2f}±{bl_coverage_std:.2f}  "
            f"repr macro={bl_macro_mean:.2f}±{bl_macro_std:.2f}  "
            f"micro={bl_micro_mean:.2f}±{bl_micro_std:.2f}  "
            f"homogeneity={bl_homogeneity_mean:.2f}±{bl_homogeneity_std:.2f}  "
            f"purity={bl_purity_mean:.2f}±{bl_purity_std:.2f}  "
            f"F1={bl_f1_mean:.2f}±{bl_f1_std:.2f}"
        )

    _log(f"  {tag} done", flush=True)
    return (did, display_name), dict(
        batch_rare_coverage_mean=bl_coverage_mean,
        batch_rare_coverage_std=bl_coverage_std,
        batch_rare_repr_macro_mean=bl_macro_mean,
        batch_rare_repr_macro_std=bl_macro_std,
        batch_rare_repr_micro_mean=bl_micro_mean,
        batch_rare_repr_micro_std=bl_micro_std,
        batch_rare_homogeneity_mean=bl_homogeneity_mean,
        batch_rare_homogeneity_std=bl_homogeneity_std,
        batch_rare_purity_mean=bl_purity_mean,
        batch_rare_purity_std=bl_purity_std,
        batch_rare_f1_macro_mean=bl_f1_mean,
        batch_rare_f1_macro_std=bl_f1_std,
    ), vlines


def rare_celltype_purity_table(
    ds_id,
    model_keywords,
    rare_quantile=0.25,
    verbose=False,
    quiet=False,
    n_workers=8,
):
    """Batch-level rare cell type metrics per model.

    For each batch, locally-rare CTs are identified via an outlier rule on within-batch
    frequencies (fallback: Q<rare_quantile>). Metrics are aggregated as mean ± std across batches.

    n_workers: number of parallel threads for Drive I/O (default 8).

    Metrics
    -------
    rare_ct_coverage:
        Fraction of rare CTs that are the majority label of >=1 metacell.

    rare_ct_repr_mean / rare_ct_repr_std:
        For each rare CT, mean fraction of cells in a rare-CT cell's metacell
        sharing its label — then macro-averaged across rare CTs.

    rare_mc_purity_mean / rare_mc_purity_std:
        Among metacells whose majority label is a rare CT, size-weighted mean
        purity (fraction of cells matching the majority label).

    Requires save_umap_data() (produces umap_cells.csv).
    All pairs are processed in parallel (n_workers threads) to minimise Drive I/O latency.
    Returns a (dataset, run) MultiIndex DataFrame compatible with show_table.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ds_ids = [ds_id] if isinstance(ds_id, str) else list(ds_id)
    tasks  = [
        (did, kw, name)
        for did in ds_ids
        for kw, name in model_keywords.items()
    ]

    results = {}  # (did, display_name) -> metrics_dict

    def _run(args):
        did, kw, name = args
        return _rare_table_one(did, kw, name, rare_quantile, verbose, quiet)

    ex = ThreadPoolExecutor(max_workers=min(n_workers, len(tasks)))
    futures = {ex.submit(_run, t): t for t in tasks}
    try:
        for fut in as_completed(futures):
            key, metrics, vlines = fut.result()
            if not quiet:
                for line in vlines:
                    print(line)
            if metrics is not None:
                results[key] = metrics
    except KeyboardInterrupt:
        if not quiet:
            print("  [rare_table] interrupted — cancelling remaining tasks", flush=True)
        for f in futures:
            f.cancel()
        ex.shutdown(wait=False)
        raise
    else:
        ex.shutdown(wait=False)

    if not results:
        return pd.DataFrame()

    # restore insertion order: ds_ids × model_keywords
    ordered_keys = [
        (did, name)
        for did in ds_ids
        for name in model_keywords.values()
        if (did, name) in results
    ]
    idx = pd.MultiIndex.from_tuples(ordered_keys, names=['dataset', 'run'])
    return pd.DataFrame([results[k] for k in ordered_keys], index=idx)


# ---------------------------------------------------------------------------
# Spatial map — single-cell coloured by cell type and niche
# ---------------------------------------------------------------------------

def _load_cell_assignments(ds_id, keyword, obs_names):
    """Return a Series (index=obs_names) with metacell_id for every cell.

    Tries cell_assignments.csv first (all cells), falls back to umap_cells.csv.
    Returns None if neither file exists or neither has a cell_id column.
    """
    for prefer in ('cell_assignments.csv', 'umap_cells.csv'):
        run_dir = _resolve_run_dir(ds_id, keyword, prefer_csv=prefer)
        if run_dir is None:
            continue
        path = os.path.join(run_dir, prefer)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if 'cell_id' in df.columns and 'metacell_id' in df.columns:
            return df.set_index('cell_id')['metacell_id'].reindex(obs_names)
    return None


def _cluster_prototypes(ds_id, keyword, k):
    """K-means on prototype UMAP embeddings from umap_protos.csv.

    Returns dict {proto_id (int): cluster_id (int)}, or None on failure.
    """
    from sklearn.cluster import KMeans

    run_dir = _resolve_run_dir(ds_id, keyword, prefer_csv='umap_protos.csv')
    if run_dir is None:
        return None
    protos_path = os.path.join(run_dir, 'umap_protos.csv')
    if not os.path.exists(protos_path):
        return None

    protos_df = pd.read_csv(protos_path)
    coords = protos_df[['umap_1', 'umap_2']].values
    k = min(k, len(protos_df))
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(coords)
    return dict(zip(protos_df['proto_id'].astype(int), labels.astype(int)))


def _load_adata(ds_id):
    """Load raw AnnData for ds_id using the same path as the trainer."""
    import scanpy as sc
    from interpretable_ssl.datasets.dataset_configs import DATASETS
    cfg = DATASETS[ds_id]
    return sc.read_h5ad(str(cfg['path'])), cfg


def _categorical_colors(labels, palette=None):
    """Return (colors_per_cell, label_to_color, ordered_categories)."""
    cats = pd.Categorical(labels).categories.tolist()
    base = palette if palette is not None else list(plt.cm.tab20.colors)
    cmap = {c: base[i % len(base)] for i, c in enumerate(cats)}
    return np.array([cmap[l] for l in labels]), cmap, cats


def _scatter_panel(ax, x, y, labels, s, alpha, gray_mask=None,
                   gray_color='#cccccc', palette=None, color_map=None):
    """Scatter one spatial panel.

    color_map: prebuilt {label: color} dict — bypasses palette-based coloring.
    gray_mask: boolean array; True cells are drawn in gray behind selected cells.
    Returns (full_cmap, visible_cats).
    """
    if color_map is not None:
        cats = [c for c in sorted(set(labels)) if c != 'unassigned']
        if 'unassigned' in set(labels):
            cats.append('unassigned')
        full_cmap = {**color_map, 'unassigned': '#bbbbbb'}
    else:
        _, full_cmap, cats = _categorical_colors(labels, palette)

    cell_colors = np.array([full_cmap.get(l, '#888888') for l in labels])

    if gray_mask is not None:
        if gray_mask.any():
            ax.scatter(x[gray_mask], y[gray_mask], c=gray_color, s=s,
                       alpha=alpha * 0.4, linewidths=0, rasterized=True)
        sel = ~gray_mask
        if sel.any():
            ax.scatter(x[sel], y[sel], c=cell_colors[sel], s=s, alpha=alpha,
                       linewidths=0, rasterized=True)
        cats = [c for c in cats if c in set(labels[sel])]
    else:
        ax.scatter(x, y, c=cell_colors, s=s, alpha=alpha, linewidths=0, rasterized=True)

    return full_cmap, cats


def _majority_label_clusters(cluster_ids, niche_labels, niche_cmap):
    """Assign each cluster its majority niche label, reusing the niche color map.

    Returns:
        cell_labels  — array of niche label strings, one per cell.
        cluster_cmap — {niche_label: color} taken from niche_cmap.
    """
    df = pd.DataFrame({'cluster': cluster_ids, 'niche': niche_labels})
    valid = df[df['cluster'] != 'unassigned']

    cluster_to_label = {}
    for cid, grp in valid.groupby('cluster'):
        cluster_to_label[cid] = grp['niche'].value_counts().idxmax()

    cell_labels = np.array([cluster_to_label.get(c, 'unassigned') for c in cluster_ids])
    cluster_cmap = {label: niche_cmap.get(label, '#888888')
                    for label in set(cell_labels) if label != 'unassigned'}
    cluster_cmap['unassigned'] = '#bbbbbb'
    return cell_labels, cluster_cmap


def fig_spatial(
    ds_id,
    cell_types=None,
    model_keywords=None,
    cluster_metacells=True,
    spatial_key='spatial',
    label_key=None,
    niche_key=None,
    section=None,
    section_key='section',
    save_path=None,
    figsize=(6, 5),
    s=4,
    alpha=0.7,
    ct_palette=None,
    niche_palette=None,
):
    """Spatial scatter: cell-type/niche rows + optional per-model metacell-cluster rows.

    Grid layout — columns: all cells | one column per entry in cell_types.
    Rows:
      • Row 0: coloured by cell type
      • Row 1: coloured by niche
      • Row 2+: one row per model in model_keywords, cells coloured by
                metacell cluster ID (k-means on prototype UMAP embeddings,
                k = number of distinct (cell_type, niche) combinations).

    Args:
        ds_id:             Dataset id (key in DATASETS).
        cell_types:        String or list — one highlighted column per entry.
        model_keywords:    Dict {keyword: display_name} as used by other fig functions.
                           Each model adds one row coloured by metacell cluster ID.
        cluster_metacells: If True (default), k-means clusters prototypes and labels
                           each cluster by majority niche. If False, colours cells
                           directly by their raw metacell_id — no clustering.
        spatial_key:       Key in adata.obsm with (x, y) coordinates.
        label_key:         adata.obs column for cell type; auto-read from config if None.
        niche_key:         adata.obs column for niche; auto-read from config if None.
        section:           Subset to cells where adata.obs[section_key] == section.
        section_key:       obs column for section filter (default 'section').
        save_path:         Optional path to save the figure (dpi=300).
        figsize:           (width, height) per panel.
        s:                 Dot size.
        alpha:             Dot alpha.
        ct_palette:        Optional colour list for cell types.
        niche_palette:     Optional colour list for niches.

    Returns:
        matplotlib Figure
    """
    adata, cfg = _load_adata(ds_id)

    if label_key is None:
        label_key = cfg.get('label_key')
    if niche_key is None:
        niche_key = cfg.get('niche_key')

    if section is not None and section_key in adata.obs.columns:
        adata = adata[adata.obs[section_key] == section].copy()

    if spatial_key not in adata.obsm:
        raise KeyError(f"'{spatial_key}' not in adata.obsm. Available: {list(adata.obsm.keys())}")

    coords = np.array(adata.obsm[spatial_key])
    x, y = coords[:, 0], coords[:, 1]

    from matplotlib.gridspec import GridSpec

    ct_labels    = adata.obs[label_key].astype(str).values if (label_key and label_key in adata.obs.columns) else None
    niche_labels = adata.obs[niche_key].astype(str).values if (niche_key and niche_key in adata.obs.columns) else None

    if ct_labels is None and niche_labels is None:
        raise ValueError(f"Neither '{label_key}' nor '{niche_key}' found in adata.obs.")

    # Build niche color map once — reused by model rows so cluster colors match niche row
    _, niche_cmap, _ = _categorical_colors(niche_labels, niche_palette) if niche_labels is not None else (None, {}, [])

    # columns: (col_title, gray_mask_or_None)
    selected_list = [] if cell_types is None else (
        [cell_types] if isinstance(cell_types, str) else list(cell_types)
    )
    col_specs = [('All cells', None)]
    if ct_labels is not None:
        for ct in selected_list:
            mask = ~np.isin(ct_labels, [ct] if isinstance(ct, str) else ct)
            col_specs.append((ct if isinstance(ct, str) else ', '.join(ct), mask))

    # fixed rows: (title, labels_array, palette, prebuilt_cmap_or_None)
    fixed_rows = []
    if ct_labels    is not None: fixed_rows.append(('Cell type', ct_labels,    ct_palette,    None))
    if niche_labels is not None: fixed_rows.append(('Niche',     niche_labels, niche_palette, None))

    # model rows: cluster labels coloured to match CT colors
    model_rows = []
    if model_keywords:
        if ct_labels is not None and niche_labels is not None:
            k = len(set(zip(ct_labels, niche_labels)))
        elif ct_labels is not None:
            k = len(set(ct_labels))
        else:
            k = len(set(niche_labels))

        for keyword, display_name in model_keywords.items():
            mc_ids = _load_cell_assignments(ds_id, keyword, adata.obs_names)
            if mc_ids is None:
                print(f"  [fig_spatial] no cell assignments for '{display_name}' — skipping")
                continue

            if cluster_metacells:
                mc_to_cluster = _cluster_prototypes(ds_id, keyword, k)
                if mc_to_cluster is None:
                    print(f"  [fig_spatial] no prototype UMAP for '{display_name}' — skipping")
                    continue
                raw_ids = mc_ids.map(mc_to_cluster)
                cell_ids = raw_ids.fillna(-1).astype(int).astype(str).values
                cell_ids[raw_ids.isna().values] = 'unassigned'
                if niche_labels is not None:
                    cell_labels, cluster_cmap = _majority_label_clusters(
                        cell_ids, niche_labels, niche_cmap)
                else:
                    cell_labels, cluster_cmap = cell_ids, None
            else:
                # majority niche label per metacell from umap_protos.csv
                run_dir = _resolve_run_dir(ds_id, keyword, prefer_csv='umap_protos.csv')
                protos_path = os.path.join(run_dir, 'umap_protos.csv') if run_dir else None
                if protos_path and os.path.exists(protos_path) and niche_key:
                    protos_df = pd.read_csv(protos_path)
                    maj_col = f'majority_{niche_key}'
                    if maj_col in protos_df.columns:
                        mc_to_niche = dict(zip(protos_df['proto_id'].astype(int),
                                               protos_df[maj_col].astype(str)))
                        cell_labels = mc_ids.map(mc_to_niche).fillna('unassigned').values
                        cluster_cmap = {**niche_cmap, 'unassigned': '#bbbbbb'}
                    else:
                        cell_labels = mc_ids.fillna(-1).astype(int).astype(str).values
                        cell_labels[mc_ids.isna().values] = 'unassigned'
                        cluster_cmap = None
                else:
                    cell_labels = mc_ids.fillna(-1).astype(int).astype(str).values
                    cell_labels[mc_ids.isna().values] = 'unassigned'
                    cluster_cmap = None

            model_rows.append((display_name, cell_labels, cluster_cmap))

    # ── layout: map cols + one legend col per row ───────────────────────────
    nrows  = len(fixed_rows) + len(model_rows)
    n_map  = len(col_specs)
    pw, ph = figsize
    legend_w = 1.8          # inches for the legend column

    # Compute data aspect ratio so panel height = panel_width × (y_range / x_range).
    # This lets each panel fill its GridSpec cell exactly with no set_aspect padding.
    xpad = (x.max() - x.min()) * 0.02
    ypad = (y.max() - y.min()) * 0.02
    xlim = (x.min() - xpad, x.max() + xpad)
    ylim = (y.min() - ypad, y.max() + ypad)
    data_aspect = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    ph_data = pw * data_aspect          # panel height matched to data shape

    # Add a small fixed title strip (0.35 in) on top of the map area
    title_h = 0.35
    fig = plt.figure(figsize=(pw * n_map + legend_w, ph_data * nrows + title_h))
    map_top = 1.0 - title_h / (ph_data * nrows + title_h)
    gs  = GridSpec(nrows, n_map + 1,
                   width_ratios=[pw] * n_map + [legend_w / pw],
                   figure=fig, hspace=0.06, wspace=0.0,
                   top=map_top, bottom=0.0, left=0.04, right=1.0)

    def _draw_row(r, row_title, labels, palette, color_map):
        all_cmap, all_cats = None, None
        for c, (col_title, gmask) in enumerate(col_specs):
            ax = fig.add_subplot(gs[r, c])
            cmap, cats = _scatter_panel(ax, x, y, labels, s, alpha,
                                        gray_mask=gmask, palette=palette,
                                        color_map=color_map)
            if c == 0:
                all_cmap, all_cats = cmap, cats
                ax.text(-0.06, 0.5, row_title, transform=ax.transAxes,
                        rotation=90, va='center', ha='center',
                        fontsize=9, fontweight='bold')
            if r == 0:
                ax.set_title(col_title, fontsize=9, pad=3)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.axis('off')

        # shared legend for this row in the dedicated legend column
        leg_ax = fig.add_subplot(gs[r, n_map])
        leg_ax.axis('off')
        if all_cmap and all_cats:
            handles = [plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=all_cmap[ct], markersize=6, label=ct)
                       for ct in all_cats]
            leg_ax.legend(handles=handles, fontsize=6, frameon=False,
                          loc='upper left', borderaxespad=0)

    for r, (row_title, labels, palette, color_map) in enumerate(fixed_rows):
        _draw_row(r, row_title, labels, palette, color_map)

    for mi, (display_name, cell_labels, cluster_cmap) in enumerate(model_rows):
        _draw_row(len(fixed_rows) + mi,
                  f'{display_name} metacell label', cell_labels, None, cluster_cmap)

    suptitle = ds_id if section is None else f'{ds_id} — {section}'
    fig.suptitle(suptitle, fontsize=12, fontweight='bold')

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


# ---------------------------------------------------------------------------
# Tumor niche purity figure (core vs surface, cell type purity + niche purity)
# ---------------------------------------------------------------------------

def _load_tumor_niche_per_cell(ds_id, keyword):
    """Load tumor_niche_per_cell.csv for a model keyword.

    Returns pd.DataFrame with columns: niche, celltype_purity, niche_purity.
    Returns None if file is not found.
    """
    run_dir = _resolve_run_dir(ds_id, keyword, prefer_csv='tumor_niche_per_cell.csv')
    if run_dir is None:
        return None
    path = os.path.join(run_dir, 'tumor_niche_per_cell.csv')
    if not os.path.exists(path):
        print(f"  [fig_tumor_niche] tumor_niche_per_cell.csv not found in {run_dir}")
        print(f"  Re-run the pipeline with load_umap=True to generate it.")
        return None
    return pd.read_csv(path)


def fig_tumor_niche_purity(
    ds_id,
    model_keywords,
    target_niches=('Tumor core', 'Tumor surface'),
    save_path=None,
    figsize=(4, 4),
    palette=None,
    swap_positions=False,
    plot_type='violin',
    stat_ref=None,
):
    """2-row figure: cell type purity | niche purity for tumor cells, split by niche.

    Mirrors fig_purity_by_celltype but for the tumor core/surface comparison task.
    Loads tumor_niche_per_cell.csv (saved by the pipeline) from each model directory.

    Layout:
      Row 0: Cell type purity — one panel per niche (core, surface), one violin per model.
      Row 1: Within-niche purity — same layout.

    Cell type purity drops for surface tumor cells in SEACell because COVET affinity links
    them to adjacent stromal cells. scProto's expression signal keeps them in tumor metacells.

    Args:
        ds_id:          dataset id string (e.g. 's28nsc').
        model_keywords: dict {keyword: display_name}. Same convention as fig_purity_by_celltype.
                        Use 'seacell_X_covet' for SEACells trained on COVET.
        target_niches:  which niche labels to show (one column per niche).
        save_path:      optional save path (dpi=300).
        figsize:        (width, height) per subplot panel.
        palette:        list of colours; defaults to tab10.
        swap_positions: reverse model order within each panel.
        plot_type:      'violin' or 'box'.
        stat_ref:       display name of the reference model for significance brackets.

    Returns:
        matplotlib Figure
    """
    if palette is None:
        palette = list(plt.cm.tab10.colors)

    all_names     = list(model_keywords.values())
    name_to_color = {n: palette[i % len(palette)] for i, n in enumerate(all_names)}

    model_data = {}
    for keyword, display_name in model_keywords.items():
        df = _load_tumor_niche_per_cell(ds_id, keyword)
        if df is None:
            print(f"Warning: no tumor_niche_per_cell.csv for keyword='{keyword}', ds='{ds_id}'")
            continue
        model_data[display_name] = df

    if not model_data:
        print(f"No tumor niche data found for ds='{ds_id}'. "
              f"Run pipeline first with niche_key set (and 'tumor_niche' not in skip_metrics).")
        return None

    names  = list(model_data.keys())
    colors = [name_to_color[n] for n in names]
    target_niches_str = [str(n) for n in target_niches]
    n_niches = len(target_niches_str)

    pw, ph = figsize
    fig, axes = plt.subplots(2, n_niches, figsize=(pw * n_niches, ph * 2), squeeze=False)

    metrics_config = [
        ('celltype_purity', 'Cell Type Purity\n(fraction of metacell = Tumor)'),
        ('niche_purity',    'Within-niche Purity\n(fraction of tumor members = same niche)'),
    ]

    for col, niche in enumerate(target_niches_str):
        for row, (metric, ylabel) in enumerate(metrics_config):
            arrays, sizes = [], []
            for name in names:
                df = model_data[name]
                sub = df[df['niche'] == niche].dropna(subset=[metric])
                arrays.append(sub[metric].values if len(sub) else None)
                sizes.append(None)  # per-cell data, no size weighting needed

            ax = axes[row, col]
            _panel = _violin_panel if plot_type == 'violin' else _box_panel
            _panel(ax, names, arrays, sizes, colors,
                   ylabel=ylabel if col == 0 else '',
                   swap_positions=swap_positions,
                   point_alpha=0)

            if stat_ref is not None:
                _draw_significance_brackets(ax, names, arrays, stat_ref, swap_positions)

            ax.set_ylim(-0.02, 1.08)
            ax.axhline(1.0, color='grey', linewidth=0.6, linestyle='--', alpha=0.5)

            if row == 0:
                ax.set_title(niche, fontsize=10, fontweight='bold')

    fig.suptitle(f'Tumor niche metacell quality  [{ds_id}]', fontsize=11, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


# ---------------------------------------------------------------------------
# Affinity niche purity: within-niche weight fraction per affinity type
# ---------------------------------------------------------------------------

def _compute_affinity_niche_scores(ds_id, niche_key, affinity_types,
                                   n_components, k_neighbors, graphs_dir):
    """Load affinities and compute per-cell within-niche weight fraction.

    Returns:
        niche_labels: np.ndarray of niche label per cell
        niches:       sorted list of unique niche labels
        scores:       dict {display_name: np.ndarray of shape (n_cells,), NaN for skipped cells}
    """
    import pickle
    import scanpy as sc
    from interpretable_ssl.datasets.dataset_configs import DATASETS

    ds_cfg = DATASETS.get(ds_id)
    if ds_cfg is None:
        raise ValueError(f"Dataset '{ds_id}' not found in DATASETS.")
    print(f"Loading adata from {ds_cfg['path']} ...")
    ad = sc.read_h5ad(str(ds_cfg['path']))

    if niche_key not in ad.obs.columns:
        raise ValueError(f"niche_key '{niche_key}' not in adata.obs. "
                         f"Available: {list(ad.obs.columns)}")

    niche_labels = ad.obs[niche_key].astype(str).values
    n_cells = len(ad)
    niches = sorted(set(niche_labels))

    scores = {}
    for aff_tag, display_name in affinity_types.items():
        fname = (f"affinity_{ds_id}{n_cells}"
                 f"_ncomp{n_components}_kneighbors{k_neighbors}_{aff_tag}.pkl")
        fpath = os.path.join(graphs_dir, fname)
        if not os.path.exists(fpath):
            print(f"  [skip] not found: {fpath}")
            continue
        print(f"  Loading {display_name} ...")
        with open(fpath, 'rb') as f:
            aff = pickle.load(f)

        aff_csr = aff.tocsr()
        cell_scores = np.full(n_cells, np.nan)
        for i in range(n_cells):
            row = aff_csr.getrow(i)
            total = row.data.sum()
            if total == 0:
                continue
            same = row.data[niche_labels[row.indices] == niche_labels[i]].sum()
            cell_scores[i] = same / total
        scores[display_name] = cell_scores

    return niche_labels, niches, scores


def fig_affinity_niche_purity(
    ds_id,
    niche_key,
    affinity_types,
    n_components=50,
    k_neighbors=50,
    graphs_dir='./graphs',
    n_cols=3,
    figsize_per_panel=(3.5, 3.0),
    palette=None,
    stat_ref=None,
    save_path=None,
):
    """One subplot per niche, violin per affinity type; also returns a summary table.

    For each cell i:
        score_i = sum(w_j for same-niche j) / sum(w_j)

    Statistical test: Mann-Whitney U (one-sided, ref > other) with Bonferroni
    correction across comparisons within each niche. Stars drawn on figure;
    p-values added to summary table.

    Args:
        ds_id:             Dataset id string (must be in DATASETS).
        niche_key:         adata.obs column with niche labels.
        affinity_types:    dict {affinity_type_tag: display_name}.
        n_components:      Must match the saved affinity (default 50).
        k_neighbors:       Must match the saved affinity (default 50).
        graphs_dir:        Directory where affinity .pkl files are stored.
        n_cols:            Number of subplot columns (default 3).
        figsize_per_panel: (width, height) per subplot panel.
        palette:           List of colors; defaults to tab10.
        stat_ref:          display_name to use as reference for significance
                           brackets (e.g. 'PCA'). None = no brackets.
        save_path:         Optional path to save the figure (dpi=300).

    Returns:
        fig:        matplotlib Figure
        summary_df: DataFrame — columns: niche, affinity, median, IQR_low,
                    IQR_high, mean, std, n, p_vs_ref, p_adj, sig
                    (p columns only present when stat_ref is given)
    """
    from matplotlib.patches import Patch
    from scipy.stats import mannwhitneyu

    if palette is None:
        palette = list(plt.cm.tab10.colors)

    niche_labels, niches, scores = _compute_affinity_niche_scores(
        ds_id, niche_key, affinity_types, n_components, k_neighbors, graphs_dir,
    )
    display_names = list(scores.keys())
    name_to_color = {name: palette[i % len(palette)] for i, name in enumerate(display_names)}
    n_comparisons = max(len(display_names) - 1, 1)  # Bonferroni denominator per niche

    def _stars(p):
        if p < 0.001: return '***'
        if p < 0.01:  return '**'
        if p < 0.05:  return '*'
        return 'ns'

    # --- summary table ---
    rows = []
    for niche in niches:
        mask = niche_labels == niche
        ref_vals = scores[stat_ref][mask][~np.isnan(scores[stat_ref][mask])] \
            if stat_ref and stat_ref in scores else None
        for name in display_names:
            vals = scores[name][mask]
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            q25, q75 = np.percentile(vals, [25, 75])
            row = {
                'niche':    niche,
                'affinity': name,
                'median':   np.median(vals),
                'IQR_low':  q25,
                'IQR_high': q75,
                'mean':     vals.mean(),
                'std':      vals.std(),
                'n':        len(vals),
            }
            if stat_ref and name != stat_ref and ref_vals is not None and len(ref_vals) > 0:
                try:
                    _, p_raw = mannwhitneyu(ref_vals, vals, alternative='greater')
                    p_adj = min(p_raw * n_comparisons, 1.0)  # Bonferroni
                    row['p_vs_ref'] = p_raw
                    row['p_adj']    = p_adj
                    row['sig']      = _stars(p_adj)
                except Exception:
                    pass
            rows.append(row)
    summary_df = pd.DataFrame(rows)

    # --- figure: one subplot per niche ---
    n_niches = len(niches)
    n_cols = min(n_cols, n_niches)
    n_rows = int(np.ceil(n_niches / n_cols))
    pw, ph = figsize_per_panel
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(pw * n_cols, ph * n_rows),
                             squeeze=False)

    violin_width = 0.35
    positions = list(range(len(display_names)))

    for idx, niche in enumerate(niches):
        ax = axes[idx // n_cols][idx % n_cols]
        mask = niche_labels == niche

        niche_vals = {}
        for ai, name in enumerate(display_names):
            vals = scores[name][mask]
            vals = vals[~np.isnan(vals)]
            niche_vals[name] = vals
            if len(vals) == 0:
                continue
            color = name_to_color[name]
            parts = ax.violinplot(vals, positions=[ai], widths=violin_width,
                                  showmedians=True, showextrema=False)
            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_alpha(0.75)
            parts['cmedians'].set_color('black')
            parts['cmedians'].set_linewidth(1.5)

            q25, med, q75 = np.percentile(vals, [25, 50, 75])
            ax.text(ai, -0.06, f'{med:.2f}\n[{q25:.2f},{q75:.2f}]',
                    ha='center', va='top', fontsize=6.5, color='#333333')

        # significance brackets vs stat_ref
        if stat_ref and stat_ref in niche_vals and len(niche_vals[stat_ref]) > 0:
            ref_pos = display_names.index(stat_ref)
            ref_arr = niche_vals[stat_ref]
            y_top = 1.08
            step = 0.10
            level = 0
            for ai, name in enumerate(display_names):
                if name == stat_ref or len(niche_vals[name]) == 0:
                    continue
                try:
                    _, p_raw = mannwhitneyu(ref_arr, niche_vals[name], alternative='greater')
                    p_adj = min(p_raw * n_comparisons, 1.0)
                except Exception:
                    continue
                stars = _stars(p_adj)
                if stars == 'ns':
                    continue
                y = y_top + step * level
                x1, x2 = min(ref_pos, ai), max(ref_pos, ai)
                ax.plot([x1, x1, x2, x2], [y, y + 0.02, y + 0.02, y],
                        color='black', linewidth=0.8)
                ax.text((x1 + x2) / 2, y + 0.02, stars,
                        ha='center', va='bottom', fontsize=8)
                level += 1

        ax.set_title(niche, fontsize=9, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels(display_names, fontsize=8, rotation=20, ha='right')
        ax.set_ylim(-0.18, 1.22)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.axhline(1.0, color='grey', linewidth=0.6, linestyle='--', alpha=0.5)
        if idx % n_cols == 0:
            ax.set_ylabel('Within-niche weight\nfraction', fontsize=8)

    for idx in range(n_niches, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    handles = [Patch(facecolor=name_to_color[n], label=n) for n in display_names]
    fig.legend(handles=handles, fontsize=8, loc='lower right',
               bbox_to_anchor=(1.0, 0.0), frameon=True)

    fig.suptitle(f'Affinity niche purity  [{ds_id}]', fontsize=11, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, summary_df


# ---------------------------------------------------------------------------
# Affinity cell-state purity: within-(celltype, niche) edge weight fraction
# ---------------------------------------------------------------------------

def fig_affinity_cellstate_purity(
    ds_id,
    niche_key,
    affinity_types,
    celltype_key=None,
    n_components=50,
    k_neighbors=50,
    graphs_dir='./graphs',
    n_cols=4,
    figsize_per_panel=(3.5, 3.0),
    min_cells=20,
    palette=None,
    stat_ref=None,
    save_path=None,
):
    """For each (celltype, niche) pair: how well does the affinity connect same-state cells?

    Score per cell i of celltype CT in niche N:

        score_i = sum(w_ij  where ct[j]==CT  AND  niche[j]==N)
                  ─────────────────────────────────────────────
                  sum(w_ij)   ← total weight, all neighbors

    "What fraction of a cell's total affinity goes to its own cell state?"
    Denominator is total edge weight, mirroring how scProto samples all edges.

    Layout: one subplot per cell type, bar per niche (x-axis), one bar color
    per affinity type. Bar height = mean score; error bar = std.

    Args:
        ds_id:             Dataset id string.
        niche_key:         adata.obs column with niche labels.
        affinity_types:    dict {affinity_type_tag: display_name}.
        celltype_key:      adata.obs column with cell-type labels.
                           If None, uses DATASETS[ds_id]['label_key'].
        n_components:      Must match the saved affinity (default 50).
        k_neighbors:       Must match the saved affinity (default 50).
        graphs_dir:        Directory where affinity .pkl files are stored.
        n_cols:            Number of subplot columns (default 4).
        figsize_per_panel: (width, height) per subplot.
        min_cells:         Skip (CT, niche) pairs with fewer cells (default 20).
        palette:           List of colors; defaults to tab10.
        stat_ref:          display_name to use as reference for significance
                           markers. None = no markers.
        save_path:         Optional save path (dpi=300).

    Returns:
        fig:        matplotlib Figure
        summary_df: DataFrame — niche, celltype, affinity, mean, std, median, n
    """
    import pickle
    import scanpy as sc
    from matplotlib.patches import Patch
    from scipy.stats import mannwhitneyu
    from interpretable_ssl.datasets.dataset_configs import DATASETS

    if palette is None:
        palette = list(plt.cm.tab10.colors)

    ds_cfg = DATASETS.get(ds_id)
    if ds_cfg is None:
        raise ValueError(f"Dataset '{ds_id}' not found in DATASETS.")
    print(f"Loading adata from {ds_cfg['path']} ...")
    ad = sc.read_h5ad(str(ds_cfg['path']))

    if celltype_key is None:
        celltype_key = ds_cfg.get('label_key', 'celltype')
    if niche_key not in ad.obs.columns:
        raise ValueError(f"niche_key '{niche_key}' not in adata.obs.")
    if celltype_key not in ad.obs.columns:
        raise ValueError(f"celltype_key '{celltype_key}' not in adata.obs.")

    ct_labels    = ad.obs[celltype_key].astype(str).values
    niche_labels = ad.obs[niche_key].astype(str).values
    n_cells      = len(ad)

    # unique sorted cell types and niches
    cell_types = sorted(set(ct_labels))
    niches     = sorted(set(niche_labels))

    # existing (CT, niche) pairs with enough cells
    from collections import defaultdict
    pair_cells = defaultdict(list)
    for i, (ct, ni) in enumerate(zip(ct_labels, niche_labels)):
        pair_cells[(ct, ni)].append(i)
    valid_pairs = {p: idxs for p, idxs in pair_cells.items() if len(idxs) >= min_cells}

    display_names  = list(affinity_types.values())
    name_to_color  = {n: palette[i % len(palette)] for i, n in enumerate(display_names)}
    n_comparisons  = max(len(display_names) - 1, 1)

    def _stars(p):
        if p < 0.001: return '***'
        if p < 0.01:  return '**'
        if p < 0.05:  return '*'
        return 'ns'

    # --- compute per-cell scores for each affinity ---
    # cell_scores[display_name] = np.ndarray(n_cells,) — NaN if no same-CT neighbors
    cell_scores = {}
    for aff_tag, display_name in affinity_types.items():
        fname = (f"affinity_{ds_id}{n_cells}"
                 f"_ncomp{n_components}_kneighbors{k_neighbors}_{aff_tag}.pkl")
        fpath = os.path.join(graphs_dir, fname)
        if not os.path.exists(fpath):
            print(f"  [skip] not found: {fpath}")
            continue
        print(f"  Loading {display_name} ...")
        with open(fpath, 'rb') as f:
            aff = pickle.load(f)
        aff_csr = aff.tocsr()

        s = np.full(n_cells, np.nan)
        for i in range(n_cells):
            row   = aff_csr.getrow(i)
            js    = row.indices
            ws    = row.data
            denom = ws.sum()
            if denom == 0:
                continue
            same_state = (ct_labels[js] == ct_labels[i]) & (niche_labels[js] == niche_labels[i])
            s[i] = ws[same_state].sum() / denom
        cell_scores[display_name] = s

    # --- summary table ---
    rows = []
    for (ct, ni), idxs in valid_pairs.items():
        idxs = np.array(idxs)
        ref_vals = cell_scores[stat_ref][idxs] if stat_ref and stat_ref in cell_scores else None
        if ref_vals is not None:
            ref_vals = ref_vals[~np.isnan(ref_vals)]
        for name in display_names:
            if name not in cell_scores:
                continue
            vals = cell_scores[name][idxs]
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                continue
            row = dict(celltype=ct, niche=ni, affinity=name,
                       mean=vals.mean(), std=vals.std(),
                       median=np.median(vals), n=len(vals))
            if stat_ref and name != stat_ref and ref_vals is not None and len(ref_vals) > 0:
                try:
                    _, p_raw = mannwhitneyu(ref_vals, vals, alternative='greater')
                    p_adj = min(p_raw * n_comparisons, 1.0)
                    row.update(p_vs_ref=p_raw, p_adj=p_adj, sig=_stars(p_adj))
                except Exception:
                    pass
            rows.append(row)
    summary_df = pd.DataFrame(rows)

    # --- figure: one subplot per cell type, boxplot per affinity across niches ---
    celltypes_present = sorted({ct for ct, _ in valid_pairs})
    n_ct   = len(celltypes_present)
    n_cols = min(n_cols, n_ct)
    n_rows = int(np.ceil(n_ct / n_cols))
    pw, ph = figsize_per_panel
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(pw * n_cols, ph * n_rows),
                             squeeze=False)

    box_width = 0.6 / max(len(display_names), 1)
    offsets   = np.linspace(-(0.6 - box_width) / 2,
                             (0.6 - box_width) / 2,
                             len(display_names))

    for idx, ct in enumerate(celltypes_present):
        ax = axes[idx // n_cols][idx % n_cols]
        ct_niches = sorted({ni for (c, ni) in valid_pairs if c == ct})
        x_pos     = np.arange(len(ct_niches))

        for ai, name in enumerate(display_names):
            if name not in cell_scores:
                continue
            color = name_to_color[name]
            for xi, ni in enumerate(ct_niches):
                idxs = np.array(valid_pairs.get((ct, ni), []))
                vals = cell_scores[name][idxs]
                vals = vals[~np.isnan(vals)]
                if len(vals) == 0:
                    continue
                bp = ax.boxplot(vals, positions=[x_pos[xi] + offsets[ai]],
                                widths=box_width, patch_artist=True,
                                showfliers=False, manage_ticks=False,
                                medianprops=dict(color='black', linewidth=1.5),
                                boxprops=dict(facecolor=color, alpha=0.75),
                                whiskerprops=dict(linewidth=0.8),
                                capprops=dict(linewidth=0.8))

        ax.set_title(ct, fontsize=9, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(ct_niches, fontsize=7, rotation=30, ha='right')
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, color='grey', linewidth=0.6, linestyle='--', alpha=0.5)
        if idx % n_cols == 0:
            ax.set_ylabel('Same-state edge\nweight fraction', fontsize=8)

    for idx in range(n_ct, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    handles = [Patch(facecolor=name_to_color[n], label=n) for n in display_names
               if n in cell_scores]
    fig.legend(handles=handles, fontsize=8, loc='lower right',
               bbox_to_anchor=(1.0, 0.0), frameon=True)
    fig.suptitle(f'Affinity cell-state purity  [{ds_id}]', fontsize=11, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, summary_df


# ---------------------------------------------------------------------------
# Affinity UMAP: one column per method, row 0 = cell type, row 1 = niche
# ---------------------------------------------------------------------------

def fig_affinity_umap(
    ds_id,
    niche_key,
    affinity_types,
    celltype_key=None,
    n_components=50,
    k_neighbors=50,
    graphs_dir='./graphs',
    celltype_palette=None,
    niche_palette=None,
    figsize_per_panel=(3.2, 3.0),
    point_size=1,
    point_alpha=0.35,
    umap_kwargs=None,
    save_path=None,
):
    """UMAP from raw affinity matrices — one column per method, two rows.

    Row 0: UMAP coloured by cell type.
    Row 1: UMAP coloured by niche / spatial domain.
    Bottom: shared colour legends (cell types left, niches right).

    The affinity pkl is used directly as UMAP connectivities (no kNN
    recomputation). Diagonal is set to 0 before passing to UMAP.

    Args:
        ds_id:             Dataset id string (key in DATASETS).
        niche_key:         adata.obs column with niche labels.
        affinity_types:    dict {affinity_tag: display_name}.
        celltype_key:      adata.obs column with cell-type labels.
                           If None, uses DATASETS[ds_id]['label_key'].
        n_components:      Must match the saved affinity file name (default 50).
        k_neighbors:       Must match the saved affinity file name (default 50).
        graphs_dir:        Directory where affinity .pkl files live.
        celltype_palette:  List of colors for cell types; defaults to tab20.
        niche_palette:     List of colors for niches; defaults to tab10.
        figsize_per_panel: (width, height) of each subplot.
        point_size:        Scatter dot size (default 1).
        point_alpha:       Scatter dot alpha (default 0.35).
        umap_kwargs:       Extra kwargs forwarded to sc.tl.umap
                           (e.g. min_dist=0.3, random_state=42).
        save_path:         Optional file path; saved at dpi=300.

    Returns:
        fig:        matplotlib Figure
        umap_coords: dict {display_name: np.ndarray (n_cells, 2)}
    """
    import pickle
    import anndata
    import scanpy as sc
    from matplotlib.patches import Patch as MPatch
    from interpretable_ssl.datasets.dataset_configs import DATASETS

    if celltype_palette is None:
        celltype_palette = list(plt.cm.tab20.colors)
    if niche_palette is None:
        niche_palette = list(plt.cm.tab10.colors)

    ds_cfg = DATASETS.get(ds_id)
    if ds_cfg is None:
        raise ValueError(f"Dataset '{ds_id}' not found in DATASETS.")
    print(f"Loading adata from {ds_cfg['path']} ...")
    ad = sc.read_h5ad(str(ds_cfg['path']))

    if celltype_key is None:
        celltype_key = ds_cfg.get('label_key', 'celltype')
    if niche_key not in ad.obs.columns:
        raise ValueError(f"niche_key '{niche_key}' not in adata.obs.")
    if celltype_key not in ad.obs.columns:
        raise ValueError(f"celltype_key '{celltype_key}' not in adata.obs.")

    ct_labels    = ad.obs[celltype_key].astype(str).values
    niche_labels = ad.obs[niche_key].astype(str).values
    n_cells      = len(ad)

    cell_types = sorted(set(ct_labels))
    niches     = sorted(set(niche_labels))

    ct_cmap    = {ct: celltype_palette[i % len(celltype_palette)]
                  for i, ct in enumerate(cell_types)}
    niche_cmap = {ni: niche_palette[i % len(niche_palette)]
                  for i, ni in enumerate(niches)}

    ct_colors    = np.array([ct_cmap[c] for c in ct_labels])    # (n, 3)
    niche_colors = np.array([niche_cmap[n] for n in niche_labels])

    _umap_kw = {'min_dist': 0.3, 'random_state': 42}
    _umap_kw.update(umap_kwargs or {})

    # --- compute UMAP for each affinity ---
    umap_coords = {}
    for aff_tag, display_name in affinity_types.items():
        fname = (f"affinity_{ds_id}{n_cells}"
                 f"_ncomp{n_components}_kneighbors{k_neighbors}_{aff_tag}.pkl")
        fpath = os.path.join(graphs_dir, fname)
        if not os.path.exists(fpath):
            print(f"  [skip] not found: {fpath}")
            continue
        print(f"  UMAP for {display_name} ...")
        with open(fpath, 'rb') as f:
            aff = pickle.load(f)
        aff_csr = aff.tocsr()
        aff_csr.setdiag(0)         # diagonal = 0 (UMAP connectivities convention)
        aff_csr.eliminate_zeros()

        ad_tmp = anndata.AnnData(np.zeros((n_cells, 1), dtype=np.float32))
        ad_tmp.obsp['connectivities'] = aff_csr
        ad_tmp.uns['neighbors'] = {
            'connectivities_key': 'connectivities',
            'distances_key': None,
            'params': {'method': 'precomputed'},
        }
        sc.tl.umap(ad_tmp, **_umap_kw)
        umap_coords[display_name] = ad_tmp.obsm['X_umap']
        del ad_tmp

    # --- layout ---
    present = [dn for dn in affinity_types.values() if dn in umap_coords]
    n_cols  = len(present)
    pw, ph  = figsize_per_panel

    # estimate legend height needed (each legend row ≈ 0.22 inches)
    ct_ncol     = min(len(cell_types), 6)
    ni_ncol     = min(len(niches), 6)
    ct_leg_rows = int(np.ceil(len(cell_types) / ct_ncol))
    ni_leg_rows = int(np.ceil(len(niches) / ni_ncol))
    legend_h    = max(ct_leg_rows, ni_leg_rows) * 0.22 + 0.5   # inches

    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(pw * n_cols, ph * 2 + legend_h),
        squeeze=False,
    )

    row_titles = ['Cell type', 'Niche']
    all_colors = [ct_colors, niche_colors]

    for ci, display_name in enumerate(present):
        xy = umap_coords[display_name]
        for ri, colors in enumerate(all_colors):
            ax = axes[ri][ci]
            ax.scatter(xy[:, 0], xy[:, 1],
                       c=colors, s=point_size, alpha=point_alpha,
                       linewidths=0, rasterized=True)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if ri == 0:
                ax.set_title(display_name, fontsize=10, fontweight='bold', pad=4)
            if ci == 0:
                ax.set_ylabel(row_titles[ri], fontsize=9, labelpad=4)

    # leave room at bottom for the two legends
    legend_frac = legend_h / (ph * 2 + legend_h)
    plt.tight_layout(rect=[0, legend_frac, 1, 1])

    # --- shared legends ---
    ct_handles = [MPatch(facecolor=ct_cmap[ct], label=ct) for ct in cell_types]
    ni_handles = [MPatch(facecolor=niche_cmap[ni], label=ni) for ni in niches]

    leg_ct = fig.legend(
        handles=ct_handles, title=celltype_key, title_fontsize=8,
        fontsize=7, frameon=True, ncol=ct_ncol,
        loc='lower left', bbox_to_anchor=(0.01, 0.0),
    )
    fig.add_artist(leg_ct)   # keep after second fig.legend call
    fig.legend(
        handles=ni_handles, title=niche_key, title_fontsize=8,
        fontsize=7, frameon=True, ncol=ni_ncol,
        loc='lower right', bbox_to_anchor=(0.99, 0.0),
    )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, umap_coords


# ---------------------------------------------------------------------------
# Affinity purity scatter: cell-type purity (x) vs cell-state purity (y)
# ---------------------------------------------------------------------------

def fig_affinity_purity_scatter(
    ds_id,
    niche_key,
    affinity_types,
    celltype_key=None,
    n_components=50,
    k_neighbors=50,
    graphs_dir='./graphs',
    palette=None,
    markers=None,
    figsize=(5, 4.5),
    show_error_bars=True,
    save_path=None,
):
    """Scatter: mean cell-type purity (x) vs mean cell-state purity (y), one point per method.

    Metric definitions (per cell i):

        cell_type_purity_i  = sum(w_ij  for j: celltype[j] == celltype[i])
                              ─────────────────────────────────────────────
                              sum(w_ij)   ← all neighbors j

        cell_state_purity_i = sum(w_ij  for j: celltype[j] == celltype[i]
                                                AND niche[j] == niche[i])
                              ─────────────────────────────────────────────
                              sum(w_ij)

    Both are in [0, 1]. Cells with zero total weight are excluded (NaN).
    The mean over all valid cells is plotted as the point coordinate.
    Error bars (optional) show ± std across cells.

    Args:
        ds_id:          Dataset id string (key in DATASETS).
        niche_key:      adata.obs column with niche / spatial-domain labels.
        affinity_types: dict {affinity_tag: display_name}.
        celltype_key:   adata.obs column with cell-type labels.
                        If None, uses DATASETS[ds_id]['label_key'].
        n_components:   Must match the saved affinity (default 50).
        k_neighbors:    Must match the saved affinity (default 50).
        graphs_dir:     Directory where affinity .pkl files live.
        palette:        List of colors; defaults to tab10.
        markers:        List of marker styles; defaults to 'o' for all.
        figsize:        Figure size (width, height).
        show_error_bars: Whether to draw ± std bars (default True).
        save_path:      Optional save path (dpi=300).

    Returns:
        fig:        matplotlib Figure
        summary_df: DataFrame — affinity, display_name,
                    mean_ct_purity, std_ct_purity,
                    mean_cs_purity, std_cs_purity, n_cells
    """
    import pickle
    import scanpy as sc
    from interpretable_ssl.datasets.dataset_configs import DATASETS

    if palette is None:
        palette = list(plt.cm.tab10.colors)

    ds_cfg = DATASETS.get(ds_id)
    if ds_cfg is None:
        raise ValueError(f"Dataset '{ds_id}' not found in DATASETS.")
    print(f"Loading adata from {ds_cfg['path']} ...")
    ad = sc.read_h5ad(str(ds_cfg['path']))

    if celltype_key is None:
        celltype_key = ds_cfg.get('label_key', 'celltype')
    if niche_key not in ad.obs.columns:
        raise ValueError(f"niche_key '{niche_key}' not in adata.obs.")
    if celltype_key not in ad.obs.columns:
        raise ValueError(f"celltype_key '{celltype_key}' not in adata.obs.")

    ct_labels    = ad.obs[celltype_key].astype(str).values
    niche_labels = ad.obs[niche_key].astype(str).values
    n_cells      = len(ad)

    display_names = list(affinity_types.values())
    if markers is None:
        markers = ['o'] * len(display_names)

    # --- compute per-cell purities for each affinity ---
    results = {}  # display_name -> {'ct': array, 'cs': array}
    for aff_tag, display_name in affinity_types.items():
        fname = (f"affinity_{ds_id}{n_cells}"
                 f"_ncomp{n_components}_kneighbors{k_neighbors}_{aff_tag}.pkl")
        fpath = os.path.join(graphs_dir, fname)
        if not os.path.exists(fpath):
            print(f"  [skip] not found: {fpath}")
            continue
        print(f"  Loading {display_name} ...")
        with open(fpath, 'rb') as f:
            aff = pickle.load(f)
        aff_csr = aff.tocsr()
        aff_csr.setdiag(0)
        aff_csr.eliminate_zeros()

        # Vectorised COO decomposition — avoids per-cell Python loop
        row_idx = np.repeat(np.arange(n_cells, dtype=np.int64),
                            np.diff(aff_csr.indptr))
        col_idx = aff_csr.indices.astype(np.int64)
        weights = aff_csr.data.astype(np.float64)

        total_w = np.bincount(row_idx, weights=weights, minlength=n_cells)

        same_ct    = ct_labels[row_idx] == ct_labels[col_idx]
        ct_w       = np.bincount(row_idx, weights=weights * same_ct,
                                  minlength=n_cells)
        ct_purity  = np.where(total_w > 0, ct_w / total_w, np.nan)

        same_state = same_ct & (niche_labels[row_idx] == niche_labels[col_idx])
        cs_w       = np.bincount(row_idx, weights=weights * same_state,
                                  minlength=n_cells)
        cs_purity  = np.where(total_w > 0, cs_w / total_w, np.nan)

        results[display_name] = {'ct': ct_purity, 'cs': cs_purity}

    # --- summary table ---
    summary_rows = []
    for aff_tag, display_name in affinity_types.items():
        if display_name not in results:
            continue
        ct_arr = results[display_name]['ct']
        cs_arr = results[display_name]['cs']
        valid  = ~(np.isnan(ct_arr) | np.isnan(cs_arr))
        summary_rows.append(dict(
            affinity=aff_tag,
            display_name=display_name,
            mean_ct_purity=np.nanmean(ct_arr),
            std_ct_purity=np.nanstd(ct_arr),
            mean_cs_purity=np.nanmean(cs_arr),
            std_cs_purity=np.nanstd(cs_arr),
            n_cells=int(valid.sum()),
        ))
    summary_df = pd.DataFrame(summary_rows)

    # --- scatter plot ---
    fig, ax = plt.subplots(figsize=figsize)

    for i, (aff_tag, display_name) in enumerate(affinity_types.items()):
        if display_name not in results:
            continue
        row = summary_df[summary_df['display_name'] == display_name].iloc[0]
        x, y = row['mean_ct_purity'], row['mean_cs_purity']
        color  = palette[i % len(palette)]
        marker = markers[i % len(markers)]

        if show_error_bars:
            ax.errorbar(x, y,
                        xerr=row['std_ct_purity'],
                        yerr=row['std_cs_purity'],
                        fmt=marker, color=color, markersize=9,
                        linewidth=1.0, capsize=3, alpha=0.85,
                        label=display_name, zorder=3)
        else:
            ax.scatter([x], [y], color=color, marker=marker,
                       s=90, label=display_name, zorder=3)

        ax.annotate(display_name, (x, y),
                    textcoords='offset points', xytext=(6, 4),
                    fontsize=8, color=color)

    ax.set_xlabel('Mean cell-type purity', fontsize=10)
    ax.set_ylabel('Mean cell-state purity', fontsize=10)
    ax.set_title(f'Affinity trade-off  [{ds_id}]', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, frameon=True, loc='lower right')
    ax.set_xlim(left=max(ax.get_xlim()[0] - 0.02, 0))
    ax.set_ylim(bottom=max(ax.get_ylim()[0] - 0.02, 0))
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, summary_df


# ---------------------------------------------------------------------------
# All-cell-type niche purity: table, per-model heatmap, trade-off scatter
# ---------------------------------------------------------------------------

_SOURCE_OBS_CACHE = {}


def _join_niche_from_source(ds_id, obs, niche_key):
    """Fallback: join niche_key onto obs (which has a 'cell_id' column) from the source h5ad.

    Needed for cell_assignments.csv files saved before niche_key was included directly
    (older runs) — reads obs only (backed mode), cached per ds_id so repeated calls for
    different keywords don't re-read the h5ad.
    """
    import scanpy as sc
    from interpretable_ssl.datasets.dataset_configs import DATASETS

    if ds_id not in _SOURCE_OBS_CACHE:
        path = str(DATASETS[ds_id]['path'])
        print(f"  [fig] '{niche_key}' missing from cell_assignments.csv — joining from {path} ...")
        _SOURCE_OBS_CACHE[ds_id] = sc.read_h5ad(path, backed='r').obs

    src_obs = _SOURCE_OBS_CACHE[ds_id]
    if niche_key not in src_obs.columns:
        return None

    obs = obs.copy()
    obs[niche_key] = src_obs.loc[obs['cell_id'].values, niche_key].values
    return obs


def _load_all_celltype_niche(ds_id, keyword, celltype_key, niche_key,
                             mc_key='metacell_id', min_cells=20):
    """Load cell_assignments.csv for a model keyword and compute per-cell purity.

    Uses all_celltype_niche_purity() from spatial_immune_task.py. Returns None if the
    run directory or required columns aren't found (even after falling back to joining
    niche labels from the source h5ad by cell_id).
    """
    from interpretable_ssl.evaluation.spatial_immune_task import all_celltype_niche_purity

    run_dir = _resolve_run_dir(ds_id, keyword, prefer_csv='cell_assignments.csv')
    if run_dir is None:
        print(f"  [fig] no run found for keyword='{keyword}', ds='{ds_id}'")
        return None

    path = os.path.join(run_dir, 'cell_assignments.csv')
    if not os.path.exists(path):
        print(f"  [fig] cell_assignments.csv not found in {run_dir}")
        return None

    obs = pd.read_csv(path)
    if niche_key not in obs.columns:
        obs = _join_niche_from_source(ds_id, obs, niche_key)
        if obs is None:
            print(f"  [fig] '{niche_key}' not available for keyword='{keyword}' "
                  f"(missing from cell_assignments.csv and from the source h5ad)")
            return None

    return all_celltype_niche_purity(
        obs, celltype_key=celltype_key, niche_key=niche_key, mc_key=mc_key, min_cells=min_cells,
    )


def _load_all_celltype_niche_by_model(ds_id, model_keywords, celltype_key, niche_key,
                                      mc_key='metacell_id', min_cells=20):
    per_cell = {}
    for keyword, name in model_keywords.items():
        df = _load_all_celltype_niche(ds_id, keyword, celltype_key, niche_key, mc_key, min_cells)
        if df is not None:
            per_cell[name] = df
    return per_cell


# title, colorbar-label-detail for each metric produced by all_celltype_niche_purity()
_PURITY_METRIC_INFO = {
    'niche_purity':                ('Niche purity',           'same cell type & niche'),
    'celltype_purity':              ('Cell-type purity',       'within this niche'),
    'niche_given_celltype_purity': ('Niche | cell-type purity', 'of same-type neighbours, fraction also same niche'),
}


def _purity_metric_labels(metric):
    if metric not in _PURITY_METRIC_INFO:
        raise ValueError(f"metric must be one of {list(_PURITY_METRIC_INFO)}, got {metric!r}")
    return _PURITY_METRIC_INFO[metric]


def fig_celltype_purity_table(ds_id, model_keywords, celltype_key='celltypes', niche_key='niches_2D',
                              mc_key='metacell_id', min_cells=20, metric='celltype_purity',
                              stat='median_iqr'):
    """Model x cell-type table for 'celltype_purity' or 'niche_purity'.

    stat='median_iqr' (default, recommended): returns (median_df, q25_df, q75_df).
    stat='mean_std': returns (mean_df, std_df) — use format_purity_table() on either
    result to render a single display-ready string table. See celltype_purity_table()
    in spatial_immune_task.py for why median_iqr is the recommended default (per-cell
    purity is often bimodal, which makes std larger than the mean and mean±std misleading).
    """
    from interpretable_ssl.evaluation.spatial_immune_task import celltype_purity_table

    per_cell = _load_all_celltype_niche_by_model(ds_id, model_keywords, celltype_key, niche_key, mc_key, min_cells)
    if not per_cell:
        print("No data loaded for any model.")
        return (None, None) if stat == 'mean_std' else (None, None, None)
    return celltype_purity_table(per_cell, metric=metric, stat=stat)


def format_purity_table(center_df, *spread_dfs, fmt='.2f'):
    """Combine a center/spread table pair (or triple) from fig_celltype_purity_table into
    one display-ready string table.

    Accepts either:
        format_purity_table(mean_df, std_df)             -> 'mean ± std'
        format_purity_table(median_df, q25_df, q75_df)    -> 'median [q25, q75]'

    Cells missing from center_df (NaN — cell type absent for that model) render as ''.
    """
    if center_df is None:
        return None

    out = pd.DataFrame(index=center_df.index, columns=center_df.columns, dtype=object)
    for r in center_df.index:
        for c in center_df.columns:
            m = center_df.loc[r, c]
            if pd.isna(m):
                out.loc[r, c] = ''
                continue

            if len(spread_dfs) == 2:
                q25, q75 = spread_dfs[0].loc[r, c], spread_dfs[1].loc[r, c]
                if pd.isna(q25) or pd.isna(q75):
                    out.loc[r, c] = f'{m:{fmt}}'
                else:
                    out.loc[r, c] = f'{m:{fmt}} [{q25:{fmt}}, {q75:{fmt}}]'
            elif len(spread_dfs) == 1:
                s = spread_dfs[0].loc[r, c]
                out.loc[r, c] = f'{m:{fmt}}' if pd.isna(s) else f'{m:{fmt}} ± {s:{fmt}}'
            else:
                out.loc[r, c] = f'{m:{fmt}}'
    return out


def fig_all_celltype_niche_heatmap(ds_id, model_keywords, celltype_key='celltypes', niche_key='niches_2D',
                                   mc_key='metacell_id', min_cells=20, metric='niche_purity',
                                   save_path=None, cell_size=0.42, cmap='YlOrRd', annot=True,
                                   fmt='.2f', fontsize=8, stat='median_iqr', show_summary_col=True,
                                   sort_by_summary=True, cell_types=None, color_gamma=0.5):
    """One heatmap per model: rows = cell type, cols = niche, color = central value of `metric`.

    metric='niche_purity' (default) is the joint (celltype AND niche) metric — a cell that
    scores well here necessarily also groups that cell type well, so the heatmap can't be
    inflated by a model that scatters cell types but happens to keep a same-type fragment
    in one niche. This answers "does this model get the cell type AND the niche right,
    together, in this specific niche?"

    metric='celltype_purity' instead shows, for cells of this type that happen to sit in
    this niche, how well the model groups them by cell type ALONE (regardless of which
    niche their metacell-mates are from). Comparing the two heatmaps side by side shows
    exactly where cell-type grouping is present but niche resolution isn't — niche_purity
    can only be as high as celltype_purity for the same (celltype, niche) cell, and niches
    don't all shape a cell type's expression equally, so the gap between the two heatmaps
    is itself informative, not just their individual values.

    Panel size scales with the number of cell types/niches (cell_size inches per grid
    cell) instead of a fixed figsize, so grids with many cell types stay readable instead
    of being crushed into a small fixed panel.

    (celltype, niche) pairs with zero ground-truth cells render as plain white (not the
    low end of the colormap) so "no data" is visually distinct from "measured near zero".

    color_gamma: PowerNorm exponent for the color scale (default 0.5, i.e. sqrt). Most
    values here sit near the low end of [0, 1] (especially for metric='niche_purity'),
    so a plain linear color scale makes almost every cell look like the same pale shade —
    color_gamma<1 stretches the low end so real differences among small values become
    visible, without changing the underlying numbers (still shown exactly in the
    annotation text). Pass color_gamma=None for the plain linear scale.

    stat: 'median_iqr' (default) colors by median and, if annot=True, writes 'median\\n[q25,q75]'
          in each cell. 'mean_std' colors by mean and writes 'mean\\n±std' instead.
          Both metrics are often bimodal per cell (some cells land in a small near-pure
          metacell, others in a large mixed one) — median/IQR is the more honest summary
          for the same reason it is in celltype_purity_table(). With 18 cell types x 9
          niches per panel the two-line annotation can still feel dense even with dynamic
          panel sizing — pass annot=False for a color-only view, or cell_types=[...] to
          restrict rows to a handful of key cell types (mirroring the paper's curated
          6-cell-type figure, fig_ct_niche_heatmap) if you want the annotated version.
    show_summary_col appends an extra "All niches" column (aggregated across that cell
    type's own niches) so a coarse per-cell-type read is available next to the fine-grained
    grid.
    sort_by_summary orders cell-type rows by that summary column (averaged across models,
    descending) instead of alphabetically, so the most-resolved cell types group together.
    cell_types: optional list — restrict rows to these cell types only.
    """
    if stat not in ('median_iqr', 'mean_std'):
        raise ValueError(f"stat must be 'median_iqr' or 'mean_std', got {stat!r}")
    _purity_metric_labels(metric)  # raises if metric is invalid

    per_cell = _load_all_celltype_niche_by_model(ds_id, model_keywords, celltype_key, niche_key, mc_key, min_cells)
    if not per_cell:
        print("No data loaded for any model.")
        return None

    if cell_types is not None:
        per_cell = {n: df[df['celltype'].isin(cell_types)] for n, df in per_cell.items()}

    SUMMARY_COL = 'All niches'
    names = list(per_cell.keys())
    center_stat, lo_stat, hi_stat = (
        ('median', lambda g: g.quantile(0.25), lambda g: g.quantile(0.75))
        if stat == 'median_iqr' else
        ('mean', lambda g: g.std(), None)
    )

    def _pivot(df, fn):
        g = df.groupby(['celltype', 'niche'])[metric]
        agg = getattr(g, fn)() if isinstance(fn, str) else fn(g)
        return agg.unstack('niche')

    center_pivots = {n: _pivot(df, center_stat) for n, df in per_cell.items()}
    lo_pivots = {n: _pivot(df, lo_stat) for n, df in per_cell.items()}
    hi_pivots = {n: _pivot(df, hi_stat) for n, df in per_cell.items()} if hi_stat else None

    all_cts = sorted({ct for p in center_pivots.values() for ct in p.index})
    all_niches = sorted({ni for p in center_pivots.values() for ni in p.columns})

    overall_by_model = {}
    for n, df in per_cell.items():
        center_pivots[n] = center_pivots[n].reindex(index=all_cts, columns=all_niches)
        lo_pivots[n] = lo_pivots[n].reindex(index=all_cts, columns=all_niches)
        g = df.groupby('celltype')[metric]
        overall_center = (g.median() if stat == 'median_iqr' else g.mean()).reindex(all_cts)
        overall_lo = (g.quantile(0.25) if stat == 'median_iqr' else g.std()).reindex(all_cts)
        overall_by_model[n] = overall_center
        if show_summary_col:
            center_pivots[n][SUMMARY_COL] = overall_center
            lo_pivots[n][SUMMARY_COL] = overall_lo
        if hi_pivots is not None:
            hi_pivots[n] = hi_pivots[n].reindex(index=all_cts, columns=all_niches)
            if show_summary_col:
                hi_pivots[n][SUMMARY_COL] = g.quantile(0.75).reindex(all_cts)

    if sort_by_summary:
        avg_summary = pd.concat(list(overall_by_model.values()), axis=1).mean(axis=1)
        all_cts = avg_summary.sort_values(ascending=False).index.tolist()
        for n in names:
            center_pivots[n] = center_pivots[n].reindex(index=all_cts)
            lo_pivots[n] = lo_pivots[n].reindex(index=all_cts)
            if hi_pivots is not None:
                hi_pivots[n] = hi_pivots[n].reindex(index=all_cts)

    vals = np.concatenate([p.values.ravel() for p in center_pivots.values()])
    vals = vals[~np.isnan(vals)]
    vmin, vmax = (float(vals.min()), float(vals.max())) if len(vals) else (0.0, 1.0)
    color_norm = PowerNorm(gamma=color_gamma, vmin=vmin, vmax=vmax) if color_gamma else None

    n_rows = len(all_cts)
    n_cols = len(all_niches) + (1 if show_summary_col else 0)
    pw = max(2.5, cell_size * n_cols + 1.6)
    ph = max(2.5, cell_size * n_rows + 1.0)

    fig, axes = plt.subplots(1, len(names), figsize=(pw * len(names), ph), squeeze=False)
    for col, name in enumerate(names):
        if stat == 'median_iqr' and annot:
            annot_matrix = center_pivots[name].copy().astype(object)
            for r in range(annot_matrix.shape[0]):
                for c in range(annot_matrix.shape[1]):
                    m, lo = center_pivots[name].values[r, c], lo_pivots[name].values[r, c]
                    hi = hi_pivots[name].values[r, c] if hi_pivots is not None else np.nan
                    if np.isnan(m):
                        annot_matrix.iat[r, c] = ''
                    elif np.isnan(lo) or np.isnan(hi):
                        annot_matrix.iat[r, c] = f'{m:{fmt}}'
                    else:
                        annot_matrix.iat[r, c] = f'{m:{fmt}}\n[{lo:{fmt}},{hi:{fmt}}]'
            std_kwarg = dict(annot_matrix=annot_matrix)
        elif annot:
            std_kwarg = dict(std_matrix=lo_pivots[name])
        else:
            std_kwarg = {}

        _draw_heatmap_ax(
            axes[0, col], center_pivots[name], title=name, cmap=cmap,
            vmin=vmin, vmax=vmax, annot=annot, fmt=fmt, fontsize=fontsize,
            show_yticklabels=(col == 0), show_ylabel=(col == 0),
            xlabel='Niche', ylabel_label='Cell Type', norm=color_norm,
            **std_kwarg,
        )
        if show_summary_col:
            axes[0, col].axvline(len(all_niches) - 0.5, color='black', linewidth=1.2)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=color_norm or plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    stat_label = 'Median' if stat == 'median_iqr' else 'Mean'
    metric_title, metric_detail = _purity_metric_labels(metric)
    fig.colorbar(sm, ax=axes[0, -1], fraction=0.046, pad=0.04,
                label=f'{stat_label} {metric_title}\n({metric_detail})')
    fig.suptitle(f'{metric_title} by cell type x niche  [{ds_id}]', fontsize=11, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def fig_celltype_niche_tradeoff(ds_id, model_keywords, celltype_key='celltypes', niche_key='niches_2D',
                                mc_key='metacell_id', min_cells=20, save_path=None,
                                figsize=(6.5, 5.5), palette=None, point_size=18, show_trend=True):
    """Scatter: one dot per (model, cell type). x = mean celltype_purity, y = mean niche_purity.

    Color = model, small dots, no per-point text — deliberately built to show the overall
    trend rather than let you identify individual cell types. If show_trend, a linear
    best-fit line is drawn per model across its cell-type dots, so you can compare how
    steeply niche purity tracks celltype purity for each model at a glance.

    The dashed y=x line is the structural upper bound (niche_purity <= celltype_purity by
    construction): a trend line hugging the diagonal means that model separates niches
    almost as well as it groups cell types; a trend line well below the diagonal means it
    groups cell types but scatters niches within them.
    """
    per_cell = _load_all_celltype_niche_by_model(ds_id, model_keywords, celltype_key, niche_key, mc_key, min_cells)
    if not per_cell:
        print("No data loaded for any model.")
        return None

    names = list(per_cell.keys())
    if palette is None:
        palette = list(plt.cm.tab10.colors)
    color_map = {n: palette[i % len(palette)] for i, n in enumerate(names)}

    summaries = {
        name: df.groupby('celltype').agg(x=('celltype_purity', 'mean'), y=('niche_purity', 'mean'))
        for name, df in per_cell.items()
    }

    fig, ax = plt.subplots(figsize=figsize)

    for name, g in summaries.items():
        c = color_map[name]
        ax.scatter(g['x'], g['y'], label=name, color=c, s=point_size, alpha=0.6,
                  edgecolors='none', zorder=2)

        if show_trend and len(g) >= 2:
            xs, ys = g['x'].values, g['y'].values
            slope, intercept = np.polyfit(xs, ys, 1)
            x_line = np.array([xs.min(), xs.max()])
            ax.plot(x_line, slope * x_line + intercept, color=c, linewidth=2.2, alpha=0.9, zorder=3)

    ax.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=1.0, zorder=1,
           label='niche = celltype purity (upper bound)')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('Cell-type purity', fontsize=11)
    ax.set_ylabel('Niche purity  (same cell type & niche)', fontsize=11)
    ax.set_title(f'Cell-type vs. niche purity trend, per model  [{ds_id}]', fontsize=11)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def fig_celltype_niche_heatmap_diff(ds_id, model_keywords, reference, celltype_key='celltypes',
                                    niche_key='niches_2D', mc_key='metacell_id', min_cells=20,
                                    metric='niche_purity', stat='median_iqr', min_n=5,
                                    cell_size=0.42, cmap='RdBu_r', fmt='+.2f', fontsize=8,
                                    annot=True, sort_by_reference=True, cell_types=None,
                                    save_path=None):
    """One heatmap per non-reference model: color = (this model's value - reference's value),
    per (celltype, niche) cell.

    This is the direct tool for "in this specific (cell type, niche) pair, does this model
    separate niches that the reference mixes together" — rather than eyeballing near-identical
    pale grids side by side (most raw values are small, so absolute heatmaps compress real
    differences into indistinguishable shades). A diverging colormap centered at 0 makes
    "this model wins here" and "reference wins here" immediately visually distinct.

    Args:
        reference: a keyword from model_keywords to diff every other model against
                   (e.g. the transcriptomics-only baseline).
        metric:    'niche_purity' or 'celltype_purity'.
        min_n:     (celltype, niche) cells where either model has fewer than this many
                   ground-truth ceels are treated as unreliable and rendered white — a
                   difference computed from a handful of cells isn't a real finding.
        sort_by_reference: order cell-type rows by their largest positive diff (averaged
                   across models, descending), so the most-improved cell types float to
                   the top instead of alphabetical order.

    Returns:
        matplotlib Figure, and prints the top improved (celltype, niche) cells per model —
        the concrete candidates for "look at this pair, others mix niches, ours separates".
    """
    if reference not in model_keywords:
        raise ValueError(f"reference={reference!r} not in model_keywords keys: {list(model_keywords)}")
    _purity_metric_labels(metric)  # raises if metric is invalid

    ref_name = model_keywords[reference]
    per_cell = _load_all_celltype_niche_by_model(ds_id, model_keywords, celltype_key, niche_key, mc_key, min_cells)
    if ref_name not in per_cell:
        print(f"No data loaded for reference '{reference}' ({ref_name}).")
        return None

    if cell_types is not None:
        per_cell = {n: df[df['celltype'].isin(cell_types)] for n, df in per_cell.items()}

    other_names = [n for k, n in model_keywords.items() if k != reference and n in per_cell]
    if not other_names:
        print("No non-reference models loaded.")
        return None

    center_fn = (lambda g: g.median()) if stat == 'median_iqr' else (lambda g: g.mean())

    def _center_pivot(df):
        return center_fn(df.groupby(['celltype', 'niche'])[metric]).unstack('niche')

    def _count_pivot(df):
        return df.groupby(['celltype', 'niche']).size().unstack('niche')

    center_pivots = {n: _center_pivot(df) for n, df in per_cell.items()}
    count_pivots = {n: _count_pivot(df) for n, df in per_cell.items()}

    all_cts = sorted({ct for p in center_pivots.values() for ct in p.index})
    all_niches = sorted({ni for p in center_pivots.values() for ni in p.columns})
    for n in per_cell:
        center_pivots[n] = center_pivots[n].reindex(index=all_cts, columns=all_niches)
        count_pivots[n] = count_pivots[n].reindex(index=all_cts, columns=all_niches).fillna(0)

    ref_center, ref_count = center_pivots[ref_name], count_pivots[ref_name]

    diffs = {}
    for name in other_names:
        diff = center_pivots[name] - ref_center
        unreliable = (count_pivots[name] < min_n) | (ref_count < min_n)
        diff = diff.mask(unreliable)
        diffs[name] = diff

    if sort_by_reference:
        stacked = pd.concat({name: d.stack() for name, d in diffs.items()}, axis=1)
        avg_diff_per_ct = stacked.mean(axis=1).groupby(level=0).max()
        all_cts = avg_diff_per_ct.reindex(all_cts).sort_values(ascending=False).index.tolist()
        for name in other_names:
            diffs[name] = diffs[name].reindex(index=all_cts)

    max_abs = np.nanmax([np.nanmax(np.abs(d.values)) for d in diffs.values() if np.isfinite(d.values).any()] or [1.0])
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    n_rows, n_cols = len(all_cts), len(all_niches)
    pw = max(2.5, cell_size * n_cols + 1.6)
    ph = max(2.5, cell_size * n_rows + 1.0)

    fig, axes = plt.subplots(1, len(other_names), figsize=(pw * len(other_names), ph), squeeze=False)
    for col, name in enumerate(other_names):
        _draw_heatmap_ax(
            axes[0, col], diffs[name], title=f'{name}\nminus {ref_name}', cmap=cmap,
            vmin=-max_abs, vmax=max_abs, annot=annot, fmt=fmt, fontsize=fontsize,
            show_yticklabels=(col == 0), show_ylabel=(col == 0),
            xlabel='Niche', ylabel_label='Cell Type', norm=norm,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    metric_title, _ = _purity_metric_labels(metric)
    fig.colorbar(sm, ax=axes[0, -1], fraction=0.046, pad=0.04,
                label=f'{metric_title} minus {ref_name}\n(blue=worse, red=better)')
    fig.suptitle(f'{metric_title} vs. {ref_name}, by cell type x niche  [{ds_id}]',
                fontsize=11, fontweight='bold')
    plt.tight_layout()

    print(f"\nTop improved (celltype, niche) pairs vs. '{ref_name}' (min_n={min_n} cells):")
    for name in other_names:
        d = diffs[name].stack().sort_values(ascending=False)
        print(f"\n  {name}:")
        print(d.head(8).round(3).to_string())

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def fig_purity_violin(ds_id, model_keywords, celltype_key='celltypes', niche_key='niches_2D',
                      mc_key='metacell_id', min_cells=20, metric='celltype_purity',
                      cell_types=None, n_cols=4, panel_size=(2.8, 2.6), palette=None,
                      save_path=None):
    """Violin (+ jittered points) of the per-cell purity distribution, one panel per cell
    type, one violin per model. Reuses _violin_panel, the same helper the paper's other
    purity figures use.

    Shows the actual distribution instead of collapsing it to mean ± std — useful because
    per-cell purity is often bimodal (some cells land in a small near-pure metacell, others
    in a large mixed one), and a mean ± std summary hides that shape entirely.

    Args:
        metric:     'celltype_purity' or 'niche_purity'.
        cell_types: optional list to restrict which cell types get a panel (default: all).
    """
    per_cell = _load_all_celltype_niche_by_model(ds_id, model_keywords, celltype_key, niche_key, mc_key, min_cells)
    if not per_cell:
        print("No data loaded for any model.")
        return None

    names = list(per_cell.keys())
    if palette is None:
        palette = list(plt.cm.tab10.colors)
    colors = [palette[i % len(palette)] for i in range(len(names))]

    all_cts = sorted({ct for df in per_cell.values() for ct in df['celltype'].unique()})
    if cell_types is not None:
        all_cts = [ct for ct in all_cts if ct in cell_types]
    if not all_cts:
        print("No cell types left to plot.")
        return None

    n_panels = len(all_cts)
    n_cols = max(1, min(n_cols, n_panels))
    n_rows = int(np.ceil(n_panels / n_cols))
    pw, ph = panel_size
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(pw * n_cols, ph * n_rows), squeeze=False)

    for i, ct in enumerate(all_cts):
        ax = axes[i // n_cols, i % n_cols]
        arrays = []
        for n in names:
            vals = per_cell[n].loc[per_cell[n]['celltype'] == ct, metric].values
            arrays.append(vals if len(vals) > 0 else None)
        sizes = [None] * len(names)
        _violin_panel(ax, names, arrays, sizes, colors, ylabel=metric if i % n_cols == 0 else '')
        ax.set_title(ct, fontsize=9)
        ax.set_ylim(-0.02, 1.02)

    for j in range(n_panels, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis('off')

    fig.suptitle(f'{metric} distribution per cell type  [{ds_id}]', fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
