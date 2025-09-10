import pandas as pd

def highlight_max_second(s):
    """
    Style: bold max, underline second max (column-wise).
    """
    # get values sorted
    sorted_vals = s.sort_values(ascending=False).unique()
    styles = [''] * len(s)

    if len(sorted_vals) > 0:
        max_val = sorted_vals[0]
        styles = [
            'font-weight: bold' if v == max_val else '' 
            for v in s
        ]
    if len(sorted_vals) > 1:
        second_val = sorted_vals[1]
        styles = [
            (st + '; text-decoration: underline' if v == second_val else st) 
            for st, v in zip(styles, s)
        ]
    return styles

def generate_table(res_dir, name_postfix=None):
    files = ['scib', 'scgraph']
    if name_postfix is not None:
        files = [f'{f}_{name_postfix}' for f in files]
    files = [f'{f}.csv' for f in files]
    scib_df = pd.read_csv(f'{res_dir}/{files[0]}', index_col=0)
    scgraph_df = pd.read_csv(f'{res_dir}/{files[1]}', index_col=0)
    df = pd.concat([scib_df, scgraph_df], axis=1)

    show_cols = ['Batch correction', 'Bio conservation', 'Total', 'Rank-PCA', 'Corr-PCA', 'Corr-Weighted']
    return df[show_cols].style.apply(highlight_max_second, axis=0).format("{:.3f}")