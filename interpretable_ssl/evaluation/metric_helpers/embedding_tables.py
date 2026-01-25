import pandas as pd
import os
from tqdm import tqdm

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
    return show_tb(df)

def show_tb(df, show_cols = ['Batch correction', 'Bio conservation', 'Total', 'Rank-PCA', 'Corr-PCA', 'Corr-Weighted']):
    if show_cols is not None:
        df = df[show_cols]
    return df.style.apply(highlight_max_second, axis=0).format("{:.3f}")

def load_tb(ds_id):
    from interpretable_ssl.configs.paths import get_dataset_model_dir
    p = get_dataset_model_dir(ds_id)
    dfs = {}

    def load_csv(fp):
        return pd.read_csv(fp, index_col=0) if os.path.exists(fp) else None

    for fol in tqdm(os.listdir(p)):
        fol_path = os.path.join(p, fol)
        if not os.path.isdir(fol_path):
            continue

        for fn in os.listdir(fol_path):
            if not fn.endswith(".csv"):
                continue

            key = fn.replace(".csv", "")
            df = load_csv(os.path.join(fol_path, fn))
            if df is None:
                continue

            dfs.setdefault(key, []).append(df)

    dfs = {k: pd.concat(v) for k, v in dfs.items()}

    scib_df = dfs.get("scib")
    scgraph_df = dfs.get("scgraph")

    try:
        df = pd.concat([scib_df, scgraph_df], axis=1)
    except:
        df = (scib_df, scgraph_df)

    return df, dfs