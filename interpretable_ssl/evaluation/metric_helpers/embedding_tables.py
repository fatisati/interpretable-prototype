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
    p = f'/home/icb/fatemehs.hashemig/models/{ds_id}/'
    dfs = {}

    def load_csv(p):
        if os.path.exists(p):
            return pd.read_csv(p, index_col = 0)
        # else:
        #     print(p)
    for key in ['scib', 'scgraph', 'mc_quality_summary', 'de_jaccard_ref', 'de_jaccard_all']:
        key_dfs = [load_csv(f"{p}{fol}/{key}.csv") for fol in tqdm(os.listdir(p))]
        key_dfs = [df for df in key_dfs if df is not None]  # remove None objects
        if len(key_dfs)>0:
            dfs[key] = pd.concat(key_dfs)
        else:
            dfs[key] = None
    scib_df, scgraph_df, mc_df = dfs['scib'], dfs['scgraph'], dfs['mc_quality_summary']
    try:
        df = pd.concat([scib_df, scgraph_df], axis=1)
    except:
        df = (scib_df, scgraph_df)
    return df, dfs