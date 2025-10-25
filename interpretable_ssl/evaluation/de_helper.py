import scanpy as sc
import pandas as pd


def get_gene_map(ad):
    return ad.var["feature_name"].to_dict()


def calc_dge(ad, label_key, filter_thr=0.05):
    group_cnt = ad.obs[label_key].value_counts()
    valid_groups = group_cnt[group_cnt > 1].index
    ad_valid = ad[ad.obs[label_key].isin(valid_groups)]
    sc.tl.rank_genes_groups(
        ad_valid, groupby=label_key, method="t-test", key_added="rank_genes"
    )
    dge_df = sc.get.rank_genes_groups_df(ad_valid, key="rank_genes", group=None)
    dge_df = dge_df[dge_df["pvals_adj"] < filter_thr]
    dge_df["group"] = dge_df["group"].map(lambda x: x.lower())
    return dge_df


def calculate_avg_jaccard(mc_de, batch_de_dict, col_name=None):
    if col_name is None:
            col_name = "avg_jaccard_index"
    else:
        col_name += '_jaccard'
    row_dict = {"index": col_name}

    for ct in mc_de["group"].unique():
        jaccard_indices = []
        mc_genes = set(mc_de[mc_de["group"] == ct]["names"])

        for b_de in batch_de_dict.values():
            b_genes = set(b_de[b_de["group"] == ct]["names"])
            intersection = len(mc_genes & b_genes)
            union = len(mc_genes | b_genes)
            jaccard_indices.append(intersection / union if union > 0 else 0)
        
        # directly store as column
        row_dict[ct] = sum(jaccard_indices) / len(jaccard_indices)

    # create DataFrame with one row, index = col_name
    return pd.DataFrame([row_dict]).set_index("index")


def get_mc_jaccard(mc_adata, ad, lk, bk, thr, name=None):
    mc_de = calc_dge(mc_adata, lk, thr)
    batch_de_dict = {}
    for b in ad.obs[bk].unique():
        b_ad = ad[ad.obs[bk] == b].copy()
        batch_de_dict[b] = calc_dge(b_ad, lk, thr)
    return calculate_avg_jaccard(mc_de, batch_de_dict, name)
