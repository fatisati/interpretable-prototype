from interpretable_ssl.evaluation.de_helper import *
from interpretable_ssl.evaluation.metric_helpers.embedding_metrics import *

def de_sudo_bulk(ad, bk, lk, ds_id):
    # sudo bulk
    pb_ad = pseudo = ad.to_df().groupby(ad.obs[[bk, lk]].agg("_".join, axis=1)).mean()

    de_df = compute_dge_consistency(
        pb_ad,
        ad,
        lk,
        bk,
        0.1,
        "sudo-bulk",
    )
    save_append(de_df, f'{home}/baselines/dge_all.csv')
