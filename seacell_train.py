from interpretable_ssl.evaluation.mc_metric_utils import *
from interpretable_ssl.datasets.dataset_configs import *
from interpretable_ssl.configs.paths import get_seacell_model_dir


def get_seacell_path(ds_id):
    return get_seacell_model_dir(ds_id)


def load_dataset(ds_id):
    conf = DATASETS[ds_id]
    return (
        sc.read_h5ad(conf["path"]),
        conf.get('batch_key', None),
        conf["label_key"],
        conf["num_prototypes"],
    )


def train_seacell(ds_id, mode):
    seacell_exists = os.path.exists(get_seacell_path(ds_id) + "/seacell_sc.h5ad")
    if mode == "train" or not (seacell_exists):
        from interpretable_ssl.evaluation.metric_helpers.metacell_metrics import (
            compute_seacells,
            agg_obs,
            save_seacell,
        )

        os.makedirs(get_seacell_path(ds_id), exist_ok=True)
        # use the current dataset id, not a fixed string
        ad, bk, lk, n_proto = load_dataset(ds_id)
        print(len(ad))
        if ad.X.max() > 20:
            sc.pp.normalize_total(ad)
            sc.pp.log1p(ad)

        sc.tl.pca(ad)
        ad, SEACell_ad, model = compute_seacells(ad, n_proto)
        agg_obs(SEACell_ad, ad, lk)
        if bk is not None:
            agg_obs(SEACell_ad, ad, bk)
        save_seacell(ad, SEACell_ad, ds_id)
    else:
        print("eval mode, seacell file founded.")
        bk, lk = DATASETS[ds_id].get('batch_key', None), DATASETS[ds_id]["label_key"]
        ad, SEACell_ad = load_seacell(ds_id)
        if bk not in SEACell_ad.obs and ad.obs[bk].nunique() == 1:
            SEACell_ad.obs[bk] = ad[0].obs[bk].item()
    if SEACell_ad.X.max() > 20:
        sc.pp.normalize_total(SEACell_ad)
        sc.pp.log1p(SEACell_ad)
    ad.obs['mc_idx'] = [int(i.split('-')[-1]) for i in ad.obs["SEACell"].values]
    save_all_mc_metrics(ad, SEACell_ad, lk, bk, get_seacell_path(ds_id))
