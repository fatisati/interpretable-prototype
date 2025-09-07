from scib_metrics.benchmark import Benchmarker
import os
import pandas as pd
import sys

sys.path.append("/home/icb/fatemehs.hashemig/Islander/src")
from scGraph import *
import scanpy as sc
import scvi
import scanpy as sc
import scanpy.external as sce

from interpretable_ssl.trainers.scpoli_original import *
from interpretable_ssl.trainers.swav import *


res_dir = "/home/icb/fatemehs.hashemig/models/"
# use this code to calculate scib and scgraph metrics for models [scProto, scPoli, scVI, pca, pca+harmoney, seacell]


def save_append(df, save_dir, name, append=True):
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    save_path = f"{save_dir}/{name}"
    if append and os.path.exists(save_path):
        saved_res = pd.read_csv(save_path, index_col=0)
        df = pd.concat([df, saved_res])
    df.to_csv(save_path)
    return df


def get_scib(adata, obsm_keys, ds, bk, lk):
    bm = Benchmarker(
        adata=adata,
        batch_key=bk,
        label_key=lk,
        embedding_obsm_keys=obsm_keys,  # evaluate the PCA space
    )
    bm.benchmark()  # runs neighbors, clustering, and metrics
    results = bm.get_results(min_max_scale=False)  # returns a tidy DataFrame
    results = results.drop(index="Metric Type")
    save_path = f"{res_dir}/{ds}/"
    return save_append(results, save_path, "scib.csv")


def get_scgraph(
    adata, obsm_keys, dataset_name, batch_key="study", label_key="cell_type", **kwargs
):
    adata.write("tmp.h5ad")
    scgraph = scGraph(
        adata_path="tmp.h5ad", batch_key=batch_key, label_key=label_key, **kwargs
    )
    scgr_res = scgraph.main(_obsm_list=obsm_keys)
    save_path = f"{res_dir}/{dataset_name}/"
    return save_append(scgr_res, save_path, "scgraph.csv")


def save_metrics(adata, emb_keys, dataset, bk, lk):
    scgraph_m = get_scgraph(adata, emb_keys, dataset, bk, lk)
    scib_m = get_scib(adata, emb_keys, dataset, bk, lk)
    return scib_m, scgraph_m


def add_trainer_emb(t, adata):
    if t.model is None:
        t.setup()
    model = t.load_model()
    # adata = t.dataset.adata
    adata.obsm[t.get_model_name()] = t.encode_adata(adata, model).detach().cpu().numpy()
    return adata


# used to be save_metrics
def save_trainer_metrics(t, dataset, append=True):
    adata = add_trainer_emb(t)
    bk, lk = t.dataset.batch_key, t.dataset.cell_type_key
    return save_metrics(adata, [t.get_model_name()], dataset, bk, lk, append)


def add_scvi_emb(adata, query_stu, bk):
    ref = adata[~adata.obs[bk].isin(query_stu)].copy()

    # 1) Setup AnnData for scVI
    #    No need for common genes step since ref_adata comes from adata
    scvi.model.SCVI.setup_anndata(ref, batch_key=bk)

    # 2) Train model on reference
    model = scvi.model.SCVI(ref, n_latent=8)
    model.train(max_epochs=100)

    # Adapt model to whole adata
    query_model = scvi.model.SCVI.load_query_data(adata, model)
    query_model.train(1)
    adata.obsm["X_scvi"] = query_model.get_latent_representation(adata)
    return adata


def get_scvi_metrics(adata, query_stu, bk, lk, dataset):
    adata = add_scvi_emb(adata, query_stu, bk)
    return save_metrics(adata, ["X_scvi"], dataset, bk, lk)


def add_pca_harmoney(adata, bk, pca_key):
    sc.tl.pca(adata)
    if pca_key != "X_pca":
        adata.obsm[pca_key] = adata.obsm["X_pca"].copy()

    # 2) Run Harmony batch correction
    sce.pp.harmony_integrate(
        adata,
        key=bk,  # your batch column
        basis=pca_key,  # which embedding to correct
        adjusted_basis=f"{pca_key}_harmoney",  # where to store corrected PCs
    )
    return adata


def save_pca_harmoney_metrics(adata, bk, lk, dataset, pca_key="X_pca"):
    adata = add_pca_harmoney(adata, bk, pca_key)
    return save_metrics(adata, [pca_key, f"{pca_key}_harmoney"], dataset, bk, lk)


def agg_obs(SEACell_ad, adata, obs_key):
    SEACell_ad.obs[obs_key] = (
        adata.obs.groupby("SEACell")[obs_key]
        .agg(lambda x: x.mode()[0])
        .reindex(SEACell_ad.obs_names)
    )
    return SEACell_ad


def get_seacell_metrics(SEACell_ad, adata, bk, lk, ds):
    # Normalize cells, log transform and compute highly variable genes
    sc.pp.normalize_per_cell(SEACell_ad)
    sc.pp.log1p(SEACell_ad)
    # sc.pp.highly_variable_genes(ad, n_top_genes=1500)

    SEACell_ad = agg_obs(SEACell_ad, adata, bk)
    SEACell_ad = agg_obs(SEACell_ad, adata, lk)

    return save_pca_harmoney_metrics(SEACell_ad, bk, lk, ds, "seacell_pca")


def calc_adata_metrics(scpoli_params, scproto_params, dataset, ds_conf):
    adata = sc.read_h5ad(ds_conf["path"])
    t1 = OriginalTrainer(debug=1, **scpoli_params)
    t2 = SwAV(debug=1, **scproto_params)
    for t in [t1, t2]:
        add_trainer_emb(t, adata)
    add_scvi_emb(adata, ds_conf["test_studies"], ds_conf["batch_key"])
    add_pca_harmoney(adata, ds_conf["batch_key"], "X_pca")
    return save_metrics(
        adata,
        [t.get_model_name() for t in [t1, t2]] + ["X_pca", "X_pca_harmoney", "X_scvi"],
        dataset,
        ds_conf["batch_key"],
        ds_conf["label_key"],
    )
