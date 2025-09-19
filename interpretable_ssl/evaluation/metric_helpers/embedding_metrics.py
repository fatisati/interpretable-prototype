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
import torch
import uuid
from interpretable_ssl.scproto_metacells import *
from interpretable_ssl.configs.defaults import get_defaults
res_dir = "/home/icb/fatemehs.hashemig/models/"
# use this code to calculate scib and scgraph metrics for models [scProto, scPoli, scVI, pca, pca+harmoney, seacell]


def save_append(df, save_dir, name, append=True, name_postfix=None):
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    if name_postfix is not None:
        name = f"{name}_{name_postfix}"
    save_path = f"{save_dir}/{name}.csv"
    print(name_postfix, save_path)
    if append and os.path.exists(save_path):
        saved_res = pd.read_csv(save_path, index_col=0)
        df = pd.concat([df, saved_res])
    df.to_csv(save_path)
    return df


def get_scib(adata, obsm_keys, bk, lk):
    bm = Benchmarker(
        adata=adata,
        batch_key=bk,
        label_key=lk,
        embedding_obsm_keys=obsm_keys,  # evaluate the PCA space
    )
    bm.benchmark()  # runs neighbors, clustering, and metrics
    results = bm.get_results(min_max_scale=False)  # returns a tidy DataFrame
    results = results.drop(index="Metric Type")
    return results
    # save_path = f"{res_dir}/{ds}/"
    # return save_append(results, save_path, "scib", name_postfix=name_postfix)


# thres_batch=100, thres_celltype=10
def get_scgraph(
    adata,
    obsm_keys,
    batch_key="study",
    label_key="cell_type",
    **kwargs,
):
    tmp_path = f"tmp_{uuid.uuid4().hex[:8]}.h5ad"

    adata.write(tmp_path)
    scgraph = scGraph(
        adata_path=tmp_path, batch_key=batch_key, label_key=label_key, **kwargs
    )
    scgr_res = scgraph.main(_obsm_list=obsm_keys)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
        print("Deleted:", tmp_path)
    return scgr_res
    # save_path = f"{res_dir}/{dataset_name}/"
    # return save_append(scgr_res, save_path, "scgraph", name_postfix=name_postfix)


def get_metrics(adata, emb_keys, bk, lk, **scgraph_kwargs):
    scg = get_scgraph(adata, emb_keys, bk, lk, **scgraph_kwargs)
    scb = get_scib(adata, emb_keys, bk, lk)
    return scg, scb


def save_metrics(adata, emb_keys, dataset, bk, lk, **scgraph_kwargs):
    scgraph_m = get_scgraph(adata, emb_keys, dataset, bk, lk, **scgraph_kwargs)
    scib_m = get_scib(adata, emb_keys, dataset, bk, lk)
    return scib_m, scgraph_m


def add_trainer_emb(t, adata):
    model = t.load_model()
    # adata = t.dataset.adata
    adata.obsm[t.get_model_name()] = t.encode_adata(adata, model).detach().cpu().numpy()
    return adata


# used to be save_metrics
def save_trainer_metrics(t, dataset, append=True):
    adata = add_trainer_emb(t)
    bk, lk = t.dataset.batch_key, t.dataset.cell_type_key
    return save_metrics(adata, [t.get_model_name()], dataset, bk, lk, append)


def add_scvi_emb(adata, query_stu, bk, pt_epochs=None, ft_epochs=None):
    d = get_defaults()
    pt_epochs = pt_epochs or d['pretraining_epochs'] + d['cvae_epochs']
    ft_epochs = ft_epochs or d['ft_epochs']
    ref = adata[~adata.obs[bk].isin(query_stu)].copy()

    # 1) Setup AnnData for scVI
    #    No need for common genes step since ref_adata comes from adata
    scvi.model.SCVI.setup_anndata(ref, batch_key=bk)

    # 2) Train model on reference
    model = scvi.model.SCVI(ref, n_latent=8)
    model.train(max_epochs=pt_epochs)

    # Adapt model to whole adata
    query_model = scvi.model.SCVI.load_query_data(adata, model)
    query_model.train(ft_epochs)
    key = "X_scvi"
    key += f"_pt{pt_epochs}" if pt_epochs != d['pretraining_epochs'] else ""
    key += f"_ft{ft_epochs}" if ft_epochs != d['ft_epochs'] else ""
    adata.obsm[key] = query_model.get_latent_representation(adata)
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


def save_pca_harmoney_metrics(
    adata, bk, lk, dataset, pca_key="X_pca", postfix=None, **scgraph_kwargs
):
    adata = add_pca_harmoney(adata, bk, pca_key)
    return save_metrics(
        adata,
        [pca_key, f"{pca_key}_harmoney"],
        dataset,
        bk,
        lk,
        name_postfix=postfix,
        **scgraph_kwargs,
    )


def get_seacell_metrics(SEACell_ad, adata, bk, lk, ds, postfix=None, **scgraph_kwargs):
    # Normalize cells, log transform and compute highly variable genes
    sc.pp.normalize_per_cell(SEACell_ad)
    sc.pp.log1p(SEACell_ad)
    # sc.pp.highly_variable_genes(ad, n_top_genes=1500)

    # SEACell_ad = agg_obs(SEACell_ad, adata, bk)
    # SEACell_ad = agg_obs(SEACell_ad, adata, lk)

    return save_pca_harmoney_metrics(
        SEACell_ad, bk, lk, ds, "seacell_pca", postfix=postfix, **scgraph_kwargs
    )


def calc_adata_metrics(dataset, ds_conf):
    adata = sc.read_h5ad(ds_conf["path"])
    add_scvi_emb(adata, ds_conf["test_studies"], ds_conf["batch_key"])
    add_pca_harmoney(adata, ds_conf["batch_key"], "X_pca")
    return save_metrics(
        adata,
        ["X_pca", "X_pca_harmoney", "X_scvi"],
        dataset,
        ds_conf["batch_key"],
        ds_conf["label_key"],
    )


def get_scproto_mc_adata(t, adata, bk, lk):
    model = t.load_model()
    protos = model.get_prototypes()
    batch = np.zeros((protos.shape[0], 1))
    batch = torch.as_tensor(batch, dtype=torch.long, device="cuda")
    sizefactor = np.ones((protos.shape[0],))
    sizefactor = torch.as_tensor(sizefactor, dtype=torch.long, device="cuda")
    metacells = model.decode(protos, batch, sizefactor)

    sample_proto_sim = t.encode_adata(adata, model, return_mapped=True)
    proto_labels = extract_proto_labels(
        adata, sample_proto_sim.detach().cpu().numpy(), [bk, lk]
    )
    metacells_adata = generate_metacell_adata(metacells, proto_labels)
    sc.tl.pca(metacells_adata)
    metacells_adata.obsm[f"{t.get_model_name()}_mc_pca"] = metacells_adata.obsm["X_pca"]
    return metacells_adata


def get_scproto_metacell_metrics(
    t, adata, ds, bk, lk, name_postfix=None, **scgraph_kwargs
):
    metacells_adata = get_scproto_mc_adata(t, adata, bk, lk)
    return save_metrics(
        metacells_adata, ["scProto_mc_pca"], ds, bk, lk, name_postfix, **scgraph_kwargs
    )


def load_seacell(ds_id):
    home = "/home/icb/fatemehs.hashemig/"
    # ad = sc.read_h5ad(f'{home}/models/{ds_id}/seacell_sc.h5ad')
    mc_ad = sc.read_h5ad(f"{home}/models/{ds_id}/seacell_agg.h5ad")
    sc.pp.normalize_total(mc_ad, target_sum=1e4)
    sc.pp.log1p(mc_ad)
    return mc_ad


def get_metacell_metrics(
    mc_ad,
    obsm_keys,
    bk,
    lk,
):
    return get_metrics(mc_ad, obsm_keys, bk, lk, thres_batch=10, thres_celltype=5)
