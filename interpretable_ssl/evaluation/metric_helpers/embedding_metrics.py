from scib_metrics.benchmark import Benchmarker
import os
import numpy as np
import pandas as pd
import sys

from interpretable_ssl.configs.paths import MODEL_DIR, get_seacell_model_dir, get_dataset_model_dir
from interpretable_ssl.scGraph import *
import scanpy as sc
import scvi
import scanpy as sc
import scanpy.external as sce
import torch
import uuid
from interpretable_ssl.scproto_metacells import *
from interpretable_ssl.configs.defaults import get_defaults
from sklearn.model_selection import train_test_split

res_dir = MODEL_DIR
# use this code to calculate scib and scgraph metrics for models [scProto, scPoli, scVI, pca, pca+harmoney, seacell]


def save_append(df, save_dir, name, append=True, name_postfix=None):
    if df is None:
        print(f"df is none, {save_dir}/{name} not saved")
        return
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


def get_scib(adata, obsm_keys, bk, lk, bio_conservation_metrics=None, batch_correction_metrics=None):
    """bio_conservation_metrics / batch_correction_metrics: pass
    scib_metrics.benchmark.BioConservation / BatchCorrection instances to
    restrict which metrics Benchmarker computes (default None = its own full
    battery). Useful for e.g. kBET/iLISI-only requests -- NMI/ARI's Leiden/KMeans
    clustering search is the slow part of benchmark(), and skipping it when only
    kBET/iLISI are wanted saves most of the runtime.
    """
    if adata.obs[bk].nunique() == 1:
        return None
    # Benchmarker's own constructor defaults these to real BioConservation()/
    # BatchCorrection() instances only when the kwarg is left UNSET -- passing
    # None explicitly bypasses that and trips its "either batch or bio metrics
    # must be defined" guard when both are None. Only forward non-None values
    # so the no-args call site keeps getting Benchmarker's real full-battery
    # defaults, unchanged from before this function took these two params.
    extra_kwargs = {}
    if bio_conservation_metrics is not None:
        extra_kwargs['bio_conservation_metrics'] = bio_conservation_metrics
    if batch_correction_metrics is not None:
        extra_kwargs['batch_correction_metrics'] = batch_correction_metrics
    bm = Benchmarker(
        adata=adata,
        batch_key=bk,
        label_key=lk,
        embedding_obsm_keys=obsm_keys,  # evaluate the PCA space
        **extra_kwargs,
    )
    bm.benchmark()  # runs neighbors, clustering, and metrics
    results = bm.get_results(min_max_scale=False)  # returns a tidy DataFrame
    results = results.drop(index="Metric Type")
    return results
    # save_path = f"{res_dir}/{ds}/"
    # return save_append(results, save_path, "scib", name_postfix=name_postfix)


def get_stage1_latent(ds_id, cvae_epochs=50, batch_size=1024):
    """Load the existing Stage-1 (scPoli pretrain) checkpoint for ds_id and encode
    every cell with it -- no Stage-2 (prototype/community) training happens here.

    Returns:
        t:  the SCProtoTrainer, with t.model holding ONLY Stage-1 weights and
            t.train_ds.adata the preprocessed AnnData used for pretraining.
        ad: t.train_ds.adata (kept as a separate name for clarity below).
        z1: (N, d) numpy array -- the Stage-1-only latent for every cell in ad,
            in the same row order as ad.
    """
    from interpretable_ssl.experiments.tasks import get_trainer

    t = get_trainer(
        experiment_name='stage1_latent_extract',
        cvae_epochs=cvae_epochs,
        dataset_id=ds_id,
        l2norm=1,
        assignment_metric='dotp',
        batch_size=batch_size,
        affinity_type='arbf',
    )
    t.load_pretrain_checkpoint()  # raises FileNotFoundError if not pretrained yet

    ad = t.train_ds.adata
    with torch.no_grad():
        z1 = t.encode_adata(ad, t.model, z_idx=1).cpu().numpy()

    print(f"[{ds_id}] Stage-1 latent: {z1.shape[0]} cells x {z1.shape[1]} dims")
    return t, ad, z1


def get_all_embeddings_for_scib(ds_id, cvae_epochs=50, batch_size=1024):
    """Assemble one AnnData with every corrected embedding as an obsm key, for a
    direct scIB-metrics comparison -- rebuttal response to Reviewer e9Ho's Q3:
    "Will you add standard metrics like ARI, NMI, ASW, kBET, iLISI...?"

    Reuses on-disk caches wherever possible: scVI comes from the per-dataset
    '_cache_X_scvi_emb.npy' cache written by
    notebooks/batch_correct_then_cluster_baselines.ipynb's
    run_all_baselines_for_dataset() -- run that at least once first so the cache
    exists; otherwise that embedding is skipped here with a warning rather than
    silently retraining scVI (slow, and this function's job is metric computation,
    not model fitting). scProto's own Stage-2 latent is reloaded from its existing
    trained checkpoint via run_mc_task(..., load_umap=True) -- also not retrained
    here. Harmony is loaded AFTER scProto (see below) at scProto's own latent
    dimension, via get_harmony_embedding_matched_dim -- matching what "the Harmony
    baseline" means everywhere else in this project now (run_correction_method's
    harmony_n_comps). It is NOT read from the old, un-dimensioned
    '_cache_X_harmony_emb.npy' -- that file (if it still exists on disk from before
    this fix) is a stale d=50 embedding and would silently be the wrong comparison
    here if used.

    Returns:
        AnnData with obsm keys among {'X_pca', 'X_stage1z', 'X_harmony', 'X_scvi',
        'X_scproto'} -- whichever are available -- ready to pass to get_scib().
    """
    from interpretable_ssl.experiments.tasks import run_mc_task, LAMBDA_PROTO_UMAP_PRECON
    from interpretable_ssl.datasets.dataset_configs import DATASETS

    t1, ad, z1 = get_stage1_latent(ds_id, cvae_epochs=cvae_epochs, batch_size=batch_size)
    ad.obsm['X_stage1z'] = z1
    if 'X_pca' not in ad.obsm:
        sc.pp.pca(ad, n_comps=50)

    emb_path = os.path.join(get_dataset_model_dir(ds_id), '_cache_X_scvi_emb.npy')
    if os.path.exists(emb_path):
        ad.obsm['X_scvi'] = np.load(emb_path)
    else:
        print(f"[{ds_id}] no cached embedding at {emb_path} -- run "
              f"batch_correct_then_cluster_baselines.ipynb's "
              f"run_all_baselines_for_dataset('{ds_id}') first to populate it. "
              f"Skipping 'scvi' in the scIB comparison for now.")

    t2, _, _ = run_mc_task(
        ds_id,
        cvae_epochs=cvae_epochs,
        train_epochs=50,
        eval_freq=3,
        patience=6,
        batch_size=batch_size,
        umap_steps_per_epoch=500,
        lambda_config=LAMBDA_PROTO_UMAP_PRECON | {'nassoc_agg': 'max'},
        affinity_type='arbf',
        load_umap=True,
        skip_eval=True,  # only need the loaded model to encode -- load_umap=True alone
                         # still runs the full eval+save pipeline unconditionally
                         # (that's what load_umap actually skips is training, not
                         # eval/save), which would otherwise re-run eval_metacell_quality/
                         # eval_task2/3 and overwrite clusters.npz/metacells.h5ad in this
                         # run's own directory -- including the canonical scProto run
                         # directories -- every time this function is called.
    )
    with torch.no_grad():
        z_scproto = t2.encode_adata(t2.train_ds.adata, t2.model, z_idx=1).cpu().numpy()
    scproto_obs_names = t2.train_ds.adata.obs_names

    missing = set(ad.obs_names) - set(scproto_obs_names)
    if missing:
        print(f"[{ds_id}] WARNING: {len(missing)} cells in the Stage-1 adata are not "
              f"present in scProto's own adata -- obs_names don't fully align between "
              f"the two loading paths. Skipping 'X_scproto' in the scIB comparison for "
              f"this dataset rather than risk silently misaligned rows.")
    else:
        z_scproto_df = pd.DataFrame(z_scproto, index=scproto_obs_names)
        ad.obsm['X_scproto'] = z_scproto_df.reindex(ad.obs_names).values

        # Harmony at scProto's own latent dimension -- see docstring. Needs
        # ad.obsm['X_scproto'] (just set above) to know that dimension, and needs
        # batch_key to run the correction at all.
        bk = DATASETS.get(ds_id, {}).get('batch_key')
        if bk is not None:
            from interpretable_ssl.evaluation.batch_correct_baselines import (
                get_harmony_embedding_matched_dim,
            )
            n_comps = ad.obsm['X_scproto'].shape[1]
            ad.obsm['X_harmony'] = get_harmony_embedding_matched_dim(ad, bk, n_comps, ds_id=ds_id)

    return ad


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
    try:
        scgraph = scGraph(
            adata_path=tmp_path, batch_key=batch_key, label_key=label_key, **kwargs
        )
        return scgraph.main(_obsm_list=obsm_keys)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    # save_path = f"{res_dir}/{dataset_name}/"
    # return save_append(scgr_res, save_path, "scgraph", name_postfix=name_postfix)


def get_mc_scg(ad, mc_adata, bk, lk, _obsm_list):
    # thres_batch=100, thres_celltype=10
    scg, tmp_path = get_scg_obj(
        ad, bk, lk, trim_rate=0.05, thres_batch=50, thres_celltype=5
    )
    scg.preprocess()
    scg.process_batches()
    scg.calculate_consensus()
    scg.adata = mc_adata
    res_df = pd.DataFrame(columns=[
        "Rank-PCA", "Corr-PCA", "Corr-Weighted",
        "Rank-PCA-std", "Corr-PCA-std", "Corr-Weighted-std",
    ])

    # self.concensus_df_pca.to_csv("concensus_df_pca_%s.csv"%self.trim_rate)
    # exit()
    for _obsm in _obsm_list:
        adata_df = scg.adata_concensus(_obsm)
        rank_per_gene = scg.rank_diff(adata_df, scg.concensus_df_pca)
        corr_per_gene = scg.corr_diff(adata_df, scg.concensus_df_pca)
        corrw_per_gene = scg.corrw_diff(adata_df, scg.concensus_df_pca)
        _row_df = pd.DataFrame(
            {
                # per-gene correlation/rank agreement, averaged: mean is the existing
                # headline number, std is the spread across genes (previously discarded)
                "Rank-PCA": rank_per_gene.mean().values,
                "Corr-PCA": corr_per_gene.mean().values,
                "Corr-Weighted": corrw_per_gene.mean().values,
                "Rank-PCA-std": rank_per_gene.std().values,
                "Corr-PCA-std": corr_per_gene.std().values,
                "Corr-Weighted-std": corrw_per_gene.std().values,
            },
            index=[_obsm],
        )
        res_df = pd.concat([res_df, _row_df], axis=0, sort=False)
    # # suppose your original df is named res_df
    # flat_df = res_df.stack().to_frame().T  # stack rows, then transpose back to 1 row
    # flat_df.columns = [f"{row}_{col}" for row, col in flat_df.columns]
    # flat_df.reset_index(drop=True, inplace=True)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
        print("Deleted:", tmp_path)
    else:
        print(f"could not remove {tmp_path}")
    coverage = mc_adata.obs[lk].nunique() / ad.obs[lk].nunique()
    res_df["covarage"] = coverage
    return res_df


def get_scg_obj(adata, bk, lk, **kwargs):
    tmp_path = f"tmp_{uuid.uuid4().hex[:8]}.h5ad"
    adata.write(tmp_path)
    scg = scGraph(adata_path=tmp_path, batch_key=bk, label_key=lk, **kwargs)
    return scg, tmp_path


def get_metrics(adata, emb_keys, bk, lk, **scgraph_kwargs):
    scg = get_scgraph(adata, emb_keys, bk, lk, **scgraph_kwargs)
    scb = get_scib(adata, emb_keys, bk, lk)
    return scg, scb


def save_metrics(adata, emb_keys, dataset, bk, lk, **scgraph_kwargs):
    scg_m, scib_m = get_metrics(adata, emb_keys, bk, lk, **scgraph_kwargs)
    baselines_dir = os.path.join(MODEL_DIR, dataset, "baselines/")
    scib_m = save_append(scib_m, baselines_dir, "scib")
    scg_m = save_append(scg_m, baselines_dir, "scgraph")
    return scib_m, scg_m


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


def add_scvi_emb(adata, query_stu, bk, pt_epochs=None, ft_epochs=None, gene_likelihood="zinb"):
    """Trains scVI (reference-then-query-finetune, matching this codebase's own
    Stage-1 pretrain-then-finetune split) and returns its latent representation.

    gene_likelihood: passed straight through to scvi.model.SCVI. Default 'zinb'
    (scvi-tools' own default) is what every existing caller of this function still
    gets -- unchanged behavior. Pass 'normal' for a Gaussian reconstruction
    likelihood (Normal NLL with learned per-gene variance) instead of the
    ZINB/NB/Poisson count-likelihood family -- still scVI's real architecture
    (same encoder, same library-size-scaled decoder mean, same batch conditioning),
    just a different, also-scvi-tools-native noise model on the SAME raw-count
    target (adata.layers['counts'], set below regardless of gene_likelihood --
    scVI's decoder always reconstructs against raw counts, scaled internally by
    library size; there is no way to point stock scVI at log-normalized data
    without also bypassing its library-size machinery, see batch_correct_baselines
    module docstring / rebuttal notebook comments for the full reasoning on why
    this is a meaningfully closer, but not perfect, match to an MSE-on-lognorm
    objective compared to the ZINB default).
    """
    adata.X = adata.layers.get("counts", adata.X)
    print(f"adata.X max: {adata.X.max()}")
    d = get_defaults()
    pt_epochs = pt_epochs or (d["pretraining_epochs"] + d["cvae_epochs"])
    ft_epochs = ft_epochs or d["ft_epochs"]
    ref = adata[~adata.obs[bk].isin(query_stu)].copy()
    train_ind, val_ind = train_test_split(
        range(len(ref)), test_size=0.1, random_state=42
    )
    train_ad = ref[train_ind].copy()

    print(f"training scvi (gene_likelihood={gene_likelihood}) with ds size: "
          f"{len(train_ad)} and {pt_epochs}, {ft_epochs}")
    # 1) Setup AnnData for scVI
    #    No need for common genes step since ref_adata comes from adata
    scvi.model.SCVI.setup_anndata(train_ad, batch_key=bk)

    # 2) Train model on reference
    model = scvi.model.SCVI(train_ad, n_latent=8, gene_likelihood=gene_likelihood)
    model.train(max_epochs=pt_epochs)

    # Adapt model to whole adata
    query_model = scvi.model.SCVI.load_query_data(adata, model)
    query_model.train(ft_epochs)
    key = "X_scvi_gauss" if gene_likelihood == "normal" else "X_scvi"
    key += (
        f"_pt{pt_epochs}"
        if pt_epochs != (d["pretraining_epochs"] + d["cvae_epochs"])
        else ""
    )
    key += f"_ft{ft_epochs}" if ft_epochs != d["ft_epochs"] else ""
    key += f'_uc_v{d["version"]}'
    return query_model.get_latent_representation(adata), key


def get_scvi_metrics(adata, query_stu, bk, lk, dataset):
    adata, key = add_scvi_emb(adata, query_stu, bk)
    return save_metrics(adata, [key], dataset, bk, lk)


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
    adata, bk, lk, dataset, pca_key="X_pca", **scgraph_kwargs
):
    adata = add_pca_harmoney(adata, bk, pca_key)
    return save_metrics(
        adata,
        [pca_key, f"{pca_key}_harmoney"],
        dataset,
        bk,
        lk,
        **scgraph_kwargs,
    )


# def get_seacell_metrics(SEACell_ad, adata, bk, lk, ds, postfix=None, **scgraph_kwargs):
#     # Normalize cells, log transform and compute highly variable genes
#     sc.pp.normalize_per_cell(SEACell_ad)
#     sc.pp.log1p(SEACell_ad)
#     # sc.pp.highly_variable_genes(ad, n_top_genes=1500)

#     # SEACell_ad = agg_obs(SEACell_ad, adata, bk)
#     # SEACell_ad = agg_obs(SEACell_ad, adata, lk)

#     return save_pca_harmoney_metrics(
#         SEACell_ad, bk, lk, ds, "seacell_pca", postfix=postfix, **scgraph_kwargs
#     )


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


def compute_similarity(z, proto):
    z = z.detach()
    proto = proto.detach()
    Z = torch.cat([z, proto], dim=0)

    k = 50
    d = torch.cdist(Z, Z)
    knn_dist = d.topk(k + 1, largest=False).values[:, 1:]
    sigma = knn_dist.median(dim=1).values

    sigma_z = sigma[: z.shape[0]].unsqueeze(1)  # (N, 1)
    sigma_p = sigma[z.shape[0] :].unsqueeze(0)  # (1, P)

    z_d = torch.cdist(z, proto)  # (N, P)

    sim = torch.exp(-2.0 * (z_d**2) / (sigma_z * sigma_p))
    return sim


def get_scproto_mc_adata(
    t,
    adata,
    bk,
    lk,
    epsilon,
    use_mean=False,
    use_max=True,
    model=None,
    similarity="normal",
    pl_version = 3,
):
    import torch.nn as nn

    if model is None:
        model = t.load_model()
    protos = model.get_prototypes().detach()

    if use_mean:
        old_emb = model.scpoli_cvae.embeddings[0]  # nn.Embedding(14, 10, max_norm=1.0)
        old_w = old_emb.weight.detach()
        mean_w = old_w.mean(dim=0, keepdim=True)  # [1, emb_dim]
        new_emb = nn.Embedding(
            old_w.shape[0] + 1, old_w.shape[1], max_norm=old_emb.max_norm
        )
        new_emb.weight.data[: old_w.shape[0]] = old_w
        new_emb.weight.data[old_w.shape[0]] = mean_w
        model.scpoli_cvae.embeddings[0] = new_emb
        # number of embeddings in the layer
        last_idx = model.scpoli_cvae.embeddings[0].num_embeddings - 1
        batch = np.full((protos.shape[0], 1), last_idx)
        device = torch.device("cuda")
        model.scpoli_cvae.embeddings[0] = model.scpoli_cvae.embeddings[0].to(device)
    elif use_max:
        max_bid = adata.obs[bk].value_counts().idxmax()
        max_bidx = t.train_ds.condition_encoders[bk][max_bid]
        batch = np.full((protos.shape[0], 1), max_bidx)
    else:
        batch = np.zeros((protos.shape[0], 1))
    batch = torch.as_tensor(batch, dtype=torch.long, device="cuda")

    if t.recon_loss == "nb":
        sf = np.ravel(adata.layers.get("counts").sum(1))
        sf = sf.mean()
        # but this is bad, maybe for each proto, use avg sizefactor of assigned cells
        # but within the batch which yu are decoding
        print("decoding protos using avg sizefactor: ", sf)
        sizefactor = np.full((protos.shape[0],), sf)
        sizefactor = torch.as_tensor(sizefactor, dtype=torch.float, device="cuda")
        metacells = model.nb_decode(protos, batch, sizefactor)
    else:
        metacells = model.decode(protos, batch)
    z_vae = t.encode_adata(adata, model, z_idx=1)

    # sample_proto_sim = t.get_proto_assignments(z_vae, model)
    if similarity == "normal" or similarity == "v12":
        sample_proto_sim = (
            -torch.cdist(z_vae.detach(), protos.detach(), p=2).cpu().numpy()
        )
    else:
        sample_proto_sim = compute_similarity(z_vae, protos).detach().cpu().numpy()
    if pl_version == 2:
        proto_labels = extract_proto_labels(
            adata, sample_proto_sim, [bk, lk], epsilon=epsilon, similarity=similarity
        )
    else:
        proto_labels = soft_proto_labels_balanced(sample_proto_sim, adata.obs, [bk, lk, 'niches_2D'], epsilon)
    metacells_adata = generate_metacell_adata(metacells, proto_labels)
    if metacells_adata.X.max() > 50:
        metacells_adata = metacells_adata.copy()
        sc.pp.normalize_total(metacells_adata, target_sum=1e4)
        sc.pp.log1p(metacells_adata)
    sc.tl.pca(metacells_adata)
    metacells_adata.obsm[f"{t.get_model_name()}_mc_pca"] = metacells_adata.obsm["X_pca"]
    metacells_adata.var_names = adata.var_names
    metacells_adata.obsm[f"{t.get_model_name()}_mc_proto"] = protos.cpu().numpy()
    return metacells_adata, sample_proto_sim, z_vae


def get_scproto_metacell_metrics(
    t, adata, ds, bk, lk, name_postfix=None, **scgraph_kwargs
):
    metacells_adata, sim, z = get_scproto_mc_adata(t, adata, bk, lk)
    return save_metrics(
        metacells_adata, ["scProto_mc_pca"], ds, bk, lk, name_postfix, **scgraph_kwargs
    )


def load_seacell(ds_id, normalize=True, build_kernel_on="X_pca", num_prototypes=None):
    from interpretable_ssl.evaluation.metric_helpers.metacell_metrics import _reconstruct_from_delta

    seacell_dir = get_seacell_model_dir(ds_id, build_kernel_on, num_prototypes=num_prototypes)
    ad = sc.read_h5ad(os.path.join(seacell_dir, "seacell_sc.h5ad"))
    if ad.uns.get("_seacell_delta"):
        # this tag's file is a delta -- merge it onto the dataset's shared
        # base to reconstruct the full per-cell ad (see save_seacell)
        ad = _reconstruct_from_delta(ad, seacell_dir)
    mc_ad = sc.read_h5ad(os.path.join(seacell_dir, "seacell_agg.h5ad"))
    if normalize and mc_ad.X.max() > 20:
        sc.pp.normalize_total(mc_ad, target_sum=1e4)
        sc.pp.log1p(mc_ad)
    return ad, mc_ad


def get_metacell_metrics(ad, mc_ad, obsm_keys, bk, lk, save_path=None):
    # SCIB
    try:
        scb = get_scib(mc_ad, obsm_keys, bk, lk)
    except Exception as e:
        print("SCIB metric failed:", e)
        scb = None

    # SCG
    try:
        scg = get_mc_scg(ad, mc_ad, bk, lk, obsm_keys)
    except Exception as e:
        print("SCG metric failed:", e)
        scg = None
    if save_path is not None:
        scg.to_csv(save_path + "/scgraph.csv")
        if scb is not None:
            scb.to_csv(save_path + "/scib.csv")
    return scg, scb
