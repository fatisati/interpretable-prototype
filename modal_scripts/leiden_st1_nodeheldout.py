"""LEAK-FREE Leiden+scPoli-Stage1 control for the node-holdout protocol.

The existing 'Leiden + scPoli-Stage1' baseline row is unfair on node-holdout:
its encoder AND its Leiden clustering saw all 16,382 cells, including the 3,277
test cells. This job builds the honest counterpart, using the SAME stage-1
checkpoint scProto's own NH runs start from (a scPoli CVAE pretrained on ONLY
the 13,105 train cells -- already on the volume, no training here):

  1. encode train cells with the frozen stage-1 encoder
  2. Leiden-at-K220 on the train-cell latent graph (build_latent_affinity,
     leiden_resolution_search -- the exact code path of the unfair baseline)
  3. encode the 3,277 held-out cells with the SAME frozen encoder
     (check_conditions_compatible asserted -- no silent query retraining)
  4. place each held-out cell by 15-NN cosine majority vote among train cells

Writes /vol/models/pancreas/leiden_st1_nh_K220/clusters.npz
  {labels: (16382,) in full-adata order, test_mask}.

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/leiden_st1_nodeheldout.py
"""
import modal

app = modal.App("scproto-leiden-st1-nh")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "numpy", "scipy", "pandas", "anndata", "scanpy",
        "torch", "scarches", "wandb", "tqdm", "statsmodels", "diptest",
        "palantir", "faiss-cpu", "scikit-learn", "matplotlib", "scib-metrics",
        "leidenalg", "python-igraph", "ipython",
    )
    .pip_install("git+https://github.com/dpeerlab/SEACells.git",
                 extra_options="--no-deps")
    .run_commands(*[
        f"mkdir -p /content/drive/MyDrive/codes/Islander/meta/{name} && "
        f"echo '{{}}' > /content/drive/MyDrive/codes/Islander/meta/{name}/cell2cat.json && "
        f"echo '{{}}' > /content/drive/MyDrive/codes/Islander/meta/{name}/batch2cat.json"
        for name in [
            "lung", "lung_fetal_donor", "lung_fetal_organoid", "brain", "breast",
            "heart", "eye", "gut_fetal", "skin", "COVID", "pancreas",
        ]
    ])
)

volume = modal.Volume.from_name("scproto-data")

TAG = "arbf_nodeheldout20_seed0"


@app.function(image=image, gpu="T4", timeout=7200, volumes={"/vol": volume})
def run():
    import os
    import sys
    import time

    os.environ["WANDB_MODE"] = "disabled"
    os.environ["HOME_DIR"] = "/vol/"
    os.environ["DATA_DIR"] = "/vol/data/"
    os.environ["MODEL_DIR"] = "/vol/models/"
    os.environ["CODE_DIR"] = "/vol/code/"
    sys.path.insert(0, "/vol/code")
    os.chdir("/vol/code")

    import h5py
    from anndata._io.specs.registry import _REGISTRY, IOSpec
    try:
        @_REGISTRY.register_read(h5py.Dataset, IOSpec("null", "0.1.0"))
        def _read_null(elem, *a, **kw):
            return None
    except TypeError:
        pass
    import anndata
    anndata.settings.allow_write_nullable_strings = True

    import numpy as np
    import pandas as pd
    import torch
    from sklearn.neighbors import NearestNeighbors

    from interpretable_ssl.experiments.tasks import _resolve_ds_id, get_trainer
    from interpretable_ssl.evaluation.batch_correct_baselines import (
        get_preprocessed_adata, build_latent_affinity, leiden_resolution_search)

    t0 = time.time()
    train_h5 = f"/vol/data/node_heldout/pancreas_{TAG}_train.h5ad"
    ds_id = _resolve_ds_id(train_h5, batch_key="tech", label_key="celltype",
                           niche_key=None, num_prototypes=220)
    t = get_trainer(experiment_name="stage1_nh_extract", cvae_epochs=50,
                    dataset_id=ds_id, l2norm=1, assignment_metric="dotp",
                    batch_size=1024, affinity_type="arbf")
    t.load_pretrain_checkpoint()   # raises if the NH pretrain is missing
    ad_tr = t.train_ds.adata
    with torch.no_grad():
        z_tr = t.encode_adata(ad_tr, t.model, z_idx=1).cpu().numpy()
    print(f"[st1-nh] encoded {len(ad_tr)} train cells ({time.time()-t0:.0f}s)")

    ad_full = get_preprocessed_adata("pancreas")
    d = np.load(f"/vol/data/node_heldout/pancreas_{TAG}_split.npz",
                allow_pickle=True)
    test_ids = [str(x) for x in d["test_ids"]]
    ad_test = ad_full[ad_full.obs_names.get_indexer(test_ids)].copy()
    assert t.check_conditions_compatible(t.model, ad_test), (
        "test cells carry unseen batch/condition values -- encode_adata would "
        "silently retrain (leak). Refusing.")
    with torch.no_grad():
        z_te = t.encode_adata(ad_test, t.model, z_idx=1).cpu().numpy()
    print(f"[st1-nh] frozen-pass encoded {len(ad_test)} held-out cells")

    # Leiden on the TRAIN latent only -- identical code path to the full-data
    # baseline (build_latent_affinity + leiden_resolution_search).
    ad_tr.obsm["X_st1nh"] = z_tr
    aff = build_latent_affinity(ad_tr, "X_st1nh")
    lab_tr, _, nk = leiden_resolution_search(aff, 220)
    print(f"[st1-nh] leiden on train latent: K={nk} ({time.time()-t0:.0f}s)")

    # Place held-out cells: 15-NN cosine majority among train cells.
    zn_tr = z_tr / (np.linalg.norm(z_tr, axis=1, keepdims=True) + 1e-12)
    zn_te = z_te / (np.linalg.norm(z_te, axis=1, keepdims=True) + 1e-12)
    nn = NearestNeighbors(n_neighbors=15, metric="cosine").fit(zn_tr)
    _, idx = nn.kneighbors(zn_te)
    lab_te = np.array([np.bincount(lab_tr[row]).argmax() for row in idx])

    s_tr = pd.Series(lab_tr, index=ad_tr.obs_names)
    s_te = pd.Series(lab_te, index=ad_test.obs_names)
    combined = pd.concat([s_tr, s_te]).reindex(ad_full.obs_names)
    assert not combined.isna().any()
    labels = combined.values.astype(int)
    test_mask = np.zeros(len(ad_full), bool)
    test_mask[ad_full.obs_names.get_indexer(test_ids)] = True

    out_dir = "/vol/models/pancreas/leiden_st1_nh_K220"
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "clusters.npz"),
             labels=labels, test_mask=test_mask)
    volume.commit()
    print(f"[st1-nh] saved {out_dir}/clusters.npz  K={nk}  "
          f"({time.time()-t0:.0f}s total)")
    return nk


@app.local_entrypoint()
def main():
    print(run.remote())
