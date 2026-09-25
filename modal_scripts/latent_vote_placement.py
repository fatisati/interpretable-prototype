"""[D47] Latent 15-NN vote placement for the d46 node-holdout runs.

DECISION, WRITTEN BEFORE RUNNING
================================
The argmax placement rule collapsed scProto's test-cell metrics (aff 0.2532
train -> 0.1591 test) while the scPoli-NH control, placed by a 15-NN latent
vote, dropped only ~6%. A graph-edge vote recovered the metrics but is
invalid (edges from full-data PCA = leak; votes on the scoring graph =
circular) -- user rejected it. This job applies the SAME rule the scPoli-NH
control uses, in scProto's OWN stage-2 latent:

    z = frozen stage-2 encoder(cell)        # trained on the 80% only
    15 nearest TRAIN cells by cosine in z   # model's opinion, not graph edges
    majority of their trained metacells     # train assignments untouched

EXPECTED: cohesion/modularity between the argmax and graph-vote columns;
purity ~0.95 (trustworthy from the quick check); batch entropy RECOVERS
toward >=1.0 (latent is batch-corrected, unlike graph edges).
VERDICT (fill in): [ ] works  [ ] fails  [ ] inconclusive because ___

Writes clusters_latvote.npz {labels, test_mask} into each run dir.
Cost: one T4, no training. Run:
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/latent_vote_placement.py
"""
import modal

app = modal.App("scproto-latvote")

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
BASE = "/vol/models/pancreas_arbf_nodeheldout20_seed0_train"
RUNS = {
    "cohO": (f"{BASE}/d46nh_cohO_pancreas_ds-panc_NP220_prtInit-wayp_aff-arbf_"
             "lumap0.0_usim-prot_cvae_e50_ecal1_pum-ema_nagg-max_lcoh3.12_"
             "gprb25_cglb2_cmini-full_cmhl500.0_cdia1.0_lcha8.95_upm-dotp_v31"),
    "calU": (f"{BASE}/d46nh_calU_pancreas_ds-panc_NP220_prtInit-wayp_aff-arbf_"
             "lumap0.06_usim-prot_cvae_e50_ecal1_pum-ema_nagg-max_lcoh3.12_"
             "gprb25_cglb2_cmini-full_cmhl500.0_cdia1.0_lcha8.95_upm-dotp_v31"),
}


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
        get_preprocessed_adata)

    t0 = time.time()
    train_h5 = f"/vol/data/node_heldout/pancreas_{TAG}_train.h5ad"
    ds_id = _resolve_ds_id(train_h5, batch_key="tech", label_key="celltype",
                           niche_key=None, num_prototypes=220)

    ad_full = get_preprocessed_adata("pancreas")
    d = np.load(f"/vol/data/node_heldout/pancreas_{TAG}_split.npz",
                allow_pickle=True)
    test_ids = [str(x) for x in d["test_ids"]]
    test_pos = ad_full.obs_names.get_indexer(test_ids)
    ad_test = ad_full[test_pos].copy()
    test_mask = np.zeros(len(ad_full), bool)
    test_mask[test_pos] = True

    out = {}
    for tag, run_dir in RUNS.items():
        t = get_trainer(experiment_name=f"latvote_{tag}", cvae_epochs=50,
                        dataset_id=ds_id, l2norm=1, assignment_metric="dotp",
                        batch_size=1024, affinity_type="arbf")
        t.load_pretrain_checkpoint()
        ck = torch.load(os.path.join(run_dir, "umap_checkpoint.pth"),
                        map_location=t.device)
        before = {k: v.detach().float().norm().item()
                  for k, v in t.model.state_dict().items()}
        missing, unexpected = t.model.load_state_dict(
            ck["model_state_dict"], strict=False)
        after = {k: v.detach().float().norm().item()
                 for k, v in t.model.state_dict().items()}
        changed = [k for k in before if abs(before[k] - after[k]) > 1e-8]
        print(f"[{tag}] stage-2 ckpt loaded (epoch {ck.get('epoch')}): "
              f"{len(changed)} tensors changed, missing={missing}, "
              f"unexpected={unexpected}")
        assert changed, f"{tag}: stage-2 weights did not load"

        ad_tr = t.train_ds.adata
        assert t.check_conditions_compatible(t.model, ad_test), (
            "test cells carry unseen conditions -- would retrain, refusing")
        with torch.no_grad():
            z_tr = t.encode_adata(ad_tr, t.model, z_idx=1).cpu().numpy()
            z_te = t.encode_adata(ad_test, t.model, z_idx=1).cpu().numpy()

        prev = np.load(os.path.join(run_dir, "clusters_full.npz"))
        lab_prev, mask_prev = prev["labels"], prev["test_mask"]
        assert (mask_prev == test_mask).all()
        # trained TRAIN assignments, in train-adata order
        s_prev = pd.Series(lab_prev, index=ad_full.obs_names)
        lab_tr = s_prev.reindex(ad_tr.obs_names).values.astype(int)

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
        moved = (labels[test_mask] != lab_prev[test_mask]).mean()

        np.savez(os.path.join(run_dir, "clusters_latvote.npz"),
                 labels=labels, test_mask=test_mask)
        print(f"[{tag}] latent-vote placement saved; {moved:.1%} of test cells "
              f"moved vs argmax ({time.time()-t0:.0f}s)")
        out[tag] = float(moved)
    volume.commit()
    return out


@app.local_entrypoint()
def main():
    print(run.remote())
