"""[D60] scProto's assignment at EPOCH 0, LEIDEN init -- stage-1 encoder + seeded waypoint
prototypes, before ANY stage-2 training.

THE QUESTION THIS ANSWERS
-------------------------
The stage-1 latent supports cross-batch agreement ~0.97 (Leiden on it). scProto
after 10 stage-2 epochs sits at 0.855. Two readings:

  (a) stage 2 DESTROYS alignment that the init already had
      -> epoch-0 cross_agree is high (~0.95+), falls during training
      -> the fix is a PRESERVATION term (regularize drift away from init),
         which is the user's reframe
  (b) scProto's assignment NEVER had it -- argmax-over-prototypes on the
      stage-1 latent is already cross-batch inconsistent at epoch 0
      -> epoch-0 cross_agree is already ~0.86
      -> the fix must CREATE alignment in the assignment, and no preservation
         term can help

The Leiden 0.9665 cannot distinguish these because it is a different
partitioning algorithm. This is the clean measurement: the exact initial state
both d59 stage-2 arms started from (same pretrain checkpoint, same SEEDED
waypoint init -- D57 made this reproducible, so what we compute here IS what
the runs saw at step 0).

CPU only, one forward pass. Saves clusters.npz (argmax labels, all 33,506
cells) + latent into files-compatible run dir d59_epoch0.

Run:
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/dump_d59_epoch0.py
"""
import modal

app = modal.App("scproto-d60-epoch0-leid")

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
        for name in ["lung", "lung_fetal_donor", "lung_fetal_organoid", "brain",
                     "breast", "heart", "eye", "gut_fetal", "skin", "COVID",
                     "pancreas"]
    ])
)

volume = modal.Volume.from_name("scproto-data")

DS = "pbmc-immune"
BATCH_KEY, LABEL_KEY, K = "study", "final_annotation", 300
TAG = "arbf_ctb20_seed0"


@app.function(image=image, gpu="T4", timeout=7200,
              volumes={"/vol": volume})
def dump():
    import os
    import sys
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
        def _rn(e, *a, **kw):
            return None
    except TypeError:
        pass
    import anndata
    anndata.settings.allow_write_nullable_strings = True

    import numpy as np
    import pandas as pd
    import torch
    from interpretable_ssl.experiments.tasks import (
        get_trainer, _resolve_ds_id, _resolve_num_prototypes)
    from interpretable_ssl.evaluation.batch_correct_baselines import get_preprocessed_adata

    train_h5 = f"/vol/data/node_heldout/{DS}_{TAG}_train.h5ad"
    resolved = _resolve_ds_id(train_h5, batch_key=BATCH_KEY, label_key=LABEL_KEY,
                              niche_key=None, num_prototypes=K)
    # [D60] LEIDEN prototype init (prot_init="leid", D22: prototypes = centroids
    # of a Leiden-CPM partition of the train graph). This is the SAME-INIT
    # epoch-0 baseline for arm d60_leid_noc -- its bars must be measured and
    # written into the D60 ADR BEFORE that arm launches (pre-registration
    # ordering). T4 because build_model hard-requires CUDA (scproto.py:779).
    t = get_trainer(experiment_name="d60_epoch0_leid", cvae_epochs=50,
                    prot_init="leid",
                    dataset_id=resolved, l2norm=1, assignment_metric="dotp",
                    batch_size=1024, affinity_type="arbf", seed=31,
                    num_prototypes=_resolve_num_prototypes(K, resolved))
    t.load_pretrain_checkpoint()
    t.init_prototypes()                      # seeded waypoint on the TRAIN graph
    t.model.eval()

    ad_full = get_preprocessed_adata(DS)
    assert t.check_conditions_compatible(t.model, ad_full)
    with torch.no_grad():
        z = t.encode_adata(ad_full, t.model, z_idx=1)
        z = z.cpu().numpy() if hasattr(z, "cpu") else np.asarray(z)
        lab = t.model.prototypes(
            torch.as_tensor(z, device=t.device)).argmax(dim=1).cpu().numpy()

    d = np.load(f"/vol/data/node_heldout/{DS}_{TAG}_split.npz", allow_pickle=True)
    test_ids = [str(x) for x in d["test_ids"]]
    test_mask = np.zeros(len(ad_full), bool)
    test_mask[ad_full.obs_names.get_indexer(test_ids)] = True

    out = f"/vol/models/{DS}/d60_epoch0_leid"
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "clusters.npz"), labels=lab.astype(int))
    np.savez(os.path.join(out, "clusters_full.npz"),
             labels=lab.astype(int), test_mask=test_mask)
    np.save(os.path.join(out, "latent_z_full.npy"), z.astype(np.float32))
    volume.commit()
    n_used = len(np.unique(lab))
    print(f"[epoch0] argmax assignment over {len(lab):,} cells: "
          f"{n_used}/{K} prototypes used -> {out}")
    return {"n_used": int(n_used), "K": K, "out": out}


@app.local_entrypoint()
def main():
    print("[D59] epoch-0 assignment: stage-1 encoder + seeded waypoint prototypes")
    print(dump.remote())
