"""[D58] Dump each arm's latent z for ALL cells, so test cells can be placed the
same way the baseline places them.

WHY
---
leiden_st1_nodeheldout.py places held-out cells like this:

    z_tr = encode(train);  z_te = encode(test)        # frozen encoder
    nn   = NearestNeighbors(15, metric="cosine").fit(z_tr)
    lab_te = majority( lab_tr[ nn.kneighbors(z_te) ] )

i.e. 15-NN COSINE IN THE LATENT among TRAIN cells, majority of their TRAIN labels.

scProto instead places test cells by `argmax dot(z, proto_k)` -- nearest PROTOTYPE
VECTOR, which uses no neighbour information at all. So the published comparison
differs in the placement rule, not only in the partition. This script makes the
matched comparison possible.

Latent k-NN is also cleaner than voting on the input affinity graph: that graph was
BUILT WITH TEST CELLS PRESENT (the "leaky/circular" concern in the project notes)
and is 92.6% within-batch, so no cross-batch information can flow through it.
Encoding a test cell and comparing it to train embeddings precomputes nothing.

CPU only -- this is a forward pass, no training. ~$0.01 for all arms.
The checkpoint is loaded DIRECTLY into the model, skipping _setup_umap_edges(),
so no edge dataset is built.

Run:
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/dump_d58_latent.py
"""
import modal

app = modal.App("scproto-d58-latent")

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
NH_TAG = "arbf_nodeheldout20_seed0"
MODEL_SUBDIR = f"/vol/models/{DS}_{NH_TAG}_train"

# arm -> its four lambdas, needed only so the trainer/model is built identically
ARMS = {
    "base_s31": (3.794, 6.399, 0.04307, 0.02042, 31),
    "gram0":    (0.0,   6.399, 0.04307, 0.02042, 31),
    "aff0":     (3.794, 0.0,   0.04307, 0.02042, 31),
    "comm0":    (3.794, 6.399, 0.0,     0.02042, 31),
    "recon0":   (3.794, 6.399, 0.04307, 0.0,     31),
}


def _env():
    import os, sys
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
    try:
        import anndata
        anndata.settings.allow_write_nullable_strings = True
    except Exception:
        pass


@app.function(image=image, cpu=4, memory=16384, timeout=3600,
              volumes={"/vol": volume})
def dump(arm: str):
    import glob
    import os
    _env()
    import numpy as np
    import torch
    from interpretable_ssl.experiments.tasks import (
        get_trainer, _resolve_ds_id, _resolve_num_prototypes,
        LAMBDA_PROTO_UMAP_PRECON)
    from interpretable_ssl.evaluation.batch_correct_baselines import get_preprocessed_adata

    gram, aff, comm, recon, seed = ARMS[arm]
    lc = dict(LAMBDA_PROTO_UMAP_PRECON) | {
        "lambda_cohesion": gram, "lambda_cohesion_aff": aff,
        "lambda_umap": comm, "lambda_recon": recon,
        "lambda_nassoc": 0.0, "lambda_modularity": 0.0, "lambda_proto_usage": 0.0,
        "lambda_swav": 0.0, "lambda_kl": 0.0, "lambda_proto_recon": 0.0,
        "cohesion_sim": "gram", "cohesion_form": "perc", "cohesion_row_norm": "l2",
        "cohesion_diag": 1.0, "cohesion_self": 1.0, "cohesion_global": 2,
        "coh_mem_init": "full", "coh_mem_hlife": 0.0, "proto_reinit_every": 0,
        "umap_proto_sg": 1, "grad_probe": 25, "seed": seed,
    }
    train_h5 = f"/vol/data/node_heldout/{DS}_{NH_TAG}_train.h5ad"
    resolved = _resolve_ds_id(train_h5, batch_key=BATCH_KEY, label_key=LABEL_KEY,
                              niche_key=None, num_prototypes=K)
    t = get_trainer(
        experiment_name=f"d58_{arm}", cvae_epochs=50, affinity_type="arbf",
        dataset_id=resolved, l2norm=1, assignment_metric="dotp",
        batch_size=1024, umap_steps_per_epoch=None, umap_similarity="proto",
        freeze_batch_embedding=1, freeze_decoder=0,
        num_prototypes=_resolve_num_prototypes(K, resolved),
        **{k: v for k, v in lc.items()
           if k not in ("umap_similarity", "affinity_type")},
    )
    dump_dir = t.get_dump_path()
    ckpt_p = os.path.join(dump_dir, "umap_checkpoint.pth")
    assert os.path.exists(ckpt_p), f"no umap_checkpoint.pth in {dump_dir}"

    # Load the TRAINED weights directly. Deliberately NOT load_umap_checkpoint(),
    # which would call _setup_umap_edges() and build the whole edge dataset -- we
    # only need the encoder.
    ck = torch.load(ckpt_p, map_location="cpu")
    for attr in ("sim_decoder_trunk", "sim_decoder_out"):
        if hasattr(t.model, attr):
            delattr(t.model, attr)
    t.model.load_state_dict(ck["model_state_dict"])
    t.model.eval()
    print(f"[{arm}] loaded umap_checkpoint (epoch {ck.get('epoch')}) from {dump_dir}")

    ad_full = get_preprocessed_adata(DS)
    assert t.check_conditions_compatible(t.model, ad_full), \
        "full adata has conditions the model never saw"
    with torch.no_grad():
        z = t.encode_adata(ad_full, t.model, z_idx=1)
        z = z.cpu().numpy() if hasattr(z, "cpu") else np.asarray(z)
    protos = t.model.get_prototypes().detach().cpu().numpy()

    out = os.path.join(dump_dir, "latent_z_full.npy")
    np.save(out, z.astype(np.float32))
    np.save(os.path.join(dump_dir, "proto_vectors_final.npy"), protos.astype(np.float32))
    with open(os.path.join(dump_dir, "latent_obs_names.txt"), "w") as f:
        f.write("\n".join(map(str, ad_full.obs_names)))
    volume.commit()
    print(f"[{arm}] z={z.shape} protos={protos.shape} -> {out}")
    return arm, list(z.shape), list(protos.shape)


@app.local_entrypoint()
def main(arms: str = "base_s31,gram0,aff0,comm0,recon0"):
    sel = [a.strip() for a in arms.split(",") if a.strip()]
    bad = [a for a in sel if a not in ARMS]
    assert not bad, f"unknown: {bad}"
    print(f"[D58-latent] dumping z for {len(sel)} arm(s) on CPU: {sel}")
    calls = {a: dump.spawn(a) for a in sel}
    for a, c in calls.items():
        try:
            print("   ", c.get())
        except Exception as e:
            print(f"   [{a}] FAILED: {type(e).__name__}: {e}")
