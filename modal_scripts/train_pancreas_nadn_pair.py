"""scProto on Pancreas, K=220, affinity_type='arbf' (the same config that produced the
published main-table numbers), with the D1 nassoc fix turned on:

  - D1 (diagonal): nassoc_diag_norm='pair' -- edges_inside/possible_pairs (density), not
    edges_inside/volume (containment). Containment is additive under merging, so it
    always prefers merging; density is superadditive in possible_pairs, so it doesn't.

This is the first check of whether the collapse problem found on the spatial (s28nsc)
dataset with the default loss also affects the non-spatial main-table datasets. Pancreas
only, one dataset, per CLAUDE.md §6 -- Lung/PBMC-immune follow only if this one looks
healthy and holds up against baselines.

Pretrain checkpoint is reused, not recomputed: `get_pretrain_dump_path` keys on
(dataset_id, cvae_epochs, batch_size, latent_dims, l2norm, model_type, beta,
num_prototypes), none of which differ from the canonical run, so the existing
`pretrain_ds-pancreas_cvae_e50` checkpoint (uploaded to the volume alongside this script)
is picked up automatically -- no `cvae_epochs=50` pretrain-from-scratch cost.

One-variable-differs control: the existing canonical run
`proto_umap_ds-panc_NP220_prtInit-wayp_aff-arbf_lprec0.01_usim-prot_cvae_e50_ecal1_lpu0.1_pum-ema_lna1_nagg-max_upm-dotp_v31`
(same LAMBDA_PROTO_UMAP_PRECON base, `nassoc_diag_norm` left at its default 'volume') is
the old-loss arm to diff against -- any change is attributable to D1.

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/train_pancreas_nadn_pair.py
"""
import time
import modal

app = modal.App("scproto-repro")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "numpy", "scipy", "pandas", "anndata", "scanpy",
        "torch", "scarches", "wandb", "tqdm", "statsmodels", "diptest",
        "palantir", "faiss-cpu", "scikit-learn", "matplotlib", "scib-metrics",
        "ipython",
    )
    .pip_install("git+https://github.com/dpeerlab/SEACells.git", extra_options="--no-deps")
    # Data_Handler.py hard-codes an absolute Colab path and eagerly reads these JSONs at
    # import time for datasets we never touch (lung/brain/etc.) -- stub them so importing
    # the scProto trainer does not crash on an unrelated legacy module.
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


@app.function(image=image, gpu="T4", timeout=3600, volumes={"/vol": volume})
def train_pancreas_nadn_pair():
    import os
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["HOME_DIR"] = "/vol/"
    os.environ["DATA_DIR"] = "/vol/data/"
    os.environ["MODEL_DIR"] = "/vol/models/"
    os.environ["CODE_DIR"] = "/vol/code/"

    import sys
    sys.path.insert(0, "/vol/code")
    os.chdir("/vol/code")  # graph_generator's get_affinity_path uses './graphs' by default

    # Same anndata compatibility patch used locally (older anndata wrote None as a
    # 'null'-encoded HDF5 dataset; this anndata build has no reader registered for it).
    import h5py
    from anndata._io.specs.registry import _REGISTRY, IOSpec
    try:
        @_REGISTRY.register_read(h5py.Dataset, IOSpec("null", "0.1.0"))
        def _read_null(elem, _reader):
            return None
    except TypeError:
        pass

    from interpretable_ssl.experiments.tasks import run_mc_task, LAMBDA_PROTO_UMAP_PRECON

    t0 = time.time()
    lc = dict(LAMBDA_PROTO_UMAP_PRECON)
    lc["nassoc_diag_norm"] = "pair"      # D1 -- the only change vs the canonical run

    trainer, metrics, _ = run_mc_task(
        "pancreas",
        cvae_epochs=50, train_epochs=20, eval_freq=2, patience=4,
        batch_size=1024, affinity_type="arbf", num_prototypes=220,
        lambda_config=lc,
        trainer_kwargs={"experiment_name": "proto_pancreas_nadnpair"},
    )
    elapsed = time.time() - t0
    print("=== TRAINING DONE ===")
    print("elapsed seconds:", elapsed)
    print("model name:", trainer.get_model_name())
    print("metrics:", metrics)

    volume.commit()  # persist the new run's output files back to the volume
    return {"elapsed": elapsed, "metrics": metrics, "model_name": trainer.get_model_name()}


@app.local_entrypoint()
def main():
    t0 = time.time()
    result = train_pancreas_nadn_pair.remote()
    print("total wall time incl. cold start (s):", time.time() - t0)
    print(result)
