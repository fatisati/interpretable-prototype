"""scProto on s28nsc, K=200, affinity_type='bk02' (BANKSY lambda=0.2), with BOTH nassoc
fixes from literature/spatial/nassoc_redesign.md turned on:
  - D1 (diagonal):     nassoc_diag_norm='pair'    -- edges_inside/possible_pairs (density),
                        not edges_inside/volume (containment). Containment is additive
                        under merging, so it always prefers merging; density is
                        superadditive in possible_pairs, so it doesn't.
  - D4 (off-diagonal):  nassoc_offdiag_norm='min'  -- edges_between/min(vol_p,vol_q)
                        instead of /sqrt(vol_p*vol_q). The geometric mean is dominated by
                        the larger prototype, so a small prototype leaking into a big one
                        was nearly invisible; billing against the smaller side fixes that.

One-variable-differs control: the existing 'proto_bk02_np200_ema15_d1pair_modal' run
(same everything, nassoc_offdiag_norm left at its default 'sqrt') is the D1-only arm to
diff against -- any change is attributable to D4.

Same BASE_KW as nassoc_density_d1_d2.py / modal_train_bk02.py so results are directly
comparable to the existing bk02/bk08 arms already on the volume (models/s28nsc/).

Run: PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/train_bk02_d1_offdiag_min.py
Cost: ~12-13 min on a T4 (see CLAUDE.md §11) -- a few dollars.
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
def train_bk02_d1_offdiag_min():
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
    lc["nassoc_diag_norm"] = "pair"      # D1
    lc["nassoc_offdiag_norm"] = "min"    # D4 -- the only change vs the existing bk02 D1-only run

    trainer, metrics, _ = run_mc_task(
        "s28nsc",
        cvae_epochs=50, train_epochs=15, eval_freq=5, patience=10,
        batch_size=1024, affinity_type="bk02", num_prototypes=200,
        label_key="celltypes", niche_key="niches_2D",
        skip_metrics=["task2", "task3"],
        lambda_config=lc,
        trainer_kwargs={"experiment_name": "proto_bk02_np200_ema15_d1_offdmin_modal"},
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
    result = train_bk02_d1_offdiag_min.remote()
    print("total wall time incl. cold start (s):", time.time() - t0)
    print(result)
