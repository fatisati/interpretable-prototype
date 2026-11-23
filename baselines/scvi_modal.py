"""Run baselines/scvi_baseline.py on a Modal GPU instead of your own machine.

Same code, same arguments -- this file only adds the GPU container around it.
Use this when a dataset is big enough that CPU training would be slow; for
quick sanity checks, run scvi_baseline.py directly on your own computer.

One-time setup:
    pip install modal
    modal setup                                  # log in, once per machine
    modal volume create scproto-data             # persistent storage, once

Before each run, upload the input file (Modal containers can't see your
local disk):
    modal volume put scproto-data datasets/processed/pancreas_hvg.h5ad pancreas_hvg.h5ad

Then run (from the repo root, so datasets/frozen/ is found and shipped along;
the h5ad's name must stay <dataset>_hvg.h5ad, which is how the frozen files
are found):
    modal run baselines/scvi_modal.py \\
        --h5ad-path pancreas_hvg.h5ad \\
        --batch-key tech --label-key celltype \\
        --out-dir runs/rare_cell_preservation/pancreas/scvi

Paths above are relative to the volume (MOUNT), not your local disk. After
the run, pull results back down:
    modal volume get scproto-data runs/rare_cell_preservation/pancreas/scvi ./runs/rare_cell_preservation/pancreas/scvi
"""
import modal

from scvi_baseline import run

MOUNT = "/data"

app = modal.App("scvi-baseline")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("scvi-tools", "scanpy", "pandas", "scikit-learn", "leidenalg", "igraph")
    # The committed gene lists + splits (evaluation/frozen.py) travel with the
    # code, so the remote run reads the same files as a local one.
    .env({"SCPROTO_FROZEN_DIR": "/root/frozen"})
    .add_local_python_source("scvi_baseline", "common", "run_record", "frozen")
    .add_local_dir("datasets/frozen", "/root/frozen")
)

volume = modal.Volume.from_name("scproto-data", create_if_missing=True)


@app.function(image=image, gpu="T4", volumes={MOUNT: volume}, timeout=3600)
def run_on_modal(**kwargs):
    kwargs["h5ad_path"] = f"{MOUNT}/{kwargs['h5ad_path']}"
    kwargs["out_dir"] = f"{MOUNT}/{kwargs['out_dir']}"
    kwargs["accelerator"] = "gpu"
    out_dir = run(**kwargs)
    volume.commit()
    return str(out_dir)


@app.local_entrypoint()
def main(
    h5ad_path: str,
    batch_key: str,
    label_key: str,
    out_dir: str,
    layer: str = None,
    test_size: float = 0.2,
    seed: int = 0,
    max_epochs: int = None,
    n_latent: int = 10,
):
    result_path = run_on_modal.remote(
        h5ad_path=h5ad_path,
        batch_key=batch_key,
        label_key=label_key,
        out_dir=out_dir,
        layer=layer,
        test_size=test_size,
        seed=seed,
        max_epochs=max_epochs,
        n_latent=n_latent,
    )
    print(f"done on Modal: {result_path}")
    print(f"pull results with: modal volume get scproto-data {out_dir} ./{out_dir}")
