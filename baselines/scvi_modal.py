"""Run baselines/scvi_baseline.py on a Modal GPU instead of your own machine.

Same code, same arguments -- this file only adds the GPU container around it.
Use this when CPU training would be slow, or to keep your laptop free for the
CPU-only stages (PCA, Leiden, SEACells).

How Modal works here (one dataset = one `modal run` = one fresh GPU container):
  - The *volume* `scproto-data` is a cloud disk. Upload a dataset's inputs once;
    every later run (any seed, any number in parallel) reads them from there.
  - Code (this repo's scvi_baseline/common/frozen/run_record) and the committed
    datasets/frozen/ files are shipped with each run automatically.
  - Each `modal run` starts its own container, so two runs started at the same
    time run in parallel and are billed separately.
  - Results come back to your local --out-dir directly (no `modal volume get`),
    and the run record is written locally, so it has the git commit like a
    local run. config.json also says it ran on Modal (`ran_on`).

One-time setup:
    pip install modal
    modal setup                                  # log in, once per machine
    modal volume create scproto-data             # persistent storage, once

Before the first run on a dataset, upload its two inputs (run prepare.py and
prepare_eval.py locally first). Paths on the volume mirror the repo:
    modal volume put -f scproto-data datasets/processed/lung_hvg.h5ad lung_hvg.h5ad
    modal volume put -f scproto-data runs/rare_cell_preservation/lung/cluster_config.json \\
        runs/rare_cell_preservation/lung/cluster_config.json

Then run (from the repo root, so datasets/frozen/ is found and shipped along;
the h5ad's name must stay <dataset>_hvg.h5ad, which is how the frozen files
are found):
    modal run baselines/scvi_modal.py \\
        --h5ad-path lung_hvg.h5ad --batch-key batch --label-key cell_type \\
        --layer counts --gene-likelihood nb \\
        --out-dir runs/rare_cell_preservation/lung/scvi

Caveat: the Leiden clustering step runs inside the container (Linux). The same
embedding + seed can give different clusters on another machine (task file,
Revisions 2026-11-28), so clusters from Modal and from a laptop are not
bit-identical. To cluster locally instead: evaluation/recluster.py.

Choices, from Modal's docs (checked 2026-11-28):
  - Package versions pinned to the local environment (Modal: "pin package
    versions"); unpinned installs would silently pick up newer scvi-tools.
  - cpu=4: Modal's default is 0.125 cores, which would make the Leiden search
    after training crawl.
  - T4: the cheapest GPU (~$0.59/h); scVI with 10 latents is small.
"""
import json
import sys
from pathlib import Path

import modal

# Imported first so the run record's start time is this command's start time.
# Locally it lives in evaluation/; in the container it is shipped next to us.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from run_record import record_run  # noqa: E402

MOUNT = "/data"
GPU = "T4"

app = modal.App("scvi-baseline")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install(
        "scvi-tools==1.5.0.post1", "torch==2.13.0", "lightning==2.6.6",
        "scanpy==1.12.4", "anndata==0.13.2", "numpy==2.4.6", "pandas==3.0.5",
        "scipy==1.18.0", "scikit-learn==1.9.0", "h5py==3.16.0",
        "leidenalg==0.12.0", "igraph==1.0.0",
    )
    # The committed gene lists + splits (evaluation/frozen.py) travel with the
    # code, so the remote run reads the same files as a local one.
    .env({"SCPROTO_FROZEN_DIR": "/root/frozen"})
    .add_local_python_source("scvi_baseline", "common", "run_record", "frozen")
    .add_local_dir("datasets/frozen", "/root/frozen")
)

volume = modal.Volume.from_name("scproto-data", create_if_missing=True)


@app.function(image=image, gpu=GPU, cpu=4.0, memory=16384, volumes={MOUNT: volume}, timeout=3 * 3600)
def run_on_modal(**kwargs):
    from scvi_baseline import run

    kwargs["h5ad_path"] = f"{MOUNT}/{kwargs['h5ad_path']}"
    kwargs["out_dir"] = f"{MOUNT}/{kwargs['out_dir']}"
    kwargs["accelerator"] = "gpu"
    out_dir = Path(run(**kwargs))
    volume.commit()
    # Send the outputs back so the caller can save them locally.
    return {p.name: p.read_bytes() for p in out_dir.iterdir() if p.is_file()}


@app.local_entrypoint()
def main(
    h5ad_path: str,
    batch_key: str,
    label_key: str,
    out_dir: str,
    layer: str = None,
    test_size: float = 0.2,
    seed: int = 0,
    max_epochs: int = 100,  # same default as scvi_baseline.py (None would mean scvi-tools' 400)
    n_latent: int = 10,
    gene_likelihood: str = "auto",
    no_early_stopping: bool = False,
):
    files = run_on_modal.remote(
        h5ad_path=h5ad_path, batch_key=batch_key, label_key=label_key, out_dir=out_dir,
        layer=layer, test_size=test_size, seed=seed, max_epochs=max_epochs,
        n_latent=n_latent, gene_likelihood=gene_likelihood,
        early_stopping=not no_early_stopping,
    )
    local = Path(out_dir)
    local.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (local / name).write_bytes(data)
    config_path = local / "config.json"
    config = json.loads(config_path.read_text())
    config["ran_on"] = f"modal gpu={GPU}"
    config["h5ad_path"] = f"datasets/processed/{h5ad_path}"  # local equivalent of the volume path
    config_path.write_text(json.dumps(config, indent=2))
    record_run(config_path)
    print(f"done on Modal, results in {local}")
