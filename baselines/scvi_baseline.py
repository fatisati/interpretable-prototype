"""scVI baseline adapter for the rare-cell-preservation task.

Thin wrapper around scvi-tools (we call the library, we don't reimplement
it): train scVI on train cells only, embed both splits with the trained
encoder, cluster twice (train-only embedding and the full embedding), and
save everything a metrics script would need -- per-cell latent coordinates,
split, and both sets of cluster labels -- to one file.

Clustering is a separate step (common.cluster_embedding_file) that reads
the saved embedding.csv. This script calls it right after training; to
change clustering later WITHOUT retraining, run evaluation/recluster.py.

Reads the split and the target cluster count from
`evaluation/prepare_eval.py`'s output -- doesn't create either itself. Run
that script first for this dataset; see docs/architecture.md for why.

Works on any h5ad file; pass --batch-key / --label-key for your dataset.
Runs on CPU locally or GPU on Modal with no code change -- see scvi_modal.py.
On an Apple Silicon Mac pass --accelerator mps to train on the Metal GPU:
scvi-tools turns the default 'auto' into CPU there on purpose.

Usage:
    python evaluation/prepare_eval.py \\
        --h5ad datasets/processed/pancreas_hvg.h5ad \\
        --batch-key tech --label-key celltype \\
        --out-dir runs/rare_cell_preservation/pancreas
    python baselines/scvi_baseline.py \\
        --h5ad datasets/processed/pancreas_hvg.h5ad \\
        --batch-key tech --label-key celltype \\
        --out-dir runs/rare_cell_preservation/pancreas/scvi
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import scanpy as sc
import scvi
from lightning.pytorch.callbacks import Callback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))  # shared pipeline code
from run_record import record_run  # first local import: records the script's start time
import frozen
import tracking
from common import cluster_embedding_file, environment_versions, load_cluster_config, load_split, looks_like_raw_counts


class TrackEpochs(Callback):
    """After each epoch: send every logged loss (train/validation ELBO, reconstruction,
    KL) to the live dashboard as a chart point, and update progress + time left
    (see evaluation/tracking.py). A callback, not a replacement logger: swapping scVI's
    own logger would empty model.history, which we use for epochs_trained."""

    def on_train_epoch_end(self, trainer, pl_module):
        tracking.log_metrics(dict(trainer.callback_metrics), step=trainer.current_epoch)
        tracking.progress(trainer.current_epoch + 1, trainer.max_epochs)


def pick_gene_likelihood(adata, layer):
    """scVI's default likelihoods (nb/zinb) assume raw integer counts --
    fall back to a continuous ('normal') likelihood when the data isn't
    really counts, instead of silently fitting the wrong noise model.
    """
    values = adata.layers[layer] if layer else adata.X
    return "nb" if looks_like_raw_counts(values) else "normal"


def run(
    h5ad_path,
    batch_key,
    label_key,
    out_dir,
    layer=None,
    test_size=0.2,
    seed=0,
    max_epochs=100,
    n_latent=10,
    accelerator="auto",
    early_stopping=True,
    gene_likelihood="auto",
):
    tracking.stage("load")
    adata = sc.read_h5ad(h5ad_path)
    # "auto" checks the data; an explicit value overrides, e.g. "nb" on the scIB
    # datasets' non-integer counts layer, which we treat as estimated counts.
    if gene_likelihood == "auto":
        gene_likelihood = pick_gene_likelihood(adata, layer)
    print(f"[scvi] {adata.n_obs:,} cells x {adata.n_vars:,} genes | "
          f"gene_likelihood={gene_likelihood} (checked {'layer=' + layer if layer else 'X'})")

    # Both read-only -- created once by prepare_eval.py, shared by every
    # baseline on this dataset. The split is committed in datasets/frozen/;
    # the cluster count lives one level above this baseline's own out_dir.
    dataset_dir = Path(out_dir).parent
    dataset = frozen.dataset_id(h5ad_path)
    frozen.check_genes(adata, dataset)  # stop if the processed file is out of date
    split_path = frozen.split_path(dataset)
    adata.obs["split"] = load_split(adata, label_key, batch_key, test_size, seed, split_path)
    train_mask = (adata.obs["split"] == "train").to_numpy()
    test_mask = ~train_mask

    # Fail before training (not after) if stage 2's cluster count is missing.
    load_cluster_config(dataset_dir / "cluster_config.json")

    scvi.settings.seed = seed
    train_adata = adata[train_mask].copy()
    scvi.model.SCVI.setup_anndata(train_adata, batch_key=batch_key, layer=layer)
    model = scvi.model.SCVI(train_adata, n_latent=n_latent, gene_likelihood=gene_likelihood)
    # max_epochs=100 by default: scvi-tools' own heuristic picks 400 for a
    # dataset this size (~19k cells), which was never a deliberate choice;
    # 100 is plenty here. early_stopping can still end training sooner
    # (watches validation loss, stops once it stops improving) -- the
    # actual count lands in config.json as epochs_trained.
    tracking.stage("training")
    model.train(max_epochs=max_epochs, accelerator=accelerator, early_stopping=early_stopping,
                callbacks=[TrackEpochs()])
    epochs_trained = len(model.history["elbo_train"])
    device_used = str(model.device)  # what training really ran on, e.g. 'cpu', 'mps:0', 'cuda:0'
    print(f"[scvi] trained {epochs_trained} epochs on {device_used} (--accelerator {accelerator})")

    # get_latent_representation accepts any AnnData with the same genes/obs
    # columns as training data -- it registers and embeds test cells through
    # the trained encoder without needing a separate setup_anndata call.
    tracking.stage("embed")
    adata.obsm["X_scVI"] = model.get_latent_representation(adata)

    result = pd.DataFrame({
        "cell_id": adata.obs_names,
        "split": adata.obs["split"].to_numpy(),
        label_key: adata.obs[label_key].to_numpy(),
        batch_key: adata.obs[batch_key].to_numpy(),
    })
    for dim in range(adata.obsm["X_scVI"].shape[1]):
        result[f"latent_{dim}"] = adata.obsm["X_scVI"][:, dim]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "embedding.csv", index=False)

    config = {
        "h5ad_path": str(h5ad_path),
        "split_path": str(split_path),
        "frozen": frozen.fingerprint(dataset),
        "batch_key": batch_key,
        "label_key": label_key,
        "layer": layer,
        "gene_likelihood": gene_likelihood,
        "test_size": test_size,
        "seed": seed,
        "n_latent": n_latent,
        "max_epochs": max_epochs,
        "epochs_trained": epochs_trained,
        "early_stopping": early_stopping,
        "accelerator": accelerator,
        "device_used": device_used,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "environment": environment_versions(),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[scvi] wrote {out_dir / 'embedding.csv'} (latents)")

    # Adds cluster_train_run / cluster_full_run to embedding.csv and the
    # clustering fields to config.json -- same code recluster.py runs.
    cluster_embedding_file(out_dir)
    return out_dir


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad", required=True, help="path to a processed h5ad file")
    p.add_argument("--batch-key", required=True, help="adata.obs column with batch/study id")
    p.add_argument("--label-key", required=True, help="adata.obs column with cell type label")
    p.add_argument("--out-dir", required=True, help="where to write embedding.csv + config.json")
    p.add_argument("--layer", default=None, help="adata.layers key to use as input (default: X)")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-epochs", type=int, default=100,
                   help="upper bound; early stopping may end sooner (default 100, see comment in run())")
    p.add_argument("--n-latent", type=int, default=10)
    p.add_argument("--accelerator", default="auto", help="'auto', 'cpu', 'gpu' (CUDA) or 'mps' (Apple Metal GPU). "
                        "'auto' picks CUDA when present and CPU otherwise -- never mps")
    p.add_argument("--no-early-stopping", dest="early_stopping", action="store_false",
                    help="disable early stopping, always run the full --max-epochs")
    p.add_argument("--gene-likelihood", default="auto", choices=["auto", "nb", "zinb", "normal"],
                   help="'auto' (default): nb if the input looks like integer counts, else normal. "
                        "Set nb to force the count likelihood on non-integer estimated counts")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tracking.start(args.out_dir, kind="baseline", params=vars(args),
                   stages=["load", "training", "embed", "cluster search train", "cluster search full"])
    run(
        h5ad_path=args.h5ad,
        batch_key=args.batch_key,
        label_key=args.label_key,
        out_dir=args.out_dir,
        layer=args.layer,
        test_size=args.test_size,
        seed=args.seed,
        max_epochs=args.max_epochs,
        n_latent=args.n_latent,
        accelerator=args.accelerator,
        early_stopping=args.early_stopping,
        gene_likelihood=args.gene_likelihood,
    )
    record_run(Path(args.out_dir) / "config.json")
