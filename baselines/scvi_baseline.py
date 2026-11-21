"""scVI baseline adapter for the rare-cell-preservation task.

Thin wrapper around scvi-tools (we call the library, we don't reimplement
it): train scVI on train cells only, embed both splits with the trained
encoder, cluster twice (train-only embedding and the full embedding), and
save everything a metrics script would need -- per-cell latent coordinates,
split, and both sets of cluster labels -- to one file.

Reads the split and the target cluster count from
`baselines/prepare_eval.py`'s output -- doesn't create either itself. Run
that script first for this dataset; see docs/architecture.md for why.

Works on any h5ad file; pass --batch-key / --label-key for your dataset.
Runs on CPU locally or GPU on Modal with no code change -- see scvi_modal.py.

Usage:
    python baselines/prepare_eval.py \\
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
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd
import scanpy as sc
import scvi

from common import load_cluster_config, load_split, looks_like_raw_counts, sweep_leiden_to_k


def pick_gene_likelihood(adata, layer):
    """scVI's default likelihoods (nb/zinb) assume raw integer counts --
    fall back to a continuous ('normal') likelihood when the data isn't
    really counts, instead of silently fitting the wrong noise model.
    """
    values = adata.layers[layer] if layer else adata.X
    return "nb" if looks_like_raw_counts(values) else "normal"


def environment_versions():
    """Exact versions of everything that can change this baseline's output
    (repo rule: see CLAUDE.md), saved into config.json next to the result."""
    versions = {"python": platform.python_version()}
    for pkg in ["scvi-tools", "torch", "scanpy", "anndata", "leidenalg", "igraph", "numpy", "pandas"]:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
    return versions


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
):
    adata = sc.read_h5ad(h5ad_path)
    gene_likelihood = pick_gene_likelihood(adata, layer)
    print(f"[scvi] {adata.n_obs:,} cells x {adata.n_vars:,} genes | "
          f"gene_likelihood={gene_likelihood} (checked {'layer=' + layer if layer else 'X'})")

    # Both read-only -- created once by prepare_eval.py, shared by every
    # baseline on this dataset, lives one level above this baseline's own
    # out_dir since neither is specific to scVI.
    dataset_dir = Path(out_dir).parent
    split_path = dataset_dir / "split.csv"
    adata.obs["split"] = load_split(adata, label_key, batch_key, test_size, seed, split_path)
    train_mask = (adata.obs["split"] == "train").to_numpy()
    test_mask = ~train_mask

    cluster_config = load_cluster_config(dataset_dir / "cluster_config.json")

    scvi.settings.seed = seed
    train_adata = adata[train_mask].copy()
    scvi.model.SCVI.setup_anndata(train_adata, batch_key=batch_key, layer=layer)
    model = scvi.model.SCVI(train_adata, n_latent=n_latent, gene_likelihood=gene_likelihood)
    # max_epochs=100 by default: scvi-tools' own heuristic picks 400 for a
    # dataset this size (~19k cells), which was never a deliberate choice;
    # 100 is plenty here. early_stopping can still end training sooner
    # (watches validation loss, stops once it stops improving) -- the
    # actual count lands in config.json as epochs_trained.
    model.train(max_epochs=max_epochs, accelerator=accelerator, early_stopping=early_stopping)
    epochs_trained = len(model.history["elbo_train"])

    # get_latent_representation accepts any AnnData with the same genes/obs
    # columns as training data -- it registers and embeds test cells through
    # the trained encoder without needing a separate setup_anndata call.
    adata.obsm["X_scVI"] = model.get_latent_representation(adata)

    train_labels, resolution_train = sweep_leiden_to_k(
        adata[train_mask], use_rep="X_scVI", target_k=cluster_config["k_train"]
    )
    full_labels, resolution_full = sweep_leiden_to_k(
        adata, use_rep="X_scVI", target_k=cluster_config["k_full"]
    )

    result = pd.DataFrame({
        "cell_id": adata.obs_names,
        "split": adata.obs["split"].to_numpy(),
        label_key: adata.obs[label_key].to_numpy(),
        batch_key: adata.obs[batch_key].to_numpy(),
    })
    for dim in range(adata.obsm["X_scVI"].shape[1]):
        result[f"latent_{dim}"] = adata.obsm["X_scVI"][:, dim]

    # cluster_train_run: only meaningful for train cells (that's the run's
    # whole population), so train cells get it and test cells get NA.
    # cluster_full_run: every cell gets a label here -- cohesion scoring for
    # a test cell needs to know which OTHER cells (including train cells)
    # share its cluster in this run, not just test cells' own labels. Which
    # rows count for *assignment* scoring (test only, per the task's
    # evaluation protocol) is decided by the metrics step that reads this
    # file, not by what gets saved here.
    result["cluster_train_run"] = pd.NA
    result.loc[train_mask, "cluster_train_run"] = train_labels

    result["cluster_full_run"] = full_labels

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "embedding.csv", index=False)

    config = {
        "h5ad_path": str(h5ad_path),
        "split_path": str(split_path),
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
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "k_train_target": cluster_config["k_train"],
        "k_full_target": cluster_config["k_full"],
        "resolution_train": resolution_train,
        "resolution_full": resolution_full,
        "n_clusters_train": int(pd.Series(train_labels).nunique()),
        "n_clusters_full": int(pd.Series(full_labels).nunique()),
    }
    config["environment"] = environment_versions()
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"[scvi] wrote {out_dir / 'embedding.csv'}")
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
    p.add_argument("--accelerator", default="auto", help="'auto', 'cpu', or 'gpu'")
    p.add_argument("--no-early-stopping", dest="early_stopping", action="store_false",
                    help="disable early stopping, always run the full --max-epochs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
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
    )
