"""Preprocess scvi-tools built-in datasets (Heart Cell Atlas, later PBMC Seurat v4
CITE-seq) for NB-loss training.

Different from prepare.py (the scIB figshare trio) in two ways:
1. Download is automatic -- scvi.data.* fetches and caches the file itself, no manual
   figshare step, no blocking observed.
2. These datasets ship TRUE raw integer counts (confirmed directly, see each dataset's
   .md), so preprocessing keeps counts as counts -- no log-normalization, and HVG
   selection uses flavor='seurat_v3', which operates on raw counts directly (the
   standard recipe for datasets headed into an NB-loss model; see
   https://docs.scvi-tools.org/en/1.3.3/tutorials/notebooks/use_cases/preprocessing.html).
   This is the opposite of prepare.py's pipeline, which works on already-log-normalized
   X because the scIB trio's raw counts turned out not to be trustworthy.

Usage (from repo root):
    python datasets/scripts/prepare_scvi_builtin.py --dataset heart
    python datasets/scripts/prepare_scvi_builtin.py --dataset all
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation"))
from run_record import record_run  # noqa: E402  records the script's start time

import numpy as np
import scipy.sparse as sp

DATASETS_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = DATASETS_DIR / "raw"
OUT_DIR = DATASETS_DIR / "processed"

N_TOP_GENES = 2000  # scvi-tools' own tutorial default for seurat_v3 HVG, not 4000 --
                     # that 4000 figure was specific to the scIB trio's MSE pipeline.


def _load_heart():
    import scvi
    return scvi.data.heart_cell_atlas_subsampled(save_path=str(RAW_DIR) + "/")


DATASETS = {
    "heart": {
        "loader": _load_heart,
        "batch_key": "donor",
        "label_key": "cell_type",
        "fine_label_key": "cell_states",
        "out_name": "heart_cell_atlas_hvg.h5ad",
    },
}


def sha256(path, chunk=2**20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def preprocess(name):
    cfg = DATASETS[name]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / cfg["out_name"]

    ad = cfg["loader"]()
    print(f"[{name}] raw: {ad.n_obs:,} cells x {ad.n_vars:,} genes | "
          f"batches ({cfg['batch_key']}): {ad.obs[cfg['batch_key']].nunique()} | "
          f"cell types ({cfg['label_key']}): {ad.obs[cfg['label_key']].nunique()} | "
          f"fine states ({cfg['fine_label_key']}): {ad.obs[cfg['fine_label_key']].nunique()}")

    X = ad.X
    data = X.data if sp.issparse(X) else np.asarray(X)[np.asarray(X) != 0]
    frac_nonint = float(np.mean(np.abs(data - np.round(data)) > 1e-3)) if data.size else 0.0
    print(f"[{name}] X: {frac_nonint:.4%} of nonzero entries are non-integer "
          f"(expect ~0%% -- this dataset should ship true raw counts)")
    if frac_nonint > 0.01:
        raise SystemExit(f"[{name}] X is not integer-valued as expected -- stopping "
                          f"before training anything on data that isn't real raw counts.")

    ad.layers["counts"] = ad.X.copy()

    # seurat_v3 HVG operates on raw counts directly -- X stays raw counts throughout,
    # no normalize_total/log1p here (that's the scVI model's own job internally).
    import scanpy as sc
    sc.pp.highly_variable_genes(
        ad, flavor="seurat_v3", n_top_genes=N_TOP_GENES,
        layer="counts", batch_key=cfg["batch_key"], subset=True,
    )
    ad.X = sp.csr_matrix(ad.X).astype(np.float32) if sp.issparse(ad.X) else ad.X.astype(np.float32)

    ad.write_h5ad(out_path)
    digest = sha256(out_path)
    print(f"[{name}] wrote {out_path}  ({ad.n_obs:,} cells x {ad.n_vars:,} genes)  "
          f"sha256={digest}")
    # Sidecar: which command + code version made this file (see evaluation/run_record.py).
    # Fresh each time -- a new h5ad is a new output.
    sidecar = out_path.with_suffix(".run.json")
    sidecar.write_text(json.dumps({"h5ad": out_path.name, "sha256": digest}, indent=2))
    record_run(sidecar)
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(DATASETS) + ["all"], required=True)
    args = p.parse_args()
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for name in names:
        preprocess(name)


if __name__ == "__main__":
    main()
