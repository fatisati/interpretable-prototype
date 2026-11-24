"""Preprocess the pancreas / lung / pbmc_immune datasets.

Source (all three): figshare article 12420968 -- "Benchmarking atlas-level data
integration in single-cell genomics - integration task datasets" (Luecken et al. 2022).
https://figshare.com/articles/dataset/Benchmarking_atlas-level_data_integration_in_single-cell_genomics_-_integration_task_datasets/12420968

Download is MANUAL: figshare blocks automated downloads on some networks/clients, so
grab the file by hand from the link above and save it -- with its exact original file
name below -- into datasets/raw/. Then run this script.

    pancreas:     datasets/raw/human_pancreas_norm_complexBatch.h5ad
    lung:         datasets/raw/Lung_atlas_public.h5ad
    pbmc_immune:  datasets/raw/Immune_ALL_human.h5ad

Usage (from repo root, after the raw file is in datasets/raw/):
    python datasets/scripts/prepare.py --dataset pancreas
    python datasets/scripts/prepare.py --dataset lung
    python datasets/scripts/prepare.py --dataset pbmc_immune
    python datasets/scripts/prepare.py --dataset all
    python datasets/scripts/prepare.py --dataset pancreas --delete-raw   # free disk after a
                                                                  # successful write

Why this is a separate, one-time step instead of each baseline preprocessing its own
copy: these datasets come from the scIB integration benchmark (Luecken et al. 2022,
Nature Methods), whose own methodology requires every method under comparison to see
IDENTICAL preprocessed input -- otherwise a difference in results could come from
preprocessing drift rather than the method itself. Run this once per dataset; every
baseline/trainer should read the saved file in datasets/processed/, never call
scanpy's normalize/HVG functions on the raw file directly.

Per the figshare dataset description (posted 2022-01-24 by Luecken, Buttner, Danese,
Interlandi, Muller, Strobl, Zappia, Dugas, Colome-Tatche, Theis, Chaichoompu): all three
raw files are already "consistently preprocessed using scran pooling and log+1
transformation" by the depositors -- so the normalize_total/log1p call below is a safety
net for an already-normalized file, not expected to actually run. This script's own job
is just: pick one fixed HVG set per dataset so every baseline reads identical genes.
Citation is required by the dataset license -- cite the paper above and the papers the
individual datasets were aggregated from (see https://github.com/theislab/scib).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation"))
from run_record import record_run  # noqa: E402  records the script's start time
import frozen  # noqa: E402  committed gene lists, see evaluation/frozen.py

import numpy as np
import scanpy as sc
import scipy.sparse as sp

DATASETS_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = DATASETS_DIR / "raw"
OUT_DIR = DATASETS_DIR / "processed"

FIGSHARE_URL = (
    "https://figshare.com/articles/dataset/Benchmarking_atlas-level_data_integration_"
    "in_single-cell_genomics_-_integration_task_datasets/12420968"
)

N_TOP_GENES = 4000

# batch_key / label_key are the obs columns every baseline/trainer should read for this
# dataset. Whether layers['counts'] is trustworthy raw counts is NOT assumed here --
# preprocess() measures it directly on every run (see NONINT_DROP_THRESHOLD below),
# because trusting a hardcoded flag is exactly how this went wrong once already: an old
# note about a different, already-reprocessed lung file claimed lung had true raw counts,
# but checking the actual raw figshare file directly (2026-11-16) showed 91.6% of its
# nonzero layers['counts'] values are non-integer -- not raw counts at all. Pancreas measured
# 42.8% non-integer the same way. Don't re-introduce a static assumption here.
DATASETS = {
    "pancreas": {
        "raw_name": "human_pancreas_norm_complexBatch.h5ad",
        "batch_key": "tech",
        "label_key": "celltype",
        "out_name": "pancreas_hvg.h5ad",
    },
    "lung": {
        "raw_name": "Lung_atlas_public.h5ad",
        "batch_key": "batch",
        "label_key": "cell_type",
        "out_name": "lung_hvg.h5ad",
    },
    "pbmc_immune": {
        "raw_name": "Immune_ALL_human.h5ad",
        "batch_key": "study",
        "label_key": "final_annotation",
        "out_name": "pbmc_immune_hvg.h5ad",
    },
}

# Fraction of nonzero layers['counts'] entries allowed to be non-integer before we
# conclude the layer isn't real raw counts and drop it instead of shipping something
# that silently looks usable for NB/ZINB likelihoods or size-factor recomputation but
# isn't.
NONINT_DROP_THRESHOLD = 0.01


def sha256(path, chunk=2**20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def preprocess(name, delete_raw, refresh=False):
    cfg = DATASETS[name]
    raw_path = RAW_DIR / cfg["raw_name"]
    if not raw_path.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        raise SystemExit(
            f"[{name}] missing {raw_path}\n"
            f"Download '{cfg['raw_name']}' by hand from:\n  {FIGSHARE_URL}\n"
            f"and save it at that exact path, then re-run this command."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / cfg["out_name"]

    ad = sc.read_h5ad(raw_path)
    print(f"[{name}] raw: {ad.n_obs:,} cells x {ad.n_vars:,} genes | "
          f"batches ({cfg['batch_key']}): {ad.obs[cfg['batch_key']].nunique()} | "
          f"cell types ({cfg['label_key']}): {ad.obs[cfg['label_key']].nunique()} | "
          f"sha256={sha256(raw_path)}")

    if ad.X.max() > 30:  # depositors say X ships scran+log1p-normalized; this firing
        print(f"[{name}] WARNING: X.max()={ad.X.max():.1f} suggests X is NOT already "
              f"log-normalized, despite the figshare description saying it is. "
              f"Applying normalize_total+log1p now -- double-check this dataset.")
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)

    if "counts" in ad.layers:
        c = ad.layers["counts"]
        data = c.data if sp.issparse(c) else np.asarray(c)[np.asarray(c) != 0]
        frac_nonint = float(np.mean(np.abs(data - np.round(data)) > 1e-3)) if data.size else 0.0
        print(f"[{name}] layers['counts']: {frac_nonint:.1%} of nonzero entries are "
              f"non-integer")
        if frac_nonint > NONINT_DROP_THRESHOLD:
            print(f"[{name}] dropping layers['counts'] -- not true raw counts (likely "
                  f"batch-scaled by the depositors), would silently corrupt any "
                  f"NB/ZINB likelihood or size-factor recomputation downstream")
            del ad.layers["counts"]
    else:
        print(f"[{name}] no counts layer in the raw file")

    # batch-aware HVG: every downstream method must see the SAME genes, picked without
    # one batch's technical variance dominating the selection. Only runs when there is
    # no committed gene list yet (or --refresh-frozen); otherwise the committed
    # datasets/frozen/<dataset>/genes.txt is used as is.
    def propose(a):
        sc.pp.highly_variable_genes(a, n_top_genes=N_TOP_GENES, batch_key=cfg["batch_key"])
        return list(a.var_names[a.var["highly_variable"]])

    ad = frozen.select_genes(
        ad, frozen.dataset_id(out_path), propose, refresh=refresh,
        method=f"scanpy highly_variable_genes (default flavor) n_top_genes={N_TOP_GENES} "
               f"batch_key={cfg['batch_key']}",
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

    if delete_raw:
        raw_path.unlink()
        print(f"[{name}] deleted {raw_path} (--delete-raw)")

    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(DATASETS) + ["all"], required=True)
    p.add_argument("--delete-raw", action="store_true",
                   help="remove the raw file after a successful write, to save disk "
                        "(re-download by hand if you ever need it again)")
    p.add_argument("--refresh-frozen", action="store_true",
                   help="recompute the gene list and overwrite datasets/frozen/<dataset>/genes.txt "
                        "(then commit the diff); default: use the committed list")
    args = p.parse_args()
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for name in names:
        preprocess(name, args.delete_raw, args.refresh_frozen)


if __name__ == "__main__":
    main()
