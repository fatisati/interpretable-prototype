"""Subsample + preprocess the HLCA core atlas (584,944 cells -- too large to use whole).

Source: cellxgene collection 6f6d381a-7701-4781-935c-db10d30de293, "An integrated cell
atlas of the human lung in health and disease (core)" (Sikkema et al. 2023, Nature
Medicine). Direct download (no figshare-style blocking observed):
    https://datasets.cellxgene.cziscience.com/688185ad-11c2-4172-a53a-f4f1f4076860.h5ad
Save as datasets/raw/hlca_core.h5ad (5.87 GB) before running this script.

Why subsample, and why NOT uniform random subsampling: the whole point of this dataset
is 3 rare, literature-documented cell types (Ionocyte 561 cells / 0.096%, Tuft 165 / 0.028%,
Neuroendocrine 159 / 0.027%, out of 584,944). A uniform random subsample to a manageable
size (e.g. 30-50k cells, ~5-9% retention) would shrink Tuft/Neuroendocrine to ~8-15 cells
each -- too few for any reliable per-type metric, destroying the exact thing this dataset
is for. Instead: cap each abundant cell type (ann_finest_level) at 1,000 cells (randomly
subsampled, seed=0), and keep EVERY cell of types already below that cap untouched. This
preserves all 3 target rare types at their full original counts while still shrinking the
total from 584,944 to ~50,819 cells.

Raw counts: confirmed directly (not assumed) that `X` ships log-normalized but `.raw.X`
is true raw integer counts (0% non-integer on a sample check) -- HLCA does this correctly,
unlike the scIB trio. `.raw.X` is copied into layers['counts'] here.

The full file doesn't fit in memory on a 16GB machine (loading X alone needs >8GB just for
one sparse-matrix component) -- this script reads in anndata's backed mode, decides which
cells to keep from obs metadata alone (no X touched), and only materializes the ~50k-cell
subset into memory.

Usage (from repo root, after datasets/raw/hlca_core.h5ad exists):
    python datasets/scripts/prepare_hlca.py
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "baselines"))
from run_record import record_run  # noqa: E402  records the script's start time

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

DATASETS_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = DATASETS_DIR / "raw" / "hlca_core.h5ad"
OUT_PATH = DATASETS_DIR / "processed" / "hlca_core_hvg.h5ad"

BATCH_KEY = "dataset"
LABEL_KEY = "ann_finest_level"
PER_TYPE_CAP = 1000
SEED = 0
N_TOP_GENES = 4000
TARGET_RARE_TYPES = ["Ionocyte", "Tuft", "Neuroendocrine"]


def pick_subsample_indices(obs):
    rng = np.random.RandomState(SEED)
    keep = []
    for _, idx in obs.groupby(LABEL_KEY).groups.items():
        idx = np.asarray(idx)
        if len(idx) > PER_TYPE_CAP:
            idx = rng.choice(idx, size=PER_TYPE_CAP, replace=False)
        keep.append(idx)
    return np.sort(np.concatenate(keep))


def main():
    print(f"[hlca] loading {RAW_PATH} in backed mode (metadata only)")
    full = ad.read_h5ad(RAW_PATH, backed="r")
    print(f"[hlca] full: {full.n_obs:,} cells x {full.n_vars:,} genes | "
          f"batches ({BATCH_KEY}): {full.obs[BATCH_KEY].nunique()} | "
          f"cell types ({LABEL_KEY}): {full.obs[LABEL_KEY].nunique()}")

    before = full.obs[LABEL_KEY].value_counts()
    idx_pos = pick_subsample_indices(full.obs.reset_index(drop=True))
    print(f"[hlca] subsample: {len(idx_pos):,} / {full.n_obs:,} cells "
          f"({100 * len(idx_pos) / full.n_obs:.1f}%), per-type cap {PER_TYPE_CAP}")

    print("[hlca] materializing subsample into memory (this reads only the kept rows)")
    sub = full[idx_pos].to_memory()

    after = sub.obs[LABEL_KEY].value_counts()
    print("[hlca] target rare-type counts, full atlas -> subsample (expect UNCHANGED):")
    for t in TARGET_RARE_TYPES:
        matches = [c for c in before.index if t in str(c)]
        for m in matches:
            b, a_ = before.get(m, 0), after.get(m, 0)
            flag = "OK" if a_ == b else "CHANGED -- investigate"
            print(f"    {m}: {b} -> {a_}  [{flag}]")

    print(f"[hlca] batch ({BATCH_KEY}) representation after subsampling:")
    print(sub.obs[BATCH_KEY].value_counts().to_string())

    R = sub.raw.X
    data = R.data if sp.issparse(R) else np.asarray(R)[np.asarray(R) != 0]
    frac_nonint = float(np.mean(np.abs(data - np.round(data)) > 1e-3)) if data.size else 0.0
    print(f"[hlca] raw.X: {frac_nonint:.4%} of nonzero entries are non-integer "
          f"(expect ~0% -- confirmed true raw counts)")
    if frac_nonint > 0.01:
        raise SystemExit("[hlca] raw.X is not integer-valued as expected -- stopping "
                          "before writing anything, this contradicts the earlier check.")

    sub.layers["counts"] = sub.raw.X.copy()  # true raw counts, confirmed above
    # sub.X stays log-normalized as shipped (cellxgene schema) -- no renormalization.

    sc.pp.highly_variable_genes(
        sub, n_top_genes=N_TOP_GENES, batch_key=BATCH_KEY,
    )
    sub = sub[:, sub.var["highly_variable"]].copy()
    sub.X = sp.csr_matrix(sub.X).astype(np.float32) if sp.issparse(sub.X) else sub.X.astype(np.float32)
    sub.layers["counts"] = sp.csr_matrix(sub.layers["counts"]).astype(np.float32)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sub.write_h5ad(OUT_PATH)

    def sha256(path, chunk=2**20):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()

    digest = sha256(OUT_PATH)
    print(f"[hlca] wrote {OUT_PATH}  ({sub.n_obs:,} cells x {sub.n_vars:,} genes)  "
          f"sha256={digest}")
    # Sidecar: which command + code version made this file (see baselines/run_record.py).
    # Fresh each time -- a new h5ad is a new output.
    sidecar = OUT_PATH.with_suffix(".run.json")
    sidecar.write_text(json.dumps({"h5ad": OUT_PATH.name, "sha256": digest}, indent=2))
    record_run(sidecar)


if __name__ == "__main__":
    main()
