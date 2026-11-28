"""Preprocess the NeurIPS 2021 bone marrow (BMMC) CITE-seq dataset for NB-loss training.

Source: Luecken et al. 2021, NeurIPS Datasets & Benchmarks, GEO GSE194122. Download
command: see docs/runbook.md (stage 1). Expected file:
    datasets/raw/neurips2021_bmmc/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad

What this script does, and why:
1. Keeps only RNA features. The file mixes 13,953 genes (feature_types == "GEX") with
   134 protein/antibody features ("ADT"). Protein counts are not gene counts and would
   be read as genes by scVI's count model, so they are dropped here.
2. Uses layers['counts'] (true raw UMI counts, checked here on every run; X ships
   already normalized). X is set to the raw counts, as in prepare_scvi_builtin.py.
3. Keeps ALL 90,261 cells, no subsampling: the rare groups (e.g. Plasma cell IGKC+,
   ~15 cells per batch) would shrink to almost nothing in a uniform subsample, and a
   per-type cap distorts how rare they are (the HLCA lesson, see datasets/hlca.md).
4. Picks 2,000 genes with the same NB-pipeline recipe as the heart atlas
   (seurat_v3 on raw counts, batch-aware), then freezes the list
   (datasets/frozen/neurips_bmmc/genes.txt).

Batch key: `batch` (12 values = site x donor). Label key: `cell_type` (45 types).

Usage (from repo root):
    python datasets/scripts/prepare_neurips_bmmc.py
    python datasets/scripts/prepare_neurips_bmmc.py --refresh-frozen
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation"))
from run_record import record_run  # noqa: E402  records the script's start time
import frozen  # noqa: E402  committed gene lists, see evaluation/frozen.py

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp

DATASETS_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = (DATASETS_DIR / "raw" / "neurips2021_bmmc"
            / "GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad")
OUT_PATH = DATASETS_DIR / "processed" / "neurips_bmmc_hvg.h5ad"

BATCH_KEY = "batch"
LABEL_KEY = "cell_type"
N_TOP_GENES = 2000  # same as the heart atlas (scvi-tools tutorial default for seurat_v3)
KEEP_OBS = [LABEL_KEY, BATCH_KEY, "Site", "DonorID", "is_train",
            "GEX_n_genes_by_counts", "GEX_pct_counts_mt", "GEX_phase"]


def sha256(path, chunk=2**20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main(refresh=False):
    if not RAW_PATH.exists():
        raise SystemExit(f"missing {RAW_PATH} -- download it first, see docs/runbook.md stage 1")

    full = ad.read_h5ad(RAW_PATH, backed="r")
    print(f"[bmmc] raw: {full.n_obs:,} cells x {full.n_vars:,} features "
          f"({full.var['feature_types'].value_counts().to_dict()}) | "
          f"batches ({BATCH_KEY}): {full.obs[BATCH_KEY].nunique()} | "
          f"cell types ({LABEL_KEY}): {full.obs[LABEL_KEY].nunique()} | "
          f"sha256={sha256(RAW_PATH)}")

    gex = np.where(full.var["feature_types"].to_numpy() == "GEX")[0]
    counts = full.layers["counts"][:, gex]
    counts = sp.csr_matrix(counts) if not sp.issparse(counts) else counts.tocsr()
    obs = full.obs[KEEP_OBS].copy()
    var = full.var.iloc[gex][["gene_id"]].copy()
    full.file.close()

    a = ad.AnnData(X=counts.astype(np.float32), obs=obs, var=var)
    a.var_names_make_unique()
    print(f"[bmmc] kept {a.n_vars:,} RNA genes (dropped ADT protein features)")

    frac_nonint = float(np.mean(np.abs(a.X.data - np.round(a.X.data)) > 1e-3))
    print(f"[bmmc] counts: {frac_nonint:.4%} of nonzero entries are non-integer (expect ~0%)")
    if frac_nonint > 0.01:
        raise SystemExit("[bmmc] counts are not integer-valued -- stopping")
    a.layers["counts"] = a.X.copy()

    def propose(x):
        sc.pp.highly_variable_genes(x, flavor="seurat_v3", n_top_genes=N_TOP_GENES,
                                    layer="counts", batch_key=BATCH_KEY)
        return list(x.var_names[x.var["highly_variable"]])

    a = frozen.select_genes(
        a, frozen.dataset_id(OUT_PATH), propose, refresh=refresh,
        method=f"RNA (GEX) features only; scanpy highly_variable_genes flavor=seurat_v3 "
               f"n_top_genes={N_TOP_GENES} layer=counts batch_key={BATCH_KEY}",
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    a.write_h5ad(OUT_PATH)
    digest = sha256(OUT_PATH)
    print(f"[bmmc] wrote {OUT_PATH} ({a.n_obs:,} cells x {a.n_vars:,} genes) sha256={digest}")
    sidecar = OUT_PATH.with_suffix(".run.json")
    sidecar.write_text(json.dumps({"h5ad": OUT_PATH.name, "sha256": digest}, indent=2))
    record_run(sidecar)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--refresh-frozen", action="store_true")
    main(p.parse_args().refresh_frozen)
