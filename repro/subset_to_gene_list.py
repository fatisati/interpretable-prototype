"""Stage 1 for the heart atlas, but keep a GIVEN gene list instead of selecting HVGs again.

Experiment tool for docs/decisions/0003-rare-metrics-and-clustering-randomness.md, not a
pipeline stage. It answers: "if both machines use exactly the same genes, do the rare-cell
metrics match?" Every step except gene selection is the same as
datasets/scripts/prepare_scvi_builtin.py (raw counts kept in X and layers['counts'],
float32, genes kept in raw-file order, like scanpy's subset=True).

Usage (from repo root; raw_dir is where scvi caches hca_subsampled_20k.h5ad):
    python repro/subset_to_gene_list.py <raw_dir> \
        repro/evidence/mac-studio/heart_cell_atlas/hvg_genes.txt \
        datasets/processed/heart_cell_atlas_hvg.h5ad
"""
import sys

import numpy as np
import scipy.sparse as sp
import scvi

raw_dir, genes_txt, out = sys.argv[1:4]
ad = scvi.data.heart_cell_atlas_subsampled(save_path=raw_dir + "/")
ad.layers["counts"] = ad.X.copy()
genes = open(genes_txt).read().split()
missing = set(genes) - set(ad.var_names)
assert not missing, f"genes not in raw file: {sorted(missing)}"
ad = ad[:, ad.var_names.isin(genes)].copy()  # keeps raw-file order, like subset=True
print("gene order equals the given list:", list(ad.var_names) == genes)
ad.X = sp.csr_matrix(ad.X).astype(np.float32) if sp.issparse(ad.X) else ad.X.astype(np.float32)
ad.write_h5ad(out)
print(ad)
