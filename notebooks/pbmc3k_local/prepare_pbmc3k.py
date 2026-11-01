"""Build data/pbmc3k.h5ad from scanpy's bundled pbmc3k (raw counts + louvain labels)."""
import os, sys, scanpy as sc, numpy as np, scipy.sparse as sp
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
raw = sc.datasets.pbmc3k()
proc = sc.datasets.pbmc3k_processed()
ad = raw[raw.obs_names.isin(proc.obs_names)].copy()
ad.obs['cell_type'] = proc.obs.loc[ad.obs_names, 'louvain'].astype(str).values
sc.pp.filter_genes(ad, min_cells=3)
ad.layers['counts'] = sp.csr_matrix(ad.X).astype(np.float32)
sc.pp.normalize_total(ad, target_sum=1e4); sc.pp.log1p(ad)
sc.pp.highly_variable_genes(ad, n_top_genes=2000)
ad = ad[:, ad.var['highly_variable']].copy()
ad.X = sp.csr_matrix(ad.X).astype(np.float32)
ad.var_names_make_unique()
os.makedirs(os.path.join(ROOT, 'data'), exist_ok=True)
ad.write_h5ad(os.path.join(ROOT, 'data', 'pbmc3k.h5ad'))
print(ad)
