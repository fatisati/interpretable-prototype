import anndata
if not hasattr(anndata, "read"):
    anndata.read = anndata.read_h5ad
