import os
import anndata

os.environ["SCIPY_ARRAY_API"] = "1"

if not hasattr(anndata, "read"):
    anndata.read = anndata.read_h5ad
