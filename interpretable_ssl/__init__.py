import os
import anndata

os.environ["SCIPY_ARRAY_API"] = "1"

if not hasattr(anndata, "read"):
    anndata.read = anndata.read_h5ad

# scanpy >= 1.11 stores obsp['distances'/'connectivities'] as scipy sparse ARRAYS;
# SEACells.build_graph.rbf() row-indexes them as sparse MATRICES (1-D sparse slices
# are not implemented for arrays). Convert after every sc.pp.neighbors call.
try:
    import scanpy as _sc
    import scipy.sparse as _sp

    if not getattr(_sc.pp.neighbors, "_csr_matrix_shim", False):
        _orig_neighbors = _sc.pp.neighbors

        def _neighbors_csr_matrix(adata, *args, **kwargs):
            out = _orig_neighbors(adata, *args, **kwargs)
            target = adata if kwargs.get("copy") is not True else out
            for _k in ("distances", "connectivities"):
                _v = target.obsp.get(_k) if hasattr(target, "obsp") else None
                if _v is not None and isinstance(_v, _sp.sparray):
                    target.obsp[_k] = _sp.csr_matrix(_v)
            return out

        _neighbors_csr_matrix._csr_matrix_shim = True
        _neighbors_csr_matrix.__doc__ = _orig_neighbors.__doc__
        _sc.pp.neighbors = _neighbors_csr_matrix
except Exception:
    pass
