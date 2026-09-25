"""[D54] Build the crcx 'bk05' affinity: aRBF on the BANKSY lambda=0.5 embedding.

Recipe = the EXACT s28nsc bk-graph construction (graph_generator's 'bk08'
branch, only lambda differs): pybanksy embedding (lambda_param=0.5,
num_neighbours=32, n_components=50 -- the PCA-to-50 happens inside BANKSY) then
the REAL SEACells SEACellGraph.rbf(k=50, graph_construction='union') on top.
Saved via save_affinity under tag 'bk05' (4-char distinct from
bk08/bk02/bk04/bk32/bkmg), so run_mc_task(..., affinity_type='bk05') finds it
with no graph_generator change (CLAUDE.md section 11). X_bk05 is also written
back into the volume h5ad so anything that re-reads the file sees it.

CPU only. Run:
  PYTHONIOENCODING=utf-8 PYTHONUTF8=1 modal run modal_scripts/build_crcx_bk05.py
"""
import modal

app = modal.App("scproto-bk05-build")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "numpy", "scipy", "pandas", "anndata", "scanpy",
        "torch", "scarches", "wandb", "tqdm", "statsmodels", "diptest",
        "palantir", "faiss-cpu", "scikit-learn", "matplotlib", "scib-metrics",
        "leidenalg", "python-igraph", "ipython", "pybanksy",
    )
    .pip_install("git+https://github.com/dpeerlab/SEACells.git",
                 extra_options="--no-deps")
    .run_commands(*[
        f"mkdir -p /content/drive/MyDrive/codes/Islander/meta/{name} && "
        f"echo '{{}}' > /content/drive/MyDrive/codes/Islander/meta/{name}/cell2cat.json && "
        f"echo '{{}}' > /content/drive/MyDrive/codes/Islander/meta/{name}/batch2cat.json"
        for name in [
            "lung", "lung_fetal_donor", "lung_fetal_organoid", "brain", "breast",
            "heart", "eye", "gut_fetal", "skin", "COVID", "pancreas",
        ]
    ])
)

volume = modal.Volume.from_name("scproto-data")


@app.function(image=image, cpu=8, memory=32768, timeout=7200,
              volumes={"/vol": volume})
def build():
    import os
    import sys
    import time

    os.environ["WANDB_MODE"] = "disabled"
    os.environ["HOME_DIR"] = "/vol/"
    os.environ["DATA_DIR"] = "/vol/data/"
    os.environ["MODEL_DIR"] = "/vol/models/"
    os.environ["CODE_DIR"] = "/vol/code/"
    sys.path.insert(0, "/vol/code")
    os.chdir("/vol/code")

    import h5py
    from anndata._io.specs.registry import _REGISTRY, IOSpec
    try:
        @_REGISTRY.register_read(h5py.Dataset, IOSpec("null", "0.1.0"))
        def _rn(e, *a, **kw):
            return None
    except TypeError:
        pass
    import anndata
    anndata.settings.allow_write_nullable_strings = True

    from interpretable_ssl.augmenters.graph_generator import (
        compute_banksy_embedding, save_affinity)
    from interpretable_ssl.evaluation.batch_correct_baselines import (
        get_preprocessed_adata)

    t0 = time.time()
    ad = get_preprocessed_adata("crcx")
    assert "spatial" in ad.obsm, "crcx adata lacks obsm['spatial']"
    print(f"[bk05] crcx loaded: {ad.n_obs} cells ({time.time()-t0:.0f}s)")

    if "X_bk05" not in ad.obsm:
        compute_banksy_embedding(ad, lambda_param=0.5, num_neighbours=32,
                                 n_components=50, obsm_key="X_bk05")
        print(f"[bk05] BANKSY lambda=0.5 embedding computed "
              f"({time.time()-t0:.0f}s)")

    from SEACells.build_graph import SEACellGraph
    sg = SEACellGraph(ad, build_on="X_bk05", verbose=True)
    A = sg.rbf(k=50, graph_construction="union")
    print(f"[bk05] SEACells rbf kernel built: {A.shape}, nnz={A.nnz} "
          f"({time.time()-t0:.0f}s)")

    path = save_affinity(A, "crcx", ad.n_obs, affinity_type="bk05")
    print(f"[bk05] saved affinity: {path}")

    # persist the embedding so file-re-reading callers see it
    h5_path = "/vol/data/spatial/crc_xenium/crc_xenium_prepped.h5ad"
    try:
        full = anndata.read_h5ad(h5_path)
        full.obsm["X_bk05"] = ad.obsm["X_bk05"]
        full.write_h5ad(h5_path)
        print("[bk05] X_bk05 written back into the volume h5ad")
    except Exception as e:
        print(f"[bk05] WARNING: could not write X_bk05 back ({e}) -- the saved "
              f"affinity pkl is still complete and sufficient for training")
    volume.commit()
    return str(path)


@app.local_entrypoint()
def main():
    print(build.remote())
