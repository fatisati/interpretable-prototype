"""
Colab boilerplate — run this at the top of every notebook:
    %run /content/drive/MyDrive/codes/interpretable-prototype/notebooks/nb_setup.py
"""
import sys, os, importlib
from importlib import reload

import anndata
if not hasattr(anndata, "read"):
    anndata.read = anndata.read_h5ad

os.environ["SCIPY_ARRAY_API"] = "1"
os.environ['HOME_DIR']     = '/content/drive/MyDrive/'
os.environ['MODEL_DIR']    = '/content/drive/MyDrive/models/'
os.environ['DATA_DIR']     = '/content/drive/MyDrive/data/'
os.environ['CODE_DIR']     = '/content/drive/MyDrive/codes/interpretable-prototype/'
os.environ['ISLANDER_SRC'] = '/content/drive/MyDrive/codes/Islander/src'

os.chdir('/content/drive/MyDrive/codes/interpretable-prototype/')

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150

def reload_interpretable_ssl():
    for m in list(sys.modules):
        if m.startswith("interpretable_ssl") or m == "constants":
            importlib.reload(sys.modules[m])

# Make tasks available directly after %run nb_setup.py
reload_interpretable_ssl()
from interpretable_ssl.experiments.tasks import (
    get_trainer,
    run_mc_task,
    LAMBDA_PROTO_UMAP,
    LAMBDA_PROTO_UMAP_PRECON,
    LAMBDA_PARAM_UMAP,
    LAMBDA_RECON_ONLY,
    LAMBDA_PROTO_RECON_ONLY,
    LAMBDA_PROTO_CTX_UMAP,
)

print("nb_setup done. Available: get_trainer, run_mc_task, LAMBDA_PROTO_UMAP, LAMBDA_PROTO_UMAP_PRECON, LAMBDA_PARAM_UMAP, LAMBDA_RECON_ONLY")
print("Configs:", {
    'LAMBDA_PROTO_UMAP': LAMBDA_PROTO_UMAP,
    'LAMBDA_PARAM_UMAP': LAMBDA_PARAM_UMAP,
    'LAMBDA_RECON_ONLY': LAMBDA_RECON_ONLY,
})
