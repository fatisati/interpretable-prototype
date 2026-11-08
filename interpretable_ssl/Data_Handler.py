import os, json, warnings
from anndata import ImplicitModificationWarning
from numba.core.errors import NumbaDeprecationWarning, NumbaWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=NumbaWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=ImplicitModificationWarning)

SEED = 42
MINCELLS = 256
CUDA_DEVICE = "5"

HOME_DIR = os.environ.get('ISLANDER_HOME', '/content/drive/MyDrive/codes/')
PROJ_DIR = rf"{HOME_DIR}/Islander"
MODEL_DIR = rf"{PROJ_DIR}/models"
META_DIR = rf"{PROJ_DIR}/meta"
DATA_DIR = rf"{PROJ_DIR}/data"
RES_DIR = rf"{PROJ_DIR}/res"

try:
    os.makedirs(RES_DIR, exist_ok=True)
    os.makedirs(rf"{RES_DIR}/scIB", exist_ok=True)
except OSError:
    pass  # Islander dir not present outside Colab; only needed for Islander datasets


def _load_meta(name, kind):
    """Load Islander meta json; return {} when the file is absent (non-Colab runs)."""
    path = rf"{META_DIR}/{name}/{kind}.json"
    try:
        with open(path, "r") as f:
            return json.loads(f.read())
    except (FileNotFoundError, OSError):
        return {}

DATA_RAW_ = {
    #
    "lung": rf"{DATA_DIR}/lung/local.h5ad",
    #
    "lung_fetal_donor": rf"{DATA_DIR}/lung_fetal_donor/donor.h5ad",
    #
    "lung_fetal_organoid": rf"{DATA_DIR}/lung_fetal_organoid/organoid.h5ad",
    #
    "brain": rf"{DATA_DIR}/brain/local.h5ad",
    #
    "breast": rf"{DATA_DIR}/breast/local.h5ad",
    #
    "heart": rf"{DATA_DIR}/heart/local.h5ad",
    #
    "eye": rf"{DATA_DIR}/eye/local.h5ad",
    #
    "gut_fetal": rf"{DATA_DIR}/gut_fetal/local.h5ad",
    #
    "skin": rf"{DATA_DIR}/skin/local.h5ad",
    #
    "COVID": rf"{DATA_DIR}/COVID/local.h5ad",
    #
    "pancreas": rf"{DATA_DIR}/pancreas/human_pancreas_norm_complexBatch.h5ad",  # need to deal with this
}

DATA_EMB_ = {
    "lung": rf"{DATA_DIR}/lung/emb.h5ad",
    "lung_hvg": rf"{DATA_DIR}/lung/emb_hvg.h5ad",
    "lung_fetal_donor": rf"{DATA_DIR}/lung_fetal_donor/emb.h5ad",
    # "lung_fetal_donor": rf"{DATA_DIR}/lung_fetal_donor/subset1.h5ad",
    # "lung_fetal_donor": rf"{DATA_DIR}/lung_fetal_donor/subset1_fibro.h5ad",
    # "lung_fetal_donor": rf"{DATA_DIR}/lung_fetal_donor/subset1_fibro_new.h5ad",
    "lung_fetal_donor_fibro": rf"{DATA_DIR}/lung_fetal_donor/subset1_fibro_new.h5ad",
    # "lung_fetal_donor": rf"{DATA_DIR}/lung_fetal_donor/subset2.h5ad",
    "lung_fetal_donor_hvg": rf"{DATA_DIR}/lung_fetal_donor/emb_hvg.h5ad",
    "lung_fetal_organoid": rf"{DATA_DIR}/lung_fetal_organoid/emb.h5ad",
    "lung_fetal_organoid_hvg": rf"{DATA_DIR}/lung_fetal_organoid/emb_hvg.h5ad",
    "brain": rf"{DATA_DIR}/brain/emb.h5ad",
    "brain_hvg": rf"{DATA_DIR}/brain/emb_hvg.h5ad",
    "breast": rf"{DATA_DIR}/breast/emb.h5ad",
    "breast_hvg": rf"{DATA_DIR}/breast/emb_hvg.h5ad",
    "heart": rf"{DATA_DIR}/heart/emb.h5ad",
    "heart_hvg": rf"{DATA_DIR}/heart/emb_hvg.h5ad",
    "eye": rf"{DATA_DIR}/eye/emb.h5ad",
    "eye_hvg": rf"{DATA_DIR}/eye/emb_hvg.h5ad",
    "gut_fetal": rf"{DATA_DIR}/gut_fetal/emb.h5ad",
    "gut_fetal_hvg": rf"{DATA_DIR}/gut_fetal/emb_hvg.h5ad",
    "skin": rf"{DATA_DIR}/skin/emb.h5ad",
    "skin_hvg": rf"{DATA_DIR}/skin/emb_hvg.h5ad",
    "COVID": rf"{DATA_DIR}/COVID/emb.h5ad",
    "COVID_hvg": rf"{DATA_DIR}/COVID/emb_hvg.h5ad",
    "pancreas": rf"{DATA_DIR}/pancreas/emb.h5ad",
    "pancreas_hvg": rf"{DATA_DIR}/pancreas/emb_hvg.h5ad",
}

META_ = {
    "lung": {"batch": "sample", "celltype": "cell_type"},
    "lung_fetal_donor": {"batch": "batch", "celltype": "new_celltype"},
    # "lung_fetal_donor": {"batch": "batch", "celltype": "broad_celltype"},
    "lung_fetal_organoid": {"batch": "batch", "celltype": "new_celltype"},
    "brain": {"batch": "donor_id", "celltype": "cell_type"},  # or "sample_id"
    "breast": {"batch": "donor_id", "celltype": "cell_type"},  # or "sample_id"
    "heart": {"batch": "donor_id", "celltype": "cell_type"},
    "eye": {"batch": "biosample_id", "celltype": "cell_type"},
    "gut_fetal": {"batch": "donor_id", "celltype": "cell_type"},  # or "Sample"
    "skin": {"batch": "donor_id", "celltype": "cell_type"},
    "COVID": {"batch": "batch", "celltype": "predicted.celltype.l2"},  # or "batch"
    "pancreas": {"batch": "tech", "celltype": "celltype"},
}

CELL2CAT_ = {
    "lung": _load_meta("lung", "cell2cat"),
    "lung_fetal_donor": _load_meta("lung_fetal_donor", "cell2cat"),
    "lung_fetal_organoid": _load_meta("lung_fetal_organoid", "cell2cat"),
    "brain": _load_meta("brain", "cell2cat"),
    "breast": _load_meta("breast", "cell2cat"),
    "heart": _load_meta("heart", "cell2cat"),
    "eye": _load_meta("eye", "cell2cat"),
    "gut_fetal": _load_meta("gut_fetal", "cell2cat"),
    "skin": _load_meta("skin", "cell2cat"),
    "COVID": _load_meta("COVID", "cell2cat"),
    "pancreas": _load_meta("pancreas", "cell2cat"),
}

BATCH2CAT_ = {
    "lung": _load_meta("lung", "batch2cat"),
    "lung_fetal_donor": _load_meta("lung_fetal_donor", "batch2cat"),
    "lung_fetal_organoid": _load_meta("lung_fetal_organoid", "batch2cat"),
    "brain": _load_meta("brain", "batch2cat"),
    "breast": _load_meta("breast", "batch2cat"),
    "heart": _load_meta("heart", "batch2cat"),
    "eye": _load_meta("eye", "batch2cat"),
    "gut_fetal": _load_meta("gut_fetal", "batch2cat"),
    "skin": _load_meta("skin", "batch2cat"),
    "COVID": _load_meta("COVID", "batch2cat"),
    "pancreas": _load_meta("pancreas", "batch2cat"),
}

Gender_Related_Genes = [
    "XIST",
    "DDX3Y",
    "EIF1AY",
    "RPS4Y1",
]

ThreePrime_Enriched_Genes = [
    # "HNRNPA1P48",
    "MTRNR2L8",
    "MTRNR2L12",
    "USP18",
    "ARPC1A",
    "LINC00685",
]

FivePrime_Enriched_Genes = [
    "GSTM2",
    "GSTM1",
    "NBPF26",
    "WBP1",
    # "MATR3-1",
    # "HIST1H2BK",
    "RPP21",
    "RPS10-NUDT3",
    "GET4",
    "MRPS24",
    "POMZP3",
    # "AC004922.1",
    "ARF5",
    "TOMM5",
    "TMEFF1",
    "GNG10",
    # "IGF2-1",
    "EEF1G",
    "C11orf98",
    "BSCL2",
    "RBM4",
    "MTG1",
    "SARNP",
    "CDK2AP1",
    "PSMA6",
    "COX16",
    "NGRN",
    "ATP6V0C",
    "PAM16",
    "ALDOA",
    "TUBB3",
    "RNASEK",
    "GABARAP",
    "PLSCR3",
    "NME2",
    "MRPL38",
    "RPL17",
    "UBE2V1",
    "TRAPPC5",
    "HNRNPL",
    "BCKDHA",
    # "SEPT5",
    "MIF",
    "PDXP",
    "GATD3B",
    "U2AF1",
    "MT-ATP8",
    "MT-ND4L",
    "MT-ND6",
]


def inverse_dict(d):
    return {v: k for k, v in d.items()}


if __name__ == "__main__":
    pass
