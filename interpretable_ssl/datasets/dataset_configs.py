from pathlib import Path
import os

DATA_DIR = Path.home() / "data/"

DATASETS = {
    "cd34": {
        "path": Path(DATA_DIR) / "seacell/cd34_multiome_rna_preprocessed.h5ad",
        "label_key": "celltype",
        "label_encoder_path": "./data/cd34_le.pkl",
        "test_studies": [],
    },
    "pbmc-immune": {
        "path": Path(DATA_DIR) / "scpoli/Immune_ALL_human_hvg.h5ad",
        "batch_key": "study",
        "label_key": "final_annotation",
        "label_encoder_path": "./data/pbmc_immune_label_encoder.pkl",
        "test_studies": ["Freytag", "Villani"],
    },
    "pancreas": {
        "path": Path(DATA_DIR) / "pancreas_hvg.h5ad",
        "batch_key": "tech",
        "label_key": "celltype",
        "label_encoder_path": "./data/pancras_label_encoder.pkl",
        "test_studies": ["celseq", "celseq2"],
    },
    "hlca": {
        "path": Path(DATA_DIR) / 'hlca/hlca_core_hvg.h5ad',
        "batch_key": "dataset",
        "label_key": "cell_type",
        "label_encoder_path": "./data/hlca_label_encoder.pkl",
        "test_studies": ["Teichmann_Meyer_2019", "Lafyatis_Rojas_2019"],
    },
    'nsc':
        {
        "path": Path(DATA_DIR) / 'spatial/NSCLC_3D_pp.h5ad',
        "batch_key": "section",
        "label_key": "celltypes",
        "label_encoder_path": "./data/NSCLC_3D.pkl",
        "test_studies": ["section_4", "section_10"],
    }
}
