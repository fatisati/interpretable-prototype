from pathlib import Path
import os

DATA_DIR = Path.home() / "data/"

DATASETS = {
    "cd34": {
        "path": Path(DATA_DIR) / "seacell/cd34_multiome_rna_preprocessed.h5ad",
        "label_key": "celltype",
        "label_encoder_path": "./data/cd34_le.pkl",
        "test_studies": [],
        "num_prototypes": 95,
        "ft_epochs": 0,
    },
    "pbmc-immune": {
        "path": Path(DATA_DIR) / "scpoli/Immune_ALL_human_hvg.h5ad",
        "batch_key": "study",
        "label_key": "final_annotation",
        "label_encoder_path": "./data/pbmc_immune_label_encoder.pkl",
        "test_studies": ["Freytag", "Villani"],
        "num_prototypes": 300,
    },
    "mpbmc-immune": {
        "path": Path(DATA_DIR) / "scpoli/immune_hvg_markers.h5ad",
        "batch_key": "study",
        "label_key": "final_annotation",
        "label_encoder_path": "./data/pbmc_immune_label_encoder.pkl",
        "test_studies": ["Freytag", "Villani"],
        "num_prototypes": 300,
    },
    "bmpbmc-immune": {
        "path": Path(DATA_DIR) / "scpoli/bmpbmc-immune.h5ad",
        "batch_key": "study",
        "label_key": "final_annotation",
        "label_encoder_path": "./data/pbmc_immune_label_encoder.pkl",
        "test_studies": ["Freytag", "Villani"],
        "num_prototypes": 300,
    },
    ".1pbmc-immune": {
        "path": Path(DATA_DIR) / "scpoli/.1pbmc-immune.h5ad",
        "batch_key": "study",
        "label_key": "final_annotation",
        "label_encoder_path": "./data/pbmc_immune_label_encoder.pkl",
        "test_studies": ["Freytag", "Villani"],
        "num_prototypes": 300,
    },
    "pancreas": {
        "path": Path(DATA_DIR) / "pancreas_hvg.h5ad",
        "batch_key": "tech",
        "label_key": "celltype",
        "label_encoder_path": "./data/pancras_label_encoder.pkl",
        "test_studies": ["celseq", "celseq2"],
        "num_prototypes": 220,
    },
     ".1pancreas": {
        "path": Path(DATA_DIR) / ".1pancreas.h5ad",
        "batch_key": "tech",
        "label_key": "celltype",
        "label_encoder_path": "./data/pancras_label_encoder.pkl",
        "test_studies": ["celseq", "celseq2"],
        "num_prototypes": 220,
    },
     "bmpancreas": {
        "path": Path(DATA_DIR) / "bmpancreas.h5ad",
        "batch_key": "tech",
        "label_key": "celltype",
        "label_encoder_path": "./data/pancras_label_encoder.pkl",
        "test_studies": ["celseq", "celseq2"],
        "num_prototypes": 220,
    },
     
    "hlca": {
        "path": Path(DATA_DIR) / "hlca/hlca_core_hvg.h5ad",
        "batch_key": "dataset",
        "label_key": "cell_type",
        "label_encoder_path": "./data/hlca_label_encoder.pkl",
        "test_studies": ["Teichmann_Meyer_2019", "Lafyatis_Rojas_2019"],
        "num_prototypes": 3000,
        "batch_size": 4096,
        "umap_checkpoint_freq": 10,
    },
    "shlca": {
        "path": Path(DATA_DIR) / "hlca/hlca_sampled.h5ad",
        "batch_key": "dataset",
        "label_key": "cell_type",
        "label_encoder_path": "./data/hlca_label_encoder.pkl",
        "test_studies": ["Teichmann_Meyer_2019", "Lafyatis_Rojas_2019"],
        "num_prototypes": 500,
        "batch_size": 1024,
        "umap_checkpoint_freq": 10,
    },
    "sihlca": {
        "path": Path(DATA_DIR) / "hlca/sihlca.h5ad",
        "batch_key": "dataset",
        "label_key": "cell_type",
        "label_encoder_path": "./data/hlca_label_encoder.pkl",
        "test_studies": ["Teichmann_Meyer_2019", "Lafyatis_Rojas_2019"],
        "num_prototypes": 500,
        "batch_size": 1024,
        "umap_checkpoint_freq": 10,
    },
    "nsc": {
        "path": Path(DATA_DIR) / "spatial/NSCLC_3D_pp.h5ad",
        "batch_key": "section",
        "label_key": "celltypes",
        "label_encoder_path": "./data/NSCLC_3D.pkl",
        "test_studies": ["section_4", "section_10"],
        "num_prototypes": 3000,
        "batch_size": 4096,
        "umap_checkpoint_freq": 10,
    },
    "snsc": {
        "path": Path(DATA_DIR) / "spatial/snsc.h5ad",
        "batch_key": "section",
        "label_key": "celltypes",
        "label_encoder_path": "./data/NSCLC_3D.pkl",
        "test_studies": ["section_4", "section_10"],
        "num_prototypes": 400,
        "batch_size": 1024,
        "umap_checkpoint_freq": 10,
    },
    "s28nsc": {
        "path": Path(DATA_DIR) / "spatial/NSCLC_3D_section_28.h5ad",
        # "batch_key": "section",
        "label_key": "celltypes",
        "label_encoder_path": "./data/NSCLC_3D_section_28.pkl",
        "num_prototypes": 700,
        "batch_size": 1024,
        "ft_epochs": 0,
    },
    'nsc_2slides':
        {
        "path": Path(DATA_DIR) / "spatial/nsc_2slides.h5ad",
        # "batch_key": "section",
        "label_key": "celltypes",
        "label_encoder_path": "./data/NSCLC_3D_section_28.pkl",
        "num_prototypes": 1500,
        "batch_size": 2048,
        "ft_epochs": 0,
    },
        
    "immnsc": {
        "path": Path(DATA_DIR) / "spatial/nsc_immune.h5ad",
        "batch_key": "section",
        "label_key": "celltypes",
        "label_encoder_path": "./data/NSCLC_3D.pkl",
        "test_studies": ["section_4", "section_10"],
        "num_prototypes": 1500,
        "batch_size": 2048,
        "umap_checkpoint_freq": 10,
    },
    "cytonsc": {
        "path": Path(DATA_DIR) / "spatial/nsc_cytotoxic.h5ad",
        "batch_key": "section",
        "label_key": "celltypes",
        "label_encoder_path": "./data/NSCLC_3D.pkl",
        "test_studies": ["section_4", "section_10"],
        "num_prototypes": 350,
        "batch_size": 1024,
        "umap_checkpoint_freq": 10,
    },
        
}

def load_ds(ds_id):
    import scanpy as sc
    conf = DATASETS[ds_id]
    return sc.read_h5ad(conf['path']), conf['batch_key'], conf['label_key'], conf['num_prototypes']