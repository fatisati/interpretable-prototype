from pathlib import Path
import os

from interpretable_ssl.configs.paths import DATA_DIR as _DATA_DIR, CODE_DIR
DATA_DIR = Path(_DATA_DIR)

DATASETS = {
    "sfib": {
        "path": Path(DATA_DIR) / "spatial/sfib.h5ad",
        "label_key": "niches_2D",
        "label_encoder_path": os.path.join(CODE_DIR, "data/sfib.pkl"),
        "num_prototypes": 100,
    },
    "lung": {
        "path": Path(DATA_DIR) / "lung_hvg.h5ad",
        "batch_key": "batch",
        "label_key": "cell_type",
        "label_encoder_path": os.path.join(CODE_DIR, "data/lung.pkl"),
        "num_prototypes": 300,
    },
    "cd34": {
        "path": Path(DATA_DIR) / "seacell/cd34_multiome_rna_preprocessed.h5ad",
        "label_key": "celltype",
        "label_encoder_path": os.path.join(CODE_DIR, "data/cd34_le.pkl"),
        "test_studies": [],
        "num_prototypes": 95,
        "ft_epochs": 0,
    },
    "pbmc-immune": {
        "path": Path(DATA_DIR) / "scpoli/Immune_ALL_human_hvg.h5ad",
        "batch_key": "study",
        "label_key": "final_annotation",
        "label_encoder_path": os.path.join(CODE_DIR, "data/pbmc_immune_label_encoder.pkl"),
        "test_studies": ["Freytag", "Villani"],
        "num_prototypes": 300,
    },
    "bmpbmc-immune": {
        "path": Path(DATA_DIR) / "scpoli/bmpbmc-immune.h5ad",
        "batch_key": "study",
        "label_key": "final_annotation",
        "label_encoder_path": os.path.join(CODE_DIR, "data/pbmc_immune_label_encoder.pkl"),
        "test_studies": ["Freytag", "Villani"],
        "num_prototypes": 300,
    },
    ".1pbmc-immune": {
        "path": Path(DATA_DIR) / "scpoli/.1pbmc-immune.h5ad",
        "batch_key": "study",
        "label_key": "final_annotation",
        "label_encoder_path": os.path.join(CODE_DIR, "data/pbmc_immune_label_encoder.pkl"),
        "test_studies": ["Freytag", "Villani"],
        "num_prototypes": 300,
    },
    "mpbmc-immune": {
        "path": Path(DATA_DIR) / "scpoli/immune_hvg_markers.h5ad",
        "batch_key": "study",
        "label_key": "final_annotation",
        "label_encoder_path": os.path.join(CODE_DIR, "data/pbmc_immune_label_encoder.pkl"),
        "test_studies": ["Freytag", "Villani"],
        "num_prototypes": 300,
    },
    ".1mpbmc-immune": {
        "path": Path(DATA_DIR) / "scpoli/.1mpbmc-immune.h5ad",
        "batch_key": "study",
        "label_key": "final_annotation",
        "label_encoder_path": os.path.join(CODE_DIR, "data/pbmc_immune_label_encoder.pkl"),
        "test_studies": ["Freytag", "Villani"],
        "num_prototypes": 300,
    },
    "bmmpbmc-immune": {
        "path": Path(DATA_DIR) / "scpoli/bmmpbmc-immune.h5ad",
        "batch_key": "study",
        "label_key": "final_annotation",
        "label_encoder_path": os.path.join(CODE_DIR, "data/pbmc_immune_label_encoder.pkl"),
        "test_studies": ["Freytag", "Villani"],
        "num_prototypes": 300,
    },
    
    "pancreas": {
        "path": Path(DATA_DIR) / "pancreas_hvg.h5ad",
        "batch_key": "tech",
        "label_key": "celltype",
        "label_encoder_path": os.path.join(CODE_DIR, "data/pancras_label_encoder.pkl"),
        "test_studies": ["celseq", "celseq2"],
        "num_prototypes": 220,
    },
    "ipanc": { #indrop1 pancreas
        "path": Path(DATA_DIR) / "pancreas_inDrop1.h5ad",
        "batch_key": "tech",
        "label_key": "celltype",
        "label_encoder_path": os.path.join(CODE_DIR, "data/pancras_label_encoder.pkl"),
        "num_prototypes": 25,
    },
     ".1pancreas": {
        "path": Path(DATA_DIR) / ".1pancreas.h5ad",
        "batch_key": "tech",
        "label_key": "celltype",
        "label_encoder_path": os.path.join(CODE_DIR, "data/pancras_label_encoder.pkl"),
        "test_studies": ["celseq", "celseq2"],
        "num_prototypes": 220,
    },
     "bmpancreas": {
        "path": Path(DATA_DIR) / "bmpancreas.h5ad",
        "batch_key": "tech",
        "label_key": "celltype",
        "label_encoder_path": os.path.join(CODE_DIR, "data/pancras_label_encoder.pkl"),
        "test_studies": ["celseq", "celseq2"],
        "num_prototypes": 220,
    },
     
    "hlca": {
        "path": Path(DATA_DIR) / "hlca/hlca_core_hvg.h5ad",
        "batch_key": "dataset",
        "label_key": "cell_type",
        "label_encoder_path": os.path.join(CODE_DIR, "data/hlca_label_encoder.pkl"),
        "test_studies": ["Teichmann_Meyer_2019", "Lafyatis_Rojas_2019"],
        "num_prototypes": 3000,
        "batch_size": 4096,
        "umap_checkpoint_freq": 10,
    },
    "shlca": {
        "path": Path(DATA_DIR) / "hlca/hlca_sampled.h5ad",
        "batch_key": "dataset",
        "label_key": "cell_type",
        "label_encoder_path": os.path.join(CODE_DIR, "data/hlca_label_encoder.pkl"),
        "test_studies": ["Teichmann_Meyer_2019", "Lafyatis_Rojas_2019"],
        "num_prototypes": 500,
        "batch_size": 1024,
        "umap_checkpoint_freq": 10,
    },
    "bmshlca": {
        "path": Path(DATA_DIR) / "hlca/bmshlca.h5ad",
        "batch_key": "dataset",
        "label_key": "cell_type",
        "label_encoder_path": os.path.join(CODE_DIR, "data/hlca_label_encoder.pkl"),
        "test_studies": ["Teichmann_Meyer_2019", "Lafyatis_Rojas_2019"],
        "num_prototypes": 500,
        "batch_size": 1024,
        "umap_checkpoint_freq": 10,
    },
    ".1shlca": {
        "path": Path(DATA_DIR) / "hlca/.1shlca.h5ad",
        "batch_key": "dataset",
        "label_key": "cell_type",
        "label_encoder_path": os.path.join(CODE_DIR, "data/hlca_label_encoder.pkl"),
        "test_studies": ["Teichmann_Meyer_2019", "Lafyatis_Rojas_2019"],
        "num_prototypes": 500,
        "batch_size": 1024,
        "umap_checkpoint_freq": 10,
    },
    "sihlca": {
        "path": Path(DATA_DIR) / "hlca/sihlca.h5ad",
        "batch_key": "dataset",
        "label_key": "cell_type",
        "label_encoder_path": os.path.join(CODE_DIR, "data/hlca_label_encoder.pkl"),
        "test_studies": ["Teichmann_Meyer_2019", "Lafyatis_Rojas_2019"],
        "num_prototypes": 500,
        "batch_size": 1024,
        "umap_checkpoint_freq": 10,
    },
    "nsc": {
        "path": Path(DATA_DIR) / "spatial/NSCLC_3D_pp.h5ad",
        "batch_key": "section",
        "label_key": "celltypes",
        "label_encoder_path": os.path.join(CODE_DIR, "data/NSCLC_3D.pkl"),
        "test_studies": ["section_4", "section_10"],
        "num_prototypes": 3000,
        "batch_size": 4096,
        "umap_checkpoint_freq": 10,
    },
    "snsc": {
        "path": Path(DATA_DIR) / "spatial/snsc.h5ad",
        "label_key": "celltypes",
        "niche_key": 'niches_2D',  # TODO: set to niche annotation column name
        "label_encoder_path": os.path.join(CODE_DIR, "data/NSCLC_3D.pkl"),
        "test_studies": ["section_4", "section_10"],
        "num_prototypes": 400,
        "batch_size": 1024,
        "umap_checkpoint_freq": 10,
    },
    "sp": {
        "path": Path(DATA_DIR) / "spatial/sp.h5ad",
        "batch_key": "section",
        "label_key": "celltypes",
        "niche_key": 'niches_2D',  # TODO: set to niche annotation column name
        "label_encoder_path": os.path.join(CODE_DIR, "data/NSCLC_3D.pkl"),
        "test_studies": ["section_4", "section_10"],
        "num_prototypes": 100,
        "batch_size": 1024,
        "umap_checkpoint_freq": 10,
    },
    "test": {
        "path": Path(DATA_DIR) / "spatial/test.h5ad",
        "batch_key": "section",
        "label_key": "celltypes",
        "label_encoder_path": os.path.join(CODE_DIR, "data/NSCLC_3D.pkl"),
        "test_studies": ["section_4", "section_10"],
        "num_prototypes": 400,
        "batch_size": 1024,
        "umap_checkpoint_freq": 10,
    },
    "s28nsc": {
        "path": Path(DATA_DIR) / "spatial/NSCLC_3D_section_28.h5ad",
        "batch_key": "section",
        "label_key": "celltypes",
        "label_encoder_path": os.path.join(CODE_DIR, "data/NSCLC_3D_section_28.pkl"),
        "num_prototypes": 800,
        "batch_size": 1024,
        "ft_epochs": 0,
        "normalized": True,
        'niche_key': 'niches_2D'
    },
    "s28f": {
        "path": Path(DATA_DIR) / "spatial/s28f.h5ad",
        "batch_key": "section",
        "label_key": "celltypes",
        "label_encoder_path": os.path.join(CODE_DIR, "data/s28f.pkl"),
        "num_prototypes": 50,
        "batch_size": 512,
        "ft_epochs": 0,
    },
    "ss28nsc": {
        "path": Path(DATA_DIR) / "spatial/ss28nsc.h5ad",
        "batch_key": "section",
        "label_key": "celltypes",
        "label_encoder_path": os.path.join(CODE_DIR, "data/NSCLC_3D_section_28.pkl"),
        "num_prototypes": 300,
        "batch_size": 1024,
        "ft_epochs": 0,
        "niche_key": 'niches_2D',  # TODO: set to niche annotation column name
    },
    "fibnsc": {
        # Fibroblasts-only subset of s28nsc (the full/un-subsampled section-28 dataset,
        # not the ss28nsc subsample), spatial context (X_ctx) precomputed on the FULL
        # tissue before cell-type filtering, via a radius-based average (build_context) --
        # matching the NSCLC paper's own method and our appendix's app:spatial_affinity
        # formula (a physical radius, not a fixed neighbour count). The radius is
        # calibrated automatically each build to match the paper's own reported density
        # (median 32 neighbours in their 2D 50um neighbourhood, nscl.pdf p.9) rather than
        # assuming our coordinate units match their micrometers -- see
        # interpretable_ssl/datasets/spatial_subsets.py:
        # calibrate_radius_for_target_median_neighbours / build_celltype_subset_with_context.
        # Single cell type by construction, so scProto's 'ctx' affinity graph built on top
        # of this X_ctx is same-cell-type only automatically (no cross-cell-type edges to
        # mask), while X_ctx itself still reflects each cell's true (mixed-cell-type)
        # neighbourhood composition. num_prototypes = n_cells/75 (15309/75), matching this
        # codebase's rough cells-per-prototype ratio on ss28nsc (28804/300 ~= 96).
        "path": Path(DATA_DIR) / "spatial/fibnsc.h5ad",
        "label_key": "celltypes",
        "niche_key": "niches_2D",
        "label_encoder_path": os.path.join(CODE_DIR, "data/NSCLC_3D_section_28.pkl"),
        "num_prototypes": 204,
        "batch_size": 512,
        "ft_epochs": 0,
    },
    "bms28nsc": { # but in reality, its 0.1 with some new probs
        "path": Path(DATA_DIR) / "spatial/bms28nsc_v1.h5ad",
        "batch_key": "section",
        "label_key": "celltypes",
        "label_encoder_path": os.path.join(CODE_DIR, "data/NSCLC_3D_section_28.pkl"),
        "num_prototypes": 700,
        "batch_size": 1024,
        "ft_epochs": 0,
        "batch_key": "section",
    },
    ".1s28nsc": {
        "path": Path(DATA_DIR) / "spatial/.1s28nsc.h5ad",
        "batch_key": "section",
        "label_key": "celltypes",
        "label_encoder_path": os.path.join(CODE_DIR, "data/NSCLC_3D_section_28.pkl"),
        "num_prototypes": 700,
        "batch_size": 1024,
        "ft_epochs": 0,
        "batch_key": "section",
    },
    'nsc_2slides':
        {
        "path": Path(DATA_DIR) / "spatial/nsc_2slides.h5ad",
        # "batch_key": "section",
        "label_key": "celltypes",
        "label_encoder_path": os.path.join(CODE_DIR, "data/NSCLC_3D_section_28.pkl"),
        "num_prototypes": 1500,
        "batch_size": 2048,
        "ft_epochs": 0,
    },
        
    "immnsc": {
        "path": Path(DATA_DIR) / "spatial/nsc_immune.h5ad",
        "batch_key": "section",
        "label_key": "celltypes",
        "label_encoder_path": os.path.join(CODE_DIR, "data/NSCLC_3D.pkl"),
        "test_studies": ["section_4", "section_10"],
        "num_prototypes": 1500,
        "batch_size": 2048,
        "umap_checkpoint_freq": 10,
    },
    "cytonsc": {
        "path": Path(DATA_DIR) / "spatial/nsc_cytotoxic.h5ad",
        "batch_key": "section",
        "label_key": "celltypes",
        "label_encoder_path": os.path.join(CODE_DIR, "data/NSCLC_3D.pkl"),
        "test_studies": ["section_4", "section_10"],
        "num_prototypes": 350,
        "batch_size": 1024,
        "umap_checkpoint_freq": 10,
    },
    "crcx": {
        # CRC Xenium (Marteau et al. 2026, Cancer Cell) -- second spatial dataset
        # for the rebuttal, different tissue (colon vs. NSCLC lung) and platform
        # (Xenium vs. CosMx). 3 whole tissue sections (g_core, l_normal, d_normal;
        # patients g/l/d, tissue_regions core/normal) kept intact -- not a random
        # per-cell subsample -- so spatial neighborhoods stay real for the affinity
        # graph and for BANKSY. X is already log-normalized (source file's own
        # log1p), layers['counts'] holds raw counts for the DGE ground-truth step.
        # See neurips_manuscript/rebuttle/notebooks/crc_xenium_data_prep.ipynb.
        "path": Path(DATA_DIR) / "spatial/crc_xenium/crc_xenium_prepped.h5ad",
        "batch_key": "patient_id",
        "label_key": "celltype",
        "niche_key": "Niche",  # NicheCompass-derived (GNN), NOT composition-clustering
                                 # like ours -- kept for reference/ground-truth DGE only.
        "label_encoder_path": os.path.join(CODE_DIR, "data/crcx_label_encoder.pkl"),
        "num_prototypes": 670,  # ~50,130 cells / 75, matching the SEACells-style
                                  # K ~= N/75 convention (appendix training.tex).
        "batch_size": 1024,
        "ft_epochs": 0,
        "normalized": True,  # X is already log-normalized -- skip Dataset's own
                              # normalize_total/log1p on load.
    },
}

def load_ds(ds_id):
    import scanpy as sc
    conf = DATASETS[ds_id]
    return sc.read_h5ad(conf['path']), conf.get('batch_key', None), conf['label_key'], conf['num_prototypes']


def register_dataset(name, path, batch_key=None, label_key='cell_type', num_prototypes=None, **kwargs):
    """Register a custom h5ad into DATASETS (in-memory only, file is never modified)."""
    from pathlib import Path
    DATASETS[name] = dict(
        path=Path(path),
        batch_key=batch_key,
        label_key=label_key,
        num_prototypes=num_prototypes,
        test_studies=[],
        **kwargs,
    )