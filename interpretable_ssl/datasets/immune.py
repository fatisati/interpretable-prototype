from interpretable_ssl.datasets.dataset import SingleCellDataset
from pathlib import Path


def get_label_encoder_path():
    return "./data/pbmc_immune_label_encoder.pkl"


class ImmuneDataset(SingleCellDataset):
    def __init__(self, adata=None, original_idx=None, **kwargs):
        super().__init__("pbmc-immune", adata, get_label_encoder_path(), original_idx, **kwargs)

    def get_data_path(self):
        return Path.home() / "data/scpoli/Immune_ALL_human_hvg.h5ad"

    def get_default_studies(self):
        return ["Freytag", "Villani"]

    def read_adata(self):
        adata = super().read_adata()
        adata.obs = adata.obs.rename(columns={"final_annotation": "cell_type"})
        return adata
