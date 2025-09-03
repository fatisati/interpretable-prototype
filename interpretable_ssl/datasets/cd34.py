from interpretable_ssl.datasets.dataset import SingleCellDataset
from pathlib import Path


def get_label_encoder_path():
    return "./data/cd34_le.pkl"


class CD34Dataset(SingleCellDataset):
    def __init__(self, adata=None, original_idx=None):
        super().__init__("cd34", adata, get_label_encoder_path(), original_idx, cell_type_key='celltype')

    def get_data_path(self):
        return Path.home() / "data/seacell/cd34_multiome_rna_preprocessed.h5ad"
    
    def split(self, test_size=0.2, random_state=None):
        return self, self
    
    def read_adata(self):
        adata = super().read_adata()
        adata.obs[self.batch_key] = 's0'
        return adata[:, adata.var['highly_variable']].copy()

    def get_train_test(self):
        return self, self