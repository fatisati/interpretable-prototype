import os
from interpretable_ssl.datasets.spatial import SpatialDataset
from interpretable_ssl.configs.paths import DATA_DIR, CODE_DIR


def get_label_encoder_path():
    return os.path.join(CODE_DIR, "data/merfish_le.pkl")


class MerfishDataset(SpatialDataset):
    def __init__(self, **kwargs):
        kwargs['name'] = 'merfish'
        kwargs['label_encoder_path'] = get_label_encoder_path()
        super().__init__(**kwargs)

    def get_data_path(self):
        return os.path.join(DATA_DIR, "spatial/merfish-mouse-brain/adata.h5ad")

    def get_test_studies(self):
        return [
            "C57BL6J-638850.66",
            "C57BL6J-638850.50",
            "C57BL6J-638850.29",
            "C57BL6J-638850.47",
            "C57BL6J-638850.17",
            "C57BL6J-638850.30",
            "C57BL6J-638850.33",
            "C57BL6J-638850.14",
        ]

