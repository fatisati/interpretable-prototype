from interpretable_ssl.datasets.dataset import SingleCellDataset
from pathlib import Path
import scanpy as sc
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import faiss
import numpy as np


class SpatialDataset(SingleCellDataset):
    def __init__(self, **kwargs):
        kwargs.setdefault('batch_key', 'batch')
        super().__init__(**kwargs)

    def get_joint_pca_spatial_representation(self, w=10):
        if "X_pca" not in self.adata.obsm:
            sc.pp.pca(self.adata, n_comps=20)
        # Get PCA features
        X_pca = self.adata.obsm["X_pca"][:, :20]  # 20 PCs is usually enough

        # Get spatial coordinates
        X_spatial = self.adata.obsm["spatial"]

        # Standardize both
        pca_scaled = StandardScaler().fit_transform(X_pca)
        spatial_scaled = StandardScaler().fit_transform(X_spatial)

        # Concatenate
        X_combined = np.concatenate([pca_scaled, spatial_scaled * w], axis=1)
        return X_combined

    def generate_graph(self, k=50):
        X = self.get_joint_pca_spatial_representation().astype("float32")
        index = faiss.IndexFlatL2(X.shape[1])
        index.add(X)
        distances, indices = index.search(X, k + 1)
        return indices[:, 1:], distances[:, 1:]
