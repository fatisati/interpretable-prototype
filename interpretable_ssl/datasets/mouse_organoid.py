from interpretable_ssl.datasets.dataset import SingleCellDataset
from pathlib import Path
import scanpy as sc
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

class MouseDataset(SingleCellDataset):
    def __init__(self, adata=None, original_idx=None):
        self.data_home = '/ictstr01/home/icb/fatemehs.hashemig/codes/interpretable-ssl/notebooks/explore_datasets/'
        super().__init__("mouse-organoid", adata, f'{self.data_home}/mouse-organoid.pkl', original_idx)
        sc.pp.pca(self.adata, n_comps=20)


    def get_data_path(self):
        return f'{self.data_home}/seqfish_processed.h5ad'

    def get_default_studies(self):
        return ['embryo3_z2', 'embryo3_z5', 'embryo1_z5']

    def get_joint_pca_spatial_representation(self, w=10):
        # Get PCA features
        X_pca = self.adata.obsm["X_pca"][:, :20]  # 20 PCs is usually enough

        # Get spatial coordinates
        X_spatial = self.adata.obsm['spatial']

        # Standardize both
        pca_scaled = StandardScaler().fit_transform(X_pca)
        spatial_scaled = StandardScaler().fit_transform(X_spatial)

        # Concatenate
        X_combined = np.concatenate([pca_scaled, spatial_scaled*w], axis=1)
        return X_combined
    
    def generate_graph(self, k = 50):
        X = self.get_joint_pca_spatial_representation()
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
        indices = nbrs.kneighbors(X, return_distance=False)[:, 1:]  # remove self
        return indices    
