from scarches.dataset.scpoli.anndata import MultiConditionAnnotatedDataset
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
import scanpy as sc
import scipy.stats as stats
from sklearn.decomposition import PCA
import logging
import random

from torch.utils.data import get_worker_info
import scipy.sparse as sp
import faiss
import os
import pickle as pkl
import scvi
from sklearn.preprocessing import StandardScaler
from interpretable_ssl.augmenters.spatial_graph import *

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
import subprocess
import time
import os
import pickle
from interpretable_ssl.augmenters.manifold_weights import *
from tqdm import tqdm


class MultiCropsDataset(MultiConditionAnnotatedDataset):
    def __init__(
        self,
        sc_ds,
        n_augmentations,
        affinity_type="inverse_dist",
        k_neighbors=50,  # seacell use 50
        n_components=50,
        supervised_ratio=0.1,
        use_bknn=0,
        save_dir=None,
        mask_probability=0.2,
        default_dispersion=0.1,
        spatial=0,
        return_idx=False,
        n_clusters=None,
        use_counts=True,
        n_proto=None,
        use_manifold_weights=False,
        **kwargs,
    ):
        self.n_proto = n_proto
        self.n_augmentations = n_augmentations
        self.adata = sc_ds.adata
        self.k_neighbors = k_neighbors
        self.n_components = n_components
        self.ds_affinities = None
        self.affinity_type = affinity_type
        self.supervised_ratio = supervised_ratio
        self.use_bknn = use_bknn

        self.save_dir = save_dir

        self.graph_name = f"affinity_{str(sc_ds)}{len(sc_ds.adata)}_ncomp{self.n_components}_kneighbors{self.k_neighbors}_{affinity_type}"

        self.spatial = spatial
        if self.spatial:
            self.graph_name += "_spatial"
        if sc_ds.fold != 0:
            self.graph_name += f"_fold{sc_ds.fold}"
        self.adata_name = self.graph_name + "_tmp.h5ad"
        self.graph_name += ".pkl"

        os.makedirs(self.save_dir, exist_ok=True)
        self.save_path = os.path.join(save_dir, self.graph_name)
        self.adata_path = os.path.join(save_dir, self.adata_name)

        self.batch_key = kwargs["condition_keys"][0]
        self.mask_probability = mask_probability

        self.default_dispersion = default_dispersion

        # Calculate mean expression and default dispersion from the data
        self.mean_expression = np.mean(self.adata.X, axis=0).flatten()
        self.dispersion = (
            1 / self.adata.varm["overdispersion"]
            if "overdispersion" in self.adata.varm
            else np.full(self.mean_expression.shape, self.default_dispersion)
        )

        self.dataset_index_map = {
            ds_key: np.where(self.adata.obs[self.batch_key].values == ds_key)[0]
            for ds_key in self.adata.obs[self.batch_key].unique()
        }
        self.ds_keys = list(self.dataset_index_map.keys())

        self.return_idx = return_idx
        self.cluster_dict = None
        self.sample_cluster_id = None
        self.n_clusters = n_clusters
        super().__init__(sc_ds.adata, **kwargs)
        self.data = (
            sc_ds.adata.layers.get("counts", sc_ds.adata.X)
            if use_counts
            else sc_ds.adata.X
        )
        if not self._is_sparse:
            self.data = torch.tensor(self.data, dtype=torch.float32)

        print(self._is_sparse, type(self.data[0]), self.data.max())
        if not self.affinities_exists():
            self.ds_affinities = self.run_graph_generator()
        self.set_affinities()
        self.use_manifold_weights = use_manifold_weights
        self.manifold = {}
        # if use_manifold_weights:
        self.calc_manifold_weights()
        self.label_key = sc_ds.label_key
        self.return_label = True
        
    def calc_manifold_weights(self):
        print("calculating manifold scores...")
        self.manifold = {
            "sigma": np.zeros(self.adata.n_obs),
            "wsigma": np.zeros(self.adata.n_obs),
            "heterogeneity": np.zeros(self.adata.n_obs),
            "mf_score": np.zeros(self.adata.n_obs),
        }
        for key in self.manifold.keys():
            self.adata.obs[key] = np.zeros(self.adata.n_obs)
        for s in tqdm(self.adata.obs[self.batch_key].unique()):
            batch_idx = self.adata.obs[self.batch_key] == s
            ad = self.adata[batch_idx].copy()

            # PCA within batch
            sc.tl.pca(ad, n_comps=self.n_components)

            # get k neighbors (full, not half)
            D, I = get_knn(ad.obsm["X_pca"], self.k_neighbors)

            # --- heterogeneity (whatever your function does on neighbors) ---
            h = homogeneity(ad.obsm["X_pca"], I)

            # --- sigma: median distance across neighbors ---
            sigma = np.median(np.sqrt(D[:, 1:]), axis=1)  # skip self, median over kNN
            w_sigma = self.get_w_sigma(sigma)
            
            # --- row marginals / weights ---
            w = compute_row_marginals(sigma, h)

            # store in dictionary
            self.manifold["sigma"][batch_idx] = sigma
            self.manifold["wsigma"][batch_idx] = w_sigma
            self.manifold["heterogeneity"][batch_idx] = h
            self.manifold["mf_score"][batch_idx] = w
            for key in self.manifold.keys():
                self.adata.obs.loc[batch_idx, key] = self.manifold[key][batch_idx]
            
        print("---done---")

    def get_w_sigma(self, sigma, alpha=0.25, clip=(0.5, 3.0)):
        # robust normalization (center & scale)
        med = np.median(sigma)
        mad = np.median(np.abs(sigma - med)) + 1e-8  # avoid div/0
        z = (sigma - med) / mad
        # self.adata.obs.loc[idx, "z"] = z
        # exponential mapping
        w = np.exp(alpha * z)

        # normalize to mean 1 within batch
        w = w / w.mean()

        # clip extreme values
        w = np.clip(w, clip[0], clip[1])
        return w

    def run_graph_generator(self):
        print(f"[{os.getpid()}] Generating affinities...")
        self.adata.write(self.adata_path)
        args = (
            self.adata_path,
            self.affinity_type,
            self.batch_key,
            self.n_components,
            self.k_neighbors,
        )
        # Spawn g_hand with input + output arguments
        config_file = self.save_path + ".inputs.pkl"
        with open(config_file, "wb") as f:
            pickle.dump(args, f)

        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "interpretable_ssl.augmenters.graph_generator",
                config_file,
                self.save_path,
            ]
        )
        # Wait for graph file
        while not os.path.exists(self.save_path):
            time.sleep(1)

        # Load result
        with open(self.save_path, "rb") as f:
            graph = pickle.load(f)

        # Cleanup temporary files
        try:
            os.remove(config_file)
            os.remove(self.adata_path)
        except FileNotFoundError:
            pass
        return graph

    def __len__(self):
        return max(
            [
                len(self.dataset_index_map[ds_id])
                for ds_id in self.dataset_index_map.keys()
            ]
        )

    def __getitem__(self, index):
        items = []
        for ds_id in self.dataset_index_map.keys():
            ds_idx = index % len(self.dataset_index_map[ds_id])
            pos_sample_ids = self.get_positive_samples(ds_idx, ds_id)
            global_idx = self.dataset_index_map[ds_id][ds_idx]
            items.append(self.assemble_from_indices(pos_sample_ids, global_idx))
        return self.combine_augmented_data(items)

    def get_positive_samples(self, local_idx, ds_id):
        row = self.ds_affinities[ds_id][local_idx]
        if not isinstance(row, np.ndarray):
            row = row.toarray().ravel()
        else:
            row = row.ravel()
        probs = row / row.sum()
        pos_idx = np.random.choice(
            len(row), size=self.n_augmentations, replace=False, p=probs
        )
        return [self.dataset_index_map[ds_id][i] for i in pos_idx]

    def assemble_from_indices(self, indices, sample_index):
        items = [
            super().__getitem__(i)
            # | ({self.label_key: self.adata.obs[self.label_key][i]} if self.return_label else {})
            | ({k: self.manifold[k][i] for k in self.manifold.keys()})
            | (
                {"index": np.array([i]), "sample_id": sample_index}
                if self.return_idx
                else {}
            )
            for i in indices
        ]
        return self.combine_augmented_data(items)

    def set_affinities(self):
        if self.ds_affinities is None:
            self.ds_affinities = self.load_affinities()

    def combine_augmented_data(self, augmented_data_list):
        """Combine the list of augmented data into a single batch."""
        keys_to_stack = [
            "x",
            "labeled",
            "sizefactor",
            "batch",
            "combined_batch",
            "study",
            # "celltypes",
            "index",
            "sample_id",
            "sigma",
            'heterogeneity',
            'mf_score',
            'wsigma'
        ]
        combined_data = {}
        for key in keys_to_stack:
            if key in augmented_data_list[0]:
                values = [data[key] for data in augmented_data_list]

                if all(isinstance(v, torch.Tensor) for v in values):
                    combined_data[key] = torch.stack(values)
                else:
                    combined_data[key] = np.stack(values)

        return combined_data

    def load_affinities(self):
        if os.path.exists(self.save_path):
            return pkl.load(open(self.save_path, "rb"))
        return None

    def get_joint_pca_spatial_representation(self, w=10):
        # Get PCA features
        X_pca = self.adata.obsm["X_pca"][:, :20]  # 20 PCs is usually enough

        # Get spatial coordinates
        if "spatial" in self.adata.obsm:
            X_spatial = self.adata.obsm["spatial"]
        elif {"x", "y"}.issubset(self.adata.obs.columns):
            X_spatial = self.adata.obs[["x", "y"]].to_numpy()
        else:
            raise ValueError(
                "No spatial coordinates found: neither 'spatial' in obsm nor 'x', 'y' in obs."
            )

        # Standardize both
        pca_scaled = StandardScaler().fit_transform(X_pca)
        spatial_scaled = StandardScaler().fit_transform(X_spatial)

        # Concatenate
        X_combined = np.concatenate([pca_scaled, spatial_scaled * w], axis=1)
        return X_combined

    def affinities_exists(self):
        return os.path.exists(self.save_path)
