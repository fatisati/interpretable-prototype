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


class MultiCropsDataset(MultiConditionAnnotatedDataset):
    def __init__(
        self,
        sc_ds,
        n_augmentations,
        augmentation_type="knn",
        k_neighbors=10,  # seacell use 50
        longest_path=3,
        dimensionality_reduction=None,
        n_components=50,
        supervised_ratio=0.1,
        use_bknn=0,
        knn_similarity="cosine",
        knn_method="faiss",
        save_dir=None,
        mask_probability=0.2,
        default_dispersion=0.1,
        spatial=0,
        return_idx=True,
        n_clusters=None,
        use_counts=True,
        n_proto=None,
        **kwargs,
    ):
        """
        Initialize the augmented dataset handler for scPoli model and trainer.

        Parameters
        ----------
        adata : `~anndata.AnnData`
            Annotated data matrix.
        n_augmentations : int
            Number of augmentations to perform for each cell in a batch.
        augmentation_type : str
            Type of augmentation to use ("knn", "cell_type", "scanpy-knn", or "negative_binomial").
        k_neighbors : int
            Number of neighbors for kNN graph (only used if augmentation_type is "knn" or "scanpy-knn").
        longest_path : int
            Maximum length of the random walk path.
        dimensity_reduction : str or None
            Type of dimensionality reduction to apply ("pca" or None).
        n_components : int or None
            Number of principal components to use for PCA if dimensity_reduction is "pca".
        kwargs : dict
            Additional arguments for the parent class.
        """
        self.n_proto = n_proto
        self.n_augmentations = n_augmentations
        self.augmentation_type = augmentation_type
        self.adata = sc_ds.adata
        self.k_neighbors = k_neighbors
        self.longest_path = longest_path
        self.dimensionality_reduction = dimensionality_reduction
        self.n_components = n_components
        self.knn_graph = None

        self.supervised_ratio = supervised_ratio
        self.use_bknn = use_bknn
        self.knn_similarity = knn_similarity
        self.knn_method = knn_method
        self.save_dir = save_dir

        self.graph_name = f"graph_{str(sc_ds)}{len(sc_ds.adata)}_{self.dimensionality_reduction}{self.n_components}_knn{self.k_neighbors}_{self.knn_method}"

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
        if self.augmentation_type not in ["cell_type", "nb"]:
            self.set_graph()
        self.mask_probability = mask_probability

        self.default_dispersion = default_dispersion

        # Calculate mean expression and default dispersion from the data
        self.mean_expression = np.mean(self.adata.X, axis=0).flatten()
        self.dispersion = (
            1 / self.adata.varm["overdispersion"]
            if "overdispersion" in self.adata.varm
            else np.full(self.mean_expression.shape, self.default_dispersion)
        )

        self.ds_index_dict = {
            ds_key: np.where(self.adata.obs[self.batch_key].values == ds_key)[0]
            for ds_key in self.adata.obs[self.batch_key].unique()
        }
        self.ds_keys = list(self.ds_index_dict.keys())
        self.current_ds_id = None
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
        if not self.graph_exists():
            self.knn_graph = self.run_graph_generator()
        self.set_graph()

    def run_graph_generator(self):
        print(f"[{os.getpid()}] Generating graph...")
        self.adata.write(self.adata_path)
        args = (
            self.adata_path,
            self.knn_method,
            self.batch_key,
            self.n_components,
            self.k_neighbors,
            self.n_proto,
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

    def __getitem__(self, index):
        if self.current_ds_id is not None:
            index = self.ds_index_dict[self.current_ds_id][index]

        items = []
        for ds_id in self.ds_index_dict.keys():
            ds_index = index % len(self.ds_index_dict[ds_id])
            global_index = self.ds_index_dict[ds_id][ds_index]
            items.append(self.normal_get_item(global_index))
        return self.combine_augmented_data(items)

    def __len__(self):
        return max(
            [len(self.ds_index_dict[ds_id]) for ds_id in self.ds_index_dict.keys()]
        )

    def normal_get_item(self, index):
        if isinstance(index, (int, np.integer)):
            # Single index
            augmented_data_list = self.augment_on_the_fly(index)
        elif isinstance(index, slice):
            # Slice of indices
            indices = range(*index.indices(len(self)))

            augmented_data_list = []
            for idx in indices:
                augmented_data_list.extend(self.augment_on_the_fly(idx))
        else:
            raise TypeError("Invalid index type")

        if self.augmentation_type == "nb" or self.augmentation_type == "mask":
            combined_data = self.combine_augmented_data(augmented_data_list)
        else:
            # Fetch the augmented data using the parent's __getitem__ method
            augmented_data_list = [
                super().__getitem__(aug_idx)
                | (
                    {"index": np.array([aug_idx]), "sample_id": index}
                    if self.return_idx
                    else {}
                )
                for aug_idx in augmented_data_list
            ]

            combined_data = self.combine_augmented_data(augmented_data_list)

        return combined_data

    def random_walk(self, start_index):
        """
        Perform a random walk on the kNN graph starting from the given index.
        """

        current_index = start_index
        path_length = np.random.randint(1, self.longest_path + 1)
        indices, distances = self.knn_graph

        for _ in range(path_length):
            neighbors = indices[current_index]
            dists = distances[current_index]

            # Handle inf and zero distances
            with np.errstate(divide="ignore", invalid="ignore"):
                weights = 1.0 / dists
                weights[np.isinf(weights)] = 0  # 1/inf = 0 (unreachable)
                weights[np.isnan(weights)] = 0  # in case of 0/0

            weights_sum = weights.sum()
            if weights_sum == 0:
                # Fallback to uniform if all weights are zero
                weights = np.ones_like(weights) / len(weights)
            else:
                weights = weights / weights_sum  # Normalize

            next_index = np.random.choice(neighbors, p=weights)
            current_index = next_index

        return current_index

    def set_graph(self):
        if self.knn_graph is None:
            self.knn_graph = self.load_graph()

    def knn_augment(self, index):
        augmented_indices = [index]  # Start with the original index
        for _ in range(self.n_augmentations - 1):
            augmented_indices.append(self.random_walk(index))
        return augmented_indices

    def augment_on_the_fly(self, index):
        if self.augmentation_type == "knn":
            return self.knn_augment(index)
        else:
            raise ValueError(f"Invalid augmentation_type: {self.augmentation_type}")

    def combine_augmented_data(self, augmented_data_list):
        """Combine the list of augmented data into a single batch."""
        keys_to_stack = [
            "x",
            "labeled",
            "sizefactor",
            "batch",
            "combined_batch",
            # "study",
            "celltypes",
            "index",
            "sample_id",
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

    def save_graph(self):
        pkl.dump(self.knn_graph, open(self.save_path, "wb"))

    def load_graph(self):
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

    def graph_exists(self):
        return os.path.exists(self.save_path)
