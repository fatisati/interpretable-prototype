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

from scipy.special import softmax
from scipy import sparse
from filelock import FileLock


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
        mc_size=100,
        adoptive_eps=False,
        p=0,
        k_pos=0,
        softm=False,
        graph_mode = 'knn',
        **kwargs,
    ):
        self.graph_mode = graph_mode
        self.sc_ds = sc_ds
        self.n_proto = n_proto
        self.n_augmentations = n_augmentations
        self.adata = sc_ds.adata
        self.k_neighbors = k_neighbors
        self.n_components = n_components
        self.aff = None
        self.affinity_type = affinity_type
        self.supervised_ratio = supervised_ratio
        self.use_bknn = use_bknn

        self.save_dir = save_dir
        self.spatial = spatial
        self.set_graph_name()
        self.adata_name = self.graph_name + "_tmp.h5ad"
        os.makedirs(self.save_dir, exist_ok=True)
        # self.save_path = os.path.join(save_dir, self.graph_name)
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
        # self.data = (
        #     sc_ds.adata.layers.get("counts", sc_ds.adata.X)
        #     if use_counts
        #     else sc_ds.adata.X
        # )
        self._is_sparse = sparse.issparse(self.data)
        if not self._is_sparse:
            self.data = torch.tensor(self.data, dtype=torch.float32)

        lock_path = self.save_path + ".lock"
        if not self.affinities_exists():
            with FileLock(lock_path):
                self.aff = self.run_graph_generator(lock_path)

        self.set_affinities()
        # Save raw affinity (no row-normalization) for edge-UMAP training and modularity.
        # normalize_aff() below overwrites self.aff with a row-normalized copy, so we
        # capture the raw weights here (diagonal removed, no row-sum scaling).
        _raw = self.aff.tocsr(copy=True)
        _raw.setdiag(0)
        _raw.eliminate_zeros()
        self.aff_raw = _raw

        self.use_manifold_weights = use_manifold_weights
        self.manifold = {}
        if self.use_manifold_weights:
            self.calc_manifold_weights()
        self.label_key = sc_ds.label_key
        self.return_label = True
        self.temperature = 0.05

        self.k_pos = k_pos  # used to be 10 in scproto_v5
        self.p = p
        if self.affinity_type == "coaff":
            self.softmax = softm
        else:
            self.softmax = False
        self.adoptive_eps = adoptive_eps
        if self.adoptive_eps:
            self.sigma = self.compute_sigma()
        else:
            self.sigma = None

        self.normalize_aff()
        
    def normalize_aff(self):
        aff = self.aff.tocsr(copy=True)
        aff.setdiag(0)
        aff.eliminate_zeros()
        for i in range(aff.shape[0]):
            start, end = aff.indptr[i], aff.indptr[i + 1]
            if start == end:
                continue

            vals = aff.data[start:end]
            idx = np.arange(start, end)

            if self.p != 0:
                # scores = softmax(vals / self.temperature) if self.softmax else vals
                scores = vals / vals.sum()
                order = np.argsort(scores)[::-1]
                cum = np.cumsum(scores[order]) / scores.sum()
                keep = order[cum <= self.p]
                if len(keep) == 0:
                    keep = order[:1]
                idx = idx[keep]
                vals = vals[keep]


            if self.k_pos != 0 and len(vals) > self.k_pos:
                k = np.argpartition(-vals, self.k_pos - 1)[: self.k_pos]
                idx = idx[k]
                vals = vals[k]

            if self.softmax:
                vals = softmax(vals / self.temperature)
            else:
                vals = vals / vals.sum()

            mask = np.zeros(end - start, dtype=bool)
            mask[idx - start] = True
            aff.data[start:end][~mask] = 0
            aff.data[idx] = vals

        aff.eliminate_zeros()
        self.aff = aff

    def compute_sigma(self):
        print("computing sigma...")
        l = self.k_neighbors // 2
        sc.tl.pca(self.adata)
        X = self.adata.obsm["X_pca"].astype("float32")
        index = faiss.IndexFlatL2(X.shape[1])
        index.add(X)
        dists, _ = index.search(X, l + 1)
        print("---done----")
        return np.sqrt(dists[:, l])

    def set_graph_name(self):
        self.graph_name = f"affinity_{str(self.sc_ds)}{len(self.sc_ds.adata)}_ncomp{self.n_components}_kneighbors{self.k_neighbors}_{self.affinity_type}"

        if self.spatial:
            self.graph_name += "_spatial"
        if self.sc_ds.fold != 0:
            self.graph_name += f"_fold{self.sc_ds.fold}"
        if self.graph_mode != 'knn':
            self.graph_name += f'_{self.graph_mode}'
        self.graph_name += ".pkl"
        self.save_path = os.path.join(self.save_dir, self.graph_name)

    def calc_manifold_weights(self):
        print("calculating manifold scores...")
        self.manifold = {
            "sigma": np.zeros(self.adata.n_obs),
            "wsigma": np.zeros(self.adata.n_obs),
            "zsigma": np.zeros(self.adata.n_obs),
            "ssigma": np.zeros(self.adata.n_obs),
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
            w_sigma, z_sigma, softmax_sigma = self.sigma_transform(sigma)

            # --- row marginals / weights ---
            w = compute_row_marginals(sigma, h)

            # store in dictionary
            self.manifold["sigma"][batch_idx] = sigma
            self.manifold["wsigma"][batch_idx] = w_sigma
            self.manifold["zsigma"][batch_idx] = z_sigma
            self.manifold["ssigma"][batch_idx] = softmax_sigma
            self.manifold["heterogeneity"][batch_idx] = h
            self.manifold["mf_score"][batch_idx] = w
            for key in self.manifold.keys():
                self.adata.obs.loc[batch_idx, key] = self.manifold[key][batch_idx]

        print("---done---")

    def sigma_transform(self, sigma, alpha=0.3, clip=(0.5, 3.0)):
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
        return w, z, softmax(alpha * z)

    def run_graph_generator(self, lock_path):
        print(
            f"[{os.getpid()}] Generating affinities..., saving data to: {self.adata_path}"
        )
        self.adata.write(self.adata_path)
        args = (
            self.adata_path,
            self.affinity_type,
            self.batch_key,
            self.n_components,
            self.k_neighbors,
            self.graph_mode
        )
        # Spawn g_hand with input + output arguments
        config_file = self.save_path + ".inputs.pkl"
        with open(config_file, "wb") as f:
            pickle.dump(args, f)

        proc = subprocess.Popen(
            [
                sys.executable,
                "-u",  # unbuffered output!
                "-m",
                "interpretable_ssl.augmenters.graph_generator",
                config_file,
                self.save_path,
                lock_path,
            ],
            stdout=None,
            stderr=None,
        )
        while not os.path.exists(self.save_path):
            time.sleep(1)

        # Load result
        with open(self.save_path, "rb") as f:
            aff = pickle.load(f)

        # Cleanup temporary files
        try:
            os.remove(config_file)
            os.remove(self.adata_path)
        except FileNotFoundError:
            pass
        return aff

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
            local_idx = index % len(self.dataset_index_map[ds_id])
            pos_global_ids, pos_probs = self.get_positive_samples(local_idx, ds_id)
            global_idx = self.dataset_index_map[ds_id][local_idx]
            items.append(
                self.assemble_from_indices(pos_global_ids, global_idx, pos_probs)
            )
        return self.combine_augmented_data(items)

    def get_positive_samples(self, local_idx, ds_id):
        global_idx = self.dataset_index_map[ds_id][local_idx]
        row = self.aff.getrow(global_idx)

        cols, probs = row.indices, row.data
        
        sampled_cols = np.random.choice(
            cols, size=self.n_augmentations - 1, replace=False, p=probs
        )

        prob_map = dict(zip(cols, probs))
        sampled_probs = np.array([prob_map[c] for c in sampled_cols])

        pos_idx = np.insert(sampled_cols, 0, global_idx)
        pos_probs = np.insert(sampled_probs, 0, 1.0)
        return pos_idx, pos_probs

    def get_positive_samples_v0(self, local_idx, ds_id):
        global_idx = self.dataset_index_map[ds_id][local_idx]
        row = self.aff.getrow(global_idx)  # 1×N sparse row

        # Extract indices and values of non-zero affinities
        cols = row.indices
        vals = row.data

        # --- NEW: restrict to Top-K neighbors ---
        if self.k_pos is not None and len(vals) > self.k_pos:
            top_idx = np.argpartition(-vals, self.k_pos - 1)[: self.k_pos]
            cols = cols[top_idx]
            vals = vals[top_idx]

        # Compute probabilities only on non-zero entries
        if self.softmax:
            probs = softmax(vals / self.temperature)
        else:
            probs = vals / vals.sum()

        # Sample among non-zero indices
        sampled_cols = np.random.choice(
            cols, size=self.n_augmentations - 1, replace=False, p=probs
        )

        # --- sort sampled neighbors by similarity ---
        mask = np.isin(cols, sampled_cols)
        sampled_vals = vals[mask]
        sampled_probs = probs[mask]

        order = np.argsort(-sampled_vals)  # descending order
        sampled_cols = sampled_cols[order]  # ensure order match
        sampled_probs = sampled_probs[order]

        pos_idx = np.insert(sampled_cols, 0, global_idx)
        pos_probs = np.insert(sampled_probs, 0, 1.0)
        return pos_idx, pos_probs

    def assemble_from_indices(self, indices, sample_index, pos_probs):
        items = [
            (
                super().__getitem__(global_idx)
                # | ({self.label_key: self.adata.obs[self.label_key][i]} if self.return_label else {})
                | ({k: self.manifold[k][global_idx] for k in self.manifold.keys()})
                | (
                    {"index": np.array([global_idx]), "sample_id": sample_index}
                    if self.return_idx
                    else {}
                )
                | (
                    {
                        "cell_idx": global_idx,
                        "sim": torch.tensor(pos_probs[idx], dtype=torch.float32),
                        # "sim": torch.tensor(1.0, dtype=torch.float32)
                    }
                )
                | (
                    {"sigma": float(self.sigma[global_idx])}
                    if self.adoptive_eps
                    else {}
                )
            )
            for idx, global_idx in enumerate(indices)
        ]
        return self.combine_augmented_data(items)

    def set_affinities(self):
        if self.aff is None:
            self.aff = self.load_affinities()

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
            "heterogeneity",
            "mf_score",
            "wsigma",
            "zsigma",
            "ssigma",
            "cell_idx",
            "sim",
            "sigma",
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
        if not os.path.exists(self.save_path):
            print("waiting for aff to be generated")
        while not os.path.exists(self.save_path):
            time.sleep(1)
        return pkl.load(open(self.save_path, "rb"))

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
