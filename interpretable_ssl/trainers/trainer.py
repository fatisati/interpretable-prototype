from interpretable_ssl.loss_manager import *
from scarches.models.scpoli import scPoli
import numpy as np
from interpretable_ssl.train_utils import *
from interpretable_ssl.configs.defaults import *
from scarches.dataset.scpoli.anndata import MultiConditionAnnotatedDataset
import os
from interpretable_ssl.evaluation.visualization import *
from tqdm import tqdm
from interpretable_ssl.evaluation.metrics import MetricCalculator
from torch.utils.data import DataLoader
from interpretable_ssl.utils import log_time
from interpretable_ssl.evaluation.cd4_marker import assign_prototype_labels
from interpretable_ssl.datasets.dataset import SingleCellDataset
from interpretable_ssl.datasets.dataset_configs import DATASETS
from interpretable_ssl.trainers.base import TrainerBase
import wandb

from interpretable_ssl.evaluation.metric_helpers.embedding_metrics import *
from interpretable_ssl.trainers.affinity import *
import subprocess
from sklearn.model_selection import train_test_split

from interpretable_ssl.evaluation.de_helper import *

import math
from filelock import FileLock, Timeout
import time
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import pairwise_distances
from interpretable_ssl.evaluation.mc_metric_utils import *
from interpretable_ssl.evaluation.dropout_recovery import *


class Trainer(TrainerBase):
    # @log_time('scpoli trainer')
    def __init__(self, dataset=None, ref_query=None, parser=None, **kwargs) -> None:
        parser_args = self.collect_parser_args(parser)
        kwargs.update(parser_args)
        self.dataset = dataset
        if "debug" not in kwargs:
            kwargs["debug"] = 0
        super().__init__(**kwargs)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.dataset is None:
            print(f"dataset is None, loading {self.dataset_id}")
            self.dataset = self.get_dataset(self.dataset_id)
        self.input_dim = self.dataset.x_dim
        if ref_query is None:
            if self.full_dataset_mode:
                self.ref = self.dataset
                self.query = None
            else:
                self.ref, self.query = self.dataset.get_train_test()
        else:
            self.ref, self.query = ref_query

        if self.study_id != "":
            mask = self.ref.adata.obs[self.condition_key] == self.study_id
            self.ref.adata = self.ref.adata[mask].copy()

        if self.full_dataset_mode == 1:
            train_ind = range(len(self.ref))
            val_ind = range(len(self.ref))
        else:
            train_ind, val_ind = train_test_split(
                range(len(self.ref)), test_size=0.1, random_state=42
            )
        self.train_, self.val_ = self.ref._create_split_instance(
            train_ind
        ), self.ref._create_split_instance(val_ind)
        self.calc_dataset_dc(self.train_)
        self.calc_dataset_dc(self.val_)
        self.dc_dict = {}
        self.dc_path = {
            "train": self.train_.get_dc_path(),
            "val": self.val_.get_dc_path(),
        }
        self.condition_key = self.ref.batch_key
        self.mc_size = math.ceil(len(self.dataset) / self.num_prototypes)

    def calc_dataset_dc(self, ds: SingleCellDataset):
        # Skip DC computation in debug mode for faster init
        if self.debug:
            return
        dc_path = ds.get_dc_path()
        if os.path.exists(dc_path):
            return
        else:
            dc_df = compute_dc(ds.adata, ds.batch_key, base=dc_path, remove_dc=False)
            dc_df.to_csv(dc_path)
        # lock_path = dc_path + ".lock"

        # if not os.path.exists(lock_path):
        #     print("calling calc dc...")
        #     open(lock_path, "w").close()
        #     ad_path = f"{dc_path.replace('.csv', '')}_tmp.h5ad"
        #     ds.adata.write(ad_path)
        #     subprocess.Popen(
        #         [
        #             sys.executable,
        #             "-u",
        #             "-m",
        #             "interpretable_ssl.evaluation.diffusion",
        #             ad_path,
        #             dc_path,
        #             lock_path,  # pass lock path
        #             self.dataset.batch_key,
        #         ],
        #         stdout=sys.stdout,
        #         stderr=sys.stderr,
        #     )
        # else:
        #     print(f"Skip: {dc_path} already being processed.")

    def get_dc(self, split):
        if split in self.dc_dict:
            return self.dc_dict[split]
        path = self.dc_path[split]
        # print(f"waiting for {path} to be generated...")
        # while not os.path.exists(path):
        #     time.sleep(1)  # wait 5 seconds before checking again
        # size = -1
        # while True:
        #     new_size = os.path.getsize(path)
        #     if new_size == size and new_size > 0:
        #         break
        #     size = new_size
        #     time.sleep(0.5)

        df = pd.read_csv(path, index_col=0)
        self.dc_dict[split] = df

        print("done")
        return df

    def calc_mc_quality(self, cell_ids, scores, split):
        dc = self.get_dc(split).loc[cell_ids]
        mc = scores.argmax(1).detach().cpu().numpy()

        df = pd.DataFrame(dc.values, index=cell_ids)
        df["mc"] = mc

        obs = self.train_.adata.obs if split == "train" else self.val_.adata.obs

        bk = self.dataset.batch_key
        lk = self.dataset.label_key
        keys = [bk, lk, "niches_2D", "niches_3D"]

        for k in keys:
            if k in obs.columns:
                df[k] = obs.loc[cell_ids, k].values

        niche_purity = calc_purity(df, "niches_2D", "mc")
        niche_purity3d = calc_purity(df, "niches_3D", "mc")
        cell_purity = calc_purity(df, lk, "mc")

        # Niche micro/macro purity
        niche_micro, niche_macro = None, None
        if "niches_2D" in df.columns:
            # Per-mc purity and majority label
            mc_stats = []
            for m, sub in df.groupby("mc"):
                vc = sub["niches_2D"].value_counts(normalize=True)
                mc_stats.append({"mc": m, "purity": vc.max(), "label": vc.idxmax(), "n": len(sub)})
            mc_df = pd.DataFrame(mc_stats)
            # Micro: weighted by n cells
            niche_micro = (mc_df["purity"] * mc_df["n"]).sum() / mc_df["n"].sum()
            # Macro: avg purity per niche label
            niche_macro = mc_df.groupby("label")["purity"].mean().mean()

        comp = []
        for _, sub in df.groupby("mc"):
            X = sub.drop(columns=["mc"] + keys, errors="ignore").values
            b = sub[bk].values
            mu = {u: X[b == u].mean(0) for u in np.unique(b)}
            Xc = np.vstack([X[b == u] - mu[u] for u in mu])
            comp.append((Xc**2).sum() / (len(Xc)))

        F = df.columns.difference(["mc"] + keys)
        X = df[F].values
        b = df[bk].values
        gmu = {u: X[b == u].mean(0) for u in np.unique(b)}
        Xg = np.zeros_like(X)
        for u in gmu:
            Xg[b == u] = X[b == u] - gmu[u]

        cent = {m: df.loc[df.mc == m, F].values.mean(0) for m in df.mc.unique()}
        C = np.vstack(list(cent.values()))
        nn = NearestNeighbors(n_neighbors=2).fit(C)
        d, _ = nn.kneighbors(C)
        sep = d[:, 1].mean()

        return np.mean(comp), sep, cell_purity, niche_purity, niche_purity3d, niche_micro, niche_macro

    def collect_parser_args(self, parser):
        if parser is not None:
            parser = self.add_parser_args(parser)
            args = parser.parse_args()
            args_dict = vars(args)

            # Remove keys from args_dict if their value is the string "None"
            keys_to_remove = [
                key for key, value in args_dict.items() if value == "None"
            ]
            for key in keys_to_remove:
                del args_dict[key]

            return args_dict
        return {}

    def add_parser_args(self, parser):
        default_values = get_defaults().copy()
        # Add arguments to parser with default values from dictionary
        for key, value in default_values.items():
            if isinstance(value, bool):
                # Handle boolean arguments with action='store_true'
                parser.add_argument(
                    f"--{key}",
                    action="store_true",
                    help=f"Set {key} to true (default is {value})",
                )
            else:
                # Handle other types of arguments
                arg_type = type(value) if value is not None else str
                if value == "":
                    value = None
                parser.add_argument(
                    f"--{key}",
                    type=arg_type,
                    default=value,
                    help=f"Set {key} (default is {value})",
                )
        return parser

    def load_model(self):
        model = self.get_model()
        path = self.get_model_path()
        model.load_state_dict(torch.load(path)["model_state_dict"])
        return model

    def load_query_model(self, adata=None):
        if adata is None:
            adata = self.query.adata
        model = self.load_model()
        model = self.adapt_model(model, adata)
        model.to(self.device)
        return model

    def dict_to_device(self, inputs_dict):
        for key in inputs_dict:
            inputs_dict[key] = inputs_dict[key].to(self.device)
        return inputs_dict

    def encode_adata(
        self,
        adata,
        model=None,
        return_mapped=False,
        return_mapped_idx=False,
        retrain_epochs=0,
        z_idx=0,
    ):
        model = self.adapt_model(model, adata, retrain_epochs)
        loader = self.prepare_scpoli_dataloader(
            adata, self.extract_scpoli(model), shuffle=False
        )
        embeddings = [
            self.encode_batch(model, batch, z_idx, return_mapped, return_mapped_idx)
            for batch in tqdm(loader)
        ]
        z = torch.cat(embeddings)
        # if getattr(model, "l2norm", False):
        #     z = F.normalize(z, dim=1)
        return z

    def adapt_model(self, model, adata, retrain_epochs=0):
        adapted_model = model
        if self.check_conditions_compatible(model, adata):
            if retrain_epochs == 0:
                return adapted_model
            else:
                adopted_wrapper = self.extract_scpoli(adapted_model, True)
        else:
            # make a new model not change the old one
            adapted_model = self.get_model()
            # because our model is not just an scpoli_cvae
            adapted_model.load_state_dict(model.state_dict())
            adopted_wrapper = scPoli.load_query_data(
                adata=adata,
                reference_model=self.extract_scpoli(adapted_model, True),
                labeled_indices=[],
            )
        if retrain_epochs > 0:
            adopted_wrapper.train(
                n_epochs=retrain_epochs, pretraining_epochs=retrain_epochs
            )

        adapted_model.attach_scpoli(adopted_wrapper)
        adapted_model.to(self.device)
        return adapted_model

    def prepare_scpoli_dataloader(self, adata, scpoli_cvae, shuffle=True):
        # adata = adata.copy()
        # because scpoli encoder gets raw counts as input
        # adata.X = adata.layers.get("counts", adata.X)

        if "condition_combined" not in adata.obs:
            adata.obs["conditions_combined"] = adata.obs[[self.condition_key]].apply(
                lambda x: "_".join(x), axis=1
            )
        dataset = MultiConditionAnnotatedDataset(
            adata,
            condition_keys=[self.condition_key],
            condition_encoders=scpoli_cvae.condition_encoders,
            conditions_combined_encoder=scpoli_cvae.conditions_combined_encoder,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=scpoli_utils.custom_collate,
            shuffle=shuffle,
        )
        return loader

    # TODO: return mapped and mapped_idx should have cleaner logic
    def encode_batch(
        self, model, batch, return_idx=0, return_mapped_idx=False, return_mapped=False
    ):

        batch = self.dict_to_device(batch)
        model.eval()

        with torch.no_grad():
            outs = model.encode(batch)
            z_swav, z_vae, logits = outs

        if return_mapped_idx:
            return torch.argmax(logits, dim=1)

        elif return_mapped:
            return logits

        else:
            return outs[return_idx]

    def extract_scpoli(self, scproto_model, return_wrapper=False):
        if return_wrapper:
            return scproto_model.scpoli_wrapper
        return scproto_model.scpoli_cvae

    def encode_ref(self, model=None):
        return self.encode_adata(self.ref.adata, model)

    def encode_query(self, ref_model=None):
        if ref_model is None:
            model = self.load_query_model()
        else:
            model = self.adapt_model(ref_model, self.query.adata)
        return self.encode_adata(self.query.adata, model)

    def check_conditions_compatible(self, model, adata):
        for key, values in self.extract_scpoli(
            model, return_wrapper=True
        ).conditions_.items():
            data_values = adata.obs[key].unique()
            is_subset = set(data_values).issubset(values)
            if not is_subset:
                return False
        return True

    def get_model_prototypes(self, model):
        return None

    def get_umap_path(self, data_part="ref"):
        pass

    def get_proto_assignments(self, z, model):
        scores = model.proto_soft_assignments(z)
        return scores.detach().cpu().numpy()

    def plot_umap(
        self, model, adata, split, save_plot=True, use_knn=True, assign_by_mc=False, k=5
    ):
        if adata.n_obs > 50000:
            idx = np.random.choice(adata.n_obs, 50000, replace=False)
            adata = adata[idx].copy()
        z = self.encode_adata(adata, model, z_idx=1)
        prototypes = self.get_model_prototypes(model)
        z_umap, prototype_umap, proto_labels = calc_umap_v2(
            z,
            prototypes,
            adata.obs[self.dataset.label_key],
            k,
            metric=self.umap_metric,
        )
        obs = adata.obs
        # if prototypes is not None:
        #     # prototype_assignments = self.encode_adata(adata, model, True, False)
        #     scores = self.get_proto_assignments(z, model)
        #     proto_df = assign_prototype_labels(
        #         adata,
        #         scores,
        #         self.num_prototypes,
        #         cell_type_column=self.dataset.label_key,
        #         use_knn=use_knn,
        #     )
        # proto_labels = proto_df.prototype_label

        if self.cell_w_mode != "uniform":
            w = adata.obs.get(self.cell_w_mode, None)
            w_label = self.cell_w_mode
        else:
            w = adata.obs.get("sigma", None)
            w_label = "pca_sigma"

        if (
            self.ref.adata.obs[self.ref.batch_key].nunique() == 1
            and "niches_2D" in self.ref.adata.obs
        ):
            last_labels = obs["niches_2D"]
        else:
            last_labels = obs[self.ref.batch_key]

        return plot_3umaps(
            z_umap,
            prototype_umap,
            obs[self.dataset.label_key],
            last_labels,
            proto_labels,
            save_plot,
            self.get_umap_path(split),
            w=w,
            w_label=w_label,
        )

    def plot_umap_simple(self, adata=None, color_key=None, model=None, max_cells=50000, figsize=(6, 5), show_proto_nums=True, soft_labels=False):
        """
        Simple UMAP plot: cells colored by label, prototypes labeled by majority vote.
        No density panel.

        Args:
            adata: AnnData to plot (default: self.dataset.adata or self.train_ds.adata)
            color_key: obs column for coloring, or list of columns for multiple plots
                       (default: self.dataset.label_key)
            model: model to use (default: self.model)
            max_cells: subsample if more cells
            figsize: figure size per subplot (width, height)
            show_proto_nums: whether to show prototype numbers

        Returns:
            fig, proto_labels (dict if multiple color_keys)
        """
        import matplotlib.pyplot as plt
        from collections import Counter

        if adata is None:
            adata = getattr(self, 'train_ds', None)
            adata = adata.adata if adata else self.dataset.adata
        if model is None:
            model = self.model
        if color_key is None:
            color_key = self.dataset.label_key

        # Handle list of color keys
        if isinstance(color_key, (list, tuple)):
            color_keys = list(color_key)
        else:
            color_keys = [color_key]

        ad = adata.copy() if adata.n_obs <= max_cells else adata[np.random.choice(adata.n_obs, max_cells, replace=False)].copy()

        z_tensor = self.encode_adata(ad, model, z_idx=1).detach()
        z = z_tensor.cpu().numpy()
        proto = model.get_prototypes().detach().cpu().numpy()
        n_protos = proto.shape[0]

        # Assignment by dotp (matches model's nn.Linear scoring used everywhere else)
        import torch.nn.functional as F
        scores_tensor = model.prototypes(z_tensor)              # (N, K)
        assignments   = scores_tensor.argmax(dim=1).cpu().numpy()
        if soft_labels:
            eps = getattr(self, 'epsilon', 1.0)
            print(f"[plot_umap_simple] soft_labels=True, using epsilon={eps:.6f}")
            S   = F.softmax(scores_tensor / eps, dim=1).detach().cpu().numpy()  # (N, K)

        # Joint UMAP (compute once, reuse for all plots)
        n_cells = len(z)
        combined = np.vstack([z, proto])
        tmp_ad = sc.AnnData(combined)
        sc.pp.neighbors(tmp_ad, use_rep='X', n_neighbors=15, metric='cosine', random_state=42)

        # Force cell↔assigned-prototype edges so UMAP keeps them together.
        from scipy.sparse import lil_matrix
        conn = tmp_ad.obsp['connectivities'].tolil()
        for cell_idx, proto_idx in enumerate(assignments):
            node = n_cells + proto_idx
            conn[cell_idx, node] = 1.0
            conn[node, cell_idx] = 1.0
        tmp_ad.obsp['connectivities'] = conn.tocsr()

        sc.tl.umap(tmp_ad, random_state=42)
        z_umap = tmp_ad.obsm['X_umap'][:n_cells]
        proto_umap = tmp_ad.obsm['X_umap'][n_cells:]

        # Create subplots
        n_plots = len(color_keys)
        n_cols = min(n_plots, 3)
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        all_proto_labels = {}

        for idx, ck in enumerate(color_keys):
            ax = axes[idx]
            labels = ad.obs[ck].values

            # Proto labels: hard majority vote, or soft weighted vote
            proto_labels = []
            proto_sizes  = []
            if soft_labels:
                import pandas as pd
                lbl_cat    = pd.Categorical(labels)
                onehot     = np.eye(len(lbl_cat.categories))[lbl_cat.codes]  # (N, L)
                lbl_weights = S.T @ onehot                                     # (K, L)
                eff_sizes   = S.sum(axis=0)                                    # (K,)
                for p in range(n_protos):
                    proto_sizes.append(eff_sizes[p])
                    proto_labels.append(lbl_cat.categories[lbl_weights[p].argmax()])
            else:
                for p in range(n_protos):
                    mask = assignments == p
                    n    = mask.sum()
                    proto_sizes.append(n)
                    proto_labels.append(Counter(labels[mask]).most_common(1)[0][0] if n > 0 else None)

            all_proto_labels[ck] = proto_labels

            # Plot cells
            unique_labels = np.unique(labels)
            cmap = plt.cm.get_cmap('tab20', len(unique_labels))
            label_to_color = {lbl: cmap(i) for i, lbl in enumerate(unique_labels)}

            for lbl in unique_labels:
                mask = labels == lbl
                ax.scatter(z_umap[mask, 0], z_umap[mask, 1], c=[label_to_color[lbl]], label=lbl, alpha=0.5, s=10)

            # Plot prototypes
            for i, plbl in enumerate(proto_labels):
                c = 'white' if plbl is None else label_to_color.get(plbl, 'white')
                size = max(50, min(300, proto_sizes[i] // 2))
                ax.scatter(proto_umap[i, 0], proto_umap[i, 1], c=[c], edgecolor='black', s=size, linewidth=1, zorder=10)
                if show_proto_nums:
                    ax.annotate(str(i), (proto_umap[i, 0], proto_umap[i, 1]), fontsize=7, ha='center', va='center')

            n_used = sum(1 for s in proto_sizes if s > 0)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.set_title(f'{ck} ({n_used}/{n_protos} protos)')

        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        # Return single proto_labels if single color_key, else dict
        if len(color_keys) == 1:
            return fig, all_proto_labels[color_keys[0]]
        return fig, all_proto_labels

    def save_umap_data(self, save_dir=None, model=None, max_cells=50000):
        """Compute UMAP and save output files to save_dir.

        umap_cells.csv      — UMAP subset (≤max_cells): cell_id, umap_1, umap_2, metacell_id, label/batch cols
        umap_protos.csv     — one row per prototype: proto_id, umap_1, umap_2, n_cells, majority_{label_key}
        proto_vectors.npy   — prototype latent vectors, shape (n_protos, latent_dim)
        cell_assignments.csv — ALL cells: cell_id, metacell_id, label/batch cols (no subsampling)
        """
        import pandas as pd
        from collections import Counter

        if save_dir is None:
            save_dir = self.get_dump_path()
        if model is None:
            model = self.model

        adata = getattr(self, 'train_ds', None)
        adata = adata.adata if adata else self.dataset.adata

        ad = adata.copy() if adata.n_obs <= max_cells else adata[np.random.choice(adata.n_obs, max_cells, replace=False)].copy()

        z_tensor = self.encode_adata(ad, model, z_idx=1).detach()
        z = z_tensor.cpu().numpy()
        proto = model.get_prototypes().detach().cpu().numpy()
        n_protos = proto.shape[0]

        assignments = model.prototypes(z_tensor).argmax(dim=1).cpu().numpy()

        n_cells = len(z)
        combined = np.vstack([z, proto])
        tmp_ad = sc.AnnData(combined)
        sc.pp.neighbors(tmp_ad, use_rep='X', n_neighbors=15, metric='cosine', random_state=42)

        # Force cell↔assigned-prototype edges so UMAP keeps them together.
        # Without this, dense clusters fill all 15 NN slots with other cells
        # and prototypes end up with no graph edge to their assigned cells.
        # Use cosine similarity as the edge weight, consistent with the metric.
        from sklearn.preprocessing import normalize
        z_norm = normalize(z, norm='l2')
        proto_norm = normalize(proto, norm='l2')
        conn = tmp_ad.obsp['connectivities'].tolil()
        for cell_idx, proto_idx in enumerate(assignments):
            weight = float(np.dot(z_norm[cell_idx], proto_norm[proto_idx]))
            node = n_cells + proto_idx
            conn[cell_idx, node] = max(conn[cell_idx, node], weight)
            conn[node, cell_idx] = max(conn[node, cell_idx], weight)
        tmp_ad.obsp['connectivities'] = conn.tocsr()

        sc.tl.umap(tmp_ad, random_state=42)
        z_umap = tmp_ad.obsm['X_umap'][:n_cells]
        proto_umap = tmp_ad.obsm['X_umap'][n_cells:]

        label_key = self.dataset.label_key
        batch_key = self.dataset.batch_key

        cells_df = pd.DataFrame({'cell_id': ad.obs_names, 'umap_1': z_umap[:, 0], 'umap_2': z_umap[:, 1], 'metacell_id': assignments})
        for col in [label_key, batch_key]:
            if col and col in ad.obs.columns:
                cells_df[col] = ad.obs[col].values
        cells_df.to_csv(os.path.join(save_dir, 'umap_cells.csv'), index=False)

        # Save assignments for ALL cells (no subsampling) so any loaded ad can be joined by barcode.
        all_ad = adata
        all_z = self.encode_adata(all_ad, model, z_idx=1).detach()
        all_assignments = model.prototypes(all_z).argmax(dim=1).cpu().numpy()
        assign_df = pd.DataFrame({'cell_id': all_ad.obs_names, 'metacell_id': all_assignments})
        for col in [label_key, batch_key]:
            if col and col in all_ad.obs.columns:
                assign_df[col] = all_ad.obs[col].values
        assign_df.to_csv(os.path.join(save_dir, 'cell_assignments.csv'), index=False)

        # Use all_assignments/all_ad for counts and majority labels so they reflect the full dataset.
        niche_key = getattr(self.dataset, 'niche_key', None)
        lk_vals = all_ad.obs[label_key].values if label_key and label_key in all_ad.obs.columns else None
        nk_vals = all_ad.obs[niche_key].values if niche_key and niche_key in all_ad.obs.columns else None
        proto_rows = []
        for p in range(n_protos):
            mask = all_assignments == p
            n = int(mask.sum())
            row = {'proto_id': p, 'umap_1': proto_umap[p, 0], 'umap_2': proto_umap[p, 1], 'n_cells': n}
            if lk_vals is not None:
                row[f'majority_{label_key}'] = Counter(lk_vals[mask]).most_common(1)[0][0] if n > 0 else None
            if nk_vals is not None:
                row[f'majority_{niche_key}'] = Counter(nk_vals[mask]).most_common(1)[0][0] if n > 0 else None
            proto_rows.append(row)
        pd.DataFrame(proto_rows).to_csv(os.path.join(save_dir, 'umap_protos.csv'), index=False)

        np.save(os.path.join(save_dir, 'proto_vectors.npy'), proto)

        print(f"UMAP data saved to {save_dir}")

    def plot_ref_umap(self, save_plot=True, name_postfix=None, model=None):

        if model is None:
            model = self.load_model()
        if name_postfix is not None:
            name = f"ref-{name_postfix}"
        else:
            name = f"ref"
        return self.plot_umap(model, self.ref.adata, name, save_plot)

    def plot_query_umap(self, save_plot=True):
        model = self.load_query_model()
        return self.plot_umap(model, self.query.adata, "query", save_plot)

    def calc_scib(self, adata, name, other={}, save=True, is_ref=False):
        latent = self.encode_adata(adata, self.model)
        res_df = MetricCalculator(
            adata,
            [latent],
            self.dump_path,
            save_path=self.get_metric_file_path(name),
        ).calculate(other, save)
        return res_df

    def save_metacell_metrics(self):
        ad = self.train_.adata.copy()

        mc_ad, sim, z = get_scproto_mc_adata(
            self,
            ad,
            self.dataset.batch_key,
            self.dataset.label_key,
            self.epsilon,
            model=self.model,
            similarity=self.lsim,
            pl_version=self.pl_version,
        )
        ad.obs["SEACell"] = sim.argmax(axis=1)
        ad.obs["mc_idx"] = ad.obs["SEACell"].values
        protos = self.model.get_prototypes()
        sample_protos = protos[ad.obs["SEACell"]]
        batch = self.train_ds.conditions.to("cuda")
        ad.layers["metacell"] = (
            self.model.decode(sample_protos, batch).detach().cpu().numpy()
        )

        ad.obsm["dc"] = self.get_dc("train").values
        save_all_mc_metrics(
            ad,
            mc_ad,
            self.dataset.label_key,
            self.dataset.batch_key,
            self.get_dump_path(),
            epsilon=self.epsilon,
            name=self.get_model_name(),
            z = z.detach().cpu().numpy()
        )
        # for seacell, g in ad.obs.groupby("SEACell"):
        #     idx = mc_ad.obs.index == f"proto_{seacell}"
        #     for k in spatial_labels + []:

        #         mc_ad.obs.loc[idx, k] = g[k].value_counts().idxmax()

    def save_metrics(self):
        adata = add_trainer_emb(self, self.dataset.adata)
        if adata.X.max() > 50:
            adata = adata.copy()
            adata.X = adata.layers["lognorm"]
        params = (
            adata,
            [self.get_model_name()],
            self.dataset.batch_key,
            self.dataset.label_key,
        )
        adata.obs[self.dataset.label_key] = adata.obs[self.dataset.label_key].astype(
            "category"
        )
        adata.obs[self.dataset.batch_key] = adata.obs[self.dataset.batch_key].astype(
            "category"
        )

        scg, scb = get_metrics(*params)
        scg.to_csv(self.get_dump_path() + "/scgraph.csv")
        if scb is not None:
            scb.to_csv(self.get_dump_path() + "/scib.csv")
        self.save_metacell_metrics()

    def get_dataset(self, dataset_id):
        ds_params = DATASETS[dataset_id]
        return SingleCellDataset(
            name=dataset_id, use_counts=self.recon_loss == "nb", **ds_params
        )

    def init_wandb(self, path=None):
        if self.debug == 1:
            return
        wandb.init(
            name=self.job_name,
            # project="interpretable-ssl",
            config={
                "num_prototypes": self.num_prototypes,
                "hidden dim": self.hidden_dim,
                "latent_dims": self.latent_dims,
                "device": self.device,
                "model path": path,
                "dataset": self.dataset,
                "train size": len(self.ref),
                "test size": len(self.query) if self.query is not None else 0,
                "batch size": self.batch_size,
            },
        )

    def set_job_name(self, path=None):
        if path is None:
            path = self.get_dump_path()
        set_job_name = (self.job_name is None) or (self.job_name == "")
        if set_job_name:
            self.job_name = f"{self.get_model_name()}/{self.dataset}"
