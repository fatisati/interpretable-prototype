import argparse
import math
import os
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import shutil
import time
from logging import getLogger
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
from interpretable_ssl.configs.larc import LARC

from swav.src.utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter,
)

from interpretable_ssl.trainers.adaptive_trainer import AdoptiveTrainer
from interpretable_ssl.augmenters.adata_augmenter import *
from scarches.models.scpoli import scPoli
import scarches.trainers.scpoli._utils as scpoli_utils
from interpretable_ssl.models.swav import *
import wandb
import multiprocessing as mp
from interpretable_ssl.evaluation.visualization import *
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from interpretable_ssl.configs.defaults import *
import sys
from interpretable_ssl.utils import *

from interpretable_ssl.evaluation.prototype_metrics import *
import torch
from collections import Counter, defaultdict
from interpretable_ssl.evaluation.cd4_marker import *
from interpretable_ssl.trainers.scproto_utils import *

from interpretable_ssl.trainers.affinity import *
from interpretable_ssl.trainers.scpoli_helpers import *
from interpretable_ssl.evaluation.mc_metric_utils import *
from interpretable_ssl.trainers.edge_umap import EdgeDataset, ParametricUMAPLoss, edge_collate_fn

logger = getLogger()


def get_gpu_type_torch():
    if not torch.cuda.is_available():
        return "No GPU available"
    return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]


def affinity_report(A):
    """Report affinity matrix statistics."""
    import scipy.sparse as sp
    A = A.tocsr(copy=True)
    A.setdiag(0)
    A.eliminate_zeros()

    nnz = np.diff(A.indptr)
    mean_deg = nnz.mean()
    med_deg = np.median(nnz)

    ent = np.zeros(A.shape[0])
    effk = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        s, e = A.indptr[i], A.indptr[i+1]
        if s == e:
            continue
        p = A.data[s:e]
        p = p / (p.sum() + 1e-12)
        ent[i] = -(p * np.log(p + 1e-12)).sum()
        effk[i] = np.exp(ent[i])

    B = A.astype(bool)
    mutual = (B.multiply(B.T)).sum()
    total = B.sum()
    mutual_ratio = float(mutual / (total + 1e-12))

    return {
        "mean_deg": float(mean_deg),
        "med_deg": float(med_deg),
        "effk_mean": float(effk.mean()),
        "effk_med": float(np.median(effk)),
        "mutual_ratio": mutual_ratio,
        "frac_empty_rows": float((nnz == 0).mean()),
    }


def _med_effk_from_logits(logits, temp):
    p = torch.softmax(logits / temp, dim=1)
    effk = 1.0 / (p * p).sum(dim=1)
    return torch.median(effk).item()


def _solve_temp_for_target_effk(logits, target_effk, iters=40, lo=1e-3, hi=50.0):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        med = _med_effk_from_logits(logits, mid)
        if med < target_effk:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def calibrate_eps_tau(model, z, effk_aff, student_factor=2.0):
    """Calibrate epsilon and tau based on target effective neighbors."""
    with torch.no_grad():
        logits = model.proto_soft_assignments(z)
    eps = _solve_temp_for_target_effk(logits, target_effk=effk_aff)
    tau = _solve_temp_for_target_effk(logits, target_effk=student_factor * effk_aff)
    return eps, tau


class SCProtoTrainer(AdoptiveTrainer):

    # @log_time('swav')
    def __init__(self, dataset=None, ref_query=None, parser=None, **kwargs):
        logger.info(get_gpu_type_torch())

        if "experiment_name" not in kwargs:
            kwargs["experiment_name"] = "scproto"
        super().__init__(dataset, ref_query, parser, **kwargs)
        self.nmb_prototypes = self.num_prototypes
        self.use_projector_out = False

        self.train_augmentation = self.augmentation_type
        self.queue = {}
        if self.condition_key is not None:
            ds_cnt = self.train_.adata.obs[self.condition_key].nunique()
        else:
            ds_cnt = 1
        self.ds_ids = range(ds_cnt)
        self.loss_keys = [
            "swav",
            "recon",
            "kl",
            "proto",
            "commit",
            "aff",
            "proto_recon",
            "r1r2",
        ]
        self.log_hist = {}

    def setup(self):
        fix_random_seeds(self.seed)
        self.dump_path = self.get_dump_path()
        if self.wandb_sweep == 0 and self.debug == 0:
            logger, self.training_stats = initialize_exp(
                self, "epoch", "loss", dump_params=self.wandb_sweep == 0
            )
        self.build_model()
        self.build_data()
        self.build_optimizer()
        self._temperature0 = float(self.temperature)
        self._epsilon0 = float(self.epsilon)
        # self.load_checkpoint()

    def calibrate_temperatures(self, effk_target=None, student_factor=2.0):
        """Calibrate epsilon and tau from data based on affinity effective neighbors."""
        if effk_target is None:
            if hasattr(self, 'aff_stats'):
                effk_target = self.aff_stats['effk_med']
            else:
                logger.warning("No affinity stats available, using default effk=10")
                effk_target = 10.0

        z = self.encode_adata(self.train_ds.adata, self.model, z_idx=1)
        eps, tau = calibrate_eps_tau(self.model, z, effk_target, student_factor)

        old_eps, old_tau = self.epsilon, self.temperature
        self.epsilon = eps
        self.temperature = tau
        # Update initial values for scheduler
        self._epsilon0 = float(eps)
        self._temperature0 = float(tau)

        logger.info(f"Calibrated: eps {old_eps:.4f} -> {eps:.4f}, tau {old_tau:.4f} -> {tau:.4f} (target effk={effk_target:.1f})")
        print(f"🎯 Calibrated: eps={eps:.4f}, tau={tau:.4f} (from effk={effk_target:.1f})")
        return eps, tau

    def _quick_niche_metrics(self, exclude_labels=("Excluded",)):
        """Quick niche purity metrics (micro/macro) for debug mode."""
        ad = self.train_ds.adata
        if "niches_2D" not in ad.obs.columns:
            return None, None, None, None

        with torch.no_grad():
            scores = self.encode_adata(ad, self.model, z_idx=2)
        mc = scores.argmax(1).detach().cpu().numpy()
        labels = np.array(ad.obs["niches_2D"].values)

        from collections import Counter, defaultdict

        # Compute stats for all cells
        mc_stats = []
        for m in np.unique(mc):
            mask = mc == m
            vc = Counter(labels[mask])
            total = mask.sum()
            majority = vc.most_common(1)[0]
            mc_stats.append({"purity": majority[1] / total, "label": majority[0], "n": total})

        if not mc_stats:
            return None, None, None, None

        # Full metrics (all labels)
        total_n = sum(s["n"] for s in mc_stats)
        niche_micro = sum(s["purity"] * s["n"] for s in mc_stats) / total_n
        label_purities = defaultdict(list)
        for s in mc_stats:
            label_purities[s["label"]].append(s["purity"])
        niche_macro = np.mean([np.mean(p) for p in label_purities.values()])

        # Filtered metrics (exclude labels)
        valid_mask = ~np.isin(labels, list(exclude_labels))
        if valid_mask.sum() == 0:
            return niche_micro, niche_macro, None, None

        mc_filt = mc[valid_mask]
        labels_filt = labels[valid_mask]
        mc_stats_filt = []
        for m in np.unique(mc_filt):
            mask = mc_filt == m
            vc = Counter(labels_filt[mask])
            total = mask.sum()
            majority = vc.most_common(1)[0]
            mc_stats_filt.append({"purity": majority[1] / total, "label": majority[0], "n": total})

        total_n_filt = sum(s["n"] for s in mc_stats_filt)
        niche_micro_filt = sum(s["purity"] * s["n"] for s in mc_stats_filt) / total_n_filt
        label_purities_filt = defaultdict(list)
        for s in mc_stats_filt:
            label_purities_filt[s["label"]].append(s["purity"])
        niche_macro_filt = np.mean([np.mean(p) for p in label_purities_filt.values()])

        return niche_micro, niche_macro, niche_micro_filt, niche_macro_filt

    def _niche_knn_acc(self, k=15):
        """Compute KNN accuracy for niche prediction in latent space vs PCA."""
        ad = self.train_ds.adata
        if "niches_2D" not in ad.obs.columns:
            return None, None

        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_score

        y = ad.obs["niches_2D"].values

        # PCA baseline (compute once and cache)
        if not hasattr(self, '_pca_knn_acc'):
            X_pca = ad.obsm.get('X_pca', ad.X[:, :50] if hasattr(ad.X, 'toarray') else ad.X[:, :50])
            if hasattr(X_pca, 'toarray'):
                X_pca = X_pca.toarray()
            self._pca_knn_acc = cross_val_score(KNeighborsClassifier(k), X_pca, y, cv=3).mean()

        # Latent
        with torch.no_grad():
            z = self.encode_adata(ad, self.model, z_idx=1).detach().cpu().numpy()
        acc_z = cross_val_score(KNeighborsClassifier(k), z, y, cv=3).mean()

        return self._pca_knn_acc, acc_z

    def niche_report(self, k=15, exclude_labels=("Excluded",)):
        """Detailed per-niche diagnostic report."""
        ad = self.train_ds.adata
        if "niches_2D" not in ad.obs.columns:
            print("No niches_2D in adata")
            return None

        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.metrics import f1_score
        import pandas as pd

        y = np.array(ad.obs["niches_2D"].values)
        niches = [n for n in np.unique(y) if n not in exclude_labels]

        # Get embeddings
        X_pca = ad.obsm.get('X_pca')
        if X_pca is None:
            X_pca = ad.X.toarray() if hasattr(ad.X, 'toarray') else ad.X
        with torch.no_grad():
            z = self.encode_adata(ad, self.model, z_idx=1).detach().cpu().numpy()

        # Get assignments
        with torch.no_grad():
            scores = self.encode_adata(ad, self.model, z_idx=2)
        assignments = scores.argmax(1).cpu().numpy()

        # Filter out excluded cells (same as training log's niche_macro_filt)
        from collections import Counter, defaultdict
        valid_mask = ~np.isin(y, list(exclude_labels))
        y_filt = y[valid_mask]
        assignments_filt = assignments[valid_mask]

        # Compute per-proto stats using filtered cells (same as training log)
        proto_stats = []
        for p in np.unique(assignments_filt):
            mask_p = assignments_filt == p
            vc = Counter(y_filt[mask_p])
            total = mask_p.sum()
            majority_label, majority_count = vc.most_common(1)[0]
            proto_stats.append({
                "proto": p,
                "label": majority_label,
                "purity": majority_count / total,
                "n": total
            })

        # Group protos by majority label
        label_to_protos = defaultdict(list)
        for ps in proto_stats:
            label_to_protos[ps["label"]].append(ps)

        # Fit KNN once (not per-niche)
        knn_pca = KNeighborsClassifier(k).fit(X_pca, y)
        knn_z = KNeighborsClassifier(k).fit(z, y)

        # Per-niche metrics
        rows = []
        for niche in niches:
            mask = y == niche
            n_cells = mask.sum()

            # Purity: avg purity of protos with this niche as majority label (same as training log)
            protos_for_niche = label_to_protos.get(niche, [])
            n_protos = len(protos_for_niche)
            purity = np.mean([ps["purity"] for ps in protos_for_niche]) if protos_for_niche else 0

            # Coverage: fraction of niche cells in protos with this majority label
            cells_in_niche_protos = sum(ps["n"] for ps in protos_for_niche)
            coverage = cells_in_niche_protos / n_cells if n_cells > 0 else 0

            # KNN recall for this niche (PCA vs latent)
            pred_pca = knn_pca.predict(X_pca[mask])
            pred_z = knn_z.predict(z[mask])
            recall_pca = (pred_pca == niche).mean()
            recall_z = (pred_z == niche).mean()

            rows.append({
                "niche": niche,
                "n_cells": n_cells,
                "n_protos": n_protos,
                "purity": purity,
                "coverage": coverage,
                "knn_pca": recall_pca,
                "knn_z": recall_z,
                "knn_delta": recall_z - recall_pca,
            })

        df = pd.DataFrame(rows).sort_values("knn_delta", ascending=False)

        print("=" * 80)
        print("PER-NICHE REPORT (sorted by KNN improvement)")
        print("=" * 80)
        print(f"{'Niche':<25} {'N':>6} {'#Proto':>6} {'Purity':>7} {'Cover':>6} {'KNN_pca':>8} {'KNN_z':>7} {'Delta':>7}")
        print("-" * 80)
        for _, r in df.iterrows():
            delta_str = f"{r['knn_delta']:+.1%}"
            print(f"{r['niche']:<25} {r['n_cells']:>6} {r['n_protos']:>6} {r['purity']:>7.1%} {r['coverage']:>6.1%} {r['knn_pca']:>8.1%} {r['knn_z']:>7.1%} {delta_str:>7}")
        print("-" * 80)
        # Only average purity over niches with ≥1 proto (same as training log)
        df_with_protos = df[df['n_protos'] > 0]
        mean_purity = df_with_protos['purity'].mean() if len(df_with_protos) > 0 else 0
        print(f"{'MEAN':<25} {'':<6} {df['n_protos'].sum():>6} {mean_purity:>7.1%} {df['coverage'].mean():>6.1%} {df['knn_pca'].mean():>8.1%} {df['knn_z'].mean():>7.1%} {df['knn_delta'].mean():+7.1%}")
        print("=" * 80)

        return df

    def build_data(self):

        # train, val = self.split_train_test(self.ref)
        # self.train_adata, self.val_adata = train, val

        # why nmb_crops is a list? i used fisrt element but not change it in case needed in furure
        scpoli_encoder = self.model.scpoli_cvae
        common_dataset_kwargs = dict(
            n_augmentations=self.nmb_views,
            affinity_type=self.affinity_type,
            k_neighbors=self.k_neighbors,
            n_components=self.n_components,
            use_bknn=self.use_bknn,
            condition_keys=[self.condition_key],
            save_dir="./graphs",
            mask_probability=self.mask_probability,
            default_dispersion=self.default_dispersion,
            spatial=self.spatial,
            n_clusters=self.num_prototypes,
            use_counts=(self.use_counts == 1),
            n_proto=self.nmb_prototypes,
            use_manifold_weights=(self.cell_w_mode != "uniform"),
            mc_size=self.mc_size,
            adoptive_eps=(self.adoptive_eps == 1),
            p=self.p,
            k_pos=self.k_pos,
            softm=(self.softm == 1),
            graph_mode=self.graph_mode,
            condition_encoders=scpoli_encoder.condition_encoders,
            conditions_combined_encoder=scpoli_encoder.conditions_combined_encoder,
            # cell_type_keys=[self.cell_type_key],
            # cell_type_encoder=model.cell_type_encoder,
        )
        self.train_ds = MultiCropsDataset(self.train_, **common_dataset_kwargs)
        self.val_.adata = add_condition_combined(
            self.val_.adata, [self.train_.batch_key]
        )
        self.test_ds = MultiCropsDataset(self.val_, **common_dataset_kwargs)

        # Print affinity report if available
        if hasattr(self.train_ds, 'aff') and self.train_ds.aff is not None:
            self.aff_stats = affinity_report(self.train_ds.aff)
            logger.info(f"Affinity stats: {self.aff_stats}")
            print(f"📊 Affinity: mean_deg={self.aff_stats['mean_deg']:.1f}, effk_med={self.aff_stats['effk_med']:.1f}, mutual={self.aff_stats['mutual_ratio']:.2%}")

        self.train_loader = self.get_data_laoder(self.train_ds)
        self.test_loader = self.get_data_laoder(self.test_ds, drop_last=False)

        self.original_train_loader = self.train_loader

    def get_data_laoder(self, ds, drop_last=True):
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=True,
            # drop_last=drop_last,
            collate_fn=scpoli_utils.custom_collate,
            shuffle=True,
        )

    def get_model_path(self):
        return os.path.join(self.get_dump_path(), self.get_checkpoint_file())

    def load_model(self):
        model = self.get_model()
        if self.ft_epochs > 0 and self.full_dataset_mode == 0:
            model = self.adapt_model(model, self.query.adata)
        checkpoint_path = self.get_model_path()
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self.device)

        # self.optimizer.load_state_dict(checkpoint["optimizer"])
        # model, _ = apex.amp.initialize(model, self.optimizer, opt_level="O1")

        # model = apex.amp.initialize(model, opt_level="O1")

        # self.model = model
        return model

    def init_prototypes(self):
        if self.prot_init == "kmeans" and self.decodable_prototypes == 0:
            logger.info("initalizing prototypes using kmeans")
            embeddings = self.encode_adata(self.train_ds.adata, self.model, z_idx=1)
            self.model.init_prototypes_kmeans(embeddings, self.nmb_prototypes)

    def build_model(self):
        self.model = self.get_model()
        self.model = self.model.cuda()
        # logger.info(self.model)
        logger.info(
            f"=======>Building model done. max value for adata fed to scpoli_wrapper: {self.model.scpoli_wrapper.adata.X.max()}"
        )

    def get_model(self):
        # if self.model_version == 1:
        # return model
        # else:
        kwargs = {
            "latent_dim": self.latent_dims,
            "nmb_prototypes": self.num_prototypes,
            "adata": self.train_.adata,
            "multi_layer_proto": self.multi_layer_protos,
            "np2": self.num_prototypes,
            "recon_loss": self.recon_loss,
            "batch_key": self.train_.batch_key,
            "l2norm": self.l2norm,
            "assignment_metric": self.assignment_metric,
            "recon_v": self.recon_v,
        }

        if self.model_type == "gm" or self.model_type == "normal":
            return scProtoGMVAE(
                temperature=self.epsilon,
                beta=self.beta,
                recon_version=self.version,
                kl_type=self.model_type,
                **kwargs,
            )
        if self.model_type == "vqvae":
            return scProtoVQVAE(
                temperature=self.temperature,
                beta=self.beta,
                recon_update_target=self.recon_update_target,
                **kwargs,
            )
        if self.model_type == "hybrid":
            return scProtoHybrid(temperature=self.temperature, beta=self.beta, **kwargs)
        else:
            return SwAVModel(**kwargs)

    def build_optimizer(self):
        opt_type = self.opt
        print(opt_type)
        # ---- Choose optimizer ----
        if opt_type == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.base_lr,
                momentum=0.9,
                weight_decay=self.wd,
            )
            # LARC only makes sense for SGD
            optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)

        elif opt_type == "adam":
            # optimizer = torch.optim.Adam(
            #     self.model.parameters(),
            #     lr=self.base_lr,
            #     weight_decay=self.wd,
            # )

            # ----- optimizer -----
            params_embedding = []
            params = []
            for name, p in self.model.scpoli_cvae.named_parameters():
                if p.requires_grad:
                    if "embedding" in name:
                        params_embedding.append(p)
                    else:
                        params.append(p)

            optimizer = torch.optim.Adam(
                [
                    {"params": params_embedding, "weight_decay": 0},
                    {"params": params},
                ],
                lr=self.base_lr,  # 1e-3,
                eps=0.01,
                weight_decay=self.wd,  # 0.04,
            )

        elif opt_type == "wadam":
            optimizer = torch.optim.AdamW(  # usually better for weighted Adam
                self.model.parameters(),
                lr=self.base_lr,
                weight_decay=self.wd,
            )

        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")

        self.optimizer = optimizer

        # ---- LR schedule (warmup → cosine) ----
        total_iters = len(self.train_loader)

        warmup_iters = total_iters * self.warmup_epochs
        warmup_lr = np.linspace(self.start_warmup, self.base_lr, warmup_iters)

        cosine_iters = total_iters * (self.pretraining_epochs - self.warmup_epochs)
        t = np.arange(cosine_iters)

        cosine_lr = self.final_lr + 0.5 * (self.base_lr - self.final_lr) * (
            1 + np.cos(np.pi * t / cosine_iters)
        )

        self.lr_schedule = np.concatenate((warmup_lr, cosine_lr))

        logger.info(f"Optimizer '{opt_type}' built successfully.")

    def get_checkpoint_file(self):
        if self.finetuning:
            checkpoint_file = "finetuned-checkpoint.pth.tar"
        elif self.training_type == "semi_supervised":
            checkpoint_file = "semi-pretrain-checkpoint.pth.tar"
        else:
            checkpoint_file = "checkpoint.pth.tar"
        return checkpoint_file

    def save_checkpoint(self, epoch):
        if self.wandb_sweep == 1:
            print("not saving checkpoint", self.debug, self.wandb_sweep)
            return
        save_dict = {
            "epoch": epoch + 1,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        # Note: fp16/amp state saving removed (was using apex.amp which is not available)
        # If you need mixed precision, use torch.cuda.amp.GradScaler instead

        checkpoint_file = self.get_checkpoint_file()
        torch.save(save_dict, os.path.join(self.dump_path, checkpoint_file))
        # Skip epoch-specific copies in debug mode to save disk space
        if not self.debug and (epoch % self.checkpoint_freq == 0 or epoch == self.pretraining_epochs - 1):
            shutil.copyfile(
                os.path.join(self.dump_path, checkpoint_file),
                os.path.join(self.dump_path, f"ckp-{epoch}.pth"),
            )

    def log_wandb_loss(self, scores, epoch):
        log_dict = scores
        log_dict["epoch"] = epoch
        if not self.debug:
            wandb.log(log_dict)
        # else:
        #     for k in log_dict:
        #         if k not in self.log_hist:
        #             self.log_hist[k] = []
        #         self.log_hist[k].append(log_dict[k])
        #     print(log_dict)

    def train(self, epochs=None):
        self.lambda_loss = self.init_lambda_loss()
        # self.setup_scheduler()
        self.create_dump_path()
        self.build_optimizer()
        cudnn.benchmark = True
        if epochs is None:
            epochs = self.pretraining_epochs
        self.n_epochs = epochs
        for epoch in range(epochs):
            logger.info(f"============ Starting epoch {epoch}============")

            if (epoch % self.umap_checkpoint_freq == 10) and (self.wandb_sweep == 0) and not self.debug:
                self.plot_umap(self.model, self.train_ds.adata, f"train-e{epoch}")

            if (
                self.queue_length > 0
                and epoch >= self.epoch_queue_starts
                and len(self.queue) == 0
            ):
                print(f"start using queue at epoch: {epoch}")
                for ds_id in self.ds_ids:
                    self.init_queue(ds_id)

            train_meters = self.train_epoch(epoch)

            # Skip test_epoch and quality metrics in debug mode for faster training
            if self.debug:
                test_meters = {}
            elif self.full_dataset_mode == 1:
                test_meters = {}
            else:
                test_meters = self.test_epoch()
                test_meters = {f"test_{key}": val for key, val in test_meters.items()}

            # Skip expensive per-epoch quality metrics in debug mode
            if not self.debug:
                scores = self.encode_adata(self.train_.adata, self.model, z_idx=2)
                cell_ids = self.train_.adata.obs_names
                (
                    train_meters["overal_compactness"],
                    train_meters["overal_separation"],
                    train_meters["celltype_purity"],
                    train_meters["niche_purity"],
                    train_meters["niche_purity3d"],
                    train_meters["niche_micro"],
                    train_meters["niche_macro"],
                ) = self.calc_mc_quality(cell_ids, scores, "train")
                ad = self.train_.adata
                ad.obs["mc"] = scores.argmax(1).detach().cpu().numpy()
                if "spatial" in ad.obsm:
                    train_meters["spatial_compactness"] = spatial_compactness(
                        ad, mc_key="mc", bk=self.dataset.batch_key
                    ).mean()
            else:
                # Quick niche metrics in debug mode
                (train_meters["niche_micro"], train_meters["niche_macro"],
                 train_meters["niche_micro_filt"], train_meters["niche_macro_filt"]) = self._quick_niche_metrics()

            # KNN accuracy diagnostic (every 2 epochs)
            if epoch % 2 == 0 or epoch == epochs - 1:
                pca_acc, z_acc = self._niche_knn_acc()
                if pca_acc is not None:
                    train_meters["knn_pca"] = pca_acc
                    train_meters["knn_z"] = z_acc

            self.log_wandb_loss(train_meters | test_meters, epoch)
            self.save_checkpoint(epoch)

            # Print epoch summary for progress tracking
            nm = train_meters.get('niche_micro_filt') or train_meters.get('niche_micro')
            nM = train_meters.get('niche_macro_filt') or train_meters.get('niche_macro')
            nm_str = f"{nm:.3f}" if nm is not None else "-"
            nM_str = f"{nM:.3f}" if nM is not None else "-"
            knn_str = ""
            if "knn_z" in train_meters:
                knn_str = f" | KNN: {train_meters['knn_z']:.1%} (pca:{train_meters['knn_pca']:.1%})"
            print(f">>> Epoch {epoch+1}/{epochs} | Loss: {train_meters.get('loss', 0):.4f} | niche_mi: {nm_str} | niche_Ma: {nM_str} | unused: {train_meters.get('proto_unused', 0):.1%}{knn_str}")

        # if self.ft_epochs > 0:
        #     self.model = self.adapt_model(self.model, self.query.adata, self.ft_epochs)
        #     self.save_checkpoint(epoch + self.ft_epochs)

        self._total_epochs_trained = getattr(self, '_total_epochs_trained', 0) + epochs
        return train_meters | test_meters

    def continue_training(self, epochs):
        """
        Continue training for additional epochs without resetting optimizer or LR schedule.

        Args:
            epochs: number of additional epochs to train
        """
        if not hasattr(self, 'optimizer') or self.optimizer is None:
            raise RuntimeError("No optimizer found. Call train() first.")

        start_epoch = getattr(self, '_total_epochs_trained', 0)

        # extend LR schedule if needed
        total_iters_needed = (start_epoch + epochs) * len(self.train_loader)
        if len(self.lr_schedule) < total_iters_needed:
            # extend with final_lr
            extra = total_iters_needed - len(self.lr_schedule)
            self.lr_schedule = np.concatenate([
                self.lr_schedule,
                np.full(extra, self.final_lr)
            ])

        self.n_epochs = start_epoch + epochs

        for epoch in range(start_epoch, start_epoch + epochs):
            logger.info(f"============ Starting epoch {epoch} (continue) ============")

            if (epoch % self.umap_checkpoint_freq == 10) and (self.wandb_sweep == 0) and not self.debug:
                self.plot_umap(self.model, self.train_ds.adata, f"train-e{epoch}")

            if (
                self.queue_length > 0
                and epoch >= self.epoch_queue_starts
                and len(self.queue) == 0
            ):
                print(f"start using queue at epoch: {epoch}")
                for ds_id in self.ds_ids:
                    self.init_queue(ds_id)

            train_meters = self.train_epoch(epoch)

            if self.debug:
                test_meters = {}
            elif self.full_dataset_mode == 1:
                test_meters = {}
            else:
                test_meters = self.test_epoch()
                test_meters = {f"test_{key}": val for key, val in test_meters.items()}

            if not self.debug:
                scores = self.encode_adata(self.train_.adata, self.model, z_idx=2)
                cell_ids = self.train_.adata.obs_names
                (
                    train_meters["overal_compactness"],
                    train_meters["overal_separation"],
                    train_meters["celltype_purity"],
                    train_meters["niche_purity"],
                    train_meters["niche_purity3d"],
                    train_meters["niche_micro"],
                    train_meters["niche_macro"],
                ) = self.calc_mc_quality(cell_ids, scores, "train")
                ad = self.train_.adata
                ad.obs["mc"] = scores.argmax(1).detach().cpu().numpy()
                if "spatial" in ad.obsm:
                    train_meters["spatial_compactness"] = spatial_compactness(
                        ad, mc_key="mc", bk=self.dataset.batch_key
                    ).mean()
            else:
                (train_meters["niche_micro"], train_meters["niche_macro"],
                 train_meters["niche_micro_filt"], train_meters["niche_macro_filt"]) = self._quick_niche_metrics()

            # KNN accuracy diagnostic (every 2 epochs)
            if epoch % 2 == 0 or epoch == start_epoch + epochs - 1:
                pca_acc, z_acc = self._niche_knn_acc()
                if pca_acc is not None:
                    train_meters["knn_pca"] = pca_acc
                    train_meters["knn_z"] = z_acc

            self.log_wandb_loss(train_meters | test_meters, epoch)
            self.save_checkpoint(epoch)

            nm = train_meters.get('niche_micro_filt') or train_meters.get('niche_micro')
            nM = train_meters.get('niche_macro_filt') or train_meters.get('niche_macro')
            nm_str = f"{nm:.3f}" if nm is not None else "-"
            nM_str = f"{nM:.3f}" if nM is not None else "-"
            knn_str = ""
            if "knn_z" in train_meters:
                knn_str = f" | KNN: {train_meters['knn_z']:.1%} (pca:{train_meters['knn_pca']:.1%})"
            print(f">>> Epoch {epoch+1}/{start_epoch + epochs} | Loss: {train_meters.get('loss', 0):.4f} | niche_mi: {nm_str} | niche_Ma: {nM_str} | unused: {train_meters.get('proto_unused', 0):.1%}{knn_str}")

        self._total_epochs_trained = start_epoch + epochs
        return train_meters | test_meters

    def train_umap_edges(self, epochs: int = None, verbose: bool = True):
        """
        Train encoder using edge-centric parametric UMAP.

        This implements the official UMAP training scheme:
        - Sample edges (i, j) proportionally to their weight p_ij
        - For each positive edge, sample K negative edges
        - Loss = attractive (positives) + repulsive (negatives)

        Args:
            epochs: Number of training epochs (default: umap_edge_epochs)
            verbose: Print progress

        Returns:
            Final training metrics
        """
        epochs = epochs or getattr(self, 'umap_edge_epochs', 200)

        # Get affinity matrix
        if hasattr(self.train_ds, 'aff_raw'):
            affinity = self.train_ds.aff_raw
        else:
            affinity = self.train_ds.aff

        # Get data tensor
        adata = self.train_ds.adata
        if hasattr(adata.X, 'toarray'):
            X = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        else:
            X = torch.tensor(adata.X, dtype=torch.float32)
        X = X.to(self.device)

        # Get UMAP parameters
        min_dist = getattr(self, 'umap_min_dist', 0.5)
        spread = getattr(self, 'umap_spread', 1.0)
        neg_rate = getattr(self, 'umap_neg_rate', 5)

        print(f"🔄 Starting edge-centric UMAP training")
        print(f"   min_dist={min_dist}, spread={spread}, neg_rate={neg_rate}")

        # Create edge dataset
        edge_dataset = EdgeDataset(
            affinity,
            n_epochs=epochs,
            negative_sample_rate=neg_rate,
        )

        # Create loss function
        loss_fn = ParametricUMAPLoss(
            min_dist=min_dist,
            spread=spread,
            negative_sample_rate=neg_rate,
        )

        # Create data loader
        loader = DataLoader(
            edge_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=edge_collate_fn,
            num_workers=0,
            drop_last=True,
        )

        # Get encoder (the scpoli encoder)
        encoder = self.model.scpoli_cvae

        # Create optimizer
        optimizer = torch.optim.Adam(encoder.parameters(), lr=self.base_lr)

        # Training loop
        for epoch in range(epochs):
            encoder.train()
            total_metrics = {
                'loss': 0, 'q_pos': 0, 'q_neg': 0, 'margin': 0,
                'loss_pos': 0, 'loss_neg': 0,
            }
            n_batches = 0

            for batch in loader:
                head = batch['head'].to(self.device)
                tail = batch['tail'].to(self.device)
                weights = batch['weight'].to(self.device)
                neg_samples = batch['neg_samples'].to(self.device)

                # Collect unique indices
                all_idx = torch.cat([head, tail, neg_samples.flatten()])
                unique_idx = torch.unique(all_idx)

                # Encode unique nodes
                X_batch = X[unique_idx]

                # Get embeddings through scpoli encoder
                # scpoli_cvae expects a dict with specific keys
                batch_dict = {'x': X_batch}
                # Add condition if needed
                if hasattr(self.model, 'condition_key') and self.model.condition_key:
                    # Use first condition (simplified)
                    batch_dict['batch'] = torch.zeros(len(X_batch), dtype=torch.long, device=self.device)

                # Forward through encoder to get latent
                # encoder_out returns (x, recon_loss, kl_loss) - take only the latent
                z_unique, _, _ = self.model.encoder_out(batch_dict)

                # Map indices back
                idx_map = {int(idx): i for i, idx in enumerate(unique_idx.cpu().numpy())}

                def gather_z(indices):
                    mapped = torch.tensor([idx_map[int(i)] for i in indices.cpu().numpy()],
                                          device=self.device, dtype=torch.long)
                    return z_unique[mapped]

                z_head = gather_z(head)
                z_tail = gather_z(tail)

                B, K = neg_samples.shape
                z_neg = gather_z(neg_samples.flatten()).view(B, K, -1)

                # Compute loss
                loss, metrics = loss_fn(z_head, z_tail, z_neg, weights)

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate
                total_metrics['loss'] += loss.item()
                for k, v in metrics.items():
                    if k in total_metrics:
                        total_metrics[k] += v
                n_batches += 1

            # Average metrics
            for k in total_metrics:
                total_metrics[k] /= max(n_batches, 1)

            # Print progress
            if verbose and ((epoch + 1) % 1 == 0 or epoch == 0):
                # Also compute KNN accuracy occasionally
                knn_str = ""
                if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                    pca_acc, z_acc = self._niche_knn_acc()
                    if pca_acc is not None:
                        knn_str = f" | KNN: {z_acc:.1%} (pca:{pca_acc:.1%})"

                print(f">>> Epoch {epoch+1}/{epochs} | "
                      f"loss={total_metrics['loss']:.4f} | "
                      f"q+={total_metrics['q_pos']:.3f} | "
                      f"q-={total_metrics['q_neg']:.3f} | "
                      f"margin={total_metrics['margin']:.3f}{knn_str}")

            # Reshuffle dataset
            edge_dataset.reshuffle()

        return total_metrics

    def init_lambda_loss(self):

        return {
            key: getattr(self, f"lambda_{key}")
            for key in self.loss_keys
            if hasattr(self, f"lambda_{key}")
        }

    def setup_scheduler(self):
        self.steps_per_epoch = len(self.train_loader)
        self.total_steps = self.pretraining_epochs * self.steps_per_epoch
        self.warmup_steps = int(0.2 * self.total_steps)

    def update_lambda(self, epoch):
        self.lambda_loss["kl"] = kl_scheduler(
            epoch, self.kl_start_epoch, self.n_epochs, max_lambda=self.lambda_kl
        )
        # self.lambda_loss['recon'] = kl_scheduler(
        #     epoch, self.recon_start_epoch, self.n_epochs, max_lambda=self.lambda_recon
        # )
        # if epoch >= self.recon_start_epoch:
        #     self.lambda_loss["recon"] = self.lambda_recon
        # else:
        #     self.lambda_loss["recon"] = 0

    def train_epoch(self, epoch):
        self.update_temp_eps(epoch)
        logger.info(
            f"epoch {epoch} | temp={self.temperature:.4f}, eps={self.epsilon:.4f}"
        )
        self.model.train()
        self.use_the_queue = 0

        meters = {
            "loss": AverageMeter(),
            "batch_time": AverageMeter(),
            "data_time": AverageMeter(),
        }

        end = time.time()
        ds_assign_cnts = {
            ds_id: np.zeros(self.nmb_prototypes, dtype=int) for ds_id in self.ds_ids
        }
        if self.kl_sched == 1:
            self.update_lambda(epoch)

        # Use tqdm progress bar in debug mode for visibility
        loader = tqdm(self.train_loader, desc=f"Epoch {epoch}", disable=not self.debug)
        for it, inputs in enumerate(loader):
            meters["data_time"].update(time.time() - end)
            iteration = epoch * len(self.train_loader) + it
            self.update_learning_rate(iteration)
            # self.update_lambda(iteration)
            # normalize prototypes
            if self.l2norm == 1:
                with torch.no_grad():
                    self.model.normalize_prototypes()

            bs = inputs["x"].size(0)
            inputs = {
                k: inputs[k].transpose(0, 1) for k in inputs.keys()
            }  # bring dataset in first to calc loss per dataset

            self.optimizer.zero_grad()

            for ds_id in self.ds_ids:
                loss, meters, assign_cnts = self.calc_ds_loss(
                    inputs, ds_id, meters, bs, self.train_ds.adata, "train"
                )
                ds_assign_cnts[ds_id] += assign_cnts.cpu().numpy()
                loss.backward()

            self._handle_prototype_freezing(epoch)
            self.optimizer.step()

            meters["batch_time"].update(time.time() - end)
            end = time.time()

            # Update tqdm progress bar with loss in debug mode
            if self.debug:
                loader.set_postfix(loss=f"{meters['loss'].avg:.4f}")
            else:
                # Log to wandb every 5 iterations in non-debug mode
                if it % 5 == 0:
                    logger.info(
                        f"Epoch: [{epoch}][{it}/{len(self.train_loader)}] "
                        f"Loss {meters['loss'].val:.4f} ({meters['loss'].avg:.4f}) "
                        f"Lr {self.optimizer.param_groups[0]['lr']:.4f}"
                    )
                    lr_gn = self.get_lr_grad()
                    self.log_wandb_loss(lr_gn, epoch)

        meters = {k: getattr(v, "avg", v) for k, v in meters.items()}
        all_assign_cnts = sum(ds_assign_cnts[ds_id] for ds_id in ds_assign_cnts)
        meters["proto_unused"] = (
            all_assign_cnts == 0
        ).sum().item() / self.nmb_prototypes

        # all_assign_cnts: shape [n_prototypes], each value = number of assigned samples
        p = all_assign_cnts / all_assign_cnts.sum()  # normalize to probabilities
        entropy = -(p * np.log(np.clip(p, 1e-8, None))).sum()
        norm_entropy = entropy / torch.log(
            torch.tensor(len(p), dtype=torch.float)
        )  # normalized [0,1]
        meters["proto_utilization"] = norm_entropy.item()

        # Skip proto collapse metric in debug mode (matrix multiplication overhead)
        if not self.debug:
            protos = F.normalize(self.model.prototypes.weight, dim=1)
            cos_sim = protos @ protos.T
            mean_off_diag = (cos_sim.sum() - cos_sim.diag().sum()) / (
                cos_sim.numel() - protos.size(0)
            )
            meters["proto_collapse"] = mean_off_diag.item()
        meters["lambda_kl"] = self.lambda_loss["kl"]
        meters["lambda_recon"] = self.lambda_loss["recon"]
        return meters

    def calc_ds_loss(self, inputs, ds_id, meters, bs, ad, split):
        ds_inputs = {k: inputs[k][ds_id] for k in inputs.keys()}
        metrics, assign_cnts = self._process_batch(ds_inputs, ds_id, ad, split)
        # averaged = self._average_metrics(metrics)
        # Update meters
        for key in metrics:
            if key not in meters:
                meters[key] = AverageMeter()
            value = (
                metrics[key].item() if hasattr(metrics[key], "item") else metrics[key]
            )
            meters[key].update(value, bs)
        # loss = (
        #     metrics["swav"] * self.lambda_swav
        #     + metrics["recon"] * self.lambda_recon
        #     + metrics["kl"] * self.lambda_kl
        #     + metrics["kl_balance"] * self.lambda_balance
        #     + metrics["proto_loss"] * self.lambda_proto
        #     + metrics["commitment_loss"] * self.lambda_commit
        #     + metrics["z_norm"] * self.lambda_l2
        #     + metrics["proto_norm"] * self.lambda_l2
        # )
        if hasattr(self, "lambda_loss"):
            loss = torch.stack(
                [metrics[k] * self.lambda_loss[k] for k in self.lambda_loss.keys()]
            ).sum()
            meters["loss"].update(loss.item(), bs)
        else:  # when init lambda loss
            loss = -1
        return loss, meters, assign_cnts

    def _process_batch(self, inputs, ds_id, ad, split):
        bs = inputs["x"].size(0)
        inputs = self.dict_to_device(inputs)
        # inputs = reshape_and_reorder_dict(inputs)
        B, n_aug = inputs["x"].shape[:2]
        inputs = {
            k: t.permute(1, 0, *range(2, t.ndim)).reshape(B * n_aug, *t.shape[2:])
            for k, t in inputs.items()
        }
        # manifold_keys = ['sigma', '', 'heterogeneity', 'mf_score']
        manifold_keys = self.train_ds.manifold.keys()
        manifold_scores = {k: inputs.pop(k, None) for k in manifold_keys}
        cell_idx = inputs.pop("cell_idx", None).cpu().numpy()
        cell_ids = [ad.obs.index[i] for i in cell_idx]
        sigma = inputs.pop("sigma").cpu().numpy() if "sigma" in inputs else None

        def sigma_to_eps(sigma, eps_min=0.02, eps_max=0.05):
            s_lo, s_hi = np.percentile(sigma, [5, 95])
            s = np.clip(sigma, s_lo, s_hi)
            s = (s - s_lo) / (s_hi - s_lo)
            return eps_min + s * (eps_max - eps_min)

        if sigma is not None:
            adoptive_eps = sigma_to_eps(sigma, 0.5 * self.epsilon, self.epsilon)
        else:
            adoptive_eps = None
        sim = inputs.pop("sim", None)
        # label = inputs.pop(self.dataset.label_key)
        z, _, scores, recon, proto_recon, propagation_sim, kl, kl_balance = self.model(
            bs, inputs
        )
        (proto_loss, commitment_loss) = propagation_sim
        # Skip affinity loss if lambda_aff is 0
        if self.lambda_loss.get("aff", 0) == 0:
            loss_aff = torch.tensor(0.0, device=scores.device)
        else:
            loss_aff = self.calc_aff_loss(scores[:bs], cell_idx[:bs])
        if self.recon_type == "swapped" or self.recon_type == "hybrid":
            swapped_recon = self.calc_swapped_recon(z, scores, bs, inputs)
            if self.recon_type == "swapped":
                recon = swapped_recon
            else:
                recon = 0.8 * recon + 0.2 * swapped_recon
        z_norm = z.norm(dim=1).mean()
        proto_norm = self.model.get_prototypes().norm(dim=1).mean()
        # z, logits, cvae_loss, resp, propagation, sim = self.parse_model_output(outputs)
        z = z.detach()
        (
            swav_loss,
            p_matched,
            q_matched,
            qproto_utilization,
            p_uncertainty,
            q_uncertainty,
            proto_entropy,
            q_effk,  # ADD
            p_effk,  # ADD
        ) = self.compute_swav_loss(
            scores,
            z,
            bs,
            ds_id,
            manifold_scores,
            None,
            sim=sim,
            adoptive_eps=adoptive_eps,
        )
        assign_cnts = get_hard_assign_cnts(scores)
        max_active = min(scores.size(0), scores.size(1))

        # Skip expensive per-batch quality metrics in debug mode
        if self.debug:
            compactness, separation, cp, np2d, np3d = 0, 0, 0, 0, 0
        else:
            compactness, separation, cp, np2d, np3d = self.calc_mc_quality(
                cell_ids, scores, split
            )
        # Uniformity loss: encourage uniform prototype usage (anti-collapse)
        P = F.softmax(scores[:bs] / self.epsilon, dim=1)  # (bs, K)
        avg_proto_usage = P.mean(dim=0)  # (K,) average assignment per prototype
        # Maximize entropy of usage distribution = minimize negative entropy
        uniform_loss = (avg_proto_usage * (avg_proto_usage + 1e-8).log()).sum()  # negative entropy
        # uniform_loss is negative (entropy is positive), so we ADD it to loss to maximize entropy

        # R1/R2 loss: proto coverage
        if self.lambda_loss.get("r1r2", 0) == 0:
            r1r2_loss = torch.tensor(0.0, device=scores.device)
        else:
            r1r2_loss = self.calc_r1r2_loss(z[:bs])

        return {
            "swav": swav_loss,
            "recon": recon,
            "proto_recon": proto_recon,
            "kl": kl,
            "kl_balance": kl_balance,
            "proto": proto_loss,
            "commit": commitment_loss,
            "p_matched": p_matched,
            "q_matched": q_matched,
            "z_norm": z_norm,
            "proto_norm": proto_norm,
            "pproto_utilization": (assign_cnts != 0).sum().item() / max_active,
            "qproto_utilization": qproto_utilization,
            "p_uncertainty": p_uncertainty,
            "q_uncertainty": q_uncertainty,
            "compactness": compactness,
            "separation": separation,
            "proto_entropy": proto_entropy,
            "aff": loss_aff,
            "q_effk": q_effk,
            "p_effk": p_effk,
            "uniform": uniform_loss,
            "r1r2": r1r2_loss,
        }, assign_cnts

    def calc_aff_loss(self, scores, cell_idx):
        """
        Binary contrastive affinity loss.

        - Positive pairs (A > 0): maximize S → pull to same proto
        - Negative pairs (A = 0): minimize S → push to diff proto
        """
        if torch.is_tensor(cell_idx):
            cell_idx = cell_idx.detach().cpu().numpy()
        cell_idx = np.asarray(cell_idx, dtype=np.int64)

        # Soft assignments
        P = torch.softmax(scores / self.epsilon, dim=1)
        S = P @ P.T  # predicted similarity (0-1)

        # Get affinity submatrix for this batch
        A = self.train_ds.aff[cell_idx][:, cell_idx]
        A = A.maximum(A.T)  # symmetrize
        A = torch.as_tensor(A.toarray(), device=S.device, dtype=S.dtype)

        # Mask diagonal
        mask = ~torch.eye(S.size(0), device=S.device, dtype=torch.bool)
        S_masked = S[mask]
        A_masked = A[mask]

        # Binary masks
        pos_mask = A_masked > 0  # connected pairs (same niche)
        neg_mask = A_masked == 0  # not connected

        pos_loss = torch.tensor(0.0, device=S.device)
        neg_loss = torch.tensor(0.0, device=S.device)

        # Positive: maximize S (pull same-niche together)
        if pos_mask.sum() > 0:
            pos_loss = -torch.log(S_masked[pos_mask] + 1e-8).mean()

        # Negative: minimize S (push diff-niche apart)
        if neg_mask.sum() > 0:
            neg_loss = -torch.log(1 - S_masked[neg_mask] + 1e-8).mean()

        # two_sided=1: both, two_sided=0: neg only
        if self.two_sided == 1:
            return pos_loss + neg_loss
        else:
            return neg_loss

    def calc_r1r2_loss(self, z):
        """
        Li et al. style R1/R2 prototype coverage losses.
        Uses scores (same as assignment) to ensure consistency.

        R1: each proto should be "best" for at least 1 cell → move proto
        R2: each cell should have high score for at least 1 proto → move encoder

        Ensures: no orphan protos, no uncovered cells.
        """
        protos = self.model.get_prototypes()  # (K, D)

        # R1: move protos toward cells (detach z)
        # Use same scoring as assignment
        scores_r1 = self.model.proto_soft_assignments(z.detach())  # (B, K)
        r1 = -scores_r1.max(dim=0).values.mean()  # max score per proto → minimize neg

        # R2: move cells toward protos (detach protos)
        # Recompute scores with detached protos
        protos_detached = protos.detach()
        if self.assignment_metric == 'sneuc':
            d2 = torch.cdist(z, protos_detached, p=2) ** 2
            scores_r2 = -d2
            scores_r2 = scores_r2 - scores_r2.max(dim=1, keepdim=True)[0]
            scores_r2 = scores_r2.clamp(min=-75)
        elif self.assignment_metric == 'dotp':
            scores_r2 = z @ protos_detached.T
        else:  # fallback to negative euclidean
            scores_r2 = -torch.cdist(z, protos_detached, p=2)
        r2 = -scores_r2.max(dim=1).values.mean()  # max score per cell → minimize neg

        return r1 + r2

    def calc_swapped_recon(self, z, scores, bs, inputs):
        loss = 0

        for view_id in self.views_for_assign:
            view_scores = scores[bs * view_id : bs * (view_id + 1)].detach()
            view_codes = view_scores.argmax(dim=1)

            # calc recon loss by closet proto to pos pairs
            subloss = 0
            aug_view_ids = np.delete(np.arange(self.nmb_views), view_id)
            for v in aug_view_ids:
                aug_z = z[bs * v : bs * (v + 1)]
                aug_inputs = {k: inputs[k][bs * v : bs * (v + 1)] for k in inputs}
                recon_loss, _, _ = self.model.quantized_recon_step(
                    aug_z, view_codes, **aug_inputs
                )
                subloss += recon_loss

            loss += subloss / len(aug_view_ids)
        return loss / len(self.views_for_assign)

    def parse_model_output(self, outputs):
        if self.model_type == "gm":
            z_swav, _, logits, cvae_loss, (propagation, sim), resp = outputs
            propagation, sim = 0, 0
        else:
            z_swav, _, logits, cvae_loss, (propagation, sim) = outputs
            resp = None

        return z_swav, logits, cvae_loss, resp, propagation, sim

    def compute_swav_loss(
        self,
        scores,
        z,
        bs,
        ds_id,
        manifold_scores=None,
        resp=None,
        sim=None,
        adoptive_eps=None,
    ):
        if sim is None or self.weighted_kl == 0:
            sim = torch.ones(z.size(0), device=z.device, dtype=z.dtype)

        (
            loss,
            p_matched,
            q_matched,
            p_uncertainty,
            q_uncertainty,
            qproto_utilization,
            proto_entropy,
        ) = (0, 0, 0, 0, 0, 0, 0)
        q_effk, p_effk = 0, 0
        # each crop mean each augmentation, just caluclate q and loss for first crops_for_assign
        for view_idx, view_id in enumerate(self.views_for_assign):
            with torch.no_grad():
                # outputs for 1 batch of data, [aug1s1, a1s2, a1s3, .., a1sb]
                view_scores = scores[bs * view_id : bs * (view_id + 1)].detach()
                # print(f'scores min, max: {view_scores.min()}, {view_scores.max()}')
                # with torch.no_grad(): not nessecary because both funcion has no grad decorator
                sinkhorn_input = self.prepare_sinkhorn_input(
                    view_idx, z, view_id, bs, view_scores, ds_id
                )
                if self.cell_w_mode != "uniform":
                    cell_weights = manifold_scores[self.cell_w_mode][
                        bs * view_id : bs * (view_id + 1)
                    ]
                else:
                    cell_weights = None  # uniform
                if self.sinkhorn_iterations == 0:
                    if adoptive_eps is not None:
                        eps = torch.as_tensor(
                            adoptive_eps[bs * view_id : bs * (view_id + 1)],
                            device=view_scores.device,
                            dtype=view_scores.dtype,
                        )
                        q = F.softmax(view_scores / eps[:, None], dim=1)
                    else:
                        q = F.softmax(view_scores / self.epsilon, dim=1)
                else:
                    q = self.sinkhorn(sinkhorn_input, cell_weights)
                qassign_cnts = get_hard_assign_cnts(q)
                max_active = min(scores.size(0), scores.size(1))
                qproto_utilization += (qassign_cnts != 0).sum().item() / max_active
                q = q[-bs:]
                # print(f'q min, max: {q.min()}, {q.max()}')
                q_uncertainty -= (q * (q.clamp_min(1e-8)).log()).sum(dim=1).mean()

            view_scores = scores[bs * view_id : bs * (view_id + 1)]
            p = F.softmax(view_scores / self.temperature, dim=1)
            p_effk += (-(p * p.clamp_min(1e-8).log()).sum(dim=1)).exp().median()  # ADD

            # per-sample entropy (H_p > 0)
            H_p = -(p * p.clamp_min(1e-8).log()).sum(dim=1).mean()
            p_uncertainty += H_p

            usage = p.mean(dim=0)  # shape [K]
            H_proto = -(usage * usage.clamp_min(1e-8).log()).sum()
            proto_entropy += H_proto

            if self.hard_clustering == 1:
                q = self.hard_clusters(q)
            q_effk += (-(q * q.clamp_min(1e-8).log()).sum(dim=1)).exp().median()  # ADD
            # check how consitent q is with other augmentations [cross entropy]
            subloss = 0
            vp_matched, vq_matched = 0, 0
            aug_view_ids = np.delete(np.arange(self.nmb_views), view_id)
            for v in aug_view_ids:
                aug_scores = scores[bs * v : bs * (v + 1)] / self.temperature
                self.check_finit(aug_scores, "p")

                # p and log(p)
                log_p = F.log_softmax(aug_scores, dim=1)
                p = log_p.exp()  # numerically consistent with log_p

                if self.div_type == "kl":
                    # ----- KL(p‖q) variant -----
                    # Only clamp q to avoid log(0)
                    log_q = q.clamp_min(1e-8).log()
                    # KL(p||q) = sum_i p_i * (log p_i - log q_i)
                    kl = (
                        torch.sum(p * (log_p - log_q), dim=1)
                        * sim[bs * v : bs * (v + 1)]
                    )
                    subloss += kl.mean()
                else:
                    # ----- default CE (≈ KL(q‖p)) -----
                    # CE = -∑ q_i * log p_i
                    ce = -torch.sum(q * log_p, dim=1) * sim[bs * v : bs * (v + 1)]
                    subloss += ce.mean()  # add (no explicit minus outside loop)

                vp_matched += get_matched_pairs_ratio(view_scores, aug_scores)
                vq_matched += get_matched_pairs_ratio(q, aug_scores)

            loss += subloss / len(aug_view_ids)
            p_matched += vp_matched / len(aug_view_ids)
            q_matched += vq_matched / len(aug_view_ids)

        return (
            loss / len(self.views_for_assign),
            p_matched / len(self.views_for_assign),
            q_matched / len(self.views_for_assign),
            qproto_utilization / len(self.views_for_assign),
            p_uncertainty / len(self.views_for_assign),
            q_uncertainty / len(self.views_for_assign),
            proto_entropy / len(self.views_for_assign),
            q_effk / len(self.views_for_assign),  # ADD
            p_effk / len(self.views_for_assign),  # ADD
        )

    def get_lr_grad(self):
        # --- Monitor learning rate and gradient norm ---
        lr = self.optimizer.param_groups[0]["lr"]

        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        grad_norm = total_norm**0.5

        return {"lr": lr, "grad_norm": grad_norm}

    def test_epoch(self):
        self.model.eval()
        with torch.no_grad():
            for inputs in self.test_loader:
                bs = inputs["x"].size(0)
                # ds_ids = range(inputs['x'].size(1))
                inputs = {
                    k: inputs[k].transpose(0, 1) for k in inputs.keys()
                }  # bring dataset in first to calc loss per dataset
                meters = {"loss": AverageMeter()}

                for ds_id in self.ds_ids:
                    _, meters, _ = self.calc_ds_loss(
                        inputs, ds_id, meters, bs, self.test_ds.adata, "val"
                    )
        meters = {k: getattr(v, "avg", v) for k, v in meters.items()}
        return meters

        # define test loader
        # pass data, get loss and metrics
        # return the dict

    def _handle_prototype_freezing(self, epoch):

        for name, p in self.model.named_parameters():
            if "prototypes" in name:
                if epoch < self.freeze_prototypes_nepochs:
                    p.grad = None
                else:
                    break

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr_schedule[iteration]

    def hard_clusters(self, out: torch.Tensor) -> torch.Tensor:
        """
        Convert output probabilities/logits into hard one-hot clusters.
        Args:
            out (torch.Tensor): shape (batch, prototypes)
        Returns:
            torch.Tensor: one-hot encoded tensor of same shape as `out`
        """
        max_indices = torch.argmax(out, dim=1)
        return F.one_hot(max_indices, num_classes=out.size(1)).to(out.dtype)

    def check_finit(self, prob, name):
        if not torch.isfinite(prob).all():
            print(f"⚠️ Invalid values in {name}! (nan/inf detected)")
            print("min:", prob.min().item(), "max:", prob.max().item())

    @torch.no_grad()
    def sinkhorn(self, out, _):
        Q = torch.exp(
            out / self.epsilon
        ).t()  # Q is K-by-B for consistency with notations from our paper

        B = Q.shape[1] * self.world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        # print(f'Q_sum {sum_Q}')
        Q /= sum_Q

        for it in range(self.sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            # print(f'row sum (proto) {sum_of_rows}')
            Q /= sum_of_rows + 1e-12
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            # print(f'col sum (samples) {torch.sum(Q, dim=0, keepdim=True)}')
            Q /= torch.sum(Q, dim=0, keepdim=True) + 1e-12
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()

    @torch.no_grad()
    def distributed_sinkhorn_marginal(self, out, cell_weights=None):
        Q = torch.exp(out / self.epsilon).t()
        self.check_finit(Q, "q")
        B = Q.shape[1]
        K = Q.shape[0]

        Q /= torch.sum(Q)

        # prototypes uniform
        row_marginals = torch.full((K, 1), 1.0 / K, device=Q.device, dtype=Q.dtype)

        # samples weighted
        if cell_weights is None:
            col_marginals = torch.full((1, B), 1.0 / B, device=Q.device, dtype=Q.dtype)
        else:
            col_marginals = cell_weights.view(1, -1)
            col_marginals = col_marginals / col_marginals.sum()

        for _ in range(self.sinkhorn_iterations):
            # match prototypes
            Q *= row_marginals / (Q.sum(dim=1, keepdim=True) + 1e-12)
            # match samples (your custom weights)
            Q *= col_marginals / (Q.sum(dim=0, keepdim=True) + 1e-12)

        return (Q * B).t()  # (B, P)

    def get_model_prototypes(self, model):
        prototypes = model.get_prototypes()
        if self.use_projector_out:
            return model.projection_head(prototypes)
        else:
            return prototypes

    @torch.no_grad()
    def prepare_sinkhorn_input(self, queue_slot, z, view_id, bs, view_logits, ds_id):
        output_logits = view_logits

        # instead of: if queue is not None -> check if ds_id is in queue
        # time to use the queue
        if ds_id in self.queue:
            # check if the queue has any real data
            if self.use_the_queue or not torch.all(
                self.queue[ds_id][queue_slot, -1, :] == 0
            ):
                self.use_the_queue = 1
                # 'get prototypes assignment scores for self.queue[i] (which contain some old embeddings)'
                queue_logits = self.model.proto_soft_assignments(
                    self.queue[ds_id][queue_slot]
                )
                output_logits = torch.cat([queue_logits, view_logits])

            # fill the queue
            self.queue[ds_id][queue_slot, bs:] = self.queue[ds_id][
                queue_slot, :-bs
            ].clone()
            self.queue[ds_id][queue_slot, :bs] = z[view_id * bs : (view_id + 1) * bs]

        return output_logits

    def init_queue(self, ds_id):
        self.queue[ds_id] = torch.zeros(
            len(self.views_for_assign),
            self.queue_length,  # // divide by wprld size
            self.latent_dims,
        ).cuda()

    def update_temp_eps(self, epoch):
        if self.sched_temp_eps == 0:
            return

        t = epoch / max(1, self.n_epochs - 1)

        # cosine decay with floor
        self.temperature = max(
            self.temperature_min,
            self._temperature0 * 0.5 * (1 + np.cos(np.pi * t)),
        )

        self.epsilon = max(
            self.epsilon_min,
            self._epsilon0 * 0.5 * (1 + np.cos(np.pi * t)),
        )
        if hasattr(self.model, "temperature"):
            self.model.temperature = self.epsilon


if __name__ == "__main__":
    swav = SCProtoTrainer()
    swav.setup()
    swav.run()
    swav.encode_ref()
