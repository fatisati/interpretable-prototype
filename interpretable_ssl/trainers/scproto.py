import math
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import shutil
import time
from logging import getLogger
from tqdm import tqdm
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader

from swav.src.utils import initialize_exp, fix_random_seeds

from interpretable_ssl.configs.larc import LARC
from interpretable_ssl.configs.defaults import *
from interpretable_ssl.utils import *
from interpretable_ssl.trainers.adaptive_trainer import AdoptiveTrainer
from interpretable_ssl.trainers.scproto_utils import *
from interpretable_ssl.trainers.affinity import *
from interpretable_ssl.trainers.scpoli_helpers import *
from interpretable_ssl.trainers.edge_umap import EdgeDataset, ParametricUMAPLoss, edge_collate_fn
from interpretable_ssl.augmenters.adata_augmenter import *
from interpretable_ssl.models.swav import *
from interpretable_ssl.evaluation.visualization import *
from interpretable_ssl.evaluation.prototype_metrics import *
from interpretable_ssl.evaluation.cd4_marker import *
from interpretable_ssl.evaluation.mc_metric_utils import *

from scarches.models.scpoli import scPoli
import scarches.trainers.scpoli._utils as scpoli_utils
import wandb

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
        self._metrics_log = {}

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

        # Print affinity report if available (use raw affinity for accurate weight stats)
        _aff_for_report = (self.train_ds.aff_raw if hasattr(self.train_ds, 'aff_raw')
                           else self.train_ds.aff)
        if _aff_for_report is not None:
            self.aff_stats = affinity_report(_aff_for_report)
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

        # ---- LR schedule (warmup -> cosine) ----
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

    # -- UMAP edge training: setup / epoch / public API ------------------

    def _setup_umap_edges(self, epochs: int = None, init_prototypes: bool = True):
        """Build and cache all objects needed for edge-centric UMAP training."""
        epochs = epochs or getattr(self, 'umap_edge_epochs', 200)

        affinity = self.train_ds.aff_raw if hasattr(self.train_ds, 'aff_raw') else self.train_ds.aff

        adata = self.train_ds.adata
        if hasattr(adata.X, 'toarray'):
            X = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        else:
            X = torch.tensor(adata.X, dtype=torch.float32)

        min_dist = getattr(self, 'umap_min_dist', 0.5)
        spread = getattr(self, 'umap_spread', 1.0)
        neg_rate = getattr(self, 'umap_neg_rate', 5)
        umap_similarity = getattr(self, 'umap_similarity', 'embedding')

        if init_prototypes:
            self.init_prototypes()

        # Build cell -> ds_id mapping
        if self.condition_key is not None:
            batch_labels = self.train_ds.adata.obs[self.condition_key]
            unique_batches = list(batch_labels.unique())
            batch_to_id = {b: i for i, b in enumerate(unique_batches)}
            cell_ds = np.array([batch_to_id[b] for b in batch_labels], dtype=np.int64)
        else:
            cell_ds = np.zeros(len(self.train_ds.adata), dtype=np.int64)

        # Single EdgeDataset over all edges; neg sampling stays within head's dataset.
        # WeightedRandomSampler picks edges proportional to affinity weight each epoch.
        from torch.utils.data import WeightedRandomSampler
        edge_dataset = EdgeDataset(affinity, n_epochs=epochs, negative_sample_rate=neg_rate, cell_ds=cell_ds)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(edge_dataset.weights).float(),
            num_samples=len(edge_dataset),
            replacement=True,
        )
        loader = DataLoader(
            edge_dataset, batch_size=self.batch_size, sampler=sampler,
            collate_fn=edge_collate_fn, num_workers=0, drop_last=False,
        )

        loss_fn = ParametricUMAPLoss(min_dist=min_dist, spread=spread, negative_sample_rate=neg_rate)

        if umap_similarity == 'proto':
            params = list(self.model.scpoli_cvae.parameters()) + list(self.model.prototypes.parameters())
        else:
            params = list(self.model.scpoli_cvae.parameters())
        optimizer = torch.optim.Adam(params, lr=self.base_lr)

        self._umap_state = {
            'X': X.to(self.device),
            'edge_dataset': edge_dataset,
            'loss_fn': loss_fn,
            'loader': loader,
            'optimizer': optimizer,
            'epoch': 0,
        }

        print(f"Starting edge-centric UMAP training (similarity={umap_similarity})")
        print(f"   min_dist={min_dist}, spread={spread}, neg_rate={neg_rate}")
        print(f"   lambda_umap={getattr(self, 'lambda_umap', 1.0)}, "
              f"lambda_recon={self.lambda_recon}, lambda_kl={self.lambda_kl}, "
              f"lambda_proto_recon={self.lambda_proto_recon}, lambda_r1r2={self.lambda_r1r2}")

    def _run_umap_epoch(self):
        """Run a single UMAP edge training epoch. Returns metrics dict."""
        s = self._umap_state
        X = s['X']
        loader = s['loader']
        loss_fn = s['loss_fn']
        optimizer = s['optimizer']

        lambda_umap = getattr(self, 'lambda_umap', 1.0)
        lambda_recon = self.lambda_recon
        lambda_kl = self.lambda_kl
        lambda_proto_recon = self.lambda_proto_recon
        lambda_r1r2 = self.lambda_r1r2
        use_proto_sim = getattr(self, 'umap_similarity', 'embedding') == 'proto'
        proto_metric = getattr(self, 'umap_proto_metric', 'dotp')

        self.model.train()
        total_metrics = {
            'loss': 0, 'umap': 0, 'q_pos': 0, 'q_neg': 0, 'margin': 0,
            'loss_pos': 0, 'loss_neg': 0, 'recon': 0, 'kl': 0,
            'proto_recon': 0, 'r1r2': 0, 'n_unused_protos': 0,
        }
        n_batches = 0
        used_proto_ids = set()

        from tqdm import tqdm
        for batch in tqdm(loader, desc='edges'):
            if self.l2norm == 1:
                with torch.no_grad():
                    self.model.normalize_prototypes()

            optimizer.zero_grad()
            proto_recon_loss = torch.tensor(0.0, device=self.device)
            r1r2_loss = torch.tensor(0.0, device=self.device)

            head        = batch['head'].to(self.device)
            tail        = batch['tail'].to(self.device)
            weights     = batch['weight'].to(self.device)
            neg_samples = batch['neg_samples'].to(self.device)
            B, neg_K = neg_samples.shape

            # Single forward pass over unique nodes in this batch
            all_idx = torch.cat([head, tail, neg_samples.flatten()])
            unique_idx = torch.unique(all_idx)
            X_batch = X[unique_idx]
            n_samples = len(X_batch)

            if hasattr(self.train_ds, 'conditions'):
                batch_cond = self.train_ds.conditions[unique_idx.cpu()].to(self.device)
            else:
                n_conds = len(self.model.scpoli_cvae.n_conditions)
                batch_cond = torch.zeros(n_samples, n_conds, dtype=torch.long, device=self.device)

            z_unique, recon_loss, kl_loss = self.model.encoder_out({'x': X_batch, 'batch': batch_cond})

            unique_idx_cpu = unique_idx.cpu().numpy()
            idx_map = {int(idx): i for i, idx in enumerate(unique_idx_cpu)}

            def _gather(indices):
                return torch.tensor([idx_map[int(i)] for i in indices.cpu().numpy()],
                                    device=self.device, dtype=torch.long)

            if use_proto_sim:
                logits = self.model.prototypes(z_unique)
                soft_assign = F.softmax(logits / self.epsilon, dim=1)
                with torch.no_grad():
                    sa_effk = (1.0 / (soft_assign * soft_assign).sum(dim=1))
                    total_metrics['effk'] = total_metrics.get('effk', 0) + sa_effk.median().item()
                    used_proto_ids.update(soft_assign.argmax(dim=1).unique().cpu().tolist())

                s_head = soft_assign[_gather(head)]
                s_tail = soft_assign[_gather(tail)]
                s_neg = soft_assign[_gather(neg_samples.flatten())].view(B, neg_K, -1)
                _eps = 1e-4
                if proto_metric == 'cosine':
                    s_head_n = F.normalize(s_head, dim=-1, p=2)
                    s_tail_n = F.normalize(s_tail, dim=-1, p=2)
                    s_neg_n = F.normalize(s_neg, dim=-1, p=2)
                    q_pos = (s_head_n * s_tail_n).sum(dim=-1).clamp(_eps, 1.0 - _eps)
                    q_neg = (s_head_n.unsqueeze(1) * s_neg_n).sum(dim=-1).clamp(_eps, 1.0 - _eps)
                else:
                    q_pos = (s_head * s_tail).sum(dim=-1).clamp(_eps, 1.0 - _eps)
                    q_neg = (s_head.unsqueeze(1) * s_neg).sum(dim=-1).clamp(_eps, 1.0 - _eps)
                loss_pos = -torch.log(q_pos).mean()
                loss_neg = -torch.log(1.0 - q_neg).sum(dim=1).mean()
                umap_loss = loss_pos + loss_neg
                metrics = {
                    'q_pos': q_pos.mean().item(), 'q_neg': q_neg.mean().item(),
                    'margin': (q_pos.mean() - q_neg.mean()).item(),
                    'loss_pos': loss_pos.item(), 'loss_neg': loss_neg.item(),
                }
            else:
                z_head = z_unique[_gather(head)]
                z_tail = z_unique[_gather(tail)]
                z_neg = z_unique[_gather(neg_samples.flatten())].view(B, neg_K, -1)
                umap_loss, metrics = loss_fn(z_head, z_tail, z_neg, weights)

            # Auxiliary losses computed once on the shared batch (not per-ds)
            if lambda_proto_recon > 0:
                scores = soft_assign if use_proto_sim else F.softmax(self.model.prototypes(z_unique) / self.epsilon, dim=1)
                protos = self.model.get_prototypes()
                K = protos.size(0)
                n_genes = X_batch.size(1)
                unique_conds, inverse_idx = torch.unique(batch_cond, dim=0, return_inverse=True)
                n_unique_conds = unique_conds.size(0)
                proto_expanded = protos.unsqueeze(0).expand(n_unique_conds, -1, -1).reshape(-1, protos.size(1))
                cond_expanded = unique_conds.unsqueeze(1).expand(-1, K, -1).reshape(-1, unique_conds.size(1))
                decoded_all = self.model.decode(proto_expanded, cond_expanded).view(n_unique_conds, K, n_genes)
                recon_x_agg = torch.zeros(n_samples, n_genes, device=self.device)
                for c in range(n_unique_conds):
                    cmask = inverse_idx == c
                    if cmask.any():
                        recon_x_agg[cmask] = scores[cmask].detach() @ decoded_all[c]
                proto_recon_loss = F.mse_loss(recon_x_agg, X_batch, reduction="none").sum(dim=-1).mean()

            if lambda_r1r2 > 0:
                r1r2_loss = self.calc_r1r2_loss(z_unique)

            loss = lambda_umap * umap_loss
            if lambda_recon > 0:
                loss = loss + lambda_recon * recon_loss
            if lambda_kl > 0:
                loss = loss + lambda_kl * kl_loss
            if lambda_proto_recon > 0:
                loss = loss + lambda_proto_recon * proto_recon_loss
            if lambda_r1r2 > 0:
                loss = loss + lambda_r1r2 * r1r2_loss

            loss.backward()
            optimizer.step()

            total_metrics['loss'] += loss.item()
            total_metrics['umap'] += umap_loss.item()
            total_metrics['recon'] += recon_loss.item()
            total_metrics['kl'] += kl_loss.item()
            total_metrics['r1r2'] += r1r2_loss.item()
            total_metrics['proto_recon'] += proto_recon_loss.item()
            for k, v in metrics.items():
                if k in total_metrics:
                    total_metrics[k] += v
            n_batches += 1

        for k in total_metrics:
            total_metrics[k] /= max(n_batches, 1)

        if use_proto_sim:
            total_metrics['n_unused_protos'] = self.nmb_prototypes - len(used_proto_ids)

        s['epoch'] += 1
        return total_metrics

    def _print_umap_epoch(self, epoch, total_epochs, metrics):
        """Print one line of UMAP training progress."""
        extra = ""
        if self.lambda_recon > 0:
            extra += f" | recon={metrics['recon']:.4f}"
        if self.lambda_kl > 0:
            extra += f" | kl={metrics['kl']:.4f}"
        if self.lambda_proto_recon > 0:
            extra += f" | proto_recon={metrics['proto_recon']:.4f}"
        if self.lambda_r1r2 > 0:
            extra += f" | r1r2={metrics['r1r2']:.4f}"
        effk_str = f" | effk={metrics['effk']:.1f}" if 'effk' in metrics else ""
        unused_str = f" | unused_proto={metrics['n_unused_protos']:.0f}" if 'n_unused_protos' in metrics else ""

        knn_str = ""
        # if epoch % 5 == 0 or epoch == total_epochs:
        #     pca_acc, z_acc = self._niche_knn_acc()
        #     if pca_acc is not None:
        #         knn_str = f" | KNN: {z_acc:.1%} (pca:{pca_acc:.1%})"

        print(f">>> Epoch {epoch}/{total_epochs} | "
              f"loss={metrics['loss']:.4f} | "
              f"q+={metrics['q_pos']:.3f} | "
              f"q-={metrics['q_neg']:.3f} | "
              f"margin={metrics['margin']:.3f}{effk_str}{unused_str}{extra}{knn_str}")

    def train_umap_edges(self, epochs: int = None, verbose: bool = True,
                         early_stop: bool = False, eval_freq: int = 10,
                         patience: int = 50, max_epochs: int = None,
                         early_stop_metric: str = 'homophily'):
        """Train encoder using edge-centric parametric UMAP (fresh start).

        Args:
            epochs: fixed number of epochs (used when early_stop=False)
            early_stop: if True, stop based on early_stop_metric instead of fixed epochs
            eval_freq: evaluate metric every N epochs (used when early_stop=True)
            patience: stop if no improvement for this many epochs (used when early_stop=True)
            max_epochs: hard cap on epochs regardless of early stopping
            early_stop_metric: 'homophily' or 'modularity' (used when early_stop=True)
        """
        epochs = epochs or getattr(self, 'pretraining_epochs', 200)
        self._setup_umap_edges(epochs)
        metrics = self.continue_train_umap_edges(
            epochs, verbose,
            early_stop=early_stop, eval_freq=eval_freq,
            patience=patience, max_epochs=max_epochs,
            early_stop_metric=early_stop_metric,
        )
        self.save_clusters()
        return metrics

    def continue_train_umap_edges(self, epochs: int = 50, verbose: bool = True,
                                  early_stop: bool = False, eval_freq: int = 10,
                                  patience: int = 50, max_epochs: int = None,
                                  early_stop_metric: str = 'homophily'):
        """Continue UMAP edge training from current model state for more epochs.

        Rebuilds optimizer/loader/edge_dataset if needed (e.g. after code reload),
        but never touches model weights or prototypes.

        Args:
            epochs: fixed number of epochs (used when early_stop=False)
            early_stop: if True, ignore epochs and stop based on early_stop_metric
            eval_freq: evaluate metric every N epochs (used when early_stop=True)
            patience: stop if no improvement for this many epochs
            max_epochs: hard cap on epochs (used with early_stop=True)
            early_stop_metric: 'homophily' or 'modularity' (used when early_stop=True)
        """
        if not hasattr(self, '_umap_state'):
            self._setup_umap_edges(epochs, init_prototypes=False)

        start = self._umap_state['epoch']

        if not early_stop:
            # --- fixed-epoch mode (original behaviour) ---
            freq = getattr(self, 'umap_checkpoint_freq', 20)
            last_epoch = start + epochs
            for i in range(epochs):
                metrics = self._run_umap_epoch()
                current_epoch = start + i + 1
                if verbose:
                    self._print_umap_epoch(current_epoch, last_epoch, metrics)
                if freq > 0 and (current_epoch % freq == 0 or current_epoch == last_epoch):
                    self.save_umap_checkpoint()
            return metrics

        # --- early-stopping mode ---
        best_score = -1.0
        no_improve_epochs = 0
        epoch_offset = 0

        print(f"Early stopping mode: metric={early_stop_metric}, eval every {eval_freq} epochs, patience={patience}" +
              (f", max_epochs={max_epochs}" if max_epochs else ""))

        # Print initial modularity before any training
        if early_stop_metric == 'modularity':
            init_result = self.modularity()
            best_score = init_result['modularity']
            print(f"[Epoch 0] initial modularity={best_score:.4f}")

        while True:
            metrics = self._run_umap_epoch()
            epoch_offset += 1
            current_epoch = start + epoch_offset

            if verbose:
                end_str = f"~{max_epochs}" if max_epochs else "?"
                self._print_umap_epoch(current_epoch, end_str, metrics)

            # Hard cap
            if max_epochs and epoch_offset >= max_epochs:
                print(f"[Early stop] Reached max_epochs={max_epochs}. Saving checkpoint.")
                self.save_umap_checkpoint()
                break

            # Eval block
            if epoch_offset % eval_freq == 0:
                if early_stop_metric == 'modularity':
                    # In recon-only mode prototypes aren't updated during training,
                    # so re-init them from current encoder before scoring.
                    recon_only = (getattr(self, 'lambda_recon', 0) > 0 and
                                  getattr(self, 'lambda_umap', 1) == 0 and
                                  getattr(self, 'lambda_proto', 0) == 0 and
                                  getattr(self, 'lambda_swav', 0) == 0 and
                                  getattr(self, 'lambda_proto_recon', 0) == 0)
                    embedding_mode = getattr(self, 'umap_similarity', 'embedding') == 'embedding'
                    if recon_only or embedding_mode:
                        self.init_prototypes()
                    result = self.modularity()
                    score = result['modularity']
                else:
                    result = self.edge_homophily()
                    score = result['homophily']

                if score > best_score:
                    improvement = score - best_score
                    best_score = score
                    no_improve_epochs = 0
                    print(f"  [Early stop] {early_stop_metric} improved to {score:.4f} (+{improvement:.4f}) → saving checkpoint")
                    self.save_umap_checkpoint()
                else:
                    no_improve_epochs += eval_freq
                    print(f"  [Early stop] No improvement ({score:.4f} vs best {best_score:.4f}), "
                          f"no-improve streak: {no_improve_epochs}/{patience}")
                    if no_improve_epochs >= patience:
                        print(f"[Early stop] Patience exhausted. Stopping at epoch {current_epoch}.")
                        break

        return metrics

    def save_umap_checkpoint(self, path=None):
        """Save model + optimizer state so training can resume after Colab restart."""
        if path is None:
            path = os.path.join(self.get_dump_path(), 'umap_checkpoint.pth')
        state = {
            'model_state_dict': self.model.state_dict(),
            'epoch': self._umap_state['epoch'] if hasattr(self, '_umap_state') else 0,
            'optimizer_state_dict': self._umap_state['optimizer'].state_dict() if hasattr(self, '_umap_state') else None,
        }
        torch.save(state, path)
        print(f"Saved UMAP checkpoint to {path} (epoch {state['epoch']})")
        return path

    def load_umap_checkpoint(self, path=None):
        """Load model weights and rebuild UMAP training state for continue_train_umap_edges.

        Usage in Colab after restart:
            t = SCProtoTrainer(dataset=ds, ...)
            t.setup()
            t.load_umap_checkpoint()
            t.continue_train_umap_edges(epochs=50)
        """
        if path is None:
            path = os.path.join(self.get_dump_path(), 'umap_checkpoint.pth')
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

        # Rebuild training objects (dataset, loader, loss_fn, optimizer)
        self._setup_umap_edges()
        self._umap_state['epoch'] = checkpoint['epoch']

        if checkpoint.get('optimizer_state_dict') is not None:
            self._umap_state['optimizer'].load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Loaded UMAP checkpoint from {path} (resuming from epoch {checkpoint['epoch']})")

    def save_modularity(self):
        import json
        import numpy as np
        result = self.modularity()
        path = os.path.join(self.get_dump_path(), 'modularity.json')
        def _convert(o):
            if isinstance(o, np.ndarray): return o.tolist()
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            raise TypeError(f'Not serializable: {type(o)}')
        with open(path, 'w') as f:
            json.dump(result, f, indent=2, default=_convert)
        print(f"Saved modularity={result['modularity']:.4f} to {path}")
        return result

    def _log_metric(self, name, value):
        """Store a scalar or dict metric value in the internal log and flush to disk."""
        self._metrics_log[name] = value
        self._save_metrics()

    def _save_metrics(self):
        """Flush current metrics log to a small JSON file on disk."""
        import json
        try:
            # When running inside _eval_baseline, write to baseline folder
            if hasattr(self, '_baseline_save_dir'):
                save_dir = self._baseline_save_dir
            else:
                save_dir = self.get_dump_path()
            path = os.path.join(save_dir, 'metrics.json')
            with open(path, 'w') as f:
                json.dump(self._metrics_log, f, indent=2)
        except Exception:
            pass  # don't break metric computation if save fails

    def save_clusters(self, assignments=None, label='proto', path=None):
        """Save cluster assignments and all accumulated metrics to disk.

        Saves a .npz file with:
          - assignments: int array (n_cells,)
          - label: assignment method name
          - metrics: dict of all scalar metrics recorded so far

        Args:
            assignments: cluster labels, or None to auto-resolve.
            label: assignment method name.
            path: save path. Defaults to <dump_path>/clusters.npz.

        Returns:
            path where file was saved.
        """
        import json

        assignments, label = self._get_assignments(assignments, label)

        if path is None:
            path = os.path.join(self.get_dump_path(), 'clusters.npz')

        # Use smallest int dtype that fits
        max_val = int(assignments.max()) if len(assignments) else 0
        if max_val < 128:
            dtype = np.int8
        elif max_val < 32768:
            dtype = np.int16
        else:
            dtype = np.int32

        # Serialise metrics to JSON string (np.savez doesn't handle nested dicts)
        metrics_json = json.dumps(self._metrics_log)

        np.savez_compressed(
            path,
            assignments=assignments.astype(dtype),
            label=np.array(label),
            metrics_json=np.array(metrics_json),
        )
        print(f"Saved clusters ({len(assignments)} cells, label='{label}') "
              f"and {len(self._metrics_log)} metrics to {path}")
        return path

    @staticmethod
    def load_metrics(path):
        """Load cluster assignments and metrics saved by save_clusters.

        Args:
            path: path to .npz file saved by save_clusters.

        Returns:
            dict with 'assignments', 'label', 'metrics'.
        """
        import json

        data = np.load(path, allow_pickle=True)
        metrics = json.loads(str(data['metrics_json']))
        return {
            'assignments': data['assignments'],
            'label': str(data['label']),
            'metrics': metrics,
        }

    def _get_assignments(self, assignments=None, label=None):
        """Resolve cluster assignments and label.

        If assignments is None:
          - if prototypes were trained (umap_similarity == 'proto'), use argmax
          - otherwise, run k-means on the embedding space

        Returns:
            (assignments, label) — numpy int array and string label.
        """
        from sklearn.cluster import KMeans

        with torch.no_grad():
            z = self.encode_adata(self.train_ds.adata, self.model, z_idx=1)

        if assignments is None:
            proto_trained = getattr(self, 'umap_similarity', 'embedding') == 'proto'
            protos = self.model.get_prototypes()
            if proto_trained and protos is not None and protos.shape[0] > 0:
                scores = self.model.prototypes(z)
                assignments = scores.argmax(dim=1).cpu().numpy()
                label = label or 'proto'
            else:
                z_np = z.cpu().numpy()
                K = self.nmb_prototypes
                km = KMeans(n_clusters=K, n_init=3, random_state=42).fit(z_np)
                assignments = km.labels_
                label = label or 'kmeans'
        else:
            label = label or 'custom'

        return assignments, label

    def clustering_score(self, assignments=None, label='proto'):
        """Compute weighted same-cluster rate for a given assignment.

        Args:
            assignments: np.ndarray of integer cluster labels (n_cells,), or None for auto.
            label: name for this assignment (used in printing).

        Returns:
            dict with score, expected_random, ratio, per_cluster scores, assignments.
        """
        assignments, label = self._get_assignments(assignments, label)

        A = self.train_ds.aff_raw if hasattr(self.train_ds, 'aff_raw') else self.train_ds.aff
        A_coo = A.tocoo()

        # Weighted same-cluster rate
        heads, tails, weights = A_coo.row, A_coo.col, A_coo.data.astype(np.float64)
        same = (assignments[heads] == assignments[tails])
        score = (weights * same).sum() / weights.sum()

        # Expected random baseline: Sigma_k (n_k / n)^2
        _, counts = np.unique(assignments, return_counts=True)
        n = len(assignments)
        expected = ((counts / n) ** 2).sum()

        # Per-cluster: fraction of member affinity that stays within cluster
        cluster_ids = np.unique(assignments)
        per_cluster = {}
        for c in cluster_ids:
            mask = assignments[heads] == c
            if mask.sum() == 0:
                continue
            c_weights = weights[mask]
            c_same = assignments[tails[mask]] == c
            per_cluster[c] = (c_weights * c_same).sum() / c_weights.sum()

        print(f"[{label}] weighted same-cluster rate: {score:.4f} "
              f"(random baseline: {expected:.4f}, ratio: {score/expected:.1f}x)")

        self._log_metric('clustering_score', score)
        self._log_metric('clustering_score/ratio', score / expected)
        self._log_metric('clustering_score/per_cluster', {int(k): float(v) for k, v in per_cluster.items()})

        return {
            'score': score,
            'expected_random': expected,
            'ratio': score / expected,
            'per_cluster': per_cluster,
            'assignments': assignments,
            'label': label,
        }

    def modularity(self, assignments=None, label='proto'):
        """Compute weighted Newman modularity for a given assignment.

        Q = (1/2m) * sum_k [ e_k - d_k^2 / (2m) ]

        where e_k = sum of edge weights within cluster k,
              d_k = sum of degrees of nodes in cluster k,
              2m  = total edge weight.

        Args:
            assignments: np.ndarray of integer cluster labels (n_cells,), or None for auto.
            label: name for this assignment (used in printing).

        Returns:
            dict with modularity, assignments, label, per_cluster_contribution.
        """
        import scipy.sparse as sp

        assignments, label = self._get_assignments(assignments, label)

        A = self.train_ds.aff_raw if hasattr(self.train_ds, 'aff_raw') else self.train_ds.aff
        A = sp.csr_matrix(A)
        A = (A + A.T) / 2

        # Degree vector and total weight
        degrees = np.array(A.sum(axis=1)).ravel()
        two_m = degrees.sum()

        if two_m == 0:
            print(f"[{label}] weighted modularity: 0.0000 (no edges)")
            return {
                'modularity': 0.0,
                'assignments': assignments,
                'label': label,
                'per_cluster_contribution': {},
            }

        # Per-cluster modularity contribution
        cluster_ids = np.unique(assignments)
        per_cluster = {}
        Q = 0.0
        for c in cluster_ids:
            mask = (assignments == c)
            # e_k: sum of edge weights within cluster k (both directions counted)
            A_sub = A[mask][:, mask]
            e_k = A_sub.sum()
            # d_k: sum of degrees of nodes in cluster k
            d_k = degrees[mask].sum()
            contrib = (e_k - d_k * d_k / two_m) / two_m
            per_cluster[int(c)] = float(contrib)
            Q += contrib

        Q = float(Q)
        print(f"[{label}] weighted modularity: {Q:.4f}")

        self._log_metric('modularity', Q)
        self._log_metric('modularity/per_cluster', per_cluster)

        return {
            'modularity': Q,
            'assignments': assignments,
            'label': label,
            'per_cluster_contribution': per_cluster,
        }

    def ncut(self, assignments=None, label='proto'):
        """Compute weighted normalized cut for a given assignment.

        NCut = sum_k  cut(C_k, V\\C_k) / vol(C_k)

        where cut(C_k, V\\C_k) = sum of edge weights crossing cluster k,
              vol(C_k)          = sum of degrees of nodes in cluster k.

        Lower is better.

        Args:
            assignments: np.ndarray of integer cluster labels (n_cells,), or None for auto.
            label: name for this assignment (used in printing).

        Returns:
            dict with ncut, assignments, label, per_cluster_contribution.
        """
        import scipy.sparse as sp

        assignments, label = self._get_assignments(assignments, label)

        A = self.train_ds.aff_raw if hasattr(self.train_ds, 'aff_raw') else self.train_ds.aff
        A = sp.csr_matrix(A)
        # Symmetrise — NCut formula assumes undirected graph
        A = (A + A.T) / 2

        degrees = np.array(A.sum(axis=1)).ravel()

        cluster_ids = np.unique(assignments)
        per_cluster = {}
        ncut_val = 0.0
        for c in cluster_ids:
            mask = (assignments == c)
            vol_k = degrees[mask].sum()
            if vol_k == 0:
                continue
            # Intra-cluster edge weight
            e_k = A[mask][:, mask].sum()
            # cut = vol_k - e_k  (edges leaving cluster = total degree - internal edges)
            cut_k = vol_k - e_k
            contrib = cut_k / vol_k
            per_cluster[int(c)] = float(contrib)
            ncut_val += contrib

        ncut_val = float(ncut_val)
        print(f"[{label}] weighted normalized cut: {ncut_val:.4f}")

        self._log_metric('ncut', ncut_val)
        self._log_metric('ncut/per_cluster', per_cluster)

        return {
            'ncut': ncut_val,
            'assignments': assignments,
            'label': label,
            'per_cluster_contribution': per_cluster,
        }

    def conductance(self, assignments=None, label='proto'):
        """Compute mean weighted conductance across clusters.

        conductance(C) = cut(C, V\C) / min(vol(C), vol(V\C))

        Lower is better (0 = perfect, 1 = worst).

        Returns:
            dict with mean_conductance, per_cluster, assignments, label.
        """
        import scipy.sparse as sp

        assignments, label = self._get_assignments(assignments, label)

        A = self.train_ds.aff_raw if hasattr(self.train_ds, 'aff_raw') else self.train_ds.aff
        A = sp.csr_matrix(A)
        A = (A + A.T) / 2

        degrees = np.array(A.sum(axis=1)).ravel()
        total_vol = degrees.sum()

        cluster_ids = np.unique(assignments)
        per_cluster = {}
        vals = []
        for c in cluster_ids:
            mask = (assignments == c)
            vol_c = degrees[mask].sum()
            if vol_c == 0:
                continue
            e_c = A[mask][:, mask].sum()
            cut_c = vol_c - e_c
            denom = min(vol_c, total_vol - vol_c)
            if denom == 0:
                continue
            cond = cut_c / denom
            per_cluster[int(c)] = float(cond)
            vals.append(cond)

        mean_cond = float(np.mean(vals)) if vals else 0.0
        print(f"[{label}] mean conductance: {mean_cond:.4f}  (lower is better)")

        self._log_metric('conductance', mean_cond)
        self._log_metric('conductance/per_cluster', per_cluster)

        return {
            'mean_conductance': mean_cond,
            'per_cluster': per_cluster,
            'assignments': assignments,
            'label': label,
        }

    def edge_homophily(self, assignments=None, label='proto'):
        """Compute weighted edge homophily.

        homophily = sum of edge weights within clusters / total edge weight

        Higher is better (1 = all edges within clusters, 0 = all edges cut).

        Returns:
            dict with homophily, assignments, label.
        """
        import scipy.sparse as sp

        assignments, label = self._get_assignments(assignments, label)

        A = self.train_ds.aff_raw if hasattr(self.train_ds, 'aff_raw') else self.train_ds.aff
        A = sp.csr_matrix(A)
        A = (A + A.T) / 2

        total_weight = A.sum()
        if total_weight == 0:
            print(f"[{label}] edge homophily: 0.0000 (no edges)")
            return {'homophily': 0.0, 'assignments': assignments, 'label': label}

        intra_weight = 0.0
        for c in np.unique(assignments):
            mask = (assignments == c)
            intra_weight += A[mask][:, mask].sum()

        h = float(intra_weight / total_weight)
        print(f"[{label}] edge homophily: {h:.4f}  (higher is better)")

        self._log_metric('edge_homophily', h)

        return {
            'homophily': h,
            'assignments': assignments,
            'label': label,
        }

    def eval_graph_structure(self, assignments=None, label='proto'):
        """Run graph structure preservation metrics and log them.

        Computes: modularity, edge_homophily.

        Returns:
            dict with all scalar metrics.
        """
        print(f"\n{'='*50}")
        print(f"Graph Structure Preservation Metrics [{label}]")
        print(f"{'='*50}")

        r_mod = self.modularity(assignments, label)
        r_hom = self.edge_homophily(assignments, label)

        summary = {
            'modularity':     r_mod['modularity'],
            'edge_homophily': r_hom['homophily'],
        }

        print(f"\nSummary:")
        print(f"  Modularity:      {summary['modularity']:.4f}  (higher is better)")
        print(f"  Edge Homophily:  {summary['edge_homophily']:.4f}  (higher is better)")
        print(f"{'='*50}\n")

        return summary

    def metacell_f1(self, assignments=None, label='proto'):
        """Compute F1 score treating clusters as metacells with majority-vote labels.

        Each cluster is assigned the majority ground-truth label of its member
        cells. Every cell then inherits its cluster's majority label as its
        prediction. F1 is computed between ground-truth and predicted labels.

        Evaluates on each available label key:
          - self.dataset.label_key (e.g. "celltype", "final_annotation")
          - "niches_2D" (if present in obs)

        Args:
            assignments: np.ndarray of integer cluster labels (n_cells,), or None for auto.
            label: name for this assignment (used in printing).

        Returns:
            dict with label, assignments, and results per label key containing
            per_class_f1, macro_f1, weighted_f1.
        """
        from sklearn.metrics import f1_score

        assignments, label = self._get_assignments(assignments, label)
        obs = self.train_ds.adata.obs

        # Gather label keys to evaluate
        keys_to_eval = []
        lk = self.dataset.label_key
        if lk in obs.columns:
            keys_to_eval.append(lk)
        if "niches_2D" in obs.columns:
            keys_to_eval.append("niches_2D")

        if not keys_to_eval:
            print(f"[{label}] metacell_f1: no label keys found in obs "
                  f"(checked '{lk}', 'niches_2D')")
            return {'label': label, 'assignments': assignments, 'results': {}}

        results = {}
        for key in keys_to_eval:
            gt = obs[key].values
            cluster_ids = np.unique(assignments)

            # Majority-vote label per cluster
            cluster_label = {}
            for c in cluster_ids:
                member_labels = gt[assignments == c]
                vals, counts = np.unique(member_labels, return_counts=True)
                cluster_label[c] = vals[counts.argmax()]

            # Predicted label for each cell = its cluster's majority label
            pred = np.array([cluster_label[c] for c in assignments])

            # Unique classes present in ground truth
            classes = np.unique(gt)

            per_class = {}
            f1_per = f1_score(gt, pred, labels=classes, average=None, zero_division=0)
            for cls, f in zip(classes, f1_per):
                per_class[cls] = float(f)

            macro = float(f1_score(gt, pred, labels=classes, average='macro', zero_division=0))
            weighted = float(f1_score(gt, pred, labels=classes, average='weighted', zero_division=0))

            results[key] = {
                'per_class_f1': per_class,
                'macro_f1': macro,
                'weighted_f1': weighted,
            }

            self._log_metric(f'f1_macro/{key}', macro)
            self._log_metric(f'f1_weighted/{key}', weighted)
            self._log_metric(f'f1_per_class/{key}', {str(k): float(v) for k, v in per_class.items()})

            print(f"[{label} | {key}] macro_f1: {macro:.4f}  weighted_f1: {weighted:.4f}")

        return {
            'label': label,
            'assignments': assignments,
            'results': results,
        }

    def spectral_assignments(self, n_clusters=None):
        """Run spectral clustering on the affinity graph.

        Args:
            n_clusters: number of clusters. Defaults to self.nmb_prototypes.

        Returns:
            np.ndarray of integer cluster labels (n_cells,).
        """
        from sklearn.cluster import SpectralClustering
        import scipy.sparse as sp

        if n_clusters is None:
            n_clusters = self.nmb_prototypes

        A = self.train_ds.aff_raw if hasattr(self.train_ds, 'aff_raw') else self.train_ds.aff
        A = sp.csr_matrix(A)

        # Symmetrise just in case
        A = (A + A.T) / 2

        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=42,
            n_init=3,
        )
        return sc.fit_predict(A)

    def _run_metrics(self, assignments, label):
        """Run all four metrics on given assignments. Returns results dict."""
        cs = self.clustering_score(assignments=assignments, label=label)
        mod = self.modularity(assignments=assignments, label=label)
        nc = self.ncut(assignments=assignments, label=label)
        f1 = self.metacell_f1(assignments=assignments, label=label)
        return {
            'clustering_score': cs,
            'modularity': mod,
            'ncut': nc,
            'metacell_f1': f1,
        }

    def eval_all(self, assignments=None, label=None):
        """Run all metrics, save clusters and metrics to disk.

        Args:
            assignments: cluster labels, or None for auto (proto/kmeans).
            label: name for the primary assignment. Auto-detected if None.

        Returns:
            dict with clustering_score, modularity, ncut, metacell_f1,
            assignments, label.
        """
        assignments, label = self._get_assignments(assignments, label)

        # Save model info for comparison tables
        self._log_metric('_info/umap_similarity', getattr(self, 'umap_similarity', 'embedding'))
        self._log_metric('_info/assignment_method', label)
        self._log_metric('_info/n_clusters', int(len(np.unique(assignments))))

        results = self._run_metrics(assignments, label)

        self.save_clusters(assignments=assignments, label=label)

        return {
            **results,
            'assignments': assignments,
            'label': label,
        }

    def _eval_baseline(self, assignments, name, n_clusters):
        """Evaluate a baseline and save to its own folder under get_save_dir().

        Args:
            assignments: np.ndarray of integer cluster labels.
            name: baseline name (used as folder name and label).
            n_clusters: number of clusters (for info logging).

        Returns:
            dict of metric results.
        """
        import json

        save_dir = os.path.join(self.get_save_dir(), name)
        os.makedirs(save_dir, exist_ok=True)

        # Swap metrics log AND disable auto-flush to model's folder
        orig_log = self._metrics_log
        self._metrics_log = {}
        self._baseline_save_dir = save_dir  # redirect _save_metrics

        self._log_metric('_info/assignment_method', name)
        self._log_metric('_info/n_clusters', int(n_clusters))

        results = self._run_metrics(assignments, name)

        # Write clusters.npz
        max_val = int(assignments.max()) if len(assignments) else 0
        dtype = np.int8 if max_val < 128 else (np.int16 if max_val < 32768 else np.int32)
        np.savez_compressed(
            os.path.join(save_dir, 'clusters.npz'),
            assignments=assignments.astype(dtype),
            label=np.array(name),
            metrics_json=np.array(json.dumps(self._metrics_log)),
        )

        print(f"Saved {name} baseline to {save_dir}")

        # Restore
        del self._baseline_save_dir
        self._metrics_log = orig_log
        return results

    def eval_spectral(self, n_clusters=None):
        """Evaluate spectral clustering baseline and save to its own folder.

        Args:
            n_clusters: number of clusters. Defaults to self.nmb_prototypes.

        Returns:
            dict of metric results.
        """
        if n_clusters is None:
            n_clusters = self.nmb_prototypes
        assignments = self.spectral_assignments(n_clusters)
        return self._eval_baseline(assignments, f'spectral_K{n_clusters}', n_clusters)

    def eval_seacells(self, n_clusters=None, max_iter=100):
        """Run SEACells archetypal analysis on our affinity and evaluate.

        Steps:
          1. Load affinity matrix, cast to symmetric CSR kernel
          2. Initialize SEACells model + archetypes
          3. Run archetypal analysis (fit)
          4. Extract hard assignments
          5. Save clustering + compute metrics for comparison

        Args:
            n_clusters: number of archetypes. Defaults to self.nmb_prototypes.
            max_iter: max fitting iterations.

        Returns:
            dict of metric results.
        """
        import SEACells.core
        import scipy.sparse as sp

        if n_clusters is None:
            n_clusters = self.nmb_prototypes

        # 1. Load affinity, cast to symmetric kernel
        A = self.train_ds.aff_raw if hasattr(self.train_ds, 'aff_raw') else self.train_ds.aff
        K = sp.csr_matrix(A)
        K = (K + K.T) / 2

        # 2. Init model with our kernel
        ad = self.train_ds.adata
        model = SEACells.core.SEACells(
            ad,
            build_kernel_on='X_pca',  # ignored when precomputed
            n_SEACells=n_clusters,
            use_gpu=False,
            verbose=True,
        )
        model.add_precomputed_kernel_matrix(K)

        # 3. Run archetypal analysis — random init (skip PCA-based waypoints)
        init_idx = np.random.choice(K.shape[0], n_clusters, replace=False)
        model.fit(max_iter=max_iter, initial_archetypes=init_idx)

        # 4. Hard assignments (A_: shape k x n → argmax per cell)
        assignments = np.array(model.A_.argmax(axis=0)).ravel()
        n_unique = len(np.unique(assignments))
        print(f"SEACells: {n_unique} unique clusters from {n_clusters} archetypes, "
              f"A_ shape={model.A_.shape}, RSS={getattr(model, 'RSS_iters', ['?'])[-1]}")

        # 5. Save + metrics
        return self._eval_baseline(assignments, f'seacells_K{n_clusters}', n_clusters)

    def eval_seacells_native(self, n_clusters=None, n_waypoint_eigs=10, max_iter=100):
        """Run SEACells with its own affinity computation (native mode).

        Unlike eval_seacells which uses our precomputed affinity, this runs
        SEACells end-to-end with its own kernel built from X_pca.

        Args:
            n_clusters: number of archetypes. Defaults to self.nmb_prototypes.
            n_waypoint_eigs: number of eigenvectors for waypoint initialization.
            max_iter: max fitting iterations.

        Returns:
            dict of metric results.
        """
        import SEACells.core
        import scanpy as sc

        if n_clusters is None:
            n_clusters = self.nmb_prototypes

        ad = self.train_ds.adata.copy()

        # Ensure PCA exists
        if 'X_pca' not in ad.obsm:
            print("Computing PCA for SEACells...")
            sc.pp.pca(ad, n_comps=50)

        # Build SEACells model with its own kernel
        model = SEACells.core.SEACells(
            ad,
            build_kernel_on='X_pca',
            n_SEACells=n_clusters,
            n_waypoint_eigs=n_waypoint_eigs,
            use_gpu=False,
            verbose=True,
        )

        # Build kernel (SEACells native)
        print("Building SEACells kernel...")
        model.construct_kernel_matrix()

        # Initialize archetypes
        print("Initializing archetypes...")
        model.initialize_archetypes()

        # Fit
        print(f"Fitting SEACells (max_iter={max_iter})...")
        model.fit(max_iter=max_iter)

        # Hard assignments
        assignments = np.array(model.A_.argmax(axis=0)).ravel()
        n_unique = len(np.unique(assignments))
        print(f"SEACells native: {n_unique} unique clusters from {n_clusters} archetypes")

        # Save + metrics
        return self._eval_baseline(assignments, f'seacells_native_K{n_clusters}', n_clusters)

    def compare_runs(self, base_dir=None):
        """Auto-discover and compare metrics from all model runs in a base directory.

        Scans every subdirectory of base_dir for metrics.json or clusters.npz.
        Only loads runs that have saved metrics. Model folder names are
        auto-shortened by stripping the longest common prefix so only the
        differing parts are shown.

        Args:
            base_dir: parent directory containing one folder per run.
                      Defaults to self.get_save_dir() (e.g. MODEL_DIR/<dataset_id>/).

        Returns:
            (df, all_metrics) — pandas DataFrame of scalar metrics + info,
            and full metrics dict.
        """
        import json
        import pandas as pd

        if base_dir is None:
            base_dir = self.get_save_dir()

        # Auto-discover runs
        all_metrics = {}
        for entry in sorted(os.listdir(base_dir)):
            run_dir = os.path.join(base_dir, entry)
            if not os.path.isdir(run_dir):
                continue
            json_path = os.path.join(run_dir, 'metrics.json')
            npz_path = os.path.join(run_dir, 'clusters.npz')
            if os.path.exists(json_path):
                with open(json_path) as f:
                    all_metrics[entry] = json.load(f)
            elif os.path.exists(npz_path):
                data = np.load(npz_path, allow_pickle=True)
                all_metrics[entry] = json.loads(str(data['metrics_json']))

        if not all_metrics:
            print(f"No runs with metrics found in {base_dir}")
            return None, {}

        # Auto-shorten names: strip longest common prefix up to last separator
        full_names = list(all_metrics.keys())
        if len(full_names) > 1:
            prefix = os.path.commonprefix(full_names)
            # Trim to last underscore/hyphen so we don't cut mid-word
            for sep in ('_', '-'):
                idx = prefix.rfind(sep)
                if idx > 0:
                    prefix = prefix[:idx + 1]
                    break
            short_names = [n[len(prefix):] or n for n in full_names]
        else:
            short_names = full_names
        name_map = dict(zip(full_names, short_names))

        print(f"Found {len(all_metrics)} runs in {base_dir}")

        # Collect scalar metrics, per-cluster dicts, and info fields
        scalar_keys = set()
        per_cluster_keys = set()
        info_keys = set()
        for m in all_metrics.values():
            for k, v in m.items():
                if k.startswith('_info/'):
                    info_keys.add(k)
                elif isinstance(v, (int, float)):
                    scalar_keys.add(k)
                elif isinstance(v, dict):
                    per_cluster_keys.add(k)
        scalar_keys = sorted(scalar_keys)
        per_cluster_keys = sorted(per_cluster_keys)
        info_keys = sorted(info_keys)

        # --- Build DataFrame ---
        rows = {}
        for full, short in name_map.items():
            row = {}
            for k in info_keys:
                row[k.replace('_info/', '')] = all_metrics[full].get(k, '')
            for k in scalar_keys:
                v = all_metrics[full].get(k)
                row[k] = v if isinstance(v, (int, float)) else None
            rows[short] = row
        df = pd.DataFrame(rows).T
        df.index.name = 'run'

        return df, all_metrics

    def compare_clusterings(self, assignments_dict=None, figsize=(14, 4)):
        """Compare clustering quality across methods.

        Args:
            assignments_dict: dict of {name: np.ndarray} cluster assignments.
                If None, compares proto assignments vs k-means on embeddings.
            figsize: figure size.

        Returns:
            dict of results per method, and the figure.
        """
        import matplotlib.pyplot as plt

        if assignments_dict is None:
            # Auto: proto vs kmeans
            from sklearn.cluster import KMeans
            with torch.no_grad():
                z = self.encode_adata(self.train_ds.adata, self.model, z_idx=1)
            z_np = z.cpu().numpy()

            # Proto assignments
            scores = self.model.prototypes(z)
            proto_assign = scores.argmax(dim=1).cpu().numpy()

            # K-means on same embeddings
            K = self.nmb_prototypes
            km = KMeans(n_clusters=K, n_init=3, random_state=42).fit(z_np)

            assignments_dict = {
                'proto': proto_assign,
                'kmeans': km.labels_,
            }

        results = {}
        for name, assigns in assignments_dict.items():
            results[name] = self.clustering_score(assigns, label=name)

        # Plot
        names = list(results.keys())
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Panel 1: overall score bar
        scores = [results[n]['score'] for n in names]
        baselines = [results[n]['expected_random'] for n in names]
        x = np.arange(len(names))
        axes[0].bar(x, scores, alpha=0.8, label='score')
        axes[0].bar(x, baselines, alpha=0.3, color='gray', label='random baseline')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names)
        axes[0].set_ylabel('weighted same-cluster rate')
        axes[0].set_title('Overall score')
        axes[0].legend(fontsize=8)

        # Panel 2: box plot of per-cluster intra-affinity
        box_data = []
        box_labels = []
        for n in names:
            vals = list(results[n]['per_cluster'].values())
            box_data.append(vals)
            box_labels.append(n)
        axes[1].boxplot(box_data, labels=box_labels)
        axes[1].set_ylabel('intra-cluster affinity fraction')
        axes[1].set_title('Per-cluster score')

        # Panel 3: ratio bar
        ratios = [results[n]['ratio'] for n in names]
        axes[2].bar(x, ratios, alpha=0.8, color='steelblue')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(names)
        axes[2].set_ylabel('score / random baseline')
        axes[2].set_title('Ratio (higher = better)')
        axes[2].axhline(1.0, color='gray', ls='--', lw=0.8)

        fig.tight_layout()
        plt.show()

        return results, fig

    def calc_r1r2_loss(self, z):
        """
        Li et al. style R1/R2 prototype coverage losses.
        Uses scores (same as assignment) to ensure consistency.

        R1: each proto should be "best" for at least 1 cell -> move proto
        R2: each cell should have high score for at least 1 proto -> move encoder

        Ensures: no orphan protos, no uncovered cells.
        """
        protos = self.model.get_prototypes()  # (K, D)

        # R1: move protos toward cells (detach z)
        # Use same scoring as assignment
        scores_r1 = self.model.proto_soft_assignments(z.detach())  # (B, K)
        r1 = -scores_r1.max(dim=0).values.mean()  # max score per proto -> minimize neg

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
        r2 = -scores_r2.max(dim=1).values.mean()  # max score per cell -> minimize neg

        return r1 + r2

    @torch.no_grad()
    def _sample_edges(self, n_edges=10000):
        """Sample positive edges and negative pairs, return indices and weights."""
        aff = self.train_ds.aff_raw if hasattr(self.train_ds, 'aff_raw') else self.train_ds.aff
        coo = aff.tocoo()
        heads, tails, weights = coo.row, coo.col, coo.data.astype(np.float64)
        n = min(n_edges, len(heads))
        idx = np.random.choice(len(heads), size=n, replace=False)
        neg_tails = np.random.randint(0, aff.shape[0], size=n)
        return heads[idx], tails[idx], weights[idx], neg_tails

    @torch.no_grad()
    def plot_p_distribution(self, n_edges=10000, bins=80, figsize=(7, 4)):
        """Plot p (input-space affinity weight) distribution for sampled edges."""
        import matplotlib.pyplot as plt
        heads, tails, weights, _ = self._sample_edges(n_edges)

        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(weights, bins=bins, alpha=0.7, density=True, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('p (affinity weight)')
        ax.set_ylabel('density')
        ax.set_title(f'p distribution (input space) — {len(weights)} edges')
        ax.axvline(np.median(weights), color='red', ls='--', label=f'median={np.median(weights):.4f}')
        ax.legend()
        plt.tight_layout()
        plt.show()
        return fig

    @torch.no_grad()
    def plot_q_distribution(self, n_edges=10000, bins=80, figsize=(10, 4)):
        """Plot q (latent-space similarity) for positive and negative pairs."""
        import matplotlib.pyplot as plt

        heads, tails, weights, neg_tails = self._sample_edges(n_edges)

        # Build X tensor if _umap_state not available
        if hasattr(self, '_umap_state'):
            X = self._umap_state['X']
            loss_fn = self._umap_state['loss_fn']
        else:
            adata = self.train_ds.adata
            if hasattr(adata.X, 'toarray'):
                X = torch.tensor(adata.X.toarray(), dtype=torch.float32).to(self.device)
            else:
                X = torch.tensor(adata.X, dtype=torch.float32).to(self.device)
            min_dist = getattr(self, 'umap_min_dist', 0.5)
            spread = getattr(self, 'umap_spread', 1.0)
            neg_rate = getattr(self, 'umap_neg_rate', 5)
            loss_fn = ParametricUMAPLoss(min_dist=min_dist, spread=spread, negative_sample_rate=neg_rate)

        use_proto_sim = getattr(self, 'umap_similarity', 'embedding') == 'proto'
        proto_metric = getattr(self, 'umap_proto_metric', 'dotp')

        all_idx = np.unique(np.concatenate([heads, tails, neg_tails]))
        X_batch = X[all_idx]

        if hasattr(self.train_ds, 'conditions'):
            batch_cond = self.train_ds.conditions[torch.tensor(all_idx)].to(self.device)
        else:
            n_conds = len(self.model.scpoli_cvae.n_conditions)
            batch_cond = torch.zeros(len(all_idx), n_conds, dtype=torch.long, device=self.device)

        self.model.eval()
        z_all, _, _ = self.model.encoder_out({'x': X_batch, 'batch': batch_cond})
        idx_map = {int(v): i for i, v in enumerate(all_idx)}
        def _map(arr):
            return torch.tensor([idx_map[int(x)] for x in arr], device=self.device)

        if use_proto_sim:
            logits = self.model.prototypes(z_all)
            sa = F.softmax(logits / self.epsilon, dim=1)
            s_h = sa[_map(heads)]
            s_t = sa[_map(tails)]
            s_n = sa[_map(neg_tails)]
            _eps = 1e-4
            if proto_metric == 'cosine':
                s_h = F.normalize(s_h, dim=-1)
                s_t = F.normalize(s_t, dim=-1)
                s_n = F.normalize(s_n, dim=-1)
            q_pos = (s_h * s_t).sum(dim=-1).clamp(_eps, 1 - _eps).cpu().numpy()
            q_neg = (s_h * s_n).sum(dim=-1).clamp(_eps, 1 - _eps).cpu().numpy()
        else:
            z_h = z_all[_map(heads)]
            z_t = z_all[_map(tails)]
            z_n = z_all[_map(neg_tails)]
            q_pos = loss_fn.compute_q(z_h, z_t).cpu().numpy()
            q_neg = loss_fn.compute_q(z_h, z_n).cpu().numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Left: q_pos vs q_neg density histograms
        ax1.hist(q_pos, bins=bins, alpha=0.6, density=True, label=f'q_pos (mean={q_pos.mean():.4f})', color='steelblue', edgecolor='black', linewidth=0.3)
        ax1.hist(q_neg, bins=bins, alpha=0.6, density=True, label=f'q_neg (mean={q_neg.mean():.4f})', color='salmon', edgecolor='black', linewidth=0.3)
        ax1.set_xlabel('q (latent similarity)')
        ax1.set_ylabel('density')
        ax1.set_title(f'q distribution — {len(q_pos)} edges')
        ax1.legend()

        # Right: p vs q scatter for positive edges
        ax2.scatter(weights, q_pos, s=1, alpha=0.3, c='steelblue', rasterized=True)
        ax2.set_xlabel('p (input affinity)')
        ax2.set_ylabel('q (latent similarity)')
        ax2.set_title('p vs q (positive edges)')
        # diagonal reference
        lims = [0, max(weights.max(), q_pos.max())]
        ax2.plot(lims, lims, 'k--', alpha=0.3, linewidth=0.8)

        plt.tight_layout()
        plt.show()
        return fig

    # ==================== Prototype Interpretation ====================

    @torch.no_grad()
    def decode_prototypes(self, batch_mode='mean'):
        """Decode all prototypes to gene expression space.

        Args:
            batch_mode: how to handle batch condition.
                'mean' - average over all batch conditions (default)
                'first' - use first batch condition
                'all' - return decoded for each batch condition

        Returns:
            If batch_mode='all': dict of {batch_id: np.ndarray [K, genes]}
            Otherwise: np.ndarray [K, genes]
        """
        protos = self.model.get_prototypes()  # [K, latent_dim]
        K = protos.shape[0]

        # Get unique batch conditions
        if hasattr(self.train_ds, 'conditions'):
            all_conds = self.train_ds.conditions.unique(dim=0)
        else:
            n_conds = len(self.model.scpoli_cvae.n_conditions)
            all_conds = torch.zeros(1, n_conds, dtype=torch.long, device=self.device)

        all_conds = all_conds.to(self.device)
        protos = protos.to(self.device)

        if batch_mode == 'all':
            results = {}
            for i, cond in enumerate(all_conds):
                cond_expanded = cond.unsqueeze(0).expand(K, -1)
                decoded = self.model.decode(protos, cond_expanded)
                results[i] = decoded.cpu().numpy()
            return results

        elif batch_mode == 'first':
            cond = all_conds[0].unsqueeze(0).expand(K, -1)
            decoded = self.model.decode(protos, cond)
            return decoded.cpu().numpy()

        else:  # mean
            decoded_sum = None
            for cond in all_conds:
                cond_expanded = cond.unsqueeze(0).expand(K, -1)
                decoded = self.model.decode(protos, cond_expanded)
                if decoded_sum is None:
                    decoded_sum = decoded
                else:
                    decoded_sum = decoded_sum + decoded
            return (decoded_sum / len(all_conds)).cpu().numpy()

    def label_prototypes(self, label_key):
        """Label each prototype based on majority vote of assigned cells.

        Args:
            label_key: column name in adata.obs for labels

        Returns:
            dict with:
                'labels': pd.Series mapping proto_id -> majority label
                'purity': pd.Series mapping proto_id -> purity (fraction of majority)
                'counts': pd.DataFrame with full crosstab
        """
        assignments, _ = self._get_assignments()
        labels = self.train_ds.adata.obs[label_key].values

        # Build crosstab: proto x label
        df = pd.DataFrame({'proto': assignments, 'label': labels})
        counts = pd.crosstab(df['proto'], df['label'])

        # Majority vote
        majority_labels = counts.idxmax(axis=1)
        purity = counts.max(axis=1) / counts.sum(axis=1)

        return {
            'labels': majority_labels,
            'purity': purity,
            'counts': counts,
        }

    def prototype_dge(self, niche_key, celltype_key, method='wilcoxon'):
        """Compute DGE on decoded prototypes per niche within each celltype.

        For each celltype, compares prototypes labeled as niche X vs rest.
        Saves ALL genes (no filtering).

        Args:
            niche_key: column for niche labels
            celltype_key: column for celltype labels
            method: scanpy rank_genes_groups method ('wilcoxon', 't-test', etc.)

        Returns:
            pd.DataFrame with columns:
                celltype, niche, gene, logfoldchange, pval, pval_adj, score
        """
        import scanpy as sc

        # Decode prototypes and create AnnData
        decoded = self.decode_prototypes(batch_mode='mean')
        proto_ad = sc.AnnData(decoded)
        proto_ad.var_names = self.train_ds.adata.var_names

        # Label prototypes
        niche_labels = self.label_prototypes(niche_key)['labels']
        ct_labels = self.label_prototypes(celltype_key)['labels']

        proto_ad.obs['niche'] = niche_labels.values
        proto_ad.obs['celltype'] = ct_labels.values
        proto_ad.obs['proto_id'] = niche_labels.index.values

        rows = []
        for ct in proto_ad.obs['celltype'].unique():
            ct_ad = proto_ad[proto_ad.obs['celltype'] == ct].copy()
            if ct_ad.n_obs < 3:
                continue

            niches = ct_ad.obs['niche'].unique()
            if len(niches) < 2:
                continue

            for niche in niches:
                n_pos = (ct_ad.obs['niche'] == niche).sum()
                n_neg = (ct_ad.obs['niche'] != niche).sum()
                if n_pos < 1 or n_neg < 1:
                    continue

                # Create binary group
                ct_ad.obs['_group'] = np.where(ct_ad.obs['niche'] == niche, 'pos', 'rest')

                try:
                    sc.tl.rank_genes_groups(
                        ct_ad, '_group', groups=['pos'], reference='rest',
                        method=method, use_raw=False
                    )
                    dge = sc.get.rank_genes_groups_df(ct_ad, group='pos')
                    dge['celltype'] = ct
                    dge['niche'] = niche
                    dge['n_pos'] = n_pos
                    dge['n_neg'] = n_neg
                    rows.append(dge)
                except Exception as e:
                    print(f"DGE failed for {ct}/{niche}: {e}")
                    continue

        if not rows:
            return pd.DataFrame()

        result = pd.concat(rows, ignore_index=True)
        result = result.rename(columns={'names': 'gene', 'logfoldchanges': 'logfoldchange',
                                         'pvals': 'pval', 'pvals_adj': 'pval_adj', 'scores': 'score'})
        return result[['celltype', 'niche', 'gene', 'logfoldchange', 'pval', 'pval_adj', 'score', 'n_pos', 'n_neg']]

    def singlecell_dge(self, niche_key, celltype_key, method='wilcoxon'):
        """Compute DGE on single cells per niche within each celltype.

        Same format as prototype_dge for direct comparison.

        Args:
            niche_key: column for niche labels
            celltype_key: column for celltype labels
            method: scanpy rank_genes_groups method

        Returns:
            pd.DataFrame with same columns as prototype_dge
        """
        import scanpy as sc

        adata = self.train_ds.adata.copy()

        rows = []
        for ct in adata.obs[celltype_key].unique():
            ct_ad = adata[adata.obs[celltype_key] == ct].copy()
            if ct_ad.n_obs < 3:
                continue

            niches = ct_ad.obs[niche_key].unique()
            if len(niches) < 2:
                continue

            for niche in niches:
                n_pos = (ct_ad.obs[niche_key] == niche).sum()
                n_neg = (ct_ad.obs[niche_key] != niche).sum()
                if n_pos < 2 or n_neg < 2:
                    continue

                ct_ad.obs['_group'] = np.where(ct_ad.obs[niche_key] == niche, 'pos', 'rest')

                try:
                    sc.tl.rank_genes_groups(
                        ct_ad, '_group', groups=['pos'], reference='rest',
                        method=method, use_raw=False
                    )
                    dge = sc.get.rank_genes_groups_df(ct_ad, group='pos')
                    dge['celltype'] = ct
                    dge['niche'] = niche
                    dge['n_pos'] = n_pos
                    dge['n_neg'] = n_neg
                    rows.append(dge)
                except Exception as e:
                    print(f"DGE failed for {ct}/{niche}: {e}")
                    continue

        if not rows:
            return pd.DataFrame()

        result = pd.concat(rows, ignore_index=True)
        result = result.rename(columns={'names': 'gene', 'logfoldchanges': 'logfoldchange',
                                         'pvals': 'pval', 'pvals_adj': 'pval_adj', 'scores': 'score'})
        return result[['celltype', 'niche', 'gene', 'logfoldchange', 'pval', 'pval_adj', 'score', 'n_pos', 'n_neg']]

    def prototype_stats(self, niche_key, celltype_key):
        """Compute statistical metrics for prototype clustering.

        Args:
            niche_key: column for niche labels
            celltype_key: column for celltype labels

        Returns:
            dict with:
                'niche_purity': purity of niche labels per prototype
                'celltype_purity': purity of celltype labels per prototype
                'chi2_niche': chi2 test for niche vs prototype association
                'chi2_celltype': chi2 test for celltype vs prototype association
                'silhouette_niche': silhouette score for niche separation (per celltype)
                'silhouette_celltype': silhouette score for celltype separation
        """
        from scipy.stats import chi2_contingency
        from sklearn.metrics import silhouette_score

        assignments, _ = self._get_assignments()
        obs = self.train_ds.adata.obs

        niche_labels = obs[niche_key].values
        ct_labels = obs[celltype_key].values

        # Purity
        niche_info = self.label_prototypes(niche_key)
        ct_info = self.label_prototypes(celltype_key)

        # Chi2 tests
        niche_tab = pd.crosstab(assignments, niche_labels)
        ct_tab = pd.crosstab(assignments, ct_labels)

        chi2_niche = chi2_contingency(niche_tab)[1] if niche_tab.shape[0] > 1 and niche_tab.shape[1] > 1 else np.nan
        chi2_ct = chi2_contingency(ct_tab)[1] if ct_tab.shape[0] > 1 and ct_tab.shape[1] > 1 else np.nan

        # Silhouette on embeddings
        with torch.no_grad():
            z = self.encode_adata(self.train_ds.adata, self.model, z_idx=1)
        z_np = z.cpu().numpy()

        # Overall celltype silhouette
        if len(np.unique(ct_labels)) >= 2:
            sil_ct = silhouette_score(z_np, ct_labels)
        else:
            sil_ct = np.nan

        # Niche silhouette per celltype
        sil_niche_per_ct = {}
        for ct in np.unique(ct_labels):
            mask = ct_labels == ct
            niches_ct = niche_labels[mask]
            if len(np.unique(niches_ct)) >= 2:
                sil_niche_per_ct[ct] = silhouette_score(z_np[mask], niches_ct)
            else:
                sil_niche_per_ct[ct] = np.nan

        return {
            'niche_purity_mean': niche_info['purity'].mean(),
            'niche_purity_per_proto': niche_info['purity'].to_dict(),
            'celltype_purity_mean': ct_info['purity'].mean(),
            'celltype_purity_per_proto': ct_info['purity'].to_dict(),
            'chi2_niche_pval': chi2_niche,
            'chi2_celltype_pval': chi2_ct,
            'silhouette_celltype': sil_ct,
            'silhouette_niche_per_ct': sil_niche_per_ct,
            'silhouette_niche_mean': np.nanmean(list(sil_niche_per_ct.values())),
        }

    def eval_prototypes(self, niche_key=None, celltype_key=None, save_dir=None):
        """Run all prototype evaluation and save results.

        Args:
            niche_key: column for niche labels. Defaults to 'niches_2D' if present.
            celltype_key: column for celltype labels. Defaults to dataset.label_key.
            save_dir: directory to save results. Defaults to get_dump_path().

        Returns:
            dict with all results
        """
        import json

        if save_dir is None:
            save_dir = self.get_dump_path()
        os.makedirs(save_dir, exist_ok=True)

        obs = self.train_ds.adata.obs
        if celltype_key is None:
            celltype_key = self.dataset.label_key
        if niche_key is None:
            niche_key = 'niches_2D' if 'niches_2D' in obs.columns else celltype_key

        print(f"Evaluating prototypes: niche_key={niche_key}, celltype_key={celltype_key}")

        results = {}

        # 1. Decode prototypes
        print("Decoding prototypes...")
        decoded = self.decode_prototypes(batch_mode='mean')
        np.save(os.path.join(save_dir, 'decoded_prototypes.npy'), decoded)
        results['decoded_shape'] = decoded.shape

        # 2. Label prototypes
        print("Labeling prototypes...")
        niche_labels = self.label_prototypes(niche_key)
        ct_labels = self.label_prototypes(celltype_key)

        label_df = pd.DataFrame({
            'proto_id': niche_labels['labels'].index,
            'niche': niche_labels['labels'].values,
            'niche_purity': niche_labels['purity'].values,
            'celltype': ct_labels['labels'].values,
            'celltype_purity': ct_labels['purity'].values,
        })
        label_df.to_csv(os.path.join(save_dir, 'prototype_labels.csv'), index=False)
        results['n_prototypes'] = len(label_df)

        # 3. Prototype DGE
        print("Computing prototype DGE...")
        proto_dge = self.prototype_dge(niche_key, celltype_key)
        proto_dge.to_csv(os.path.join(save_dir, 'prototype_dge.csv'), index=False)
        results['proto_dge_rows'] = len(proto_dge)

        # 4. Single-cell DGE
        print("Computing single-cell DGE...")
        sc_dge = self.singlecell_dge(niche_key, celltype_key)
        sc_dge.to_csv(os.path.join(save_dir, 'singlecell_dge.csv'), index=False)
        results['sc_dge_rows'] = len(sc_dge)

        # 5. Statistical tests
        print("Computing statistics...")
        stats = self.prototype_stats(niche_key, celltype_key)
        results.update({k: v for k, v in stats.items() if not isinstance(v, dict)})

        # Save stats
        stats_save = {k: (v if not isinstance(v, dict) else {str(kk): vv for kk, vv in v.items()})
                      for k, v in stats.items()}
        with open(os.path.join(save_dir, 'prototype_stats.json'), 'w') as f:
            json.dump(stats_save, f, indent=2, default=float)

        # 6. Save summary
        with open(os.path.join(save_dir, 'eval_summary.json'), 'w') as f:
            json.dump({k: (v if not isinstance(v, (np.floating, np.integer)) else float(v))
                       for k, v in results.items()}, f, indent=2)

        print(f"Results saved to {save_dir}")
        print(f"  - decoded_prototypes.npy: {decoded.shape}")
        print(f"  - prototype_labels.csv: {len(label_df)} prototypes")
        print(f"  - prototype_dge.csv: {len(proto_dge)} rows")
        print(f"  - singlecell_dge.csv: {len(sc_dge)} rows")
        print(f"  - prototype_stats.json")

        return results

    def eval_singlecell(self, niche_key=None, celltype_key=None):
        """Run DGE and stats on raw single cells (ground truth baseline).

        Saves results to <save_dir>/gt/ folder.

        Args:
            niche_key: column for niche labels. Defaults to 'niches_2D'.
            celltype_key: column for celltype labels. Defaults to dataset.label_key.

        Returns:
            dict with dge DataFrame and stats
        """
        import json
        import scanpy as sc
        from scipy.stats import chi2_contingency
        from sklearn.metrics import silhouette_score

        save_dir = os.path.join(self.get_save_dir(), 'gt')
        os.makedirs(save_dir, exist_ok=True)

        adata = self.train_ds.adata
        obs = adata.obs
        if celltype_key is None:
            celltype_key = self.dataset.label_key
        if niche_key is None:
            niche_key = 'niches_2D' if 'niches_2D' in obs.columns else celltype_key

        print(f"Evaluating single cells: niche_key={niche_key}, celltype_key={celltype_key}")

        # 1. DGE
        print("Computing single-cell DGE...")
        sc_dge = self.singlecell_dge(niche_key, celltype_key)
        sc_dge.to_csv(os.path.join(save_dir, 'singlecell_dge.csv'), index=False)

        # 2. Stats
        print("Computing statistics...")
        niche_labels = obs[niche_key].values
        ct_labels = obs[celltype_key].values

        # Chi2
        niche_tab = pd.crosstab(ct_labels, niche_labels)
        chi2_pval = chi2_contingency(niche_tab)[1] if niche_tab.shape[0] > 1 and niche_tab.shape[1] > 1 else np.nan

        # Silhouette on PCA
        if 'X_pca' not in adata.obsm:
            sc.pp.pca(adata, n_comps=min(50, adata.n_vars - 1))
        X = adata.obsm['X_pca']

        sil_ct = silhouette_score(X, ct_labels) if len(np.unique(ct_labels)) >= 2 else np.nan
        sil_niche_per_ct = {}
        for ct in np.unique(ct_labels):
            mask = ct_labels == ct
            if len(np.unique(niche_labels[mask])) >= 2:
                sil_niche_per_ct[ct] = silhouette_score(X[mask], niche_labels[mask])

        stats = {
            'chi2_niche_celltype_pval': chi2_pval,
            'silhouette_celltype': sil_ct,
            'silhouette_niche_per_ct': {str(k): v for k, v in sil_niche_per_ct.items()},
            'silhouette_niche_mean': np.nanmean(list(sil_niche_per_ct.values())),
            'n_cells': len(adata),
        }
        with open(os.path.join(save_dir, 'singlecell_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2, default=float)

        print(f"Saved to {save_dir}: singlecell_dge.csv ({len(sc_dge)} rows), singlecell_stats.json")
        return {'dge': sc_dge, 'stats': stats}

    def compare_dge_plots(self, ct=None, niche=None):
        """Load DGE results and plot volcano + logFC distribution comparison.

        Args:
            ct: celltype to show in volcano (if None, picks first available)
            niche: niche to show in volcano (if None, picks first available)

        Returns:
            dict with proto_dge, sc_dge DataFrames
        """
        import matplotlib.pyplot as plt

        # Load DGE files
        proto_path = os.path.join(self.get_dump_path(), 'prototype_dge.csv')
        sc_path = os.path.join(self.get_save_dir(), 'gt', 'singlecell_dge.csv')

        if not os.path.exists(proto_path):
            print(f"Run t.eval_prototypes() first. Missing: {proto_path}")
            return None
        if not os.path.exists(sc_path):
            print(f"Run t.eval_singlecell() first. Missing: {sc_path}")
            return None

        proto_dge = pd.read_csv(proto_path)
        sc_dge = pd.read_csv(sc_path)

        # Pick ct/niche if not specified
        if ct is None:
            ct = proto_dge['celltype'].iloc[0]
        if niche is None:
            niche = proto_dge[proto_dge['celltype'] == ct]['niche'].iloc[0]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # --- Row 1: Volcano plots ---
        for ax, dge, title in [(axes[0, 0], sc_dge, 'Single-cell'), (axes[0, 1], proto_dge, 'Metacell')]:
            d = dge[(dge['celltype'] == ct) & (dge['niche'] == niche)].copy()
            if len(d) == 0:
                ax.set_title(f'{title}: No data for {ct}/{niche}')
                continue
            d['neg_log_p'] = -np.log10(d['pval'].clip(lower=1e-300))

            ax.scatter(d['logfoldchange'], d['neg_log_p'], s=3, alpha=0.4, c='gray')
            sig = d[(d['pval_adj'] < 0.05) & (abs(d['logfoldchange']) > 0.5)]
            ax.scatter(sig['logfoldchange'], sig['neg_log_p'], s=8, c='red', alpha=0.7)

            ax.set_xlabel('Log Fold Change')
            ax.set_ylabel('-log10(p)')
            ax.set_title(f'{title}: {ct} / {niche}\n({len(sig)} sig genes)')
            ax.axhline(-np.log10(0.05), ls='--', c='blue', alpha=0.3)
            ax.axvline(0, ls='--', c='black', alpha=0.3)

        # --- Row 2: LogFC distributions ---
        ax = axes[1, 0]
        ax.hist(sc_dge['logfoldchange'].dropna(), bins=50, alpha=0.6,
                label=f'Single-cell (std={sc_dge["logfoldchange"].std():.2f})', density=True)
        ax.hist(proto_dge['logfoldchange'].dropna(), bins=50, alpha=0.6,
                label=f'Metacell (std={proto_dge["logfoldchange"].std():.2f})', density=True)
        ax.set_xlabel('Log Fold Change')
        ax.set_ylabel('Density')
        ax.set_title('LogFC Distribution (all genes)')
        ax.legend()

        # --- Row 2 right: Signal comparison scatter ---
        ax = axes[1, 1]
        rows = []
        for (c, n), p_grp in proto_dge.groupby(['celltype', 'niche']):
            s_grp = sc_dge[(sc_dge['celltype'] == c) & (sc_dge['niche'] == n)]
            if len(s_grp) < 10 or len(p_grp) < 5:
                continue
            rows.append({
                'sc_signal': s_grp.nlargest(20, 'logfoldchange')['logfoldchange'].mean(),
                'proto_signal': p_grp.nlargest(20, 'logfoldchange')['logfoldchange'].mean(),
            })
        if rows:
            df_sig = pd.DataFrame(rows)
            ax.scatter(df_sig['sc_signal'], df_sig['proto_signal'], s=40, alpha=0.7)
            lim = max(df_sig['sc_signal'].max(), df_sig['proto_signal'].max()) * 1.1
            ax.plot([0, lim], [0, lim], 'k--', alpha=0.3)
            pct = (df_sig['proto_signal'] > df_sig['sc_signal']).mean() * 100
            ax.set_title(f'Signal Strength\n(Metacell stronger in {pct:.0f}% cases)')
        ax.set_xlabel('Single-cell signal (mean top20 logFC)')
        ax.set_ylabel('Metacell signal (mean top20 logFC)')

        plt.tight_layout()
        plt.show()

        # Compute comparison scores
        from interpretable_ssl.evaluation.niche_recovery import jaccard, rbo
        score_rows = []
        for (c, n), p_grp in proto_dge.groupby(['celltype', 'niche']):
            s_grp = sc_dge[(sc_dge['celltype'] == c) & (sc_dge['niche'] == n)]
            if len(s_grp) < 10 or len(p_grp) < 5:
                continue
            p_genes = p_grp.nlargest(20, 'logfoldchange')['gene'].tolist()
            s_genes = s_grp.nlargest(20, 'logfoldchange')['gene'].tolist()
            merged = p_grp.merge(s_grp, on='gene', suffixes=('_proto', '_sc'))
            corr = merged['logfoldchange_proto'].corr(merged['logfoldchange_sc']) if len(merged) > 5 else np.nan
            score_rows.append({
                'celltype': c, 'niche': n,
                'jaccard': jaccard(p_genes, s_genes),
                'rbo': rbo(p_genes, s_genes),
                'logfc_corr': corr,
                'n_proto': len(p_grp), 'n_sc': len(s_grp),
            })
        scores_df = pd.DataFrame(score_rows)

        print(f"\nDGE Comparison Scores:")
        print(f"  Jaccard:   mean={scores_df['jaccard'].mean():.3f}, median={scores_df['jaccard'].median():.3f}")
        print(f"  RBO:       mean={scores_df['rbo'].mean():.3f}, median={scores_df['rbo'].median():.3f}")
        print(f"  LogFC corr: mean={scores_df['logfc_corr'].mean():.3f}, median={scores_df['logfc_corr'].median():.3f}")

        return {'proto_dge': proto_dge, 'sc_dge': sc_dge, 'scores': scores_df}

    def compare_dge_all_niches(self, celltype=None):
        """Plot volcano comparison for all niches (one row per niche).

        Args:
            celltype: filter to specific celltype (None = all)

        Returns:
            dict with proto_dge, sc_dge, scores DataFrames
        """
        import matplotlib.pyplot as plt

        # Load DGE files
        proto_path = os.path.join(self.get_dump_path(), 'prototype_dge.csv')
        sc_path = os.path.join(self.get_save_dir(), 'gt', 'singlecell_dge.csv')

        if not os.path.exists(proto_path) or not os.path.exists(sc_path):
            print("Run t.eval_prototypes() and t.eval_singlecell() first")
            return None

        proto_dge = pd.read_csv(proto_path)
        sc_dge = pd.read_csv(sc_path)

        # Get unique celltype/niche combos
        if celltype:
            proto_dge = proto_dge[proto_dge['celltype'] == celltype]
            sc_dge = sc_dge[sc_dge['celltype'] == celltype]

        combos = proto_dge.groupby(['celltype', 'niche']).size().reset_index()[['celltype', 'niche']]
        n_combos = len(combos)

        if n_combos == 0:
            print("No data found")
            return None

        # Create figure: 2 columns (single-cell, metacell), n_combos rows
        fig, axes = plt.subplots(n_combos, 2, figsize=(10, 3 * n_combos))
        if n_combos == 1:
            axes = axes.reshape(1, -1)

        for i, (_, row) in enumerate(combos.iterrows()):
            ct, niche = row['celltype'], row['niche']

            for j, (dge, title) in enumerate([(sc_dge, 'Single-cell'), (proto_dge, 'Metacell')]):
                ax = axes[i, j]
                d = dge[(dge['celltype'] == ct) & (dge['niche'] == niche)].copy()

                if len(d) == 0:
                    ax.set_title(f'{title}: No data')
                    continue

                d['neg_log_p'] = -np.log10(d['pval'].clip(lower=1e-300))

                ax.scatter(d['logfoldchange'], d['neg_log_p'], s=2, alpha=0.4, c='gray')
                sig = d[(d['pval_adj'] < 0.05) & (abs(d['logfoldchange']) > 0.5)]
                ax.scatter(sig['logfoldchange'], sig['neg_log_p'], s=6, c='red', alpha=0.7)

                ax.set_xlabel('LogFC')
                ax.set_ylabel('-log10(p)')
                ax.set_title(f'{title}: {ct[:15]} / {niche[:20]}\n({len(sig)} sig)')
                ax.axhline(-np.log10(0.05), ls='--', c='blue', alpha=0.3)
                ax.axvline(0, ls='--', c='black', alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Compute scores
        from interpretable_ssl.evaluation.niche_recovery import jaccard, rbo
        score_rows = []
        for (c, n), p_grp in proto_dge.groupby(['celltype', 'niche']):
            s_grp = sc_dge[(sc_dge['celltype'] == c) & (sc_dge['niche'] == n)]
            if len(s_grp) < 5 or len(p_grp) < 3:
                continue
            p_genes = p_grp.nlargest(20, 'logfoldchange')['gene'].tolist()
            s_genes = s_grp.nlargest(20, 'logfoldchange')['gene'].tolist()
            merged = p_grp.merge(s_grp, on='gene', suffixes=('_proto', '_sc'))
            corr = merged['logfoldchange_proto'].corr(merged['logfoldchange_sc']) if len(merged) > 5 else np.nan
            score_rows.append({
                'celltype': c, 'niche': n,
                'jaccard': jaccard(p_genes, s_genes),
                'rbo': rbo(p_genes, s_genes),
                'logfc_corr': corr,
            })
        scores_df = pd.DataFrame(score_rows)

        print(f"\nScores: jaccard={scores_df['jaccard'].mean():.3f}, rbo={scores_df['rbo'].mean():.3f}, corr={scores_df['logfc_corr'].mean():.3f}")

        return {'proto_dge': proto_dge, 'sc_dge': sc_dge, 'scores': scores_df}

    def eval_baseline_clustering(self, name, niche_key=None, celltype_key=None):
        """Evaluate a baseline clustering method (spectral, seacells, etc.)

        Loads clustering from baseline folder, computes average gene expression
        per cluster, runs DGE, and saves results.

        Args:
            name: baseline name (e.g., 'spectral_K50', 'seacells_K50')
            niche_key: column for niche labels
            celltype_key: column for celltype labels

        Returns:
            dict with dge DataFrame and stats
        """
        import scanpy as sc

        # Paths
        baseline_dir = os.path.join(self.get_save_dir(), name)
        clusters_path = os.path.join(baseline_dir, 'clusters.npz')

        if not os.path.exists(clusters_path):
            print(f"Baseline not found: {clusters_path}")
            print(f"Run t.eval_spectral() or t.eval_seacells() first")
            return None

        # Load clustering
        data = np.load(clusters_path, allow_pickle=True)
        assignments = data['assignments']

        # Setup keys
        obs = self.train_ds.adata.obs
        if celltype_key is None:
            celltype_key = self.dataset.label_key
        if niche_key is None:
            niche_key = 'niches_2D' if 'niches_2D' in obs.columns else celltype_key

        print(f"Evaluating baseline '{name}': niche_key={niche_key}, celltype_key={celltype_key}")

        # Compute average gene expression per cluster
        X = self.train_ds.adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()

        n_clusters = len(np.unique(assignments))
        avg_expr = np.zeros((n_clusters, X.shape[1]))
        cluster_ids = np.unique(assignments)

        for i, c in enumerate(cluster_ids):
            mask = assignments == c
            avg_expr[i] = X[mask].mean(axis=0)

        # Create AnnData for clusters
        cluster_ad = sc.AnnData(avg_expr)
        cluster_ad.var_names = self.train_ds.adata.var_names

        # Label clusters by majority vote
        niche_labels = obs[niche_key].values
        ct_labels = obs[celltype_key].values

        cluster_niche = []
        cluster_ct = []
        for c in cluster_ids:
            mask = assignments == c
            # Majority vote
            vals, counts = np.unique(niche_labels[mask], return_counts=True)
            cluster_niche.append(vals[counts.argmax()])
            vals, counts = np.unique(ct_labels[mask], return_counts=True)
            cluster_ct.append(vals[counts.argmax()])

        cluster_ad.obs['niche'] = cluster_niche
        cluster_ad.obs['celltype'] = cluster_ct
        cluster_ad.obs['cluster_id'] = cluster_ids

        # Run DGE (same logic as prototype_dge)
        rows = []
        for ct in cluster_ad.obs['celltype'].unique():
            ct_ad = cluster_ad[cluster_ad.obs['celltype'] == ct].copy()
            if ct_ad.n_obs < 3:
                continue
            niches = ct_ad.obs['niche'].unique()
            if len(niches) < 2:
                continue

            for niche in niches:
                n_pos = (ct_ad.obs['niche'] == niche).sum()
                n_neg = (ct_ad.obs['niche'] != niche).sum()
                if n_pos < 1 or n_neg < 1:
                    continue

                ct_ad.obs['_group'] = np.where(ct_ad.obs['niche'] == niche, 'pos', 'rest')
                try:
                    sc.tl.rank_genes_groups(ct_ad, '_group', groups=['pos'], reference='rest',
                                            method='wilcoxon', use_raw=False)
                    dge = sc.get.rank_genes_groups_df(ct_ad, group='pos')
                    dge['celltype'] = ct
                    dge['niche'] = niche
                    dge['n_pos'] = n_pos
                    dge['n_neg'] = n_neg
                    rows.append(dge)
                except Exception as e:
                    print(f"DGE failed for {ct}/{niche}: {e}")

        if not rows:
            print("No DGE results")
            return None

        dge_df = pd.concat(rows, ignore_index=True)
        dge_df = dge_df.rename(columns={'names': 'gene', 'logfoldchanges': 'logfoldchange',
                                         'pvals': 'pval', 'pvals_adj': 'pval_adj', 'scores': 'score'})
        dge_df = dge_df[['celltype', 'niche', 'gene', 'logfoldchange', 'pval', 'pval_adj', 'score', 'n_pos', 'n_neg']]

        # Save
        dge_df.to_csv(os.path.join(baseline_dir, 'baseline_dge.csv'), index=False)
        np.save(os.path.join(baseline_dir, 'avg_expression.npy'), avg_expr)

        print(f"Saved to {baseline_dir}: baseline_dge.csv ({len(dge_df)} rows), avg_expression.npy")

        return {'dge': dge_df, 'avg_expr': avg_expr, 'cluster_ad': cluster_ad}

    def compare_all_methods(self, niche_key=None, celltype_key=None, name_map=None):
        """Compare DGE results across all methods: prototype, baselines, single-cell.

        Computes jaccard/rbo/logfc_corr for each method vs single-cell ground truth.

        Args:
            niche_key: column for niche labels
            celltype_key: column for celltype labels
            name_map: dict mapping folder names to display names, e.g.
                      {'scproto_ds-s28f_v31': 'Proto UMAP', 'spectral_K50': 'Spectral'}

        Returns:
            DataFrame with method, mean_jaccard, mean_rbo, mean_logfc_corr, logfc_std
        """
        import matplotlib.pyplot as plt
        from interpretable_ssl.evaluation.niche_recovery import jaccard, rbo

        obs = self.train_ds.adata.obs
        if celltype_key is None:
            celltype_key = self.dataset.label_key
        if niche_key is None:
            niche_key = 'niches_2D' if 'niches_2D' in obs.columns else celltype_key

        # Load single-cell ground truth
        sc_path = os.path.join(self.get_save_dir(), 'gt', 'singlecell_dge.csv')
        if not os.path.exists(sc_path):
            print("Run t.eval_singlecell() first")
            return None
        sc_dge = pd.read_csv(sc_path)

        # Find all methods in model dir
        methods = {}

        for entry in os.listdir(self.get_save_dir()):
            entry_dir = os.path.join(self.get_save_dir(), entry)
            if not os.path.isdir(entry_dir):
                continue

            # Check for prototype_dge.csv (trained models)
            proto_path = os.path.join(entry_dir, 'prototype_dge.csv')
            if os.path.exists(proto_path):
                methods[entry] = pd.read_csv(proto_path)
                continue

            # Check for baseline_dge.csv (baselines)
            baseline_path = os.path.join(entry_dir, 'baseline_dge.csv')
            if os.path.exists(baseline_path):
                methods[entry] = pd.read_csv(baseline_path)

        if not methods:
            print("No methods found. Run t.eval_prototypes() and t.eval_baseline_clustering() first")
            return None

        # Compute scores for each method
        results = []
        for method_name, method_dge in methods.items():
            scores = []
            for (ct, niche), m_grp in method_dge.groupby(['celltype', 'niche']):
                s_grp = sc_dge[(sc_dge['celltype'] == ct) & (sc_dge['niche'] == niche)]
                if len(s_grp) < 10 or len(m_grp) < 3:
                    continue

                m_genes = m_grp.nlargest(20, 'logfoldchange')['gene'].tolist()
                s_genes = s_grp.nlargest(20, 'logfoldchange')['gene'].tolist()
                merged = m_grp.merge(s_grp, on='gene', suffixes=('_method', '_sc'))
                corr = merged['logfoldchange_method'].corr(merged['logfoldchange_sc']) if len(merged) > 5 else np.nan

                scores.append({
                    'jaccard': jaccard(m_genes, s_genes),
                    'rbo': rbo(m_genes, s_genes),
                    'logfc_corr': corr,
                })

            if scores:
                scores_df = pd.DataFrame(scores)
                results.append({
                    'method': method_name,
                    'mean_jaccard': scores_df['jaccard'].mean(),
                    'mean_rbo': scores_df['rbo'].mean(),
                    'mean_logfc_corr': scores_df['logfc_corr'].mean(),
                    'logfc_std': method_dge['logfoldchange'].std(),
                    'n_comparisons': len(scores),
                })

        results_df = pd.DataFrame(results).sort_values('mean_jaccard', ascending=False)

        # Apply name mapping
        if name_map:
            results_df['display_name'] = results_df['method'].map(lambda x: name_map.get(x, x))
        else:
            results_df['display_name'] = results_df['method']

        # Plot comparison
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        x = range(len(results_df))
        names = results_df['display_name'].tolist()

        for ax, col, title in zip(axes, ['mean_jaccard', 'mean_rbo', 'mean_logfc_corr', 'logfc_std'],
                                   ['Jaccard ↑', 'RBO ↑', 'LogFC Corr ↑', 'LogFC Std ↑']):
            bars = ax.bar(x, results_df[col], alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=45, ha='right')
            ax.set_title(title)
            # Highlight best
            best_idx = results_df[col].idxmax() if '↑' in title else results_df[col].idxmin()
            best_pos = results_df.index.get_loc(best_idx)
            bars[best_pos].set_color('green')

        plt.tight_layout()
        plt.show()

        print("\nMethod Comparison (vs single-cell ground truth):")
        print(results_df.to_string(index=False))

        return results_df

    def find_discovered_genes(self, pval_thr=0.05, logfc_thr=0.5):
        """Find genes significant in metacell but not in single-cell.

        These are genes that metacells "discover" by reducing noise/sparsity.

        Args:
            pval_thr: adjusted p-value threshold for significance
            logfc_thr: absolute log fold change threshold

        Returns:
            DataFrame with discovered genes and their stats
        """
        proto_path = os.path.join(self.get_dump_path(), 'prototype_dge.csv')
        sc_path = os.path.join(self.get_save_dir(), 'gt', 'singlecell_dge.csv')

        if not os.path.exists(proto_path) or not os.path.exists(sc_path):
            print("Run t.eval_prototypes() and t.eval_singlecell() first")
            return None

        proto_dge = pd.read_csv(proto_path)
        sc_dge = pd.read_csv(sc_path)

        rows = []
        for (ct, niche), p_grp in proto_dge.groupby(['celltype', 'niche']):
            s_grp = sc_dge[(sc_dge['celltype'] == ct) & (sc_dge['niche'] == niche)]
            if len(s_grp) == 0:
                continue

            # Significant in prototype
            p_sig = p_grp[(p_grp['pval_adj'] < pval_thr) & (abs(p_grp['logfoldchange']) > logfc_thr)]

            # Significant genes in single-cell
            s_sig_genes = set(s_grp[(s_grp['pval_adj'] < pval_thr) & (abs(s_grp['logfoldchange']) > logfc_thr)]['gene'])

            # Metacell-only genes
            for _, gene_row in p_sig.iterrows():
                if gene_row['gene'] not in s_sig_genes:
                    sc_match = s_grp[s_grp['gene'] == gene_row['gene']]
                    rows.append({
                        'celltype': ct,
                        'niche': niche,
                        'gene': gene_row['gene'],
                        'proto_logfc': gene_row['logfoldchange'],
                        'proto_pval': gene_row['pval_adj'],
                        'sc_logfc': sc_match['logfoldchange'].values[0] if len(sc_match) > 0 else np.nan,
                        'sc_pval': sc_match['pval_adj'].values[0] if len(sc_match) > 0 else np.nan,
                    })

        df = pd.DataFrame(rows)

        if len(df) == 0:
            print("No metacell-discovered genes found")
            return df

        # Sort by proto_logfc
        df = df.sort_values('proto_logfc', ascending=False).reset_index(drop=True)

        # Save
        save_path = os.path.join(self.get_dump_path(), 'discovered_genes.csv')
        df.to_csv(save_path, index=False)

        print(f"Found {len(df)} metacell-discovered genes")
        print(f"Saved to {save_path}")
        print(f"\nTop 15 by logFC:")
        print(df.head(15).to_string(index=False))

        return df

    def plot_discovered_genes(self, top_n=20, figsize=(14, 10)):
        """Plot discovered genes: bar chart + volcano highlighting.

        Args:
            top_n: number of top genes to show in bar chart
            figsize: figure size

        Returns:
            Figure and discovered genes DataFrame
        """
        import matplotlib.pyplot as plt

        # Load discovered genes
        save_path = os.path.join(self.get_dump_path(), 'discovered_genes.csv')
        if not os.path.exists(save_path):
            print("Run t.find_discovered_genes() first")
            return None

        df = pd.read_csv(save_path)

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # --- Top-left: Top upregulated genes ---
        ax = axes[0, 0]
        top_up = df[df['proto_logfc'] > 0].nlargest(top_n, 'proto_logfc')
        colors = ['#2ecc71' if 'Tumor' in n else '#3498db' for n in top_up['niche']]
        bars = ax.barh(range(len(top_up)), top_up['proto_logfc'], color=colors, alpha=0.8)
        ax.set_yticks(range(len(top_up)))
        ax.set_yticklabels([f"{row['gene']} ({row['niche'][:15]})" for _, row in top_up.iterrows()], fontsize=8)
        ax.set_xlabel('Log Fold Change (metacell)')
        ax.set_title(f'Top {len(top_up)} Upregulated\n(green=Tumor surface, blue=Desmoplastic)')
        ax.invert_yaxis()

        # --- Top-right: Top downregulated genes ---
        ax = axes[0, 1]
        top_down = df[df['proto_logfc'] < 0].nsmallest(top_n, 'proto_logfc')
        colors = ['#e74c3c' if 'Desmo' in n else '#9b59b6' for n in top_down['niche']]
        bars = ax.barh(range(len(top_down)), top_down['proto_logfc'], color=colors, alpha=0.8)
        ax.set_yticks(range(len(top_down)))
        ax.set_yticklabels([f"{row['gene']} ({row['niche'][:15]})" for _, row in top_down.iterrows()], fontsize=8)
        ax.set_xlabel('Log Fold Change (metacell)')
        ax.set_title(f'Top {len(top_down)} Downregulated\n(red=Desmoplastic, purple=other)')
        ax.invert_yaxis()

        # --- Bottom-left: Metacell vs Single-cell logFC scatter ---
        ax = axes[1, 0]
        valid = df.dropna(subset=['sc_logfc'])
        ax.scatter(valid['sc_logfc'], valid['proto_logfc'], s=30, alpha=0.6, c='steelblue')

        # Label top genes
        for _, row in valid.nlargest(5, 'proto_logfc').iterrows():
            ax.annotate(row['gene'], (row['sc_logfc'], row['proto_logfc']), fontsize=7)
        for _, row in valid.nsmallest(5, 'proto_logfc').iterrows():
            ax.annotate(row['gene'], (row['sc_logfc'], row['proto_logfc']), fontsize=7)

        lim = max(abs(valid['sc_logfc']).max(), abs(valid['proto_logfc']).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3)
        ax.axhline(0, c='gray', lw=0.5)
        ax.axvline(0, c='gray', lw=0.5)
        ax.set_xlabel('Single-cell logFC (not significant)')
        ax.set_ylabel('Metacell logFC (significant)')
        ax.set_title('Metacell amplifies weak signals')

        # --- Bottom-right: P-value comparison ---
        ax = axes[1, 1]
        valid = df.dropna(subset=['sc_pval'])
        ax.scatter(-np.log10(valid['sc_pval'].clip(lower=1e-10)),
                   -np.log10(valid['proto_pval'].clip(lower=1e-10)),
                   s=30, alpha=0.6, c='coral')

        ax.axhline(-np.log10(0.05), c='red', ls='--', alpha=0.5, label='p=0.05')
        ax.axvline(-np.log10(0.05), c='blue', ls='--', alpha=0.5)
        ax.set_xlabel('Single-cell -log10(p)')
        ax.set_ylabel('Metacell -log10(p)')
        ax.set_title('Metacell finds significance\n(top-left = discovered)')
        ax.legend(fontsize=8)

        plt.tight_layout()
        plt.show()

        return fig, df

    def discovered_genes_table(self, top_n=30):
        """Generate a formatted table of discovered genes for presentation.

        Args:
            top_n: number of genes to include

        Returns:
            DataFrame formatted for presentation
        """
        save_path = os.path.join(self.get_dump_path(), 'discovered_genes.csv')
        if not os.path.exists(save_path):
            print("Run t.find_discovered_genes() first")
            return None

        df = pd.read_csv(save_path)

        # Select top up and down
        top_up = df[df['proto_logfc'] > 0].nlargest(top_n // 2, 'proto_logfc')
        top_down = df[df['proto_logfc'] < 0].nsmallest(top_n // 2, 'proto_logfc')
        selected = pd.concat([top_up, top_down])

        # Format for presentation
        table = selected[['gene', 'niche', 'proto_logfc', 'proto_pval', 'sc_logfc', 'sc_pval']].copy()
        table['proto_logfc'] = table['proto_logfc'].round(2)
        table['sc_logfc'] = table['sc_logfc'].round(2)
        table['proto_pval'] = table['proto_pval'].apply(lambda x: f"{x:.1e}")
        table['sc_pval'] = table['sc_pval'].apply(lambda x: f"{x:.1e}" if pd.notna(x) else "NA")
        table['direction'] = table['proto_logfc'].apply(lambda x: '↑' if x > 0 else '↓')

        table = table.rename(columns={
            'gene': 'Gene',
            'niche': 'Niche',
            'proto_logfc': 'Metacell LogFC',
            'proto_pval': 'Metacell p-val',
            'sc_logfc': 'Single-cell LogFC',
            'sc_pval': 'Single-cell p-val',
            'direction': '↑↓'
        })

        return table[['Gene', '↑↓', 'Niche', 'Metacell LogFC', 'Metacell p-val', 'Single-cell LogFC', 'Single-cell p-val']]

    def plot_gene_comparison(self, genes=None, top_n=10):
        """Plot side-by-side comparison of specific genes: metacell vs single-cell.

        Args:
            genes: list of gene names to plot. If None, uses top discovered genes.
            top_n: if genes is None, use top_n discovered genes

        Returns:
            Figure
        """
        import matplotlib.pyplot as plt

        save_path = os.path.join(self.get_dump_path(), 'discovered_genes.csv')
        if not os.path.exists(save_path):
            print("Run t.find_discovered_genes() first")
            return None

        df = pd.read_csv(save_path)

        if genes is None:
            # Get top by absolute logfc
            top_up = df[df['proto_logfc'] > 0].nlargest(top_n // 2, 'proto_logfc')
            top_down = df[df['proto_logfc'] < 0].nsmallest(top_n // 2, 'proto_logfc')
            selected = pd.concat([top_up, top_down])
        else:
            selected = df[df['gene'].isin(genes)]

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(selected))
        width = 0.35

        bars1 = ax.bar(x - width/2, selected['proto_logfc'], width, label='Metacell (significant)', color='#2ecc71', alpha=0.8)
        bars2 = ax.bar(x + width/2, selected['sc_logfc'], width, label='Single-cell (not sig)', color='#95a5a6', alpha=0.8)

        ax.set_ylabel('Log Fold Change')
        ax.set_title('Discovered Genes: Metacell vs Single-cell')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{row['gene']}\n({row['niche'][:10]})" for _, row in selected.iterrows()],
                          rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.axhline(0, c='black', lw=0.5)

        # Add significance markers
        for i, (_, row) in enumerate(selected.iterrows()):
            if row['proto_pval'] < 0.01:
                ax.text(i - width/2, row['proto_logfc'] + 0.05 * np.sign(row['proto_logfc']), '**', ha='center', fontsize=10)
            elif row['proto_pval'] < 0.05:
                ax.text(i - width/2, row['proto_logfc'] + 0.05 * np.sign(row['proto_logfc']), '*', ha='center', fontsize=10)

        plt.tight_layout()
        plt.show()

        return fig

    def plot_method_comparison_summary(self):
        """Create a summary figure comparing all methods for presentation.

        Returns:
            Figure
        """
        import matplotlib.pyplot as plt

        # Run compare_all_methods but capture the data
        result = self.compare_all_methods()
        if result is None:
            return None

        # Additional summary plot
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # Load discovered genes count
        disc_path = os.path.join(self.get_dump_path(), 'discovered_genes.csv')
        n_discovered = len(pd.read_csv(disc_path)) if os.path.exists(disc_path) else 0

        # Panel 1: Method comparison bar
        ax = axes[0]
        methods = result['method'].tolist()
        jaccard = result['mean_jaccard'].tolist()
        colors = ['#2ecc71' if 'proto' in m.lower() else '#3498db' for m in methods]
        ax.bar(methods, jaccard, color=colors, alpha=0.8)
        ax.set_ylabel('Jaccard Score')
        ax.set_title('Agreement with Ground Truth')
        ax.set_xticklabels(methods, rotation=45, ha='right')

        # Panel 2: Signal strength
        ax = axes[1]
        logfc_std = result['logfc_std'].tolist()
        ax.bar(methods, logfc_std, color=colors, alpha=0.8)
        ax.set_ylabel('LogFC Std')
        ax.set_title('Signal Strength (higher=cleaner)')
        ax.set_xticklabels(methods, rotation=45, ha='right')

        # Panel 3: Discovered genes summary
        ax = axes[2]
        ax.text(0.5, 0.7, f"{n_discovered}", fontsize=48, ha='center', va='center', transform=ax.transAxes, fontweight='bold', color='#2ecc71')
        ax.text(0.5, 0.35, "Genes Discovered", fontsize=14, ha='center', va='center', transform=ax.transAxes)
        ax.text(0.5, 0.15, "(significant in metacell,\nnot in single-cell)", fontsize=10, ha='center', va='center', transform=ax.transAxes, color='gray')
        ax.axis('off')

        plt.tight_layout()
        plt.show()

        return fig