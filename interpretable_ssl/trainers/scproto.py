import argparse
import math
import os
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import shutil
import time
from logging import getLogger

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import apex
from apex.parallel.LARC import LARC

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
logger = getLogger()


def get_gpu_type_torch():
    if not torch.cuda.is_available():
        return "No GPU available"
    return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]


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

        # self.load_checkpoint()

    def build_data(self):

        # train, val = self.split_train_test(self.ref)
        # self.train_adata, self.val_adata = train, val

        # why nmb_crops is a list? i used fisrt element but not change it in case needed in furure
        scpoli_encoder = self.model.scpoli_cvae
        common_dataset_kwargs = dict(
            n_augmentations=self.nmb_views[0],
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
        if self.ft_epochs > 0:
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
        }

        if self.model_type == "gm" or self.model_type == "normal":
            return scProtoGMVAE(
                temperature=self.temperature,
                beta=self.beta,
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
                lr=self.base_lr, #1e-3,
                eps=0.01,
                weight_decay=self.wd #0.04,
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
        if self.debug or self.wandb_sweep == 1:
            print("not saving checkpoint", self.debug, self.wandb_sweep)
            return
        save_dict = {
            "epoch": epoch + 1,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.use_fp16:
            save_dict["amp"] = apex.amp.state_dict()

        checkpoint_file = self.get_checkpoint_file()
        torch.save(save_dict, os.path.join(self.dump_path, checkpoint_file))
        if epoch % self.checkpoint_freq == 0 or epoch == self.pretraining_epochs - 1:
            shutil.copyfile(
                os.path.join(self.dump_path, checkpoint_file),
                os.path.join(self.dump_path, f"ckp-{epoch}.pth"),
            )

    def log_wandb_loss(self, scores, epoch):
        log_dict = scores
        log_dict["epoch"] = epoch
        if not self.debug:
            wandb.log(log_dict)
        else:
            for k in log_dict:
                if k not in self.log_hist:
                    self.log_hist[k] = []
                self.log_hist[k].append(log_dict[k])
            print(log_dict)

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

            if (epoch % self.umap_checkpoint_freq == 10) and (self.wandb_sweep == 0):
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
            test_meters = self.test_epoch()
            test_meters = {f"test_{key}": val for key, val in test_meters.items()}

            scores = self.encode_adata(self.train_.adata, self.model, z_idx=2)
            cell_ids = self.train_.adata.obs_names
            train_meters["overal_compactness"], train_meters["overal_separation"], train_meters["celltype_purity"], train_meters["niche_purity"], train_meters["niche_purity3d"] = (
                self.calc_mc_quality(cell_ids, scores, "train")
            )
            ad = self.train_.adata
            ad.obs['mc'] = scores.argmax(1).detach().cpu().numpy()
            train_meters["spatial_compactness"] = spatial_compactness(ad, mc_key='mc').mean()

            self.log_wandb_loss(train_meters | test_meters, epoch)
            self.save_checkpoint(epoch)

        # if self.ft_epochs > 0:
        #     self.model = self.adapt_model(self.model, self.query.adata, self.ft_epochs)
        #     self.save_checkpoint(epoch + self.ft_epochs)

        return train_meters | test_meters

    def init_lambda_loss(self):

       
        return {
            key: getattr(self, f"lambda_{key}")
            for key in self.loss_keys
            if hasattr(self, f"lambda_{key}")
        }
        # meters = {
        #     "loss": AverageMeter(),
        # }
        # for it, inputs in enumerate(self.train_loader):
        #     bs = inputs["x"].size(0)
        #     inputs = {k: inputs[k].transpose(0, 1) for k in inputs.keys()}

        #     for ds_id in self.ds_ids:
        #         loss, meters, assign_cnts = self.calc_ds_loss(
        #             inputs, ds_id, meters, bs, self.train_ds.adata, "train"
        #         )
        # meters = {k: getattr(v, "avg", v) for k, v in meters.items()}
        # lambda_loss = {key: getattr(self, f"lambda_{key}") / meters[key] for key in }
        # for loss_key in self.loss_keys:
        #         setattr(self, f"lambda_{loss_key}", lambda_loss[loss_key])
        #         lambda_loss[loss_key] = 
        # print(lambda_loss)
        # if self.normalize_loss:
        #     return lambda_loss
        # else:
        #     return lambda_weight

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
        for it, inputs in enumerate(self.train_loader):
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
            if it % 5 == 0:
                lr_gn = self.get_lr_grad()

            self.optimizer.step()

            meters["batch_time"].update(time.time() - end)
            end = time.time()
            if it % 5 == 0:
                logger.info(
                    f"Epoch: [{epoch}][{it}] "
                    f"Time {meters['batch_time'].val:.3f} ({meters['batch_time'].avg:.3f}) "
                    f"Data {meters['data_time'].val:.3f} ({meters['data_time'].avg:.3f}) "
                    f"Loss {meters['loss'].val:.4f} ({meters['loss'].avg:.4f}) "
                    f"Lr {self.optimizer.param_groups[0]['lr']:.4f}"
                )
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

        # get proto collapse metric
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
                [metrics[k] * self.lambda_loss[k] for k in self.loss_keys]
            ).sum()
            meters["loss"].update(loss.item(), bs)
        else:  # when init lambda loss
            loss = -1
        return loss, meters, assign_cnts

    def _process_batch(self, inputs, ds_id, ad, split):
        bs = inputs["x"].size(0)
        inputs = self.dict_to_device(inputs)
        # inputs = reshape_and_reorder_dict(inputs)
        B, n_aug = inputs['x'].shape[:2]
        inputs = {k: t.permute(1, 0, *range(2, t.ndim)).reshape(B * n_aug, *t.shape[2:]) for k, t in inputs.items()}
        # manifold_keys = ['sigma', '', 'heterogeneity', 'mf_score']
        manifold_keys = self.train_ds.manifold.keys()
        manifold_scores = {k: inputs.pop(k, None) for k in manifold_keys}
        cell_idx = inputs.pop("cell_idx", None).cpu().numpy()
        cell_ids = [ad.obs.index[i] for i in cell_idx]
        sim = inputs.pop("sim", None)
        # label = inputs.pop(self.dataset.label_key)
        z, _, scores, recon, (proto_loss, commitment_loss), kl, kl_balance = self.model(
            inputs
        )
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
            proto_entropy
        ) = self.compute_swav_loss(scores, z, bs, ds_id, manifold_scores, None, sim=sim)
        assign_cnts = get_hard_assign_cnts(scores)
        max_active = min(scores.size(0), scores.size(1))
        compactness, separation, cp, np2d, np3d = self.calc_mc_quality(cell_ids, scores, split)
        return {
            "swav": swav_loss,
            "recon": recon,
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
            'proto_entropy': proto_entropy
        }, assign_cnts

    def calc_swapped_recon(self, z, scores, bs, inputs):
        loss = 0

        for view_id in self.views_for_assign:
            view_scores = scores[bs * view_id : bs * (view_id + 1)].detach()
            view_codes = view_scores.argmax(dim=1)

            # calc recon loss by closet proto to pos pairs
            subloss = 0
            aug_view_ids = np.delete(np.arange(np.sum(self.nmb_views)), view_id)
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
        self, scores, z, bs, ds_id, manifold_scores=None, resp=None, sim=None
    ):
        if sim is None or self.weighted_kl == 0:
            sim = torch.ones(z.size(0), device=z.device, dtype=z.dtype)

        loss, p_matched, q_matched, p_uncertainty, q_uncertainty, qproto_utilization, proto_entropy = (
            0,
            0,
            0,
            0,
            0,
            0,
            0
        )

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
            
            # per-sample entropy (H_p > 0)
            H_p = - (p * p.clamp_min(1e-8).log()).sum(dim=1).mean()
            p_uncertainty += H_p
            
            usage = p.mean(dim=0)     # shape [K]
            H_proto = - (usage * usage.clamp_min(1e-8).log()).sum()
            proto_entropy += H_proto

            if self.hard_clustering == 1:
                q = self.hard_clusters(q)

            # check how consitent q is with other augmentations [cross entropy]
            subloss = 0
            vp_matched, vq_matched = 0, 0
            aug_view_ids = np.delete(np.arange(np.sum(self.nmb_views)), view_id)
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
            Q /= (sum_of_rows + 1e-12)
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            # print(f'col sum (samples) {torch.sum(Q, dim=0, keepdim=True)}')
            Q /= (torch.sum(Q, dim=0, keepdim=True) + 1e-12)
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


if __name__ == "__main__":
    swav = SCProtoTrainer()
    swav.setup()
    swav.run()
    swav.encode_ref()
