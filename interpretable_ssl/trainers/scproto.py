import argparse
import math
import os
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
from interpretable_ssl.trainers.swav_utils import *

from interpretable_ssl.trainers.affinity import *
from interpretable_ssl.trainers.scpoli_helpers import *

logger = getLogger()


class SCProtoTrainer(AdoptiveTrainer):

    # @log_time('swav')
    def __init__(self, dataset=None, ref_query=None, parser=None, **kwargs):
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

    def setup(self):
        fix_random_seeds(self.seed)
        self.dump_path = self.get_dump_path()
        if self.wandb_sweep == 0 and self.debug == 0:
            logger, self.training_stats = initialize_exp(
                self, "epoch", "loss", dump_params=self.wandb_sweep == 0
            )
        self.build_model()
        self.build_data()
        if self.debug == 0:
            self.init_prototypes()
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
            drop_last=drop_last,
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
        logger.info(f"Building model done.")

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

        if self.model_type == "gm":
            return scProtoGMVAE(temperature=self.temperature, beta=self.beta, **kwargs)
        if self.model_type == "vqvae":
            return scProtoVQVAE(temperature=self.temperature, beta=self.beta, **kwargs)
        if self.model_type == "hybrid":
            return scProtoHybrid(temperature=self.temperature, beta=self.beta, **kwargs)
        else:
            return SwAVModel(**kwargs)

    def build_optimizer(self):
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.base_lr,
            momentum=0.9,
            weight_decay=self.wd,
        )
        self.optimizer = LARC(
            optimizer=self.optimizer, trust_coefficient=0.001, clip=False
        )

        warmup_lr_schedule = np.linspace(
            self.start_warmup,
            self.base_lr,
            len(self.train_loader) * self.warmup_epochs,
        )
        iters = np.arange(
            len(self.train_loader) * (self.pretraining_epochs - self.warmup_epochs)
        )
        cosine_lr_schedule = np.array(
            [
                self.final_lr
                + 0.5
                * (self.base_lr - self.final_lr)
                * (
                    1
                    + math.cos(
                        math.pi
                        * t
                        / (
                            len(self.train_loader)
                            * (self.pretraining_epochs - self.warmup_epochs)
                        )
                    )
                )
                for t in iters
            ]
        )
        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))

        logger.info("Building optimizer done.")

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
            print(log_dict)

    def train(self, epochs=None):
        self.create_dump_path()
        self.build_optimizer()
        cudnn.benchmark = True
        if epochs is None:
            epochs = self.pretraining_epochs

        for epoch in range(epochs):
            logger.info(f"============ Starting epoch {epoch}============")

            if (epoch % self.umap_checkpoint_freq == 10) and (self.wandb_sweep == 0):
                self.plot_umap(self.model, self.train_ds.adata, f"train-e{epoch}")

            if (
                self.queue_length > 0
                and epoch >= self.epoch_queue_starts
                and len(self.queue) == 0
            ):
                print(f'start using queue at epoch: {epoch}')
                for ds_id in self.ds_ids:
                    self.init_queue(ds_id)

            train_meters = self.train_epoch(epoch)
            test_meters = self.test_epoch()
            test_meters = {f"test_{key}": val for key, val in test_meters.items()}

            self.log_wandb_loss(train_meters | test_meters, epoch)
            self.save_checkpoint(epoch)

        # if self.ft_epochs > 0:
        #     self.model = self.adapt_model(self.model, self.query.adata, self.ft_epochs)
        #     self.save_checkpoint(epoch + self.ft_epochs)

        return train_meters | test_meters

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

        for it, inputs in enumerate(self.train_loader):
            meters["data_time"].update(time.time() - end)
            iteration = epoch * len(self.train_loader) + it
            self.update_learning_rate(iteration)

            if self.l2norm == 1:
                with torch.no_grad():
                    self.model.normalize_prototypes()

            bs = inputs["x"].size(0)
            inputs = {
                k: inputs[k].transpose(0, 1) for k in inputs.keys()
            }  # bring dataset in first to calc loss per dataset

            self.optimizer.zero_grad()

            for ds_id in self.ds_ids:
                loss, meters, assign_cnts = self.calc_ds_loss(inputs, ds_id, meters, bs)
                ds_assign_cnts[ds_id] += assign_cnts.cpu().numpy()
                loss.backward()

            self._handle_prototype_freezing(epoch)
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

        meters = {k: getattr(v, "avg", v) for k, v in meters.items()}
        meters["p_empty_protos_ratio"] = (
            np.mean(
                [(ds_assign_cnts[ds_id] == 0).sum().item() for ds_id in ds_assign_cnts]
            )
            / self.nmb_prototypes
        )
        eps = 1
        meters["p_ams"] = meters["p_matched"] / (meters["p_empty_protos_ratio"] + eps)
        return meters

    def calc_ds_loss(self, inputs, ds_id, meters, bs):
        ds_inputs = {k: inputs[k][ds_id] for k in inputs.keys()}
        metrics, assign_cnts = self._process_batch(ds_inputs, ds_id)
        # averaged = self._average_metrics(metrics)
        # Update meters
        for key in metrics:
            if key not in meters:
                meters[key] = AverageMeter()
            value = (
                metrics[key].item() if hasattr(metrics[key], "item") else metrics[key]
            )
            meters[key].update(value, bs)

        loss = (
            metrics["swav"] * self.lambda_swav
            + metrics["recon"] * self.lambda_recon
            + metrics["kl"] * self.lambda_kl
            + metrics["kl_balance"] * self.lambda_balance
            + metrics["propagation"] * self.propagation_reg
            + metrics["similarity"] * self.prot_emb_sim_reg
            + metrics["z_norm"] * self.lambda_l2
            + metrics["proto_norm"] * self.lambda_l2
            # + metrics["align_loss"] * self.lambda_align
        )
        meters["loss"].update(loss.item(), bs)
        return loss, meters, assign_cnts

    def _process_batch(self, inputs, ds_id):
        bs = inputs["x"].size(0)
        inputs = self.dict_to_device(inputs)
        inputs = reshape_and_reorder_dict(inputs)
        # manifold_keys = ['sigma', '', 'heterogeneity', 'mf_score']
        manifold_keys = self.train_ds.manifold.keys()
        manifold_scores = {k: inputs.pop(k, None) for k in manifold_keys}
        # label = inputs.pop(self.dataset.label_key)
        z, _, logits, recon, (propagation, sim), kl, kl_balance = self.model(inputs)
        z_norm = z.norm(dim=1).mean()
        proto_norm = self.model.get_prototypes().norm(dim=1).mean()
        # z, logits, cvae_loss, resp, propagation, sim = self.parse_model_output(outputs)
        z = z.detach()
        assign_cnts = get_assign_cnts(logits)
        swav_loss, align_loss, matched_pairs_ratio, q_matched, assignment_metrics = (
            self.compute_swav_loss(logits, z, bs, ds_id, manifold_scores, None)
        )
        assignment_metrics["p_proto_utilization"] = (
            assign_cnts != 0
        ).sum().item() / min(logits.size(0), logits.size(1))
        # entropy = self.peaky_softmax_loss(scores)
        # match, prob_ent, p_ent = self.calculate_pair_matching(scores, bs)

        return {
            "swav": swav_loss,
            "recon": recon,
            "kl": kl,
            "kl_balance": kl_balance,
            "propagation": propagation,
            "similarity": sim,
            "align_loss": align_loss,
            "p_matched": matched_pairs_ratio,
            "q_matched": q_matched,
            "z_mean": abs(z.mean().detach().item()),
            "z_norm": z_norm,
            "proto_norm": proto_norm,
        } | assignment_metrics, assign_cnts

    def parse_model_output(self, outputs):
        if self.model_type == "gm":
            z_swav, _, logits, cvae_loss, (propagation, sim), resp = outputs
            propagation, sim = 0, 0
        else:
            z_swav, _, logits, cvae_loss, (propagation, sim) = outputs
            resp = None

        return z_swav, logits, cvae_loss, resp, propagation, sim

    def compute_swav_loss(self, logits, z, bs, ds_id, manifold_scores=None, resp=None):

        loss, assignment_metrics_list, avg_matched_pairs, avg_q_matched = 0, [], 0, 0
        align_loss = 0

        # each crop mean each augmentation, just caluclate q and loss for first crops_for_assign
        for view_idx, view_id in enumerate(self.views_for_assign):
            with torch.no_grad():
                # outputs for 1 batch of data, [aug1s1, a1s2, a1s3, .., a1sb]
                view_logits = logits[bs * view_id : bs * (view_id + 1)].detach()
                
                # with torch.no_grad(): not nessecary because both funcion has no grad decorator
                sinkhorn_input = self.prepare_sinkhorn_input(
                    view_idx, z, view_id, bs, view_logits, ds_id
                )
                if self.cell_w_mode != "uniform":
                    cell_weights = manifold_scores[self.cell_w_mode][
                        bs * view_id : bs * (view_id + 1)
                    ]
                    
                else:
                    cell_weights = None  # uniform
                q = self.sinkhorn(sinkhorn_input, cell_weights)

                assignment_metrics_list.append(get_assignment_metrics(q, "q"))
                q = q[-bs:]

            # check how consitent q is with other augmentations [cross entropy]
            subloss = 0
            matched_pairs_ratio, q_matched = 0, 0
            if self.hard_clustering == 1:
                q = self.hard_clusters(q)

            aug_view_ids = np.delete(np.arange(np.sum(self.nmb_views)), view_id)
            for v in aug_view_ids:
                aug_logits = (
                    logits[bs * v : bs * (v + 1)] / self.temperature
                )  # logits for the v-th crop
                self.check_finit(aug_logits, "p")
                log_probs = F.log_softmax(aug_logits, dim=1)
                subloss -= torch.mean(torch.sum(q * log_probs, dim=1))

                matched_pairs_ratio += get_matched_pairs_ratio(view_logits, aug_logits)
                q_matched += get_matched_pairs_ratio(q, aug_logits)

            loss += subloss / len(aug_view_ids)
            avg_matched_pairs += matched_pairs_ratio / len(aug_view_ids)
            avg_q_matched += q_matched / len(aug_view_ids)

        avg = lambda metrics: {
            k: torch.tensor([m[k] for m in metrics]).float().mean().item()
            for k in metrics[0]
        }
        avg_assign_metrics = avg(assignment_metrics_list)
        q_matched = avg_q_matched / len(self.views_for_assign)
        p_matched = avg_matched_pairs / len(self.views_for_assign)
        eps = 1
        avg_assign_metrics["q_ams"] = (
            q_matched / (avg_assign_metrics["q_empty_protos_ratio"] + eps) * 100
        )

        return (
            loss / len(self.views_for_assign),
            align_loss / len(self.views_for_assign),
            p_matched,
            q_matched,
            avg_assign_metrics,
        )

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
                    _, meters, _ = self.calc_ds_loss(inputs, ds_id, meters, bs)
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
        Q = torch.exp(out / self.epsilon).t() # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * self.world_size # number of samples to assign
        K = Q.shape[0] # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(self.sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B # the colomns must sum to 1 so that Q is an assignment
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
