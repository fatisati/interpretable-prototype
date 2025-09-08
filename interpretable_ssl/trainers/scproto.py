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
from interpretable_ssl.utils import log_time

from interpretable_ssl.evaluation.prototype_metrics import *
import torch
from collections import Counter, defaultdict
from interpretable_ssl.evaluation.cd4_marker import *
from interpretable_ssl.trainers.swav_utils import *
from sklearn.model_selection import train_test_split


logger = getLogger()


class SCProtoTrainer(AdoptiveTrainer):

    # @log_time('swav')
    def __init__(
        self, dataset=None, ref_query=None, parser=None, **kwargs
    ):

        super().__init__(dataset, ref_query, parser, **kwargs)
        self.nmb_prototypes = self.num_prototypes
        self.use_projector_out = False
        
        self.train_augmentation = self.augmentation_type
        self.queue = {}
        ds_cnt = self.ref.adata.obs[self.condition_key].nunique()
        self.ds_ids = range(ds_cnt)

    def setup(self):
        fix_random_seeds(self.seed)
        self.dump_path = self.get_dump_path()
        # self.create_dump_path()
        if self.wandb_sweep == 0:
            logger, self.training_stats = initialize_exp(
                self, "epoch", "loss", dump_params=self.wandb_sweep == 0
            )
        # self.init_scpoli()
        self.build_model()

        self.build_data()

        self.build_optimizer()
        self.init_mixed_precision()
        # self.load_checkpoint()

    def build_data(self):

        # train, val = self.split_train_test(self.ref)
        # self.train_adata, self.val_adata = train, val

        # why nmb_crops is a list? i used fisrt element but not change it in case needed in furure
        scpoli_encoder = self.model.scpoli_cvae
        common_dataset_kwargs = dict(
            n_augmentations=self.nmb_crops[0],
            augmentation_type=self.train_augmentation,
            k_neighbors=self.k_neighbors,
            longest_path=self.longest_path,
            dimensionality_reduction=self.dimensionality_reduction,
            n_components=self.n_components,
            supervised_ratio=self.supervised_ratio,
            use_bknn=self.use_bknn,
            condition_keys=[self.condition_key],
            knn_similarity=self.knn_similarity,
            save_dir="./graphs",
            mask_probability=self.mask_probability,
            default_dispersion=self.default_dispersion,
            spatial=self.spatial,
            n_clusters=self.num_prototypes,
            condition_encoders=scpoli_encoder.condition_encoders,
            conditions_combined_encoder=scpoli_encoder.conditions_combined_encoder,
            # cell_type_keys=[self.cell_type_key],
            # cell_type_encoder=model.cell_type_encoder,
        )

        train_ind, val_ind = train_test_split(
            range(len(self.ref)), test_size=0.1, random_state=42
        )
        train, val = self.ref._create_split_instance(
            train_ind
        ), self.ref._create_split_instance(val_ind)
        self.train_ds = MultiCropsDataset(train, **common_dataset_kwargs)
        self.test_ds = MultiCropsDataset(val, **common_dataset_kwargs)

        self.train_loader = self.get_data_laoder(self.train_ds)
        self.test_loader = self.get_data_laoder(self.test_ds)

        self.original_train_loader = self.train_loader
        if self.multi_layer_protos == 1:
            self.cell_type_ds = MultiCropsDataset(
                self.ref,
                self.nmb_crops[0],
                "cell_type",
                k_neighbors=self.k_neighbors,
                longest_path=self.longest_path,
                dimensionality_reduction=self.dimensionality_reduction,
                n_components=self.n_components,
                supervised_ratio=self.supervised_ratio,
                condition_keys=[self.condition_key],
                # cell_type_keys=[self.cell_type_key],
                condition_encoders=scpoli_encoder.condition_encoders,
                conditions_combined_encoder=scpoli_encoder.conditions_combined_encoder,
                # cell_type_encoder=model.cell_type_encoder,
            )
            self.cell_type_loader = self.get_data_laoder(self.cell_type_ds)
            # def get_train_loader(ld1, ld2):
            #     return zip(ld1, ld2)

            # self.train_loader = get_train_loader(self.original_train_loader, self.cell_type_loader)
        logger.info(f"Building data done with {len(self.train_ds)} samples loaded.")

    def get_data_laoder(self, ds):
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=scpoli_utils.custom_collate,
            shuffle=True,
        )

    def get_model(self):
        # if self.model_version == 1:
        # return model
        # else:
        kwargs = {
            "latent_dim": self.latent_dims,
            "nmb_prototypes": self.num_prototypes,
            "adata": self.ref.adata,
            "multi_layer_proto": self.multi_layer_protos,
            "np2": self.num_prototypes,
            "recon_loss": self.recon_loss,
            "batch_key": self.ref.batch_key,
            "l2norm": self.l2norm,
            "assignment_metric": self.assignment_metric,
        }

        if self.decodable_prototypes == 1:
            return SwAVDecodableProto(
                self.latent_dims,
                self.num_prototypes,
                self.ref.adata,
                self.multi_layer_protos,
                self.num_prototypes,
            )

        elif self.model_type == "gm":
            return scProtoGMVAE(use_rbf=self.use_rbf, **kwargs)
        else:
            return SwAVModel(**kwargs)

    def get_model_path(self):
        return os.path.join(self.get_dump_path(), self.get_checkpoint_file())

    def load_model(self):
        model = self.get_model()
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
            embeddings = self.encode_ref(self.model)
            self.model.init_prototypes_kmeans(embeddings, self.nmb_prototypes)

    def build_model(self):
        self.model = self.get_model()
        self.model = self.model.cuda()
        if self.debug == 0:
            self.init_prototypes()
        # logger.info(self.model)
        logger.info(f"Building model done. with prot init {self.prot_init}")

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
                            len(self.original_train_loader)
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
        cudnn.benchmark = True
        if epochs is None:
            epochs = self.pretraining_epochs

        for epoch in range(epochs):
            logger.info(f"============ Starting epoch {epoch}============")

            if (epoch % self.umap_checkpoint_freq == 0) and (self.wandb_sweep == 0):
                self.plot_umap(self.model, self.original_ref.adata, f"ref-e{epoch}")

            if (
                self.queue_length > 0
                and epoch >= self.epoch_queue_starts
                and len(self.queue) == 0
            ):
                for ds_id in self.ds_ids:
                    self.init_queue(ds_id)

            train_meters = self.train_epoch(epoch)
            test_meters = self.test_epoch()
            test_meters = {f"test_{key}": val for key, val in test_meters.items()}

            self.log_wandb_loss(train_meters | test_meters, epoch)
            self.save_checkpoint(epoch)
        return train_meters | test_meters

    def softmax_probs(self, s):
        return F.softmax(s / self.temperature)

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
            metrics["swav"]
            + metrics["cvae"] * self.cvae_loss_scaler
            + metrics["propagation"] * self.propagation_reg
            # + metrics["similarity"] * self.prot_emb_sim_reg
        )
        meters["loss"].update(loss.item(), bs)
        return loss, meters, assign_cnts

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
            if it % 50 == 0:
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

    def _process_batch(self, inputs, ds_id):
        bs = inputs["x"].size(0)
        inputs = self.dict_to_device(inputs)
        inputs = reshape_and_reorder_dict(inputs)
        z, _, logits, cvae_loss, (propagation, sim) = self.model(inputs)
        z = z.detach()
        assign_cnts = get_assign_cnts(logits)
        swav_loss, matched_pairs_ratio, q_matched, assignment_metrics = (
            self.compute_swav_loss(logits, z, bs, ds_id)
        )

        # entropy = self.peaky_softmax_loss(scores)
        # match, prob_ent, p_ent = self.calculate_pair_matching(scores, bs)

        return {
            "swav": swav_loss,
            "cvae": cvae_loss,
            "propagation": propagation,
            "similarity": sim,
            "p_matched": matched_pairs_ratio,
            "q_matched": q_matched,
            # "entropy": entropy,
            # "match": match,
            # "prob_ent": prob_ent,
            # "p_ent": p_ent,
        } | assignment_metrics, assign_cnts

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

    def compute_swav_loss(self, logits, z, bs, ds_id):

        loss, assignment_metrics_list, avg_matched_pairs, avg_q_matched = 0, [], 0, 0

        # each crop mean each augmentation, just caluclate q and loss for first crops_for_assign
        for view_idx, view_id in enumerate(self.views_for_assign):

            # outputs for 1 batch of data, [aug1s1, a1s2, a1s3, .., a1sb]
            view_logits = logits[bs * view_id : bs * (view_id + 1)]
            with torch.no_grad():
                sinkhorn_input = self.prepare_sinkhorn_input(
                    view_idx, z, view_id, bs, view_logits, ds_id
                )
                q = self.distributed_sinkhorn(sinkhorn_input)

            assignment_metrics_list.append(get_assignment_metrics(q, "q"))
            q = q[-bs:]
            # check how consitent q is with other augmentations [cross entropy]
            subloss = 0
            matched_pairs_ratio = 0
            q_matched = 0
            if self.hard_clustering == 1:
                q = self.hard_clusters(q)

            aug_view_ids = np.delete(np.arange(np.sum(self.nmb_crops)), view_id)
            for v in aug_view_ids:
                aug_logits = (
                    logits[bs * v : bs * (v + 1)] / self.temperature
                )  # logits for the v-th crop
                subloss -= torch.mean(
                    torch.sum(q * F.log_softmax(aug_logits, dim=1), dim=1)
                )
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
            p_matched,
            q_matched,
            avg_assign_metrics,
        )

    @torch.no_grad()
    def distributed_sinkhorn(self, out):
        Q = torch.exp(out / self.epsilon).t()
        B = Q.shape[1]
        K = Q.shape[0]

        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(self.sinkhorn_iterations):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B
        return Q.t()

    def get_model_prototypes(self, model):
        prototypes = model.get_prototypes()
        if self.use_projector_out:
            return model.projection_head(prototypes)
        else:
            return prototypes

    def finetune(self):
        print(f"-------finetuning: {self.fine_tuning_epochs}----------")
        # old_aug_type = self.augmentation_type

        # scpoli_query = scPoli.load_query_data(
        #     adata=self.ref.adata,
        #     reference_model=self.get_scpoli(),
        #     labeled_indices=[],
        # )
        # self.model.set_scpoli_model(scpoli_query.model)
        self.model = self.adapt_model(self.model, self.ref.adata)
        self.train_augmentation = "cell_type"
        self.build_data()
        self.build_optimizer()
        # self.setup()
        self.train(self.fine_tuning_epochs)
        # self.augmentation_type = old_aug_type

    def prepare_sinkhorn_input(self, view_idx, z, view_id, bs, view_logits, ds_id):
        output_logits = view_logits.detach()

        # time to use the queue
        if ds_id in self.queue:
            # check if the queue has any real data
            if self.use_the_queue or not torch.all(
                self.queue[ds_id][view_idx, -1, :] == 0
            ):
                self.use_the_queue = 1
                # 'get prototypes assignment scores for self.queue[i] (which contain some old embeddings)'
                queue_logits = self.model.proto_soft_assignments(
                    self.queue[ds_id][view_idx]
                )
                output_logits = torch.cat([queue_logits, view_logits])

            # fill the queue
            self.queue[ds_id][view_idx, bs:] = self.queue[ds_id][view_idx, :-bs].clone()
            self.queue[ds_id][view_idx, :bs] = z[view_id * bs : (view_id + 1) * bs]

        return output_logits


if __name__ == "__main__":
    swav = SCProtoTrainer()
    swav.setup()
    swav.run()
    swav.encode_ref()
