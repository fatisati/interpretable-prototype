import torch.nn as nn
import torch
import torch
from sklearn.cluster import KMeans
from scarches.models.scpoli import scPoli
from scarches.models.scpoli._utils import one_hot_encoder
from scarches.models.trvae.losses import nb

import torch.nn.functional as F
import itertools
from torch.distributions import NegativeBinomial
from interpretable_ssl.trainers.gmvae_utils import *
from collections import defaultdict
from interpretable_ssl.models.gmvae_loss import *

# encoder
# possibly a projection head
# prototype layer


def softplus_inverse(y):
    return torch.log(torch.exp(y) - 1.0)


class SwavBase(nn.Module):
    def __init__(
        self,
        scpoli_cvae,
        latent_dim,
        nmb_prototypes,  # , propagation_reg=0.5, prot_emb_sim_reg=0.5
        multi_layer_proto=False,
        np2=None,
        l2norm=1,
        assignment_metric="dotp",
        recon_v = 2,
        bs = None
    ):
        super().__init__()
        self.scpoli_cvae = scpoli_cvae
        self.prototypes = nn.Linear(latent_dim, nmb_prototypes, bias=False)
        if multi_layer_proto:
            print("initializing cell proto layer")
            self.cell_protos = nn.Linear(latent_dim, np2, bias=False)
        self.projection_head = None
        self.l2norm = l2norm == 1
        self.nmb_prototypes = nmb_prototypes
        self.assignment_metric = assignment_metric
        self.latent_dim = latent_dim
        self.recon_v = recon_v
        self.bs = bs
        # self.propagation_reg = propagation_reg
        # self.prot_emb_sim_reg = prot_emb_sim_reg

    def init_prototypes_kmeans(self, embeddings, nmb_prots):
        # Run KMeans on embeddings (convert to numpy for compatibility)
        kmeans = KMeans(n_clusters=nmb_prots)
        kmeans.fit(embeddings.cpu().numpy())

        # Get cluster centers and convert them back to a PyTorch tensor
        cluster_centers = torch.tensor(kmeans.cluster_centers_)
        self.set_prototypes(cluster_centers)

    def compute_cvae_loss(self, recon_loss, kl_loss, mmd_loss):
        calc_alpha_coeff = 0.5
        cvae_loss = recon_loss + calc_alpha_coeff * kl_loss + mmd_loss
        return cvae_loss

    def encoder_out(self, batch):
        x, recon_loss, kl_loss, mmd_loss = self.scpoli_cvae(**batch)
        cvae_loss = self.compute_cvae_loss(recon_loss, kl_loss, mmd_loss)
        return x, recon_loss + mmd_loss, kl_loss

    def forward(self, batch):
        x, recon_loss, kl_loss = self.encoder_out(batch)

        if self.projection_head is not None:
            x = self.projection_head(x)

        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)

        propagation_sim = self.propagation_sim_loss(x)

        # TO DO: recheck this with original scpoli
        # calc_alpha_coeff = 0.5
        # cvae_loss = recon_loss + calc_alpha_coeff * kl_loss + mmd_loss

        # return 2 x so it would be match the other model output
        if hasattr(self, "cell_protos"):
            return (
                x,
                x,
                (self.prototypes(x), self.cell_protos(x)),
                cvae_loss,
                propagation_sim,
            )
        else:
            proto_assignments = self.proto_soft_assignments(x)
            return x, x, proto_assignments, recon_loss, propagation_sim, kl_loss

    def proto_pos_cos(self, z):
        sim = self.proto_cos_sim(z)
        sim_pos = (sim + 1) / 2
        return sim_pos

    def proto_cos_sim(self, z):
        z = F.normalize(z, dim=1)
        prototypes = F.normalize(self.prototypes.weight, dim=1)
        sim = torch.matmul(z, prototypes.T)
        return sim

    def proto_neg_euclidean(self, z):
        prototypes = self.get_prototypes()
        dist = torch.cdist(z, prototypes)  # (B, K)
        return -dist

    def proto_soft_assignments(self, z):
        if self.assignment_metric == "dotp":
            return self.prototypes(z)
        elif self.assignment_metric == "pcos":
            return self.proto_pos_cos(z)
        elif self.assignment_metric == "cos":
            return self.proto_cos_sim(z)
        elif self.assignment_metric == "neuc":
            return self.proto_neg_euclidean(z)

    # move prototypes toward uncovered embeddings to improve coverage
    def propagation(self, z: torch.Tensor):
        # prototypes: [nmb_prototypes, latent_dim]
        protos = self.prototypes.weight  # each row = prototype

        # pairwise Euclidean distances: [n_samples, nmb_prototypes]
        dists = torch.cdist(z.detach(), protos, p=2)

        # for each sample → distance to closest prototype
        min_dists, _ = dists.min(dim=1)  # shape [n_samples]

        # TODO: try this maybe smoother gradient
        # threshold = torch.quantile(min_dists, 0.95)
        # loss = min_dists[min_dists >= threshold].mean()

        # return the maximum of these minima
        # min_dists.topk(5).values.mean()
        return min_dists.topk(k=int(0.1 * len(min_dists))).values.mean()

    # move embeddings (z) toward their nearest prototypes
    def z_commit(self, z: torch.Tensor):
        # detach prototypes → commitment-type loss
        protos = self.prototypes.weight.detach()

        # pairwise Euclidean distances [n_samples, n_prototypes]
        dists = torch.cdist(z, protos, p=2)

        # for each prototype, take closest embedding
        # shape nmb_proto
        # for each proto, we have the distance to the closest embedding
        min_dists, _ = dists.min(dim=0)

        # focus on underused prototypes: min_dists.topk(k=int(0.1 * len(min_dists)))
        # for those proto which this distance is high, move samples to the prototypes - move samples to unused prototypes
        return min_dists.topk(k=int(0.1 * len(min_dists))).values.mean()
        # return min_dists.mean()

    def propagation_sim_loss(self, z):
        return self.propagation(z), self.z_commit(z)

    def encode(self, batch):
        out = self.forward(1, batch)
        return out[:3]

    def get_prototypes(self):
        return self.prototypes.weight

    def normalize_prototypes(self):
        w = self.get_prototypes().data.clone()
        w = nn.functional.normalize(w, dim=1, p=2)
        self.set_prototypes(w)

    def set_prototypes(self, w):
        with torch.no_grad():
            self.prototypes.weight.copy_(w)

    def prototypes_avg_distance(self):
        """
        Calculate the average of the average distances for each tensor in a (p, d) tensor.

        Args:
            tensor (torch.Tensor): Input tensor of shape (p, d).

        Returns:
            float: The average of the average distances for all p tensors.
        """
        tensor = self.get_prototypes()
        # Calculate pairwise distances using broadcasting
        pairwise_diff = tensor.unsqueeze(1) - tensor.unsqueeze(0)  # Shape: (p, p, d)
        pairwise_distances = torch.norm(pairwise_diff, dim=2)  # Shape: (p, p)

        # Average distance for each tensor (exclude self-distance by setting diagonal to 0)
        pairwise_distances.fill_diagonal_(0)
        avg_distances_per_tensor = pairwise_distances.sum(dim=1) / (tensor.shape[0] - 1)

        # Average of these distances
        overall_avg_distance = avg_distances_per_tensor.mean().item()

        return overall_avg_distance

    def get_all_batch_embeddings(self, device="cuda"):
        """
        Generate all possible batch embeddings by iterating through every combination of indices.
        Returns:
            Tensor of shape (num_combinations, embedding_dim), where embedding_dim is the sum of embedding dimensions.
        """
        # Generate all possible indices for each embedding layer
        all_indices = [
            torch.arange(emb.num_embeddings, device=device)
            for emb in self.scpoli_cvae.embeddings
        ]
        # Create all possible combinations of indices
        combinations = list(itertools.product(*all_indices))

        # Generate embeddings for each combination
        embeddings_list = []
        for combination in combinations:
            # Pass each index through its respective embedding layer
            embedding = torch.cat(
                [
                    self.scpoli_cvae.embeddings[i](torch.tensor([index], device=device))
                    for i, index in enumerate(combination)
                ],
                dim=-1,
            )
            embeddings_list.append(embedding)

        # Stack all embeddings into a single tensor
        return torch.vstack(embeddings_list)

    def freeze_batch_embedding(self):
        for name, p in self.named_parameters():
            if "scpoli_cvae.embeddings" in name:
                p.requires_grad = False
                print(f"Froze: {name}")


# TODO: refactor input params with swav base
class SwAVModel(SwavBase):
    def __init__(
        self,
        latent_dim,
        nmb_prototypes,
        adata,
        multi_layer_proto=False,
        np2=None,
        recon_loss="nb",
        batch_key="study",
        l2norm=1,
        assignment_metric="dot-product",
        **kwargs
    ):  # , propagation_reg=0.5, prot_emb_sim_reg=0.5
        # self.cell_type_key = "cell_type"
        self.condition_key = batch_key
        self.scpoli_wrapper = self.init_scpoli(adata, latent_dim, recon_loss)
        self.recon_loss = recon_loss
        super().__init__(
            self.scpoli_wrapper.model,
            latent_dim,
            nmb_prototypes,
            multi_layer_proto,
            np2,
            l2norm,
            assignment_metric,
            **kwargs
        )  # , propagation_reg, prot_emb_sim_reg

    def init_scpoli(self, adata, latent_dim, recon_loss="nb"):
        return scPoli(
            adata=adata,
            condition_keys=self.condition_key,
            # cell_type_keys=self.cell_type_key,
            latent_dim=latent_dim,
            recon_loss=recon_loss,
        )

    def attach_scpoli(self, scpoli_wrapper):
        self.scpoli_wrapper = scpoli_wrapper
        self.scpoli_cvae = scpoli_wrapper.model


class scProtoGMVAE(SwAVModel):
    def __init__(self, temperature, beta, recon_version, kl_type="gm", **kwargs):
        # kwargs["recon_loss"] = "nb"
        super().__init__(**kwargs)
        # self.log_sigma2_p = torch.nn.Parameter(torch.tensor(-2.0))
        self.BETA_EPS = 1e-8
        self.register_buffer("log_sigma2_p", torch.as_tensor(-0.0))
        self.register_buffer(
            "proto_priors",
            torch.full((self.nmb_prototypes,), 1.0 / self.nmb_prototypes),
        )
        self.beta = beta

        self.temperature = temperature
        self.gm_vparam = softplus_inverse(
            torch.ones(self.nmb_prototypes, self.latent_dim)
        )
        self.gm_vparam = self.gm_vparam.to(self.get_prototypes().device)
        self.kl_type = kl_type
        self.recon_version = recon_version
    def forward(self, bs, batch):
        z, recon, kl, resp, kl_balance, proto_recon = self.calc_z_and_cvae_loss(bs, **batch)
        propagation_sim = self.propagation_sim_loss(z)
        return (
            z,
            z,
            self.proto_soft_assignments(z),
            recon,
            proto_recon,
            propagation_sim,
            kl,
            kl_balance,
        )

    def calc_z_and_cvae_loss(
        self,
        bs,
        x=None,
        batch=None,
        combined_batch=None,
        sizefactor=None,
        celltypes=None,
        labeled=None,
    ):
        batch_embeddings = torch.hstack(
            [self.scpoli_cvae.embeddings[i](batch[:, i]) for i in range(batch.shape[1])]
        )
        if self.recon_loss == "nb":
            model_input = torch.log(1 + x)
        else:
            model_input = x
        z_mu, z_logvar = self.scpoli_cvae.encoder(model_input, batch_embeddings)
        if self.l2norm:
            z_mu = nn.functional.normalize(z_mu, dim=1, p=2)

        
        scores = torch.softmax(
            self.proto_soft_assignments(z_mu[:bs]) / self.temperature,
            dim=1
        )#.detach()  # (B, K) v28

        protos = self.get_prototypes().detach()  # (K, D) detach is in version 28
        proto_vec = protos.unsqueeze(0).expand(bs, -1, -1)
        batch_vec = batch[:bs].unsqueeze(1).expand(bs, protos.size(0), -1)

        recon_x = self.decode(
            proto_vec.reshape(bs * protos.size(0), -1),
            batch_vec.reshape(bs * protos.size(0), -1),
        ).view(bs, protos.size(0), -1)  # (B, K, G)

        if self.recon_version == 26:
            mse = (recon_x - x[:bs].unsqueeze(1)).pow(2).sum(dim=-1)
            proto_recon = (scores * mse).sum(dim=1).mean()
        else:
            recon_x = (scores.detach().unsqueeze(-1) * recon_x).sum(dim=1)
            proto_recon = torch.nn.functional.mse_loss(
                recon_x, x[:bs], reduction="none"
            ).sum(dim=-1).mean()
            
        z = self.scpoli_cvae.sampling(z_mu, z_logvar)
        recon = self.calc_recon(
            z, batch, x, bs, sizefactor=sizefactor, combined_batch=combined_batch
        )

        gm_mu = self.get_gm_mu()
        gm_vparam = self.gm_vparam.to(gm_mu.device)
        # TODO: change tempreture?
        resp = responsibilities(z_mu, gm_mu, gm_vparam, self.temperature)
        z_var = torch.exp(z_logvar)
        z_vparam = softplus_inverse(z_var)

        if self.kl_type == "gm":
            kl, kl_dict = gm_kl(z_mu[:bs], z_vparam[:bs], gm_mu, gm_vparam, resp[:bs])
        else:
            kl = (
                torch.distributions.kl_divergence(
                    torch.distributions.Normal(z_mu[:bs], torch.sqrt(z_var[:bs])),
                    torch.distributions.Normal(
                        torch.zeros_like(z_mu[:bs]), torch.ones_like(z_var[:bs])
                    ),
                )
                .sum(dim=1)
                .mean()
            )

        # ---------- total ----------
        # loss = recon + self.beta * kl
        return z_mu, recon, kl, resp, kl, proto_recon

    def calc_recon(self, z, batch, x, bs, **kwargs):
        if self.recon_v > 1:
            z = z[:bs]
            batch = batch[:bs]
            x = x[:bs]
        if self.recon_loss == "nb":
            return self.nb_recon(z, batch, x=x, **kwargs)
        else:  # mse
            return self.mse_recon(z, batch, x)

    def mse_recon(self, z, batch, x, reduction="all", q=0.25):
        recon_x = self.decode(z, batch)
        mse_loss = torch.nn.functional.mse_loss(recon_x, x, reduction="none")
        per_cell_loss = mse_loss.sum(dim=-1)  # (N,)

        if reduction == "lowerq":
            thr = torch.quantile(per_cell_loss, q)
            recon_loss = per_cell_loss[per_cell_loss <= thr].mean()
        else:
            recon_loss = per_cell_loss.mean()

        return recon_loss


    # same as scpoli nb loss
    def nb_recon(self, z, batch, sizefactor, combined_batch, x):
        dec_mean = self.nb_decode(z, batch, sizefactor)
        dispersion = F.linear(
            one_hot_encoder(combined_batch, self.scpoli_cvae.n_conditions_combined),
            self.scpoli_cvae.theta,
        )
        dispersion = torch.exp(dispersion)
        recon_loss = -nb(x=x, mu=dec_mean, theta=dispersion).sum(dim=-1).mean()
        return recon_loss

    def nb_decode(self, z, batch, sizefactor):
        dec_mean_gamma = self.decode(z, batch)
        size_factor_view = sizefactor.unsqueeze(1).expand(
            dec_mean_gamma.size(0), dec_mean_gamma.size(1)
        )
        dec_mean = dec_mean_gamma * size_factor_view
        return dec_mean

    def decode(self, z, batch):
        batch_embeddings = torch.hstack(
            [self.scpoli_cvae.embeddings[i](batch[:, i]) for i in range(batch.shape[1])]
        )
        decoder_out = self.scpoli_cvae.decoder(z, batch_embeddings)
        return decoder_out[0]

    def proto_soft_assignments(self, z):
        if self.assignment_metric == "dotp":
            return self.prototypes(z)
        elif self.assignment_metric == "ddotp":
            return F.linear(z, self.prototypes.weight.detach())
        elif self.assignment_metric == "dneuc":
            protos = self.get_prototypes()
            return -torch.cdist(z, protos.detach(), p=2)
        elif self.assignment_metric == "neuc":
            protos = self.get_prototypes()
            return -torch.cdist(z, protos, p=2)
        elif self.assignment_metric == "sneuc":  # stable neuc
            protos = self.get_prototypes()  # (K, d)
            d2 = torch.cdist(z, protos, p=2) ** 2  # ||z - c||^2   (B, K)
            s = -d2  # negative distance
            s = s - s.max(dim=1, keepdim=True)[0]  # row-wise stabilization
            s = s.clamp(min=-75)
            return s
        elif self.assignment_metric == "student":  # Student-t kernel (like t-SNE/UMAP)
            protos = self.get_prototypes()  # (K, d)
            d2 = torch.cdist(z, protos, p=2) ** 2  # ||z - c||^2   (B, K)
            # Student-t with df=1: 1 / (1 + d²)
            # Return log for numerical stability with softmax
            s = -torch.log1p(d2)  # log(1 / (1 + d²)) = -log(1 + d²)
            return s
        elif self.assignment_metric == "nneuc":
            protos = self.get_prototypes()
            d2 = torch.cdist(z, protos, p=2).pow(2)
            # z: (B, D), protos: (K, D)
            if not hasattr(self, "cell_scale"):
                self.cell_scale = (
                    d2.median(dim=1, keepdim=True).values
                    .detach()
                    .clamp_min(1e-8)
                )
            # Scale by median distance for stability
            s = -d2 / self.cell_scale
            return s
        else:  # cos
            return F.cosine_similarity(
                z.unsqueeze(1), self.prototypes.weight.unsqueeze(0), dim=-1
            )

    def get_gm_mu(self):
        return self.prototypes.weight

    def get_gm_vparam(self):
        gm_mu = self.get_gm_mu()
        return self.gm_vparam.to(gm_mu.device)

    def coverage_loss(self, z, top_proto_ratio=0.1, top_sample_ratio=0.1):
        proto_cover = self.move_prototypes(z)
        sample_cover = self.move_samples(z)
        return proto_cover + sample_cover

    def select_uncovered(self, resp, dists, top_proto_ratio=0.1, top_sample_ratio=0.1):
        # protos = self.get_prototypes()
        # resp = torch.softmax(resp_logits / self.temperature, dim=1)
        usage = resp.sum(dim=0)
        num_proto = int(self.nmb_prototypes * top_proto_ratio)
        _, proto_idx = torch.topk(usage, num_proto, largest=False)

        # dists = torch.cdist(z, protos, p=2)
        min_dists, _ = dists.min(dim=1)
        num_samples = int(resp.size(0) * top_sample_ratio)
        _, sample_idx = torch.topk(min_dists, num_samples, largest=True)
        return proto_idx, sample_idx

    def move_prototypes(self, z, *args):
        k = int(0.1 * self.nmb_prototypes)
        return self.soft_align(z.detach(), self.get_prototypes(), 1)

    def move_samples(self, z, *args):
        k = int(0.1 * z.size(0))
        return self.soft_align(z, self.get_prototypes().detach(), 0)

    def soft_align(self, z, protos, dim):
        resp_logits = responsibilities(
            z, protos, self.get_gm_vparam(), return_logits=True
        )
        weights = torch.softmax(resp_logits / 0.05, dim=1)
        dist = torch.cdist(z, protos, p=2)
        w_dist = (weights * dist).sum(dim=dim)

        threshold = torch.quantile(w_dist, 0.9)
        mask = (w_dist > threshold).float()
        loss = (w_dist * mask).sum() / mask.sum().clamp_min(1.0)
        return loss


class scProtoVQVAE(scProtoGMVAE):
    def __init__(self, temperature, beta, recon_update_target="encoder", **kwargs):
        super().__init__(temperature, beta, **kwargs)
        self.recon_update_target = recon_update_target

    def forward(self, inputs):
        # recon loss, proto loss, commitment loss
        z, recon_loss, proto_loss, commit_loss, perplexity = self.calc_z_and_cvae_loss(
            **inputs
        )
        # propagation_sim = self.propagation_sim_loss(z)
        return (
            z,
            z,
            self.proto_soft_assignments(z),
            recon_loss,
            (proto_loss, commit_loss),
            perplexity,
            0,
        )

    def calc_z_and_cvae_loss(
        self,
        x=None,
        batch=None,
        combined_batch=None,
        sizefactor=None,
        celltypes=None,
        labeled=None,
    ):
        batch_embeddings = torch.hstack(
            [self.scpoli_cvae.embeddings[i](batch[:, i]) for i in range(batch.shape[1])]
        )
        x_log = torch.log(1 + x)
        z_mu, z_logvar = self.scpoli_cvae.encoder(x_log, batch_embeddings)
        if self.l2norm:
            z_mu = nn.functional.normalize(z_mu, dim=1, p=2)

        proto_assign = torch.cdist(z_mu, self.get_prototypes())  # [B, K]

        proto_ind = proto_assign.argmin(dim=1)  # [B]

        recon_loss, proto_loss, commit_loss = self.quantized_recon_step(
            z_mu, proto_ind, batch, x, sizefactor, combined_batch
        )

        # Compute usage histogram over prototypes
        usage = torch.bincount(
            proto_ind, minlength=self.get_prototypes().shape[0]
        ).float()
        usage = usage / usage.sum()  # normalize to probabilities p_k

        # Compute perplexity = exp(entropy)
        perplexity = torch.exp(-torch.sum(usage * torch.log(usage + 1e-10)))

        return z_mu, recon_loss, proto_loss, commit_loss, perplexity

    def quantized_recon_step(
        self, z, proto_ind, batch, x, sizefactor, combined_batch, **kwargs
    ):
        proto = self.get_prototypes()[proto_ind]  # [B, D]
        zq = z + (proto - z).detach()  # straight-through estimator

        recon_loss = self.calc_recon(
            zq, batch, x, sizefactor=sizefactor, combined_batch=combined_batch
        )
        # if proto_assign is not None:
        #     proto_loss = torch.mean(proto_assign.unsqueeze(-1) * (proto.unsqueeze(0) - z.detach().unsqueeze(1))**2)
        # else:
        proto_loss = torch.mean((proto - z.detach()) ** 2)  # codebook update
        commit_loss = torch.mean((z - proto.detach()) ** 2)  # encoder commitment
        return recon_loss, proto_loss, commit_loss


class scProtoHybrid(scProtoVQVAE):
    def __init__(self, temperature, beta, **kwargs):
        super().__init__(temperature, beta, **kwargs)
        hidden_dim = self.latent_dim * 4
        self.swav_projector = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

    def forward(self, batch):
        # recon loss, proto loss, commitment loss
        z, recon_loss, proto_loss, commit_loss, perplexity = self.calc_z_and_cvae_loss(
            **batch
        )
        # propagation_sim = self.propagation_sim_loss(z)
        logits, z_swav = self.proto_soft_assignments(z, True)
        return z_swav, z, logits, recon_loss, (proto_loss, commit_loss), perplexity, 0

    def proto_soft_assignments(self, z, return_z=False):
        z_swav = self.swav_projector(z)
        z_swav = nn.functional.normalize(z_swav, dim=1, p=2)

        proto_swav = self.swav_projector(self.get_prototypes())
        proto_swav = nn.functional.normalize(proto_swav, dim=1, p=2)
        proto_swav = proto_swav.detach()
        if return_z:
            return z_swav @ proto_swav.T, z_swav
        else:
            return z_swav @ proto_swav.T
