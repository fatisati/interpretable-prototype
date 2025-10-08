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
        assignment_metric="dot-product",
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
        return x, cvae_loss

    def forward(self, batch):
        x, cvae_loss = self.encoder_out(batch)

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
            return x, x, proto_assignments, cvae_loss, propagation_sim

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
        min_dists, _ = dists.min(dim=0)
        # encourage all prototypes to have at least one nearby embedding
        # focus on underused prototypes: min_dists.topk(k=int(0.1 * len(min_dists)))
        return min_dists.topk(k=int(0.1 * len(min_dists))).values.mean()
        # return min_dists.mean()

    def propagation_sim_loss(self, z):
        return self.propagation(z), self.z_commit(z)

    def encode(self, batch):
        encoder_out, x, x_mapped, _, _ = self.forward(batch)
        return encoder_out, x, x_mapped

    def get_prototypes(self):
        return self.prototypes.weight

    def normalize_prototypes(self):
        w = self.get_prototypes().data.clone()
        w = nn.functional.normalize(w, dim=1, p=2)
        self.set_prototypes(w)
        # if hasattr(self, "cell_protos"):
        #     wc = self.cell_protos.weight.data
        #     wc = nn.functional.normalize(wc, dim=1, p=2)
        #     with torch.no_grad():
        #         self.cell_protos.weight.copy_(wc)

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

    def reconstruct_nb(self, decoder_outputs, mean_size_factor, batch):
        """
        Reconstruct gene expression data using a default size factor and batch.

        Args:
            decoder_outputs (tuple): Outputs from the decoder (dec_mean_gamma, y1).
            mean_size_factor (float): Mean size factor to use for all cells.
            batch (int): Batch/condition index to use for all cells.
            model (torch.nn.Module): The model containing `theta` and other parameters.

        Returns:
            torch.Tensor: Reconstructed gene expression data.
        """
        # Unpack decoder outputs
        dec_mean_gamma, _ = decoder_outputs

        # Repeat mean_size_factor to match batch size
        size_factors = torch.full(
            (dec_mean_gamma.size(0),), mean_size_factor, device=dec_mean_gamma.device
        )

        # Expand size factors to match decoder output dimensions
        size_factor_view = size_factors.unsqueeze(1).expand(
            dec_mean_gamma.size(0), dec_mean_gamma.size(1)
        )

        # Compute the scaled mean (dec_mean)
        dec_mean = dec_mean_gamma * size_factor_view

        # Repeat batch index to match batch size
        batch_indices = torch.full(
            (dec_mean_gamma.size(0),),
            batch,
            dtype=torch.long,
            device=dec_mean_gamma.device,
        )

        # Compute the dispersion (theta)
        one_hot_batches = one_hot_encoder(
            batch_indices, self.scpoli_cvae.n_conditions_combined
        )
        dispersion = F.linear(one_hot_batches, self.scpoli_cvae.theta)
        dispersion = torch.exp(dispersion)  # Ensure positivity

        # Define the Negative Binomial distribution
        probs = dispersion / (dispersion + dec_mean)
        nb_dist = NegativeBinomial(total_count=dispersion, probs=probs)

        # Sample reconstructed gene expression data
        reconstructed_data = nb_dist.sample()

        return reconstructed_data

    def reconstruct_mse(self, outputs):
        recon_x, y1 = outputs
        reconstructed_input = torch.exp(recon_x) - 1
        return reconstructed_input

    def decode(
        self,
        input_tensor,
        recon_loss="nb",
        use_avg_batch_embedding=False,
        use_batch=None,
    ):
        # Get all possible embeddings
        batch_embeddings = (
            self.get_all_batch_embeddings()
        )  # Shape: (num_combinations, embedding_dim)

        if use_avg_batch_embedding:
            # Average all embeddings
            avg_embedding = batch_embeddings.mean(dim=0)
            batch_embeddings = [avg_embedding]

        if use_batch is not None:
            batch_embeddings = [batch_embeddings[use_batch]]
        # Decode input tensor with each embedding and average results
        decoded_results = []
        for i, batch_embedding in enumerate(batch_embeddings):
            # Repeat the embedding for the batch size
            batch_embedding_repeated = batch_embedding.expand(input_tensor.size(0), -1)
            # Decode using the input tensor and the batch embedding
            output = self.scpoli_cvae.decoder(
                input_tensor, batch_embedding_repeated
            )  # Define decoder logic
            if recon_loss == "nb":
                size_factor = 520.0436341421945
                decoded = self.reconstruct_nb(output, size_factor, i)
            else:
                decoded = self.reconstruct_mse(output)

            decoded_results.append(decoded)

        # Stack all decoded results and average along the "embedding" dimension
        return torch.stack(decoded_results, dim=0).mean(dim=0)

    def decode_proto(
        self, recon_loss="nb", use_avg_batch_embedding=False, use_batch=None
    ):
        print("new decode")
        """
        Decode the input tensor with all possible batch embeddings, then average the results.
        Args:
            input_tensor (torch.Tensor): Input tensor to decode, shape (batch_size, input_dim).
        Returns:
            Averaged decoded tensor of shape (batch_size, output_dim).
        """
        # Move input tensor to GPU
        input_tensor = self.get_prototypes()
        return self.decode(input_tensor, recon_loss, use_avg_batch_embedding, use_batch)


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
    ):  # , propagation_reg=0.5, prot_emb_sim_reg=0.5
        # self.cell_type_key = "cell_type"
        self.condition_key = batch_key
        self.scpoli_wrapper = self.init_scpoli(adata, latent_dim, recon_loss)
        super().__init__(
            self.scpoli_wrapper.model,
            latent_dim,
            nmb_prototypes,
            multi_layer_proto,
            np2,
            l2norm,
            assignment_metric,
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
    def __init__(self, temperature, beta, **kwargs):
        kwargs["recon_loss"] = "nb"
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

    # same as scpoli nb loss
    def cacl_recon_loss(self, z, batch, sizefactor, combined_batch, x):
        dec_mean = self.decode(z, batch, sizefactor)
        dispersion = F.linear(
            one_hot_encoder(combined_batch, self.scpoli_cvae.n_conditions_combined),
            self.scpoli_cvae.theta,
        )
        dispersion = torch.exp(dispersion)
        recon_loss = -nb(x=x, mu=dec_mean, theta=dispersion).sum(dim=-1).mean()
        return recon_loss

    def decode(self, z, batch, sizefactor):
        batch_embeddings = torch.hstack(
            [self.scpoli_cvae.embeddings[i](batch[:, i]) for i in range(batch.shape[1])]
        )
        dec_mean_gamma, _ = self.scpoli_cvae.decoder(z, batch_embeddings)
        size_factor_view = sizefactor.unsqueeze(1).expand(
            dec_mean_gamma.size(0), dec_mean_gamma.size(1)
        )
        dec_mean = dec_mean_gamma * size_factor_view
        return dec_mean

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

        z = self.scpoli_cvae.sampling(z_mu, z_logvar)
        recon = self.cacl_recon_loss(z, batch, sizefactor, combined_batch, x)
        # proto_kl, cluster_kl, responsibilities = self.calc_kl_loss(z_mu, z_logvar)
        # proto_kl, cluster_kl, responsibilities = self.calc_kl_loss(z_mu, z_logvar)
        gm_mu = self.get_gm_mu()
        gm_vparam = self.gm_vparam.to(gm_mu.device)
        # TODO: change tempreture?
        resp = responsibilities(z_mu, gm_mu, gm_vparam, self.temperature)
        z_var = torch.exp(z_logvar)
        z_vparam = softplus_inverse(z_var)
        kl, kl_dict = gm_kl(z_mu, z_vparam, gm_mu, gm_vparam, resp)
        # ---------- total ----------
        loss = recon + self.beta * kl
        return z_mu, loss, resp

    def forward(self, batch):
        z, cvae_loss, resp = self.calc_z_and_cvae_loss(**batch)
        propagation_sim = self.propagation_sim_loss(z)
        return z, z, self.proto_soft_assignments(z), cvae_loss, propagation_sim

    def proto_soft_assignments(self, z):
        if self.l2norm:
            return self.prototypes(z)
        elif self.assignment_metric == "neuc":
            protos = self.get_prototypes()
            return -torch.cdist(z, protos, p=2).pow(2)
        else:
            return F.cosine_similarity(
                z.unsqueeze(1), self.prototypes.weight.unsqueeze(0), dim=-1
            )

    def get_gm_mu(self):
        return self.prototypes.weight
