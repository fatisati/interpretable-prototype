"""scVI-backed encoder/decoder backbone for scProto's Stage 2.

Why this exists
---------------
Reviewer F5RB's two open points are really one question: is scProto's advantage the
Stage-2 prototype/community objective, or is it inherited from whichever Stage-1
batch-correction encoder it happens to sit on top of? The clean way to answer that is
to hold the Stage-1 encoder fixed and vary ONLY what happens afterwards:

    arm A   pretrained scVI  ->  Leiden / SEACells on its latent   (batch-correct-then-cluster)
    arm B   pretrained scVI  ->  scProto Stage 2 continues training that same encoder

Same dataset, same pretrained weights, same affinity graph, same K, same metrics --
so any difference between A and B is attributable to the Stage-2 loss alone. This
module supplies the piece that did not exist yet: a backbone that lets scProto's
Stage-2 training loop drive a *scVI* encoder instead of scPoli's cVAE.

How it plugs in
---------------
`SCProtoTrainer`'s Stage-2 loop (`_run_umap_epoch`) only ever touches its model
through a small surface:

    model.encoder_out({'x': X, 'batch': cond})   -> (z, recon_loss, kl_loss)
    model.prototypes / get_prototypes / normalize_prototypes / set_prototypes
    model.decode(z, cond)                        -> gene-space reconstruction
    model.scpoli_cvae                            -> the module whose parameters are optimised
                                                    (+ condition encoders, read by build_data)

`ScviProtoModel` implements exactly that surface on top of a `scvi.model.SCVI`
module, so no line of the Stage-2 training loop, evaluation, or checkpointing needs
to change.

Data space convention (important)
---------------------------------
scVI's encoder is trained on `log1p(raw counts)` -- that transform lives *inside*
`VAE.inference`. To keep a pretrained scVI encoder exactly valid when scProto feeds it
data directly, `ScviProtoTrainer` sets `adata.X = log1p(counts)` (raw counts kept in
`adata.layers['counts']`), and this module treats every `x` it receives as already
being in that space:

  * the encoder consumes `x` as-is (skipping scVI's internal log1p -- it is already applied),
  * `decode()` returns `log1p(px_rate)`, i.e. the same space as `x`.

That second point matters for more than tidiness: Stage 2's prototype-reconstruction
term is a plain MSE between `decode(prototypes)` and the mini-batch `x`. In raw-count
space that MSE is 4-6 orders of magnitude larger than the community loss it is
supposed to accompany, and `lambda_proto_recon=0.01` would silently turn into the only
term being optimised. In log1p space it has the same magnitude as scProto's own
MSE-on-log-normalised setup, so the canonical lambda config keeps its intended
balance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from interpretable_ssl.models.swav import SwavBase


# --------------------------------------------------------------------------------
# scPoli-cVAE-shaped facade over a scvi-tools VAE
# --------------------------------------------------------------------------------


class ScviCVAE(nn.Module):
    """Wraps a `scvi.model.SCVI` module (`scvi_model.module`) behind the handful of
    calls scProto expects from `model.scpoli_cvae`.

    Registering the scVI `VAE` as a submodule is what makes the rest of the pipeline
    work unchanged: `_setup_umap_edges` builds its optimiser from
    `model.scpoli_cvae.parameters()`, so scVI's encoder/decoder weights are exactly
    the parameters Stage 2 trains, and `model.state_dict()` (used by
    save/load_umap_checkpoint) carries them.

    Args:
        scvi_module:       the `VAE` module of a `scvi.model.SCVI` (trained or not).
        batch_categories:  batch labels in scVI's OWN category order -- index i here
                           must be the batch index scVI was trained with, otherwise
                           the decoder would be conditioned on the wrong batch.
        batch_key:         adata.obs column holding those labels.
        ref_log_library:   log library size used when decoding a *prototype*. A
                           prototype is a point in latent space, not a cell, so it has
                           no library of its own; a single dataset-level reference
                           (median log library) keeps every prototype decoded on one
                           common scale, which is what the prototype-reconstruction
                           and metacell-expression code both assume.
        gene_likelihood:   'zinb' | 'nb' | 'poisson' -- only used when reconstruction
                           loss is actually requested (see `recon_kl`).
    """

    def __init__(self, scvi_module, batch_categories, batch_key,
                 ref_log_library, gene_likelihood="zinb"):
        super().__init__()
        self.vae = scvi_module
        self.batch_key = batch_key
        self.batch_categories = [str(c) for c in batch_categories]
        self.gene_likelihood = gene_likelihood

        # scPoli-compatible condition encoding. scProto's dataset layer
        # (MultiCropsDataset -> scarches MultiConditionAnnotatedDataset) builds its
        # `conditions` index tensor from these dicts, so defining them from scVI's own
        # category order is what keeps `batch` indices consistent between the dataset
        # and the pretrained decoder.
        self.condition_encoders = {batch_key: {c: i for i, c in enumerate(self.batch_categories)}}
        self.conditions_combined_encoder = dict(self.condition_encoders[batch_key])
        self.n_conditions = [len(self.batch_categories)]
        self.n_conditions_combined = len(self.batch_categories)

        self.register_buffer("ref_log_library", torch.as_tensor(float(ref_log_library)))

        self._check_assumptions()

    def _check_assumptions(self):
        """Fail loudly on any scVI configuration this facade would silently misread.

        Each of these is a default in scvi-tools; the checks exist because getting one
        wrong produces plausible-looking numbers rather than an error.
        """
        batch_rep = getattr(self.vae, "batch_representation", "one-hot")
        if batch_rep != "one-hot":
            raise ValueError(
                f"scVI module uses batch_representation='{batch_rep}'; this backbone "
                f"assumes the default 'one-hot' conditioning (batch fed to the decoder "
                f"as a categorical index). Retrain scVI with the default, or extend "
                f"ScviCVAE.decode/encode to build the batch embedding the same way "
                f"scvi-tools does for that mode."
            )
        if not getattr(self.vae, "log_variational", True):
            raise ValueError(
                "scVI module was built with log_variational=False, i.e. its encoder was "
                "trained on raw counts, not log1p(counts). This backbone feeds the "
                "encoder log1p(counts) directly (see the module docstring) -- with "
                "log_variational=False that input is in the wrong space."
            )
        if getattr(self.vae, "use_size_factor_key", False):
            raise ValueError(
                "scVI module was built with use_size_factor_key=True; its decoder then "
                "expects a raw size factor rather than a log library size, so the "
                "library value passed in decode() would be interpreted incorrectly."
            )

    # -- shape helpers -------------------------------------------------------

    @staticmethod
    def _as_cat(batch):
        """scProto passes batch conditions as (N, n_condition_keys); scVI's FCLayers
        want a single (N, 1) categorical index. Only one condition key is ever used
        here (the dataset's batch_key), so take the first column."""
        if batch is None:
            raise ValueError("batch condition tensor is required -- scVI's decoder is batch-conditioned.")
        if batch.dim() == 1:
            return batch.view(-1, 1).long()
        return batch[:, :1].long()

    # -- encoder -------------------------------------------------------------

    def encode(self, x_log1p, batch):
        """Return (q_m, q_v) for `x_log1p`, which must ALREADY be log1p(counts) --
        exactly the tensor `VAE.inference` would hand to `z_encoder` internally.

        Handles both scvi-tools encoder return conventions: `(dist, latent)` on
        versions where `Encoder.return_dist=True`, and `(q_m, q_v, latent)` on older
        ones."""
        cat = self._as_cat(batch)
        out = self.vae.z_encoder(x_log1p, cat)
        if len(out) == 2:
            qz, _z = out
            return qz.loc, qz.scale.pow(2)
        q_m, q_v, _z = out
        return q_m, q_v

    @staticmethod
    def sampling(mu, var_or_logvar, is_logvar=False):
        """Reparameterised sample. Kept for interface parity with scPoli's cVAE;
        Stage 2 itself trains on the posterior mean (`q_m`), like scProto does."""
        std = torch.exp(0.5 * var_or_logvar) if is_logvar else var_or_logvar.clamp_min(1e-8).sqrt()
        return mu + std * torch.randn_like(std)

    # -- decoder -------------------------------------------------------------

    def _decoder_forward(self, z, batch, log_library):
        cat = self._as_cat(batch)
        px_scale, px_r, px_rate, px_dropout = self.vae.decoder(
            self.vae.dispersion, z, log_library, cat
        )
        px_r = self._resolve_px_r(px_r, cat)
        return px_scale, px_r, px_rate, px_dropout

    def _resolve_px_r(self, px_r, cat):
        """Mirror `VAE.generative`'s dispersion handling: the decoder only returns a
        per-cell px_r for dispersion='gene-cell'; every other mode reads it off the
        model-level parameter."""
        dispersion = self.vae.dispersion
        if dispersion == "gene-cell":
            return torch.exp(px_r)
        if dispersion == "gene":
            return torch.exp(self.vae.px_r)
        if dispersion == "gene-batch":
            one_hot = F.one_hot(cat.view(-1), num_classes=self.n_conditions_combined).float()
            return torch.exp(F.linear(one_hot, self.vae.px_r))
        raise ValueError(
            f"unsupported scVI dispersion mode '{dispersion}' -- expected one of "
            f"'gene', 'gene-batch', 'gene-cell'."
        )

    def decode(self, z, batch):
        """Decode latent points to the SAME space the model is fed (log1p counts).

        Used for (a) Stage 2's prototype-reconstruction term and (b) writing
        metacell expression profiles (`save_metacells`). The library size is the
        dataset-level reference rather than a per-cell one -- see `ref_log_library`.
        """
        log_lib = self.ref_log_library.expand(z.shape[0], 1)
        _px_scale, _px_r, px_rate, _px_dropout = self._decoder_forward(z, batch, log_lib)
        return torch.log1p(px_rate)

    # -- likelihood terms (only computed when a lambda actually needs them) ----

    def recon_kl(self, x_log1p, batch, q_m, q_v):
        """scVI's own reconstruction NLL + the analytic KL to N(0, I).

        `x_log1p` is inverted back to counts with expm1 (exact inverse of the
        transform applied to adata.X), so the likelihood is evaluated on real counts
        the way scVI defines it -- no approximation of scVI's objective.
        """
        from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial

        counts = torch.expm1(x_log1p).clamp_min(0.0)
        log_lib = torch.log(counts.sum(dim=1, keepdim=True).clamp_min(1.0))
        _px_scale, px_r, px_rate, px_dropout = self._decoder_forward(z=self._z_from_q(q_m),
                                                                    batch=batch,
                                                                    log_library=log_lib)

        if self.gene_likelihood == "zinb":
            dist = ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)
        elif self.gene_likelihood == "nb":
            dist = NegativeBinomial(mu=px_rate, theta=px_r)
        elif self.gene_likelihood == "poisson":
            dist = torch.distributions.Poisson(px_rate)
        else:
            raise ValueError(f"unsupported gene_likelihood '{self.gene_likelihood}'")

        recon = -dist.log_prob(counts).sum(dim=-1).mean()
        kl = 0.5 * (q_v + q_m.pow(2) - 1.0 - q_v.clamp_min(1e-8).log()).sum(dim=1).mean()
        return recon, kl

    @staticmethod
    def _z_from_q(q_m):
        # Deterministic latent for the reconstruction term, matching how Stage 2 uses
        # the posterior mean everywhere else.
        return q_m


# --------------------------------------------------------------------------------
# The scProto model, backed by scVI
# --------------------------------------------------------------------------------


class ScviProtoWrapperShim:
    """Stand-in for scPoli's `scpoli_wrapper` object.

    `Trainer.extract_scpoli(model, return_wrapper=True)` and
    `check_conditions_compatible()` expect a wrapper exposing `.model`, `.adata` and
    `.conditions_`; `SCProtoTrainer.build_model` logs `.adata.X.max()`. Providing
    those three attributes is enough for the whole trainer to run unmodified, and
    keeps `encode_adata` on its fast path (conditions always compatible, so it never
    tries scPoli's query-model surgery).
    """

    def __init__(self, model, adata, batch_key, batch_categories):
        self.model = model
        self.adata = adata
        self.conditions_ = {batch_key: [str(c) for c in batch_categories]}


class ScviProtoModel(SwavBase):
    """scProto (prototype layer + Stage-2 interface) on a scVI encoder/decoder.

    compute_recon_kl: reconstruction/KL terms cost a full decoder pass per step and
    are unused by the canonical Stage-2 config (`lambda_recon=lambda_kl=0`), so they
    are skipped unless a caller actually weights them -- `ScviProtoTrainer` sets this
    from the lambda config rather than leaving it to be remembered by hand.
    """

    def __init__(self, scvi_cvae, latent_dim, nmb_prototypes, l2norm=1,
                 assignment_metric="dotp", compute_recon_kl=False):
        super().__init__(
            scpoli_cvae=scvi_cvae,
            latent_dim=latent_dim,
            nmb_prototypes=nmb_prototypes,
            l2norm=l2norm,
            assignment_metric=assignment_metric,
        )
        self.compute_recon_kl = bool(compute_recon_kl)
        self.recon_loss = "nb"          # informational: scVI's native count likelihood
        self.scpoli_wrapper = None      # set by attach_wrapper (not a submodule)
        self.proto_niche_labels = None  # niche-constrained mode is not supported here

    # -- wiring --------------------------------------------------------------

    def attach_wrapper(self, wrapper):
        object.__setattr__(self, "scpoli_wrapper", wrapper)

    def attach_scpoli(self, wrapper):
        """Interface parity with SwAVModel.attach_scpoli. `adapt_model` only calls
        this when an adata carries conditions the model has not seen -- impossible
        here, since the model is always built from the dataset it is trained on."""
        raise NotImplementedError(
            "ScviProtoModel cannot be re-attached to a different scPoli wrapper. "
            "This is only reached if encode_adata() is called with an adata whose "
            "batch values are not a subset of the training batches."
        )

    # -- Stage-2 interface ---------------------------------------------------

    def encoder_out(self, batch):
        """(z, recon_loss, kl_loss) for one mini-batch -- the single entry point
        Stage 2's training loop uses to get embeddings and encoder-side losses."""
        x = batch["x"]
        cond = batch["batch"]
        q_m, q_v = self.scpoli_cvae.encode(x, cond)

        if self.compute_recon_kl:
            recon, kl = self.scpoli_cvae.recon_kl(x, cond, q_m, q_v)
        else:
            zero = torch.zeros((), device=q_m.device)
            recon, kl = zero, zero

        z = F.normalize(q_m, dim=1, p=2) if self.l2norm else q_m
        return z, recon, kl

    def decode(self, z, batch):
        return self.scpoli_cvae.decode(z, batch)

    def proto_soft_assignments(self, z, cell_niche_idx=None, proto_niche_labels=None):
        """Same signature as scProtoGMVAE's (the Stage-2 loop and evaluation code both
        call it with the niche arguments), delegating to SwavBase for the actual
        metric. Niche-constrained assignment is spatial-only and not part of this
        experiment -- fail loudly rather than silently ignoring a mask."""
        if cell_niche_idx is not None or proto_niche_labels is not None:
            raise NotImplementedError(
                "niche-constrained prototype assignment is not supported by the scVI backbone."
            )
        return SwavBase.proto_soft_assignments(self, z)

    def forward(self, bs, batch):
        """Inference path only -- returns the same tuple layout as scProtoGMVAE.forward
        so `Trainer.encode_batch` (which reads out[:3]) works unchanged.

        Stage 2 never routes its losses through here (it calls `encoder_out` +
        the prototype layer directly), so the loss slots are returned as zeros
        rather than being computed on every encode pass.
        """
        batch = dict(batch)
        batch.pop("cell_niche_idx", None)
        z, recon, kl = self.encoder_out(batch)
        scores = self.proto_soft_assignments(z)
        zero = torch.zeros((), device=z.device)
        return z, z, scores, recon, zero, (zero, zero), kl, zero

    def encode(self, batch):
        out = self.forward(1, batch)
        return out[:3]

    # -- freezing helpers (scVI parameter names, not scPoli's) ----------------

    def freeze_batch_embedding(self):
        """No-op with a reason: stock scVI has no trainable batch-embedding table --
        batch enters as one-hot categorical input to the decoder's FCLayers. The
        closest analogue is freezing the decoder (below)."""
        print("freeze_batch_embedding: skipped -- scVI conditions on one-hot batch "
              "indices, there is no batch-embedding table to freeze.")

    def freeze_decoder(self):
        n = 0
        for name, p in self.named_parameters():
            if "scpoli_cvae.vae.decoder" in name or name.endswith("scpoli_cvae.vae.px_r"):
                p.requires_grad = False
                n += 1
        print(f"freeze_decoder: froze {n} scVI decoder parameter tensors")


# --------------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------------


def build_scvi_proto_model(scvi_model, adata, batch_key, nmb_prototypes,
                           l2norm=1, assignment_metric="dotp",
                           compute_recon_kl=False, gene_likelihood="zinb"):
    """Wrap an existing `scvi.model.SCVI` (trained or freshly constructed) into a
    Stage-2-ready scProto model.

    The batch category order is read from scVI's own AnnData manager when available,
    so the condition indices scProto's dataset produces are the same integers scVI's
    decoder was trained with. Falling back to `adata.obs[batch_key]` categories is
    only for scVI versions whose registry layout differs -- it reproduces the same
    order scvi-tools derives itself (pandas categorical order).
    """
    import numpy as np

    module = scvi_model.module
    categories = _get_scvi_batch_categories(scvi_model, adata, batch_key)

    counts = _get_counts_matrix(adata)
    lib = np.asarray(counts.sum(axis=1)).ravel()
    ref_log_library = float(np.log(np.maximum(np.median(lib), 1.0)))

    cvae = ScviCVAE(
        scvi_module=module,
        batch_categories=categories,
        batch_key=batch_key,
        ref_log_library=ref_log_library,
        gene_likelihood=gene_likelihood,
    )
    model = ScviProtoModel(
        scvi_cvae=cvae,
        latent_dim=module.n_latent,
        nmb_prototypes=nmb_prototypes,
        l2norm=l2norm,
        assignment_metric=assignment_metric,
        compute_recon_kl=compute_recon_kl,
    )
    model.attach_wrapper(ScviProtoWrapperShim(model, adata, batch_key, categories))
    print(f"[scvi backbone] latent_dim={module.n_latent}  K={nmb_prototypes}  "
          f"n_batches={len(categories)}  ref_log_library={ref_log_library:.3f}  "
          f"recon/kl terms={'on' if compute_recon_kl else 'off'}")
    return model


def _get_scvi_batch_categories(scvi_model, adata, batch_key):
    try:
        registry = scvi_model.adata_manager.get_state_registry("batch")
        return [str(c) for c in registry.categorical_mapping]
    except Exception as e:  # noqa: BLE001 -- registry layout varies across versions
        print(f"[scvi backbone] could not read scVI's batch registry ({type(e).__name__}: {e}); "
              f"falling back to adata.obs['{batch_key}'] categorical order.")
        import pandas as pd
        return [str(c) for c in pd.Categorical(adata.obs[batch_key]).categories]


def _get_counts_matrix(adata):
    if "counts" not in adata.layers:
        raise ValueError(
            "adata.layers['counts'] is required -- scVI's library size and likelihood "
            "are defined on raw counts. ScviProtoTrainer.get_dataset populates this "
            "layer; if you built the AnnData yourself, add it."
        )
    return adata.layers["counts"]
