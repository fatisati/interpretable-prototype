"""scProto Stage 2 on top of a pretrained scVI encoder.

`ScviProtoTrainer` is `SCProtoTrainer` with Stage 1 swapped out: instead of pretraining
scPoli's cVAE, it trains (or reloads) a plain `scvi.model.SCVI` at its own default
settings and hands that encoder/decoder to Stage 2. Everything after Stage 1 --
waypoint prototype init, temperature calibration, the edge-centric community loss,
nassoc, usage, prototype reconstruction, early stopping on modularity, evaluation,
checkpointing -- is inherited untouched, which is the point: it makes
"scVI + scProto Stage 2" differ from "scVI + Leiden/SEACells" in exactly one place.

See `interpretable_ssl/models/scvi_backbone.py` for the model side, including why
`adata.X` is log1p(counts) here.

Resuming
--------
Every expensive step writes to disk and is reloaded if present:
  * scVI weights          -> {MODEL_DIR}/{ds}/scvi_stage1/{tag}/   (scvi's own save/load)
  * Stage-1 model state   -> get_pretrain_dump_path()/pretrain_checkpoint.pth
  * Stage-2 model state   -> get_dump_path()/umap_checkpoint.pth   (saved every eval_freq)
so a Colab disconnect costs at most the epochs since the last evaluation.
"""

import os

import numpy as np
import scipy.sparse as sp
import torch

from interpretable_ssl.trainers.scproto import SCProtoTrainer
from interpretable_ssl.trainers.scpoli_helpers import add_condition_combined
from interpretable_ssl.datasets.dataset import SingleCellDataset
from interpretable_ssl.datasets.dataset_configs import DATASETS
from interpretable_ssl.configs.paths import MODEL_DIR
from interpretable_ssl.models.scvi_backbone import build_scvi_proto_model


def log1p_matrix(X):
    """log1p that preserves sparsity (X is cells x genes raw counts)."""
    if sp.issparse(X):
        out = X.tocsr(copy=True).astype(np.float32)
        out.data = np.log1p(out.data)
        return out
    return np.log1p(np.asarray(X, dtype=np.float32))


def _count_stats(X):
    data = X.data if sp.issparse(X) else np.asarray(X).ravel()
    if data.size == 0:
        return 0.0, 1.0
    sample = data if data.size <= 200000 else np.random.default_rng(0).choice(data, 200000, replace=False)
    return float(data.max()), float(np.mean(sample == np.round(sample)))


def check_counts(adata, ds_id, batch_key=None):
    """Verify adata.X holds counts, not log-normalised expression.

    Hard failure is reserved for the one case that silently corrupts everything: X
    already being log-normalised. scVI's likelihood, its library size and the log1p
    space the Stage-2 losses live in would all be wrong, and nothing downstream would
    complain.

    A non-integer fraction is reported, not raised on. Several of the benchmark h5ads
    used here (pancreas among them) carry a small share of non-integer entries -- these
    matrices are what this codebase's existing scVI baselines were already trained on
    (`embedding_metrics.add_scvi_emb` reads the same X), so refusing them here would
    make this run inconsistent with the numbers it is meant to be compared against.
    The negative-binomial likelihood is defined for non-integer values anyway. Per-batch
    stats are printed so a single re-normalised batch would be visible rather than
    averaged away.
    """
    mx, frac_int = _count_stats(adata.X)
    if mx < 30:
        raise ValueError(
            f"[{ds_id}] adata.X looks log-normalised (max={mx:.2f}), not counts. scVI "
            f"needs raw counts -- check that the dataset h5ad carries a 'counts' layer."
        )

    print(f"[{ds_id}] counts check: max={mx:.0f}, integer fraction={frac_int:.3f}")
    if frac_int < 0.99:
        print(f"[{ds_id}] NOTE: {(1 - frac_int):.1%} of non-zero entries are not whole "
              f"numbers. Values are still count-scale (max={mx:.0f}), and this is the "
              f"same matrix the existing scVI baselines used, so the run proceeds.")
        if batch_key is not None and batch_key in adata.obs.columns:
            for b in adata.obs[batch_key].unique():
                bmx, bfrac = _count_stats(adata[adata.obs[batch_key] == b].X)
                print(f"    {str(b):<15} max={bmx:>12.1f}  integer fraction={bfrac:.3f}")


class ScviProtoTrainer(SCProtoTrainer):

    # scVI knobs, overridable per run via kwargs (they are plain attributes -- they do
    # not appear in ABBREVIATIONS, so they never enter the run-directory name; use
    # experiment_name to distinguish configurations on disk).
    SCVI_DEFAULT_EPOCHS = 50
    SCVI_DEFAULT_LIKELIHOOD = "zinb"

    def __init__(self, dataset=None, ref_query=None, parser=None, **kwargs):
        kwargs.setdefault("experiment_name", "scviproto")
        self.scvi_epochs = kwargs.pop("scvi_epochs", self.SCVI_DEFAULT_EPOCHS)
        self.scvi_gene_likelihood = kwargs.pop("scvi_gene_likelihood", self.SCVI_DEFAULT_LIKELIHOOD)
        self.scvi_tag = kwargs.pop("scvi_tag", None)
        self._scvi_model = None
        super().__init__(dataset, ref_query, parser, **kwargs)

    # ------------------------------------------------------------------
    # data: counts kept in a layer, X moved to scVI's own encoder input space
    # ------------------------------------------------------------------

    def get_dataset(self, dataset_id):
        ds = SingleCellDataset(name=dataset_id, use_counts=True, **DATASETS[dataset_id])
        ad = ds.adata
        check_counts(ad, dataset_id, batch_key=DATASETS[dataset_id].get("batch_key"))
        if "counts" not in ad.layers:
            ad.layers["counts"] = ad.X.copy()
        ad.X = log1p_matrix(ad.layers["counts"])

        # scarches' MultiConditionAnnotatedDataset (the base of MultiCropsDataset)
        # requires obs['conditions_combined']. On the scPoli path it appears as a side
        # effect of constructing the scPoli model on this adata; nothing in the scVI
        # path creates it, so build_data would fail on a KeyError. Added here, before
        # the train/val split, so both splits inherit it -- and via the codebase's own
        # helper so the values match the convention conditions_combined_encoder is
        # keyed on (for a single condition key, the batch label itself).
        add_condition_combined(ad, [ds.batch_key])
        print(f"[{dataset_id}] adata.X set to log1p(counts) -- scVI's own encoder input "
              f"space; raw counts kept in layers['counts'], log-normalised values in "
              f"layers['lognorm'].")
        return ds

    def build_data(self):
        """Guarantee obs['conditions_combined'] on every split, then build normally.

        get_dataset() already adds it before the train/val split, which covers the
        usual path. This second call covers the cases that bypass get_dataset
        entirely -- a SingleCellDataset handed to the constructor, or a ref/query pair
        built elsewhere -- where the column would otherwise be missing and scarches'
        MultiConditionAnnotatedDataset would fail with a bare KeyError deep inside its
        label encoder.
        """
        for split in (self.train_, self.val_):
            if split is not None:
                add_condition_combined(split.adata, [split.batch_key])
        return super().build_data()

    # ------------------------------------------------------------------
    # Stage 1: scVI instead of scPoli
    # ------------------------------------------------------------------

    def get_scvi_dir(self):
        tag = self.scvi_tag or f"d{self.latent_dims}_{self.scvi_gene_likelihood}_e{self.scvi_epochs}"
        return os.path.join(MODEL_DIR, self.dataset_id, "scvi_stage1", tag)

    def get_pretrain_dump_path(self):
        """Separate namespace from scPoli pretrain checkpoints -- same dataset, same
        latent size, completely different weights; they must never resolve to one path."""
        tag = self.scvi_tag or f"d{self.latent_dims}_{self.scvi_gene_likelihood}_e{self.scvi_epochs}"
        return os.path.join(MODEL_DIR, self.dataset_id, "pretrain_scvi", tag)

    def build_scvi_model(self, adata=None, train=False):
        """Construct a `scvi.model.SCVI` on this trainer's adata.

        train=False just builds the architecture (used to give Stage 2 a correctly
        shaped module before weights are loaded). train=True runs the actual Stage-1
        fit, reusing an existing save directory when one is there.
        """
        import scvi

        if adata is None:
            # SingleCellDataset._create_split_instance hands out AnnData *views*.
            # scvi's setup_anndata writes registry entries into .uns/.obs, which
            # silently materialises a view mid-call; doing it up front instead keeps
            # one concrete object shared by scVI, the Stage-2 dataset and every
            # obsm[...] written later (they must stay row-aligned).
            if self.train_.adata.is_view:
                self.train_.adata = self.train_.adata.copy()
            adata = self.train_.adata
        save_dir = self.get_scvi_dir()

        if train and os.path.exists(os.path.join(save_dir, "model.pt")):
            print(f"[scvi stage1] reusing trained scVI at {save_dir}")
            model = scvi.model.SCVI.load(save_dir, adata=adata)
            self._scvi_model = model
            return model

        scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=self.condition_key)
        model = scvi.model.SCVI(
            adata,
            n_latent=self.latent_dims,
            gene_likelihood=self.scvi_gene_likelihood,
        )
        if train:
            print(f"[scvi stage1] training scVI (n_latent={self.latent_dims}, "
                  f"gene_likelihood={self.scvi_gene_likelihood}, "
                  f"max_epochs={self.scvi_epochs}) on {adata.n_obs} cells")
            model.train(max_epochs=self.scvi_epochs)
            os.makedirs(save_dir, exist_ok=True)
            model.save(save_dir, overwrite=True)
            print(f"[scvi stage1] saved to {save_dir}")
        self._scvi_model = model
        return model

    def get_model(self):
        scvi_model = self._scvi_model or self.build_scvi_model(train=False)
        return build_scvi_proto_model(
            scvi_model=scvi_model,
            adata=self.train_.adata,
            batch_key=self.condition_key,
            nmb_prototypes=self.num_prototypes,
            l2norm=self.l2norm,
            assignment_metric=self.assignment_metric,
            compute_recon_kl=(self.lambda_recon > 0 or self.lambda_kl > 0),
            gene_likelihood=self.scvi_gene_likelihood,
        )

    def pretrain_encoder(self, scpoli_train_kwargs=None):
        """Stage 1 = fit scVI, then copy its weights into the Stage-2 model.

        Weights are copied rather than the model rebuilt so that the prototype layer
        (and anything else Stage 2 may already have attached) survives.
        """
        scvi_model = self.build_scvi_model(train=True)
        # strict=True: a shape/name mismatch here means the Stage-2 module was built
        # with different scVI hyperparameters than the one just trained -- silently
        # continuing would leave a randomly-initialised encoder in place.
        self.model.scpoli_cvae.vae.load_state_dict(scvi_model.module.state_dict(), strict=True)
        self.model.to(self.device)
        print("[scvi stage1] weights loaded into the Stage-2 model")

    # ------------------------------------------------------------------
    # latent access
    # ------------------------------------------------------------------

    def get_latent(self, adata=None, l2norm=None):
        """(N, d) latent for `adata` (defaults to the training adata) from the model in
        its CURRENT state -- right after Stage 1 this is plain scVI's latent, after
        Stage 2 it is the updated one.

        l2norm: None keeps whatever the model trains with (1 in the canonical config,
        i.e. embeddings on the unit sphere). Pass False for the raw posterior mean --
        that is what a plain "run scVI, then cluster its latent" baseline would use,
        and it is how scVI latents are consumed everywhere else in this codebase.
        """
        adata = self.train_ds.adata if adata is None else adata
        prev = self.model.l2norm
        if l2norm is not None:
            self.model.l2norm = bool(l2norm)
        try:
            with torch.no_grad():
                return self.encode_adata(adata, self.model, z_idx=1).cpu().numpy()
        finally:
            self.model.l2norm = prev
