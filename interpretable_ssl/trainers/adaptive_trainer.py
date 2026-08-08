from interpretable_ssl.trainers.trainer import *
from torch.utils.data import random_split
from logging import getLogger
from interpretable_ssl.utils import log_time
from interpretable_ssl.trainers.cvae_trainer import CvaeTrainer
from sklearn.neighbors import NearestNeighbors

logger = getLogger()


class AdoptiveTrainer(Trainer):
    # @log_time('adoptive trainer')
    def __init__(
        self, dataset=None, ref_query=None, parser=None, **kwargs
    ) -> None:

        super().__init__(dataset, ref_query, parser, **kwargs)
        self.finetune_ds = None
        self.original_ref = self.ref
        self.partial_ref = None
        self.finetuning = False
        self.transfer_learning_mode = False

    def split_train_data(self, finetune_size=0.1):
        self.partial_ref, self.finetune_ds = self.original_ref.split(finetune_size)
        # self.tune_nmb_crops([self.partial_ref.adata, self.finetune_ds.adata])

    def finetune(self):
        pass

    def train_semi_supervised(self):
        self.split_train_data()
        self.ref = self.partial_ref
        self.setup()
        self.train()
        self.finetuning = True
        self.ref = self.finetune_ds
        self.finetune()

    def transfer_learning(self):
        # pretrain on one dataset
        # finetune on another dataset
        # make sure about namings

        # pretrain
        self.dataset = self.get_dataset(self.pretrain_dataset_id)
        if self.full_dataset_mode == 1:
            self.ref = self.dataset
            self.query = None
        else:
            self.ref, self.query = self.dataset.get_train_test()
        self.setup()
        self.train()

        # finetune
        self.finetuning = True
        self.dataset = self.get_dataset(self.finetune_dataset_id)
        self.ref, self.query = self.dataset.get_train_test()
        # self.setup()
        self.finetune()

    def train_fully_supervised(self):
        pass

    PRETRAIN_PARAM_KEYS = [
        'dataset_id', 'cvae_epochs', 'batch_size',
        'latent_dims', 'l2norm', 'model_type', 'beta', 'num_prototypes',
    ]
    PRETRAIN_ABBREVIATIONS = {
        'dataset_id': 'ds', 'cvae_epochs': 'cvae_e', 'batch_size': 'BS',
        'latent_dims': 'LD', 'l2norm': 'l2norm', 'model_type': 'model',
        'beta': 'beta', 'num_prototypes': 'NP',
    }

    def _get_pretrain_param_default(self, key, defaults):
        """Default value to compare against when deciding whether `key` earns a token
        in the pretrain checkpoint name. Plain `defaults.get(key)` (configs/defaults.py's
        flat, dataset-agnostic fallback) is wrong for `num_prototypes` specifically --
        it legitimately varies per dataset (see dataset_configs.py: pancreas=220,
        lung=300, cd34=95, ...), so comparing against one flat number would tag every
        dataset whose own default isn't that number, even when num_prototypes was never
        actually overridden for it.
        """
        if key == 'num_prototypes':
            from interpretable_ssl.datasets.dataset_configs import DATASETS
            ds_conf = DATASETS.get(self.dataset_id)
            if ds_conf and 'num_prototypes' in ds_conf:
                return ds_conf['num_prototypes']
            # Unregistered/custom dataset: no per-dataset default to compare against --
            # fall back to "never tag this key" (self.params.get(key) == itself),
            # matching the pre-existing behavior for such datasets exactly.
            return self.params.get(key)
        return defaults.get(key)

    def _get_pretrain_name(self):
        from interpretable_ssl.configs.defaults import get_defaults
        defaults = get_defaults()
        parts = ['pretrain']
        for key in self.PRETRAIN_PARAM_KEYS:
            val = self.params.get(key)
            default_val = self._get_pretrain_param_default(key, defaults)
            if val is not None and val != default_val:
                abbr = self.PRETRAIN_ABBREVIATIONS[key]
                parts.append(f"{abbr}-{val}" if isinstance(val, str) else f"{abbr}{val}")
        return "_".join(parts)

    def get_pretrain_dump_path(self):
        from interpretable_ssl.configs.constants import MODEL_DIR
        return os.path.join(MODEL_DIR, self.dataset_id, 'pretrain', self._get_pretrain_name())

    def pretrain_encoder(self, scpoli_train_kwargs=None):
        kwargs = scpoli_train_kwargs or {}
        self.extract_scpoli(self.model, True).train(
            n_epochs=self.cvae_epochs,
            pretraining_epochs=self.cvae_epochs,
            **kwargs
        )

    def save_pretrain_checkpoint(self, path=None):
        import torch, os
        if path is None:
            os.makedirs(self.get_pretrain_dump_path(), exist_ok=True)
            path = os.path.join(self.get_pretrain_dump_path(), 'pretrain_checkpoint.pth')
        pretrain_params = {k: self.params.get(k) for k in self.PRETRAIN_PARAM_KEYS}
        pretrain_params['condition_key'] = self.condition_key
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'pretrain_params': pretrain_params,
        }, path)
        print(f"Saved pretrain checkpoint to {path}")
        return path

    def load_pretrain_checkpoint(self, path=None):
        import torch, os
        if path is None:
            path = os.path.join(self.get_pretrain_dump_path(), 'pretrain_checkpoint.pth')
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrain checkpoint from {path}")
        print(f"  pretrain_params: {checkpoint.get('pretrain_params')}")
        return checkpoint

    def init_prototypes(self):
        pass

    def train(self):
        pass

    def setup(self):
        pass

    def get_umap_path(self, data_part="ref"):
        img_name = f"/{data_part}-umap"
        if self.finetuning:
            img_name = f"{img_name}_finetuned"
        # umap_paths = [self.get_dump_path() + f"/{img_name}.png"]
        umap_paths = []
        if self.save_temp_res == 1 and (self.debug!=1):
            umap_paths.append(self.get_temp_res_path() + f"/{img_name}.png")
        return umap_paths

    def load_adopt(self):
        model = self.load_model()
        self.adapt_model(model, self.finetune_ds.adata)
        return model

    def save_checkpoint(self, epoch):
        pass
        
    def run(self):
        
        if self.mode == 'eval':
            self.setup()
            try:
                self.model = self.load_model()
                self.save_metrics()
                return
            except Exception as e:
                print(e, self.get_dump_path())
                return
                
            
        if not (self.debug==1):
            self.set_job_name(self.dump_path)
            self.init_wandb(self.dump_path)
        
        if self.cvae_epochs > 0:
            self.pretrain_encoder()
        
        # if hasattr(self.model, "freeze_batch_embedding"):
        #     self.model.freeze_batch_embedding()
        self.init_prototypes()

        # Auto-calibrate eps/tau after encoder pretrain + proto init
        if getattr(self, 'auto_eps_tau', 0) == 1 and hasattr(self, 'calibrate_temperatures'):
            self.calibrate_temperatures()

        if self.cvae_epochs > 0 and not self.debug:
            self.plot_umap(self.model, self.ref.adata, "pretrained-ref")
        self.train()
        
        if self.ft_epochs > 0 and (self.full_dataset_mode == 0):
            self.model = self.adapt_model(self.model, self.query.adata, self.ft_epochs)
            self.save_checkpoint(self.pretraining_epochs + self.ft_epochs)
        
        # Skip expensive metrics and UMAP in debug mode
        if not self.debug:
            self.save_metrics()
            self.ref = self.original_ref
            self.plot_umap(self.model, self.original_ref.adata, "ref")
            self.plot_umap(self.model, self.dataset.adata, "all", True)
