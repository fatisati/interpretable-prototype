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

    def pretrain_encoder(self):
        self.extract_scpoli(self.model, True).train(
            n_epochs=self.cvae_epochs,
            pretraining_epochs=self.cvae_epochs,
        )

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
        
        if not (self.debug==1):
            self.set_job_name(self.dump_path)
            self.init_wandb(self.dump_path)
        
        if self.cvae_epochs > 0:
            self.pretrain_encoder()
        
        # if hasattr(self.model, "freeze_batch_embedding"):
        #     self.model.freeze_batch_embedding()
        self.init_prototypes()

        if self.cvae_epochs > 0:
            self.plot_umap(self.model, self.ref.adata, "pretrained-ref")
        self.train()
        
        if self.ft_epochs > 0:
            self.model = self.adapt_model(self.model, self.query.adata, self.ft_epochs)
            self.save_checkpoint(self.pretraining_epochs + self.ft_epochs)
        
        self.save_metrics()
        self.ref = self.original_ref
        self.plot_umap(self.model, self.original_ref.adata, "ref")
        self.plot_umap(self.model, self.dataset.adata, "all", True)
