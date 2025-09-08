from interpretable_ssl.trainers.trainer import *
from torch.utils.data import random_split
from logging import getLogger
from interpretable_ssl.utils import log_time
from interpretable_ssl.trainers.cvae_trainer import CvaeTrainer

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
            eta=5,
        )

    def init_prototypes(self):
        pass

    def train_pretrain_encoder(self):
        if self.fine_tuning_epochs > 0:
            self.split_train_data()
            self.ref = self.partial_ref
        self.setup()
        self.pretrain_encoder()
        self.init_prototypes()
        self.plot_umap(self.model, self.ref.adata, "pretrained-ref")
        # pretrain with swav
        self.train()

        if self.fine_tuning_epochs > 0:
            self.prot_decoding_loss_scaler = 0
            self.finetuning = True
            self.ref = self.finetune_ds
            self.finetune()

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
        if self.save_temp_res == 1:
            umap_paths.append(self.get_temp_res_path() + f"/{img_name}.png")
        return umap_paths

    def load_adopt(self):
        model = self.load_model()
        self.adapt_model(model, self.finetune_ds.adata)
        return model

    def run(self):
        if not self.debug:
            self.set_job_name(self.dump_path)
            self.init_wandb(self.dump_path)
        if self.training_type == "semi_supervised":
            self.train_semi_supervised()
        elif self.training_type == "transfer_learning":
            self.transfer_learning()
        elif self.training_type == "fully_supervised":
            self.train_fully_supervised()
        elif self.training_type == "pretrain_encoder":
            self.train_pretrain_encoder()
        else:
            print(f"training type: {self.training_type}")
            self.train()

        self.ref = self.original_ref
        self.plot_umap(self.model, self.original_ref.adata, "ref")
        self.plot_umap(self.model, self.dataset.adata, "all", True)
        self.plot_umap(self.model, self.query.adata, "query", True)
