import os
import wandb
from interpretable_ssl.utils import get_device
from interpretable_ssl.trainers.base import TrainerBase

from interpretable_ssl.utils import log_time

from interpretable_ssl.datasets.dataset import SingleCellDataset
from interpretable_ssl.datasets.dataset_configs import DATASETS
class Trainer(TrainerBase):
    # @log_time('trainer')
    def __init__(self, debug=False, dataset=None, ref_query=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.device = get_device()
        self.dataset = dataset
        if self.dataset is None:
            print(f"dataset is None, loading {self.dataset_id}")
            if self.no_data == "False":
                self.dataset = self.get_dataset(self.dataset_id)
        self.input_dim = self.dataset.x_dim

        if ref_query is None:
            print(self.dataset.batch_key)
            self.ref, self.query = self.dataset.get_train_test()
        else:
            self.ref, self.query = ref_query
        self.debug = debug
        self.ref_latent, self.query_latent, self.all_latent = None, None, None

        if (not self.debug) and (self.wandb_sweep != 1):
            self.set_job_name()
            # self.init_wandb()
        self.condition_key = self.ref.batch_key
        
    def get_model(self):
        pass

    def get_dataset(self, dataset_id):
        ds_params = DATASETS[dataset_id]
        return SingleCellDataset(name = dataset_id, **ds_params)

    def set_job_name(self, path=None):
        if path is None:
            path = self.get_dump_path()
        set_job_name = (self.job_name is None) or (self.job_name == "")
        if set_job_name:
            self.job_name = f"{self.get_model_name()}/{self.dataset}"

    def init_wandb(self, path=None):
        if self.debug == 1:
            return 
        wandb.init(
            name=self.job_name,
            # project="interpretable-ssl",
            config={
                "num_prototypes": self.num_prototypes,
                "hidden dim": self.hidden_dim,
                "latent_dims": self.latent_dims,
                "device": self.device,
                "model path": path,
                "dataset": self.dataset,
                "train size": len(self.ref),
                "test size": len(self.query),
                "batch size": self.batch_size,
            },
        )
