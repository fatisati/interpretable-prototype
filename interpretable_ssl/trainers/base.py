from pathlib import Path
from interpretable_ssl.configs.defaults import *
from interpretable_ssl.configs.constants import *
import os
from constants import *
from interpretable_ssl.utils import log_time
from interpretable_ssl.model_name import generate_model_name

from scarches.dataset.scpoli.anndata import MultiConditionAnnotatedDataset
import scarches.trainers.scpoli._utils as scpoli_utils
from torch.utils.data import DataLoader


class TrainerBase:
    # @log_time('trainer base')
    def __init__(self, **kwargs) -> None:
        # Get the default values from the function
        defaults = get_defaults().copy()
        self.default_values = get_defaults().copy()
        # Update defaults with any provided keyword arguments
        defaults.update(kwargs)

        # Assign each default value to an instance variable
        for key, value in defaults.items():
            setattr(self, key, value)
        
        
        self.set_experiment_name()
        self.params = self.__dict__.copy()
        self.create_dump_path()
        self.create_temp_res_path()
        
        # if self.training_type != "semi_supervised" and self.training_type != "fully_supervised":
        #     self.pretraining_epochs += self.fine_tuning_epochs

    def get_metric_file_path(self, split):
        if self.model_name_version == 3:
            base = f"{split}-scib"
        elif self.model_name_version >= 3.5:
            base = f"{split}-metrics"
        if self.finetuning:
            base = f"{base}-semi-supervised"
        filename = f"{base}.csv"
        return os.path.join(self.get_dump_path(), filename)

    def check_scib_metrics_exist(self):
        path = self.get_metric_file_path("ref")
        if os.path.exists(path):
            print(path, " exists")
            return True

        name = "semi-supervised" if self.finetuning else ""
        return any(
            name in file and file.endswith(".csv")
            for _, _, files in os.walk(self.get_dump_path())
            for file in files
        )

    def create_dump_path(self):
        if self.wandb_sweep == 1:
            return
        self.dump_path = self.get_dump_path()
        # if not os.path.exists(self.dump_path):
        os.makedirs(self.dump_path, exist_ok=True)

    def create_temp_res_path(self):
        if self.wandb_sweep == 1:
            return
        temp_res_path = self.get_temp_res_path()
        if self.save_temp_res == 1 and not os.path.exists(temp_res_path):
            os.makedirs(temp_res_path)

    def set_experiment_name(self):
        if self.experiment_name == "":
            self.experiment_name = None
        if self.experiment_name is not None:
            return
        self.experiment_name = f"swav"

    def generate_name_based_on_changes(self):
        return generate_model_name(get_defaults().copy(), self.params)

    def get_model_name(self):
        return self.generate_name_based_on_changes()

    def get_save_dir(self):
        if self.training_type == "transfer_learning":
            return f"{MODEL_DIR}/{self.pretrain_dataset_id}_{self.finetune_dataset_id}/"
        return f"{MODEL_DIR}/{self.dataset_id}/"

    def get_temp_res_path(self):
        return f"{self.temp_res_path}/{self.get_model_name()}/"

    def get_dump_path(self):
        name = self.get_model_name()
        save_dir = self.get_save_dir()
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        return f"{save_dir}{name}"

    def get_model_path(self):
        return self.get_dump_path() + ".pth"

    def get_abbreviation(self, key):

        if self.model_name_version < 4:
            return key

        if key in ABBREVIATIONS:
            return ABBREVIATIONS[key]
        return key

    def prepare_scpoli_dataloader(self, adata, scpoli_cvae, shuffle=True):
        if "condition_combined" not in adata.obs:
            adata.obs["conditions_combined"] = adata.obs[[self.condition_key]].apply(
                lambda x: "_".join(x), axis=1
            )
        dataset = MultiConditionAnnotatedDataset(
            adata,
            condition_keys=[self.condition_key],
            condition_encoders=scpoli_cvae.condition_encoders,
            conditions_combined_encoder=scpoli_cvae.conditions_combined_encoder,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=scpoli_utils.custom_collate,
            shuffle=shuffle,
        )
        return loader
