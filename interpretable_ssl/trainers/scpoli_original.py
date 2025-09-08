# load data
# define model
# set training parameters
# train loop

from interpretable_ssl.datasets.immune import ImmuneDataset
from scarches.models.scpoli import scPoli
from interpretable_ssl import utils
from interpretable_ssl.trainers.trainer import Trainer
import torch
import wandb
import sys
from interpretable_ssl.trainers.adaptive_trainer import AdoptiveTrainer


class OriginalTrainer(AdoptiveTrainer):
    def __init__(
        self, **kwargs
    ):
        if 'experiment_name' not in kwargs:
            kwargs['experiment_name'] = 'scpoli'
            
        super().__init__(**kwargs)
        print("input data size: ", len(self.ref))
        self.model = self.get_model()

    def get_model(self):

        adata = self.ref.adata
        if self.training_type == "pretrain":
            cell_type_key = None
        else:
            cell_type_key = self.dataset.label_key
        condition_key = self.dataset.batch_key

        return scPoli(
            adata=adata,
            condition_keys=condition_key,
            cell_type_keys=cell_type_key,
            latent_dim=self.latent_dims,
            recon_loss=self.recon_loss,
        )

    def train(self):
        epochs = self.pretraining_epochs + self.fine_tuning_epochs
        self.model.train(
            n_epochs=epochs,
            pretraining_epochs=self.pretraining_epochs,
            eta=5,
        )
        model_path = self.get_model_path()
        utils.save_model_checkpoint(
            self.model.model,
            epochs,
            model_path,
        )
        # self.save_metrics()

    def get_model_path(self):
        return self.get_dump_path() + "/model.pth"

    def load_model(self):
        model = self.get_model()
        path = self.get_model_path()
        model.model.load_state_dict(torch.load(path)["model_state_dict"])
        return model

    def adapt_model(self, ref_model, adata, retrain_epochs=0):
        query_model = self.get_model()
        query_model.model.load_state_dict(ref_model.model.state_dict())
        scpoli_query = scPoli.load_query_data(
            adata=adata,
            reference_model=query_model,
            labeled_indices=[],
        )
        if retrain_epochs > 0:
            scpoli_query.train(n_epochs=retrain_epochs, 
                              pretraining_epochs=retrain_epochs)
        return scpoli_query

    def encode_batch(self, model, batch, return_maped=False, return_mapped_idx=None):
        batch = self.dict_to_device(batch)
        scpoli_model = self.extract_scpoli(model)
        scpoli_model.to(self.device)
        scpoli_model.eval()
        with torch.no_grad():
            x, _, _, _ = scpoli_model(**batch)
        return x

    def extract_scpoli(self, pretrained_model, return_model=True):
        if return_model:
            return pretrained_model.model
        return pretrained_model
