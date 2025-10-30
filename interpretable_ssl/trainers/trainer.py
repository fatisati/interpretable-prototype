from interpretable_ssl.models.scpoli import *
from interpretable_ssl.loss_manager import *
from scarches.models.scpoli import scPoli
import numpy as np
from interpretable_ssl.train_utils import *
from interpretable_ssl.configs.defaults import *
from scarches.dataset.scpoli.anndata import MultiConditionAnnotatedDataset
import os
from interpretable_ssl.evaluation.visualization import *
from tqdm import tqdm
from interpretable_ssl.evaluation.metrics import MetricCalculator
from torch.utils.data import DataLoader
from interpretable_ssl.utils import log_time
from interpretable_ssl.evaluation.cd4_marker import assign_prototype_labels
from interpretable_ssl.datasets.dataset import SingleCellDataset
from interpretable_ssl.datasets.dataset_configs import DATASETS
from interpretable_ssl.trainers.base import TrainerBase
import wandb

from interpretable_ssl.evaluation.metric_helpers.embedding_metrics import *
from interpretable_ssl.trainers.affinity import *
import subprocess
from sklearn.model_selection import train_test_split

from interpretable_ssl.evaluation.de_helper import *


class Trainer(TrainerBase):
    # @log_time('scpoli trainer')
    def __init__(self, dataset=None, ref_query=None, parser=None, **kwargs) -> None:
        parser_args = self.collect_parser_args(parser)
        kwargs.update(parser_args)
        kwargs.update(DATASETS[kwargs['dataset_id']])
        self.dataset = dataset
        if "debug" not in kwargs:
            kwargs["debug"] = 0
        super().__init__(**kwargs)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.dataset is None:
            print(f"dataset is None, loading {self.dataset_id}")
            self.dataset = self.get_dataset(self.dataset_id)
        self.input_dim = self.dataset.x_dim
        if ref_query is None:
            self.ref, self.query = self.dataset.get_train_test()
        else:
            self.ref, self.query = ref_query

        train_ind, val_ind = train_test_split(
            range(len(self.ref)), test_size=0.1, random_state=42
        )
        self.train_, self.val_ = self.ref._create_split_instance(
            train_ind
        ), self.ref._create_split_instance(val_ind)
        self.condition_key = self.ref.batch_key
        if self.study_id != "":
            mask = self.ref.adata.obs[self.condition_key] == self.study_id
            self.ref.adata = self.ref.adata[mask].copy()

    def collect_parser_args(self, parser):
        if parser is not None:
            parser = self.add_parser_args(parser)
            args = parser.parse_args()
            args_dict = vars(args)

            # Remove keys from args_dict if their value is the string "None"
            keys_to_remove = [
                key for key, value in args_dict.items() if value == "None"
            ]
            for key in keys_to_remove:
                del args_dict[key]

            return args_dict
        return {}

    def add_parser_args(self, parser):
        default_values = get_defaults().copy()
        # Add arguments to parser with default values from dictionary
        for key, value in default_values.items():
            if isinstance(value, bool):
                # Handle boolean arguments with action='store_true'
                parser.add_argument(
                    f"--{key}",
                    action="store_true",
                    help=f"Set {key} to true (default is {value})",
                )
            else:
                # Handle other types of arguments
                arg_type = type(value) if value is not None else str
                if value == "":
                    value = None
                parser.add_argument(
                    f"--{key}",
                    type=arg_type,
                    default=value,
                    help=f"Set {key} (default is {value})",
                )
        return parser

    def load_model(self):
        model = self.get_model()
        path = self.get_model_path()
        model.load_state_dict(torch.load(path)["model_state_dict"])
        return model

    def load_query_model(self, adata=None):
        if adata is None:
            adata = self.query.adata
        model = self.load_model()
        model = self.adapt_model(model, adata)
        model.to(self.device)
        return model

    def dict_to_device(self, inputs_dict):
        for key in inputs_dict:
            inputs_dict[key] = inputs_dict[key].to(self.device)
        return inputs_dict

    def encode_adata(
        self,
        adata,
        model=None,
        return_mapped=False,
        return_mapped_idx=False,
        retrain_epochs=0,
        z_idx=0,
    ):
        model = self.adapt_model(model, adata, retrain_epochs)
        loader = self.prepare_scpoli_dataloader(
            adata, self.extract_scpoli(model), shuffle=False
        )
        embeddings = [
            self.encode_batch(model, batch, z_idx, return_mapped, return_mapped_idx)
            for batch in tqdm(loader)
        ]
        z = torch.cat(embeddings)
        # if getattr(model, "l2norm", False):
        #     z = F.normalize(z, dim=1)
        return z

    def adapt_model(self, model, adata, retrain_epochs=0):
        adapted_model = model
        if self.check_conditions_compatible(model, adata):
            if retrain_epochs == 0:
                return adapted_model
            else:
                adopted_wrapper = self.extract_scpoli(adapted_model, True)
        else:
            # make a new model not change the old one
            adapted_model = self.get_model()
            # because our model is not just an scpoli_cvae
            adapted_model.load_state_dict(model.state_dict())
            adopted_wrapper = scPoli.load_query_data(
                adata=adata,
                reference_model=self.extract_scpoli(adapted_model, True),
                labeled_indices=[],
            )
        if retrain_epochs > 0:
            adopted_wrapper.train(
                n_epochs=retrain_epochs, pretraining_epochs=retrain_epochs
            )

        adapted_model.attach_scpoli(adopted_wrapper)
        adapted_model.to(self.device)
        return adapted_model

    def prepare_scpoli_dataloader(self, adata, scpoli_cvae, shuffle=True):
        # adata = adata.copy()
        # because scpoli encoder gets raw counts as input
        # adata.X = adata.layers.get("counts", adata.X)

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

    # TODO: return mapped and mapped_idx should have cleaner logic
    def encode_batch(
        self, model, batch, return_idx=0, return_mapped_idx=False, return_mapped=False
    ):

        batch = self.dict_to_device(batch)
        model.eval()

        with torch.no_grad():
            outs = model.encode(batch)
            z_swav, z_vae, logits = outs

        if return_mapped_idx:
            return torch.argmax(logits, dim=1)

        elif return_mapped:
            return logits

        else:
            return outs[return_idx]

    def extract_scpoli(self, scproto_model, return_wrapper=False):
        if return_wrapper:
            return scproto_model.scpoli_wrapper
        return scproto_model.scpoli_cvae

    def encode_ref(self, model=None):
        return self.encode_adata(self.ref.adata, model)

    def encode_query(self, ref_model=None):
        if ref_model is None:
            model = self.load_query_model()
        else:
            model = self.adapt_model(ref_model, self.query.adata)
        return self.encode_adata(self.query.adata, model)

    def check_conditions_compatible(self, model, adata):
        for key, values in self.extract_scpoli(
            model, return_wrapper=True
        ).conditions_.items():
            data_values = adata.obs[key].unique()
            is_subset = set(data_values).issubset(values)
            if not is_subset:
                return False
        return True

    def get_model_prototypes(self, model):
        return None

    def get_umap_path(self, data_part="ref"):
        pass

    def get_proto_assignments(self, z, model):
        scores = model.proto_soft_assignments(z)
        return scores.detach().cpu().numpy()

    def plot_umap(self, model, adata, split, save_plot=True, use_knn=True):
        z = self.encode_adata(adata, model, z_idx=1)
        prototypes = self.get_model_prototypes(model)
        z_umap, prototype_umap = calculate_umap(z, prototypes)
        obs = adata.obs
        if prototypes is not None:
            # prototype_assignments = self.encode_adata(adata, model, True, False)
            scores = self.get_proto_assignments(z, model)
            proto_df = assign_prototype_labels(
                adata,
                scores,
                self.num_prototypes,
                cell_type_column=self.dataset.label_key,
                use_knn=use_knn,
            )
            proto_labels = proto_df.prototype_label
        else:
            proto_labels = None
        if self.cell_w_mode != "uniform":
            w = adata.obs.get(self.cell_w_mode, None)
            w_label = self.cell_w_mode
        else:
            w = adata.obs.get("sigma", None)
            w_label = "pca_sigma"
        return plot_3umaps(
            z_umap,
            prototype_umap,
            obs[self.dataset.label_key],
            obs[self.ref.batch_key],
            proto_labels,
            save_plot,
            self.get_umap_path(split),
            w=w,
            w_label=w_label,
        )

    def plot_ref_umap(self, save_plot=True, name_postfix=None, model=None):

        if model is None:
            model = self.load_model()
        if name_postfix is not None:
            name = f"ref-{name_postfix}"
        else:
            name = f"ref"
        return self.plot_umap(model, self.ref.adata, name, save_plot)

    def plot_query_umap(self, save_plot=True):
        model = self.load_query_model()
        return self.plot_umap(model, self.query.adata, "query", save_plot)

    def calc_scib(self, adata, name, other={}, save=True, is_ref=False):
        latent = self.encode_adata(adata, self.model)
        res_df = MetricCalculator(
            adata,
            [latent],
            self.dump_path,
            save_path=self.get_metric_file_path(name),
        ).calculate(other, save)
        return res_df

    def save_metacell_metrics(self):
        ad = self.train_.adata.copy()
        mc_adata, sim = get_scproto_mc_adata(
            self,
            ad,
            self.dataset.batch_key,
            self.dataset.label_key,
        )
        ad.X = ad.layers['lognorm']
        mc_scg, mc_scb = get_metacell_metrics(
            ad,
            mc_adata,
            [f"{self.get_model_name()}_mc_pca", f"{self.get_model_name()}_mc_proto"],
            self.dataset.batch_key,
            self.dataset.label_key,
        )
        save_append(mc_scg, self.get_dump_path(), "scgraph")
        if mc_scb is not None:
            save_append(mc_scb, self.get_dump_path(), "scib")
        ad.obs["SEACell"] = sim.argmax(axis=1)
        tmp_path = f"tmp_{uuid.uuid4().hex[:8]}.h5ad"
        ad.write(tmp_path)
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "interpretable_ssl.evaluation.metric_helpers.mc_quality",
                tmp_path,
                self.dataset.batch_key,
                self.dataset.label_key,
                self.get_dump_path(),
                self.get_model_name(),
            ]
        )
        process.wait()  # ✅ wait for the subprocess to finish
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            print("Deleted:", tmp_path)
        
        get_mc_jaccard(
            mc_adata,
            self.dataset.adata,
            self.dataset.label_key,
            self.dataset.batch_key,
            0.05,
            self.get_model_name(),
        ).to_csv(self.get_dump_path() + "/de_jaccard_all.csv")
        
        get_mc_jaccard(
            mc_adata,
            self.ref.adata,
            self.dataset.label_key,
            self.dataset.batch_key,
            0.05,
            self.get_model_name(),
        ).to_csv(self.get_dump_path() + "/de_jaccard_ref.csv")

    def save_metrics(self):
        adata = add_trainer_emb(self, self.dataset.adata)
        if adata.X.max() > 50:
            adata = adata.copy()
            adata.X = adata.layers['lognorm']
        params = (
            adata,
            [self.get_model_name()],
            self.dataset.batch_key,
            self.dataset.label_key,
        )
        scg, scb = get_metrics(*params)
        scg.to_csv(self.get_dump_path() + "/scgraph.csv")
        if scb is not None:
            scb.to_csv(self.get_dump_path() + "/scib.csv")
        self.save_metacell_metrics()

    def get_dataset(self, dataset_id):
        ds_params = DATASETS[dataset_id]
        return SingleCellDataset(name=dataset_id, **ds_params)

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
                "test size": len(self.query) if self.query is not None else 0,
                "batch size": self.batch_size,
            },
        )

    def set_job_name(self, path=None):
        if path is None:
            path = self.get_dump_path()
        set_job_name = (self.job_name is None) or (self.job_name == "")
        if set_job_name:
            self.job_name = f"{self.get_model_name()}/{self.dataset}"
