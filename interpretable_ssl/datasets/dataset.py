import scanpy as sc
from torch.utils.data import Dataset
import torch
from sklearn.model_selection import train_test_split
import interpretable_ssl.utils as utils
import pickle as pkl
import inspect
from interpretable_ssl.utils import log_time
import os
import itertools


class SingleCellDataset(Dataset):

    def __init__(
        self,
        name,
        adata=None,
        label_encoder_path=None,
        original_idx=None,
        fold=0,
        test_study_cnt=2,
        batch_key = 'study'
    ):
        # self.device = utils.get_device()
        self.name = name
        if not adata:
            self.adata = self.read_adata()
        else:
            self.adata = adata
        self.label_encoder_path = label_encoder_path
        self.le = self.load_label_encoder()

        self.num_classes = len(set(self.adata.obs["cell_type"].cat.categories))
        self.x_dim = self.adata[0].X.shape[1]

        # Store the initialization arguments
        self.init_args = {
            "name": name,
            "adata": adata,
            "label_encoder_path": label_encoder_path,
        }

        self.original_idx = original_idx
        if self.original_idx is None:
            self.original_idx = list(range(len(self.adata)))
        self.fold = fold
        self.study_list = None
        self.test_study_cnt = test_study_cnt
        self.batch_key = batch_key

    def __str__(self) -> str:
        return self.name

    def get_data_path(self):
        pass

    def read_adata(self):
        data_path = self.get_data_path()
        print(f"loading {str(self)} data")
        data = sc.read_h5ad(data_path)
        print("done")
        return data

    def load_label_encoder(self):
        if os.path.exists(self.label_encoder_path):
            return pkl.load(open(self.label_encoder_path, "rb"))
        else:
            return utils.fit_label_encoder(self.adata, self.label_encoder_path)

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        x = self.get_x(idx).squeeze(0)
        y = self.get_y(idx).squeeze(0)
        return x, y

    def get_unique_studies(self):
        unique_studies = self.adata.obs.study.unique()
        return unique_studies

    def get_x(self, i):
        x = self.adata[i].X.toarray()
        return torch.tensor(x)

    def get_y(self, i):
        y = self.le.transform(self.adata[i].obs["cell_type"])
        return torch.tensor(y)

    def split(self, test_size=0.2, random_state=None):
        """Split the dataset into train and test datasets."""
        train_idx, test_idx = train_test_split(
            range(self.adata.n_obs), test_size=test_size, random_state=random_state
        )
        return (
            self._create_split_instance(train_idx),
            self._create_split_instance(test_idx),
        )

    def _create_split_instance(self, indices):
        """Create a new instance of the current class with the given indices of adata."""
        # adata_split = self.adata[indices].copy()
        adata_split = self.adata[indices]

        # Get the signature of the __init__ method of the current class
        init_signature = inspect.signature(self.__class__.__init__)

        # Filter the init_args to include only those that are accepted by the __init__ method
        filtered_args = {
            key: value
            for key, value in self.init_args.items()
            if key in init_signature.parameters
        }

        # Update the adata in the arguments
        filtered_args["adata"] = adata_split
        filtered_args["original_idx"] = indices
        return self.__class__(**filtered_args)

    @log_time("get train test")
    def get_train_test(self):
        test_studies = self.get_test_studies()
        test_idx = self.adata.obs[self.batch_key].isin(test_studies)
        return self._create_split_instance(~test_idx), self._create_split_instance(
            test_idx
        )

    def get_default_studies(self):
        pass

    def set_study_list(self):
        # Get unique values
        unique_studies = list(self.adata.obs.study.unique())

        # Generate all possible combinations of selecting 2
        combinations = list(itertools.combinations(unique_studies, self.test_study_cnt))

        # Convert to a list of lists
        combinations = [list(combo) for combo in combinations]
        # Define the target pair
        target_pair = self.get_default_studies()

        # Sort the list to place target_pair at index 0
        combinations.sort(key=lambda x: x != target_pair)
        return combinations

    def get_test_studies(self):
        if self.fold == 0:
            return self.get_default_studies()
        if self.study_list is None:
            self.study_list = self.set_study_list()
        if self.fold > len(self.study_list):
            raise ValueError(
                f"Fold {self.fold} is greater than the number of possible folds"
            )
        test_studies = self.study_list[self.fold]
        print(f"Test studies: {test_studies}")
        return test_studies
