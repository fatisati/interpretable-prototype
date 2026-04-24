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
import numpy as np
import scipy

class SingleCellDataset(Dataset):

    def __init__(
        self,
        name,
        path=None,
        test_studies=None,
        adata=None,
        label_encoder_path=None,
        fold=0,
        test_study_cnt=2,
        batch_key=None,
        label_key="cell_type",
        niche_key=None,
        use_counts=False,
        **kwargs
    ):
        # self.device = utils.get_device()
        self.use_counts = use_counts
        self.name = name
        self.batch_key = batch_key

        self.label_key = label_key
        self.niche_key = niche_key
        self.path = path
        self.test_studies = test_studies
        if adata is None:
            self.adata = self.read_adata()
        else:
            self.adata = adata
        self.label_encoder_path = label_encoder_path
        self.le = self.load_label_encoder()

        self.num_classes = len(set(self.adata.obs[self.label_key].cat.categories))
        self.x_dim = self.adata[0].X.shape[1]

        # Store the initialization arguments
        self.init_args = {
            "name": name,
            "adata": adata,
            "label_encoder_path": label_encoder_path,
        }

        self.fold = fold
        self.study_list = None
        self.test_study_cnt = test_study_cnt

    def get_dc_path(self):
        return f'./dc/{self.name}{len(self.adata)}.csv'
    
    def __str__(self) -> str:
        return self.name

    def requires_hvg(self):
        if "highly_variable" in self.adata.var.columns:
            n_hvg = self.adata.var["highly_variable"].sum()
            if n_hvg == self.adata.n_vars:
                print(f"✅ Already subsetted to HVGs ({n_hvg} genes).")
                return False
            else:
                print(
                    f"ℹ️ Found {n_hvg} HVGs out of {self.adata.n_vars} total genes."
                )
                return True
        else:
            print("⚠️ No HVG column found.")
            return False

    def read_adata(self):
        print(f"loading {str(self)} data")
        self.adata = sc.read_h5ad(self.path)

        if self.batch_key is None:
            self.batch_key = 'batch'
            self.adata.obs['batch'] = ['b0'] * len(self.adata)
        
        if self.adata.X.max() < 30:
            if 'lognorm' not in self.adata.layers:
                self.adata.layers["lognorm"] = self.adata.X.copy()
            if self.use_counts:
                self.adata.X = self.adata.layers.get('counts', self.adata.X)
        # data is not normalized
        else:
            # copy to calc lognorm data
            ad = self.adata.copy()
            sc.pp.normalize_total(ad, target_sum=1e4)
            sc.pp.log1p(ad)
            # this should be false, only in dropout setup, which we want to keep correct lognorm values
            if 'lognorm' not in self.adata.layers:
                self.adata.layers["lognorm"] = ad.X.copy()
            # if we do not want raw counts, update adata.X
            if not self.use_counts: 
                self.adata.X = ad.X
            
        # check if hvg not applied apply it
        if self.requires_hvg():
            self.adata.raw = self.adata.copy()
            self.adata = self.adata[:, self.adata.var["highly_variable"].values].copy()
            
        if not (scipy.sparse.isspmatrix_csr(self.adata.X) and self.adata.X.dtype == np.float32) and type(self.adata.X) != np.ndarray:
            self.adata.X = self.adata.X.tocsr().astype(np.float32)
        return self.adata

    def load_label_encoder(self):
        if self.label_encoder_path and os.path.exists(self.label_encoder_path):
            return pkl.load(open(self.label_encoder_path, "rb"))
        else:
            return utils.fit_label_encoder(self.adata, self.label_encoder_path, self.label_key)

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        x = self.get_x(idx).squeeze(0)
        y = self.get_y(idx).squeeze(0)
        return x, y

    def get_x(self, i):
        x = self.adata[i].X.toarray()
        return torch.tensor(x)

    def get_y(self, i):
        y = self.le.transform(self.adata[i].obs[self.label_key])
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

    def get_init_args(self):
        sig = inspect.signature(self.__init__)
        return {p: getattr(self, p) for p in sig.parameters if p not in ["self", 'kwargs']}

    def _create_split_instance(self, indices):
        """Create a new instance of the current class with the given indices of adata."""
        # adata_split = self.adata[indices].copy()
        adata_split = self.adata[indices]

        # Get the signature of the __init__ method of the current class
        init_dict = self.get_init_args()

        init_dict["adata"] = adata_split
        return self.__class__(**init_dict)

    # @log_time("get train test")
    def get_train_test(self):
        if self.batch_key is None or (self.adata.obs[self.batch_key].nunique() == 1):
            print("1 batch dataset")
            return self, None
        test_studies = self.get_fold_test_studies()
        test_idx = self.adata.obs[self.batch_key].isin(test_studies)
        return self._create_split_instance(~test_idx), self._create_split_instance(
            test_idx
        )

    def set_study_list(self):

        # Get unique values
        unique_studies = list(self.adata.obs[self.batch_key].unique())

        # Generate all possible combinations of selecting 2
        combinations = list(itertools.combinations(unique_studies, self.test_study_cnt))

        # Convert to a list of lists
        combinations = [list(combo) for combo in combinations]
        # Define the target pair
        target_pair = self.test_studies

        # Sort the list to place target_pair at index 0
        combinations.sort(key=lambda x: x != target_pair)
        return combinations

    def get_fold_test_studies(self):
        if self.fold == 0:
            return self.test_studies
        if self.study_list is None:
            self.study_list = self.set_study_list()
        if self.fold > len(self.study_list):
            raise ValueError(
                f"Fold {self.fold} is greater than the number of possible folds"
            )
        test_studies = self.study_list[self.fold]
        print(f"Test studies: {test_studies}")
        return test_studies
