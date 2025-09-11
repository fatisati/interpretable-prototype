import torch
from sklearn.preprocessing import LabelEncoder
import pickle as pkl
from torch.utils.data import random_split
import scanpy as sc


import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)


def log_time(class_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Log start time
            start_time = time.time()
            logging.info(f"Starting '__init__' of class '{class_name}'")

            # Execute the function
            result = func(*args, **kwargs)

            # Log end time and duration
            end_time = time.time()
            duration = end_time - start_time
            logging.info(
                f"Finished '__init__' of class '{class_name}' in {duration:.4f} seconds"
            )

            return result

        return wrapper

    return decorator


# @log_time('get device')
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_home():
    return "/home/icb/fatemehs.hashemig/"


def save_model_checkpoint(model, epoch, save_path):
    print(f"saving model at {save_path}")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
        },
        save_path,
    )


def save_model(model, path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def get_pancras_model_dir():
    return get_home() + "/models/pancras/"


def fit_label_encoder(adata, save_path):

    # fit label encoder
    le = LabelEncoder()
    le.fit(adata.obs["cell_type"])

    # save it
    pkl.dump(le, open(save_path, "wb"))
    return le


def get_model_dir():
    return get_home() + "models/"


def sample_dataset(dataset, sample_ratio):
    sample, _ = random_split(
        dataset,
        [sample_ratio, 1 - sample_ratio],
        generator=torch.Generator().manual_seed(42),
    )
    return sample


def plot_umap(adata, rep):
    sc.pp.neighbors(adata, use_rep=rep)
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=["cell_type"])


def tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def add_prefix_key(dict, prefix):
    new_dict = {}
    for key in dict:
        new_dict[f"{prefix}_{key}"] = dict[key]
    return new_dict


def reshape_and_reorder_dict(data_dict):
    """
    Reshape and reorder the tensors in the dictionary.
    Handles tensors with different shapes by applying reshaping accordingly.
    """
    reshaped_dict = {}

    for key, tensor in data_dict.items():
        # Store the reshaped tensor in the dictionary
        reshaped_dict[key] = reshape_and_reorder_tensor(tensor)
    return reshaped_dict


def reshape_and_reorder_tensor(tensor):
    batch_size, num_augmentations = tensor.shape[:2]
    feature_dims = tensor.shape[2:]

    # Permute the tensor to bring augmentations to the first dimension
    permuted_tensor = tensor.permute(1, 0, *range(2, len(tensor.shape)))

    # Reshape to combine the augmentation and batch dimensions
    reshaped_tensor = permuted_tensor.reshape(
        num_augmentations * batch_size, *feature_dims
    )
    return reshaped_tensor
