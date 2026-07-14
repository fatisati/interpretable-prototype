"""
Central configuration for all paths.
Set these environment variables before running on Colab:

    import os
    os.environ['HOME_DIR'] = '/content/drive/MyDrive/'
    os.environ['MODEL_DIR'] = '/content/drive/MyDrive/models/'
    os.environ['DATA_DIR'] = '/content/drive/MyDrive/data/'
    os.environ['CODE_DIR'] = '/content/drive/MyDrive/codes/interpretable-prototype/'

Or modify the defaults below directly.
"""
import os

# Base directories - set via environment variables or modify defaults
HOME_DIR = os.environ.get('HOME_DIR', '/home/icb/fatemehs.hashemig/')
MODEL_DIR = os.environ.get('MODEL_DIR', os.path.join(HOME_DIR, 'models/'))
DATA_DIR = os.environ.get('DATA_DIR', os.path.join(HOME_DIR, 'data/'))
CODE_DIR = os.environ.get('CODE_DIR', os.path.join(HOME_DIR, 'codes/interpretable-prototype/'))

# External libraries
ISLANDER_SRC = os.environ.get('ISLANDER_SRC', os.path.join(HOME_DIR, 'Islander/src'))


def get_home():
    return HOME_DIR


def get_model_dir():
    return MODEL_DIR


def get_data_dir():
    return DATA_DIR


def get_code_dir():
    return CODE_DIR


def get_seacell_model_dir(ds_id, build_kernel_on="X_pca"):
    dir_name = "seacell" if build_kernel_on == "X_pca" else f"seacell_{build_kernel_on}"
    return os.path.join(MODEL_DIR, ds_id, dir_name)


def get_sure_model_dir(ds_id):
    return os.path.join(MODEL_DIR, ds_id, "sure")


def get_metaq_model_dir(ds_id):
    return os.path.join(MODEL_DIR, ds_id, "metaq")


def get_affinity_path(
    ds_name,
    n_cells,
    n_components=50,
    k_neighbors=50,
    affinity_type='arbf',
    graph_dir='./graphs',
    spatial=False,
    fold=0,
    graph_mode='knn',
):
    name = f"affinity_{ds_name}{n_cells}_ncomp{n_components}_kneighbors{k_neighbors}_{affinity_type}"
    if spatial:
        name += "_spatial"
    if fold != 0:
        name += f"_fold{fold}"
    if graph_mode != 'knn':
        name += f"_{graph_mode}"
    name += ".pkl"
    return os.path.join(graph_dir, name)


def get_dataset_model_dir(ds_id):
    return os.path.join(MODEL_DIR, ds_id)
