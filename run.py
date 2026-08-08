from constants import *
from interpretable_ssl.configs.defaults import *
from experiment_runner import *

if __name__ == "__main__":
    print("runner started...")
    runner = ExperimentRunner("swav_template.sbatch")
    pancreas = {
        "dataset_id": ["pancreas"],
        "num_prototypes": [220],
        # "batch_size": [256],
    }
    hlca = {
        "dataset_id": ["hlca"],
        "num_prototypes": [3000],
        "umap_checkpoint_freq": [10],
        "batch_size": [4096],
    }
    nsc = {
        "dataset_id": ["nsc"],
        "num_prototypes": [3000],
        "batch_size": [4096],
        "umap_checkpoint_freq": [10],
    }
    cd34 = {"dataset_id": ["cd34"], "num_prototypes": [95], "ft_epochs": [0]}
    large_batch = hlca | {"batch_size": [4096]}
    # small_batch = hlca | {
    #     "batch_size": [256],
    #     "queue_length": [15 * 256],
    #     "epoch_queue_starts": [15],
    #     "base_lr": [0.6],
    #     "final_lr": [0.0006],
    #     "warmup_epochs": [0],
    # }

    affinities = {"affinity_type": ["arbf", "coaff", "inverse_dist", "umap"]}
    cwmode = {"cell_w_mode": ["ssigma", "sigma", "wsigma", "mf_score", "uniform"]}

    swav_base = {
        "experiment_name": ["swav_base"],
        # "model_type": ["vqvae", 'gm'],
        "l2norm": [1],
        "lambda_swav": [1.0],
        "lambda_recon": [0.0],
        "lambda_kl": [0.0],
        "lambda_balance": [0],
        "lambda_proto": [0.0],
        "lambda_commit": [0.0],
        "lambda_l2": [0.0],
        "assignment_metric": ["dotp", "ddotp"],
    }
    swav_commit = {
        "experiment_name": ["swav_commit"],
        # "model_type": ["vqvae", 'gm'],
        "l2norm": [1],
        "lambda_swav": [1.0],
        "lambda_recon": [0.0],
        "lambda_kl": [0.0],
        "lambda_balance": [0],
        "lambda_proto": [5.0],
        "lambda_commit": [5.0],
        "lambda_l2": [0.0],
        "assignment_metric": ["dotp"],
    }
    vqvae_swav = {
        "experiment_name": ["vqvae_swav_v2"],
        "model_type": ["vqvae"],
        "l2norm": [1],
        "lambda_swav": [1.0],
        "lambda_recon": [0.01],
        "lambda_kl": [0.0],
        "lambda_balance": [0],
        "lambda_proto": [1.0],
        "lambda_commit": [0.25],
        "lambda_l2": [0.0],
        "assignment_metric": ["ddotp"],
    }

    test = {
        "experiment_name": ["test"],
        "model_type": ["vqvae"],
        "l2norm": [1],
        "lambda_swav": [1.0],
        "lambda_recon": [0.01],
        "lambda_kl": [0.0],
        "lambda_balance": [0],
        "lambda_proto": [1.0],
        "lambda_commit": [0.25],
        "lambda_l2": [0.0],
        "assignment_metric": ["ddotp"],
    }

    vqvae_base = {
        "experiment_name": ["vqvae_base"],
        "model_type": ["vqvae"],
        "l2norm": [1],
        "lambda_swav": [0.0],
        "lambda_recon": [1.0],
        "lambda_kl": [0.0],
        "lambda_balance": [0],
        "lambda_proto": [0.04],
        "lambda_commit": [0.03],
        "lambda_l2": [0.0],
        "recon_loss": ["mse"],
        "normalize_loss": [1],
        "recon_type": ["normal"],
    }

    adam = {
        "opt": ["adam"],
        "wd": [1e-5],
        "base_lr": [3e-4],  # good for finetune
        "final_lr": [1e-5],
        "warmup_epochs": [5],
        "start_warmup": [0.0],
    }

    wadam = {
        "opt": ["wadam"],
        "wd": [1e-5],
        "base_lr": [1e-3, 5e-4],  # good for finetune
        "final_lr": [1e-5],
        "warmup_epochs": [10],
        "start_warmup": [0.0],
    }

    sgd = {
        "wd": [1e-6],
        "base_lr": [0.05],  # good for finetune
        "final_lr": [0.005],
        "warmup_epochs": [10],
        "start_warmup": [0.0],
    }

    gmvae_base = {
        "experiment_name": ["gmvae_base"],
        "model_type": ["gm"],
        "l2norm": [0],
        "lambda_swav": [0.0],
        "lambda_recon": [1.0],
        "lambda_kl": [0.1],
        "lambda_balance": [0],
        "lambda_proto": [0.0],
        "lambda_commit": [0.0],
        "lambda_l2": [0.0],
        "normalize_loss": [1],
        # 'kl_sched': [0],
        # 'cvae_epochs': [5],
    }

    my_swav = {
        "experiment_name": ["gmvae_swav"],
        # "sinkhorn_iterations": [3],
        "lambda_swav": [1.0],
        # "assignment_metric": ['dopt'],
        "div_type": ["ce"],
        "epsilon": [0.5],
        "temperature": [1.0],  # also try 0.3, 0.6
        # 'weighted_kl': [1],
        "affinity_type": ["spatial"],
        # 'l2norm': [1]
    }

    swav_only = {
        "experiment_name": ["swav_original"],
        "lambda_swav": [1.0],
        "div_type": ["ce"],
        "epsilon": [0.05],
        "temperature": [0.1],
        "affinity_type": ["spatial", "coaff"],
        "dataset_id": ["s28nsc"],
        "cvae_epochs": [0],
        # 'umap_metric': ['cosine'],
        "weighted_kl": [0],
        # "assignment_metric": ['dopt'],
        # 'l2norm': [1]
    }
    from interpretable_ssl.datasets.dataset_configs import DATASETS

    ds = {"dataset_id": ["s28nsc"]}  # ['nsc', 'cd34', 'pbmc-immune']
    all_datasets = list(DATASETS.keys())

    gmvae_base = {
        "experiment_name": ["gmvae_base"],
        "cvae_epochs": [50],
        "lambda_kl": [1.0],
        "lambda_recon": [1.0],
        "lambda_swav": [0.0],
    }

    vqvae_base = {
        "experiment_name": ["vqvae_v4"],
        "model_type": ["vqvae"],
        "l2norm": [0, 1],
        "lambda_recon": [1.0],
        "lambda_kl": [0.0],
        "lambda_balance": [0],
        "lambda_proto": [0.05],
        "lambda_commit": [0.1],
        "lambda_l2": [0.0],
        "recon_loss": ["mse"],
    }

    vae_base = {
        "experiment_name": ["scpoli"],
        "cvae_epochs": [50],
        "lambda_kl": [0.1, 0.0],
        "lambda_recon": [1.0],
        "full_dataset_mode": [1],
        "pretraining_epochs": [100],
    }

    pancreas = [".1pancreas", "bmpancreas", "pancreas"]
    immune = ["bmpbmc-immune", ".1pbmc-immune", "pbmc-immune"]
    scproto = {
        "experiment_name": [
            "scproto_v10"
        ],  # normalize + topk simialr, v9: change eps bases, v10: incldeu eps sceduler
        "cvae_epochs": [50],
        "lambda_kl": [0.1],
        "lambda_recon": [1.0],
        "lambda_swav": [1.0],
        "sinkhorn_iterations": [0],
        "lambda_p_uncertainty": [0.1],
        "full_dataset_mode": [1],
        "pretraining_epochs": [50],
        "affinity_type": ["arbf"],
        "adoptive_eps": [1],
    }
    critical_ds = ["bmpbmc-immune", "bmpancreas"]
    all_ds = [".1pbmc-immune", ".1pancreas", "pbmc-immune", "pancreas"]
    experiments = [
        
        {'dataset_id': ['bmpbmc-immune'], 'model': ['seacell'], 'mode': ['eval']},
        scproto
        | {
            "dataset_id": ['bmpbmc-immune'],
            "p": [0.5],
            "lambda_aff": [1.0],
            "affinity_type": ["arbf"],
            "two_sided": [0],
            'mode': ['eval']
        },
        # scproto
        # | {
        #     # "dataset_id": ["pancreas", "0.1pancreas", "0.3pancreas", 'pbmc-immune', 'shlca', 'snsc', 'nsc', 's28nsc', 'cd34', 'hlca'],
        #     'dataset_id': ['.1pbmc-immune', 'bmpbmc-immune', 'pbmc-immune'], #'pancreas', 'bmpancreas', '.1pancreas', 'shlca'],
        #     "full_dataset_mode": [1],
        #     "affinity_type": ["icoaff", "iarbf"],
        #     "pretraining_epochs": [100],
        #     'epsilon': [0.02],
        #     'temperature': [0.04, 0.1]
        # },
    ]

    evaluate_job_count(experiments)
    for item_to_test in experiments:
        runner.run_multiple_experiments(item_to_test.copy(), True)
