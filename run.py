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
        "cvae_epochs": [0],
        # "hard_clustering": [0, 1]
        # 'freeze_prototypes_nepochs': [1, 50, 100]
        # 'epoch_queue_starts': [15],
        # 'queue_length': [4*1024]
    }

    large_batch = hlca | {"experiment_name": ["large-batch"], "batch_size": [4096]}
    small_batch = hlca | {
        "experiment_name": ["small-batch"],
        "batch_size": [256],
        "queue_length": [15 * 256],
        "epoch_queue_starts": [15],
        "base_lr": [0.6],
        "final_lr": [0.0006],
        "warmup_epochs": [0],
    }
    assign_sharpness = [
        {"temperature": [0.2], "epsilon": [0.1, 0.15]},
        # {"temperature": [0.3], "epsilon": [0.15]},
    ]
    affinities = {"affinity_type": ["inverse_dist", "arbf"]}

    base = {
        "affinity_type": ["arbf"],
        # "cell_w_mode": ["sigma", "heterogeneity", "mf_score", "uniform"],
        # "propagation_reg": [0],
    }
    prop = {'propagation_reg': [0],  "cell_w_mode": ["sigma",'uniform']}
    cell_w_modes = {"cell_w_mode": ["heterogeneity", "mf_score"],}
    l2norm = {'experiment_name': ['trvar_v2'], 'epsilon': [0.5], 'temperature': [1.0], 'l2norm': [0], "propagation_reg": [0]}
    experiments = [
        l2norm,
        l2norm | pancreas
        # affinities,
        # affinities | pancreas,
        # base | {'cell_w_mode': ['sigma']},
        # base | {'cell_w_mode': ['sigma']} | pancreas,
    ]

    evaluate_job_count(experiments)
    for item_to_test in experiments:
        runner.run_multiple_experiments(item_to_test.copy(), True)
