from constants import *
from interpretable_ssl.configs.defaults import *
from experiment_runner import *

if __name__ == "__main__":
    print("runner started...")
    runner = ExperimentRunner("swav_template.sbatch")
    full_loss = {
        "cvae_loss_scaler": [0.01],
        "propagation_reg": [1],
        "experiment_name": ["t2"],
    }
    scpoli_base = {"model": ["scpoli"]}
    imm_1ds = {"study_id": ["10X"], "num_prototypes": [150]}
    panc_1ds = {
        "dataset_id": ["pancreas"],
        "num_prototypes": [50],
        "study_id": ["inDrop3"],
        "batch_size": [128],
    }

    cd34 = {"dataset_id": ["cd34"], "num_prototypes": [90], "batch_size": [128]}
    experiments = [
        full_loss
        # scpoli_base | imm_1ds,
        # scpoli_base | panc_1ds,
        # scpoli_base | cd34, 
        # scpoli_base | {"dataset_id": ["pancreas", "pbmc-immune"]},
        
        # full_loss | cd34,
        # full_loss.copy() | {"study_id": ["10X"], "num_prototypes": [150]},
        # full_loss.copy() | {"dataset_id": ["pancreas"], 'num_prototypes': [50], 'study_id': ['inDrop3'], 'batch_size': [128]},
        # full_loss.copy() | {"dataset_id": ["pancreas"], 'num_prototypes': [220], 'batch_size': [128]},
        # | {
        #     "assignment_metric": [
        #         "dotp",  # raw linear projection
        #         "pcos",  # (cosine similarity + 1)/2
        #         "cos",  # cosine similarity in [-1,1]
        #         "neuc",  # negative Euclidean distance
        #     ]
        # base.copy() | {"hard_clustering": [0, 1]},
    ]

    evaluate_job_count(experiments)
    for item_to_test in experiments:
        runner.run_multiple_experiments(item_to_test.copy(), True)
