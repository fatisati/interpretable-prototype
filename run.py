from constants import *
from interpretable_ssl.configs.defaults import *
from experiment_runner import *

if __name__ == "__main__":
    print("runner started...")
    runner = ExperimentRunner("swav_template.sbatch")
    experiments = [
        {
            "experiment_name": ['multi-ds'],
            "cvae_loss_scaler": [0.0],
            "propagation_reg": [0],
        },
        {
            "experiment_name": ['multi-ds'],
        },
    ]

    evaluate_job_count(experiments)
    for item_to_test in experiments:
        runner.run_multiple_experiments(item_to_test, True)
