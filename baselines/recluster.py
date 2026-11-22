"""Re-cluster a baseline's saved embedding without retraining the baseline.

For embedding-output baselines (scVI, Harmony, ComBat): reads
<run-dir>/embedding.csv, clusters to the shared K from the dataset's
cluster_config.json, and rewrites the cluster columns in embedding.csv
plus the clustering fields in config.json. Runs exactly the same code the
training script runs after training (common.cluster_embedding_file).

Not for SEACells (clustering is built into the method) or BBKNN (outputs a
graph, not an embedding).

After this, that baseline's metrics/ folder is stale: re-run
compute_metrics.py. (This script doesn't delete metrics/ itself -- that
folder belongs to the metrics stage, see docs/architecture.md.)

Usage:
    python baselines/recluster.py --run-dir runs/rare_cell_preservation/heart_cell_atlas/scvi
"""
import argparse
from pathlib import Path

from run_record import record_run  # first local import: records the script's start time
from common import cluster_embedding_file

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="baseline run folder containing embedding.csv + config.json")
    args = p.parse_args()
    cluster_embedding_file(args.run_dir)
    record_run(Path(args.run_dir) / "config.json")
    print("[recluster] done -- metrics/ for this run is now stale: re-run baselines/compute_metrics.py")
