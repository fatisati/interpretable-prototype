from interpretable_ssl.evaluation.mc_metric_utils import *
import sys
import scanpy as sc



if __name__ == "__main__":
    ad_path, bk, label_key, save_dir, model_name = sys.argv[1:6]
    ad = sc.read_h5ad(ad_path)
    summarize_metacell_quality(ad, bk, label_key, save_dir, model_name)
