from interpretable_ssl.evaluation.metric_helpers.metacell_metrics import *
import sys
import scanpy as sc

if __name__ == "__main__":
    ad_path, label_key, save_dir, model_name = sys.argv[1:5]
    ad = sc.read_h5ad(ad_path)

    summary_df, _ = mc_quality_metrics(
        ad=ad, cell_type_key=label_key
    )
    summary_df.index = [model_name]
    summary_df.to_csv(save_dir + '/mc_quality.csv')
