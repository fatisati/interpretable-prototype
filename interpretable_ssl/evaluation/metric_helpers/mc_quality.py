from interpretable_ssl.evaluation.metric_helpers.metacell_metrics import *
import sys
import scanpy as sc

if __name__ == "__main__":
    ad_path, bk, label_key, save_dir, model_name = sys.argv[1:6]
    ad = sc.read_h5ad(ad_path)

    try:
        avg_metrics, _ = avg_mc_quality_metrics(ad, bk, label_key)
    except Exception as e:
        print("avg_mc_quality_metrics failed:", e)
    summary = {}

    for col in avg_metrics.columns:
        vals = pd.to_numeric(avg_metrics[col], errors="coerce").dropna().values
        if len(vals) == 0:
            continue

        center = np.mean(vals) if "purity" in col else np.median(vals)
        q25, q75 = np.percentile(vals, [25, 75])
        iqr = q75 - q25

        summary[f"{col}_center"] = round(center, 3)
        summary[f"{col}_summary"] = f"{center:.3f} ± {iqr:.3f}"

    summary_df = pd.DataFrame([summary]) 
    summary_df.index = [model_name]
    summary_df.to_csv(save_dir + '/mc_quality_summary.csv')
    
    avg_metrics.to_csv(save_dir + f'/mc_quality.csv')
