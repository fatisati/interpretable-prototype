import torch
import numpy as np
import pandas as pd
from copy import deepcopy
from scib_metrics.benchmark import Benchmarker
from interpretable_ssl.models.linear import *
from interpretable_ssl.evaluation.knn import *

import sys

from interpretable_ssl.configs.paths import ISLANDER_SRC
sys.path.append(ISLANDER_SRC)
from scGraph import *
import os

class MetricCalculator:
    def __init__(self, input_adata, latents, dump_path, keys=["latent"], save_path=None) -> None:

        self.batch_key = "batch"
        self.label_key = "cell_type"
        self.save_path = save_path
        self.dump_path = dump_path
        self.keys = keys
        self.adata = self.prepare_adata(input_adata, latents)
        # self.latents = latents

    def prepare_adata(self, input_adata, latents):

        # Create a deep copy of the input AnnData object
        adata = input_adata
        for key, latent in zip(self.keys, latents):
            # print(key)
            # Convert the latent tensor to a numpy array if it's a PyTorch tensor
            if isinstance(latent, torch.Tensor):
                latent = latent.detach().cpu().numpy()
                # Store the latent embeddings in the AnnData object
            adata.obsm[key] = latent
        adata = self.remove_single_sample_celltypes(adata)
        return adata

    def remove_single_sample_celltypes(self, adata):
        """
        Removes cells from AnnData where the cell type has only one sample.

        Parameters:
        - adata: AnnData object
        - celltype_column: The column name in adata.obs that contains cell type information (default: 'cell_type')

        Returns:
        - adata_filtered: AnnData object with filtered cells
        """
        # Count the occurrences of each cell type
        celltype_counts = adata.obs[self.label_key].value_counts()

        # Identify cell types that have more than one sample
        valid_celltypes = celltype_counts[celltype_counts > 1].index

        # Filter adata to keep only cells with valid cell types
        adata_filtered = adata[adata.obs[self.label_key].isin(valid_celltypes)].copy()
        
        return adata_filtered

    def calculate_scib(self):
        # Initialize the Benchmarker
        benchmarker = Benchmarker(
            self.adata,
            batch_key=self.batch_key,
            label_key=self.label_key,
            embedding_obsm_keys=self.keys,
        )

        # Perform the benchmark
        benchmarker.benchmark()
        # Get the results as a dictionary
        results_df = benchmarker.get_results(min_max_scale=False)
        return results_df

    def check_duplicate_category_adata(self, adata):
        for col in adata.obs.columns:
            if adata.obs[col].dtype.name == "category":
                print(col, adata.obs[col].cat.categories.duplicated().sum())

    def calculate_scgraph(self):
        
        adata_tmp_path = os.path.join(self.dump_path, "adata_tmp.h5ad")
        self.check_duplicate_category_adata(self.adata)
        self.adata.write(adata_tmp_path)

        scgraph = scGraph(
            adata_path=adata_tmp_path,
            batch_key="batch",
            label_key="cell_type",
            hvg=False,
            trim_rate=0.05,
            thres_batch=100,
            thres_celltype=10,
        )
        return scgraph.main(_obsm_list=self.keys)

    def extract_f1_score(self, df, key, prefix=""):
        return {
            f"{prefix} macro": df.loc[df["Class"] == "micro", "F1 Score"].values[0],
            f"{prefix} micro": df.loc[df["Class"] == "macro", "F1 Score"].values[0],
            f"{prefix} weighted": df.loc[df["Class"] == "weighted", "F1 Score"].values[
                0
            ],
            "model": key,
        }

    def linear_results(self, epochs=100):
        def get_linear_results(embedding, key):
            classifier = LinearClassifier(
                embedding, self.adata.obs[self.label_key], batch_size=128, epochs=epochs
            )
            classifier.train()
            df, _ = classifier.evaluate()

            return self.extract_f1_score(df, key, 'linear classifier f1')

        return self.get_results(get_linear_results)

    def knn_results(self):
        def get_knn_results(emb, key):
            scores = knn_classifier_with_f1_report(emb, self.adata.obs[self.label_key])
            return self.extract_f1_score(scores, key, "knn f1")

        return self.get_results(get_knn_results)

    def get_results(self, get_res_func):
        # score_list = [
        #     get_res_func(emb, key) for emb, key in zip(self.adata.obsm, self.keys)
        # ]

        score_list = []
        
        for key in self.keys:
            emb = self.adata.obsm[key]
            score_list.append(get_res_func(emb, key))
            
        # Create a DataFrame from the list of F1 scores
        result_df = pd.DataFrame(score_list)

        # Set the model name as the index
        result_df.set_index("model", inplace=True)

        return result_df

    def save(self, results):
        # Save the results to a CSV file if save_path is provided
        if self.save_path is not None:
            results.to_csv(self.save_path, index=False)
            print(f"Results saved to {self.save_path}")

    def concat_results(self, scib_res, scg_res):
        scib_clean = scib_res.loc[scib_res.index != "Metric Type"]
        scib_clean = scib_clean.rename(columns={"Total": "scib total"})
        result = pd.concat([scib_clean, scg_res], axis=1, join="inner")
        return result

    def calculate(self, other_metrics={}, save=True):
        scib_res = self.calculate_scib()
        scgraph_res = self.calculate_scgraph()
        final_res = self.concat_results(scib_res, scgraph_res)
        for key, val in other_metrics.items():
            final_res[key] = val
        if save:
            self.save(final_res)
        return final_res


def calculate_scib_metrics_using_benchmarker(
    input_adata, latent, save_path=None, batch_key="batch", label_key="cell_type"
):
    # Convert the latent tensor to a numpy array if it's a PyTorch tensor
    if isinstance(latent, torch.Tensor):
        latent = latent.detach().cpu().numpy()

    # Create a deep copy of the input AnnData object
    adata = deepcopy(input_adata)

    # Store the latent embeddings in the AnnData object
    adata.obsm["latent"] = latent

    # Initialize the Benchmarker
    benchmarker = Benchmarker(
        adata,
        batch_key=batch_key,
        label_key=label_key,
        embedding_obsm_keys=["latent"],
        n_jobs=-1,  # Adjust the number of jobs according to your system
    )

    # Perform the benchmark
    benchmarker.benchmark()

    # Get the results as a dictionary
    results_df = benchmarker.get_results(min_max_scale=False)

    # Save the results to a CSV file if save_path is provided
    if save_path is not None:
        results_df.to_csv(save_path, index=False)
        print(f"Results saved to {save_path}")

    return results_df


def affinity_preservation_metrics(t, temperature=None, max_cells=10000, k_neighbors=15):
    """
    Measure how well the learned soft assignments preserve the original affinity structure.

    Uses soft assignments P = softmax(scores/temp) to compute predicted similarity S = P @ P.T,
    then compares S to original affinity A.

    Args:
        t: SCProtoTrainer (after setup/train)
        temperature: softmax temperature (default: t.epsilon)
        max_cells: subsample for efficiency
        k_neighbors: for neighbor overlap metric

    Returns:
        dict with metrics:
            - pearson: correlation between S and A (non-zero entries)
            - spearman: rank correlation
            - neighbor_overlap: fraction of top-k affinity neighbors in top-k predicted neighbors
            - weighted_same_proto: affinity-weighted fraction assigned to same prototype
            - mse: mean squared error between S and A
    """
    import torch.nn.functional as F
    from scipy.stats import pearsonr, spearmanr
    import scipy.sparse as sp

    ad = t.train_ds.adata
    A = t.train_ds.aff

    if temperature is None:
        temperature = t.epsilon

    # subsample if needed
    if ad.n_obs > max_cells:
        idx = np.random.choice(ad.n_obs, max_cells, replace=False)
        idx = np.sort(idx)
        ad = ad[idx].copy()
        A = A[idx][:, idx]
    else:
        idx = np.arange(ad.n_obs)

    # get scores and soft assignments
    scores = t.encode_adata(ad, t.model, z_idx=2)
    P = F.softmax(scores / temperature, dim=1)  # (n, n_proto)

    # predicted similarity
    S = (P @ P.T).detach().cpu().numpy()

    # convert A to dense if sparse
    if sp.issparse(A):
        A = A.toarray()

    # mask for non-zero affinity entries (excluding diagonal)
    np.fill_diagonal(A, 0)
    np.fill_diagonal(S, 0)
    mask = A > 0

    a_vals = A[mask]
    s_vals = S[mask]

    # correlation metrics
    pearson_corr, _ = pearsonr(a_vals, s_vals) if len(a_vals) > 2 else (0, 1)
    spearman_corr, _ = spearmanr(a_vals, s_vals) if len(a_vals) > 2 else (0, 1)

    # MSE
    mse = ((a_vals - s_vals) ** 2).mean()

    # neighbor overlap: for each cell, check overlap of top-k neighbors
    n = A.shape[0]
    overlaps = []
    for i in range(n):
        topk_aff = np.argsort(-A[i])[:k_neighbors]
        topk_pred = np.argsort(-S[i])[:k_neighbors]
        overlap = len(set(topk_aff) & set(topk_pred)) / k_neighbors
        overlaps.append(overlap)
    neighbor_overlap = np.mean(overlaps)

    # weighted same-prototype assignment
    hard_assign = scores.argmax(dim=1).detach().cpu().numpy()
    same_proto = (hard_assign[:, None] == hard_assign[None, :]).astype(float)
    np.fill_diagonal(same_proto, 0)  # exclude self-pairs

    # Affinity RECALL: of high-affinity pairs, what fraction are in same proto?
    # (This is what you had as weighted_same_proto)
    affinity_recall = (A * same_proto).sum() / (A.sum() + 1e-12)

    # Affinity PRECISION: of same-proto pairs, what fraction have high affinity?
    # This penalizes putting non-similar cells in same proto
    affinity_precision = (A * same_proto).sum() / (same_proto.sum() + 1e-12)

    # Mean intra-cluster affinity: average affinity within each cluster
    intra_affinities = []
    for p in range(scores.shape[1]):
        mask = hard_assign == p
        if mask.sum() < 2:
            continue
        cluster_A = A[np.ix_(mask, mask)]
        # Mean of upper triangle (exclude diagonal)
        triu_vals = cluster_A[np.triu_indices_from(cluster_A, k=1)]
        if len(triu_vals) > 0:
            intra_affinities.append(triu_vals.mean())
    mean_intra_affinity = np.mean(intra_affinities) if intra_affinities else 0.0

    # soft agreement: affinity-weighted soft similarity
    soft_agreement = (A * S).sum() / (A.sum() + 1e-12)

    # effective number of prototypes (entropy-based)
    proto_counts = np.bincount(hard_assign, minlength=scores.shape[1])
    proto_probs = proto_counts / proto_counts.sum()
    proto_entropy = -(proto_probs * np.log(proto_probs + 1e-12)).sum()
    effective_k = np.exp(proto_entropy)

    # proto utilization
    n_active_protos = (proto_counts > 0).sum()

    # niche purity (if available)
    niche_purity = None
    niche_purity_stats = {}
    if 'niches_2D' in ad.obs.columns:
        niches = ad.obs['niches_2D'].values
        purities = []
        for p in range(scores.shape[1]):
            mask = hard_assign == p
            if mask.sum() == 0:
                continue
            niche_counts = pd.Series(niches[mask]).value_counts(normalize=True)
            purities.append(niche_counts.max())
        if purities:
            niche_purity = np.mean(purities)
            niche_purity_stats = {
                'niche_purity_mean': np.mean(purities),
                'niche_purity_median': np.median(purities),
                'niche_purity_std': np.std(purities),
            }

    result = {
        'pearson': pearson_corr,
        'spearman': spearman_corr,
        'mse': mse,
        'neighbor_overlap': neighbor_overlap,
        'affinity_recall': affinity_recall,      # high-aff pairs in same proto (was weighted_same_proto)
        'affinity_precision': affinity_precision, # same-proto pairs that have high aff
        'mean_intra_affinity': mean_intra_affinity,  # avg affinity within clusters
        'soft_agreement': soft_agreement,
        'effective_k': effective_k,
        'n_active_protos': n_active_protos,
        'n_total_protos': scores.shape[1],
    }
    result.update(niche_purity_stats)
    return result


# ============================================================================
# Metacell / Prototype Niche Evaluation Metrics
# ============================================================================

def get_proto_assignments(t, ad=None):
    """Get hard prototype assignments for cells."""
    if ad is None:
        ad = t.train_ds.adata
    scores = t.encode_adata(ad, t.model, z_idx=2)
    hard_assign = scores.argmax(dim=1).detach().cpu().numpy()
    return hard_assign


def build_metacell_adata(t, ad=None, ct_key='celltype', niche_key='niches_2D'):
    """
    Build a metacell AnnData where each metacell (prototype) is the mean
    expression of assigned cells.

    Returns:
        mc_ad: AnnData with metacells as obs, genes as var
        cell_ad: original adata with 'proto_idx' column added
    """
    import scanpy as sc

    if ad is None:
        ad = t.train_ds.adata.copy()
    else:
        ad = ad.copy()

    # Get prototype assignments
    hard_assign = get_proto_assignments(t, ad)
    ad.obs['proto_idx'] = hard_assign.astype(str)

    # Get expression matrix
    X = ad.X.toarray() if hasattr(ad.X, 'toarray') else ad.X

    # Aggregate per prototype
    protos = np.unique(hard_assign)
    mc_expr = []
    mc_obs = []

    for p in protos:
        mask = hard_assign == p
        if mask.sum() == 0:
            continue

        # Mean expression
        mc_expr.append(X[mask].mean(axis=0))

        # Majority vote for labels
        if ct_key in ad.obs.columns:
            ct = ad.obs.loc[mask, ct_key].mode().iloc[0]
        else:
            ct = 'unknown'

        if niche_key in ad.obs.columns:
            niche = ad.obs.loc[mask, niche_key].mode().iloc[0]
        else:
            niche = 'unknown'

        mc_obs.append({
            'proto_id': str(p),
            ct_key: ct,
            niche_key: niche,
            'n_cells': int(mask.sum()),
        })

    mc_expr = np.array(mc_expr)
    mc_obs_df = pd.DataFrame(mc_obs)

    import anndata
    mc_ad = anndata.AnnData(
        X=mc_expr,
        obs=mc_obs_df,
        var=ad.var.copy()
    )
    mc_ad.obs.index = mc_ad.obs['proto_id']

    return mc_ad, ad


def niche_purity_stats(t, ad=None, niche_key='niches_2D'):
    """
    Compute niche purity statistics per prototype.

    Purity = fraction of cells in each proto that belong to majority niche.

    Returns:
        dict with mean, median, std, and per-proto purities
    """
    if ad is None:
        ad = t.train_ds.adata

    hard_assign = get_proto_assignments(t, ad)
    niches = ad.obs[niche_key].values

    purities = []
    proto_stats = []

    for p in np.unique(hard_assign):
        mask = hard_assign == p
        n_cells = mask.sum()
        if n_cells == 0:
            continue

        niche_counts = pd.Series(niches[mask]).value_counts()
        majority_niche = niche_counts.index[0]
        purity = niche_counts.iloc[0] / n_cells

        purities.append(purity)
        proto_stats.append({
            'proto_id': p,
            'n_cells': n_cells,
            'majority_niche': majority_niche,
            'purity': purity,
            'n_niches': len(niche_counts),
        })

    purities = np.array(purities)

    return {
        'mean': purities.mean(),
        'median': np.median(purities),
        'std': purities.std(),
        'min': purities.min(),
        'max': purities.max(),
        'n_protos': len(purities),
        'per_proto': pd.DataFrame(proto_stats),
    }


def niche_purity_per_celltype(t, ad=None, ct_key='celltype', niche_key='niches_2D'):
    """
    Compute niche purity within each cell type.

    For each cell type, computes purity of prototypes that mostly contain that cell type.
    """
    if ad is None:
        ad = t.train_ds.adata

    hard_assign = get_proto_assignments(t, ad)
    niches = ad.obs[niche_key].values
    celltypes = ad.obs[ct_key].values

    results = {}

    for ct in np.unique(celltypes):
        ct_mask = celltypes == ct
        ct_protos = np.unique(hard_assign[ct_mask])

        purities = []
        for p in ct_protos:
            proto_mask = hard_assign == p
            # Only consider protos where this celltype is majority
            proto_cts = pd.Series(celltypes[proto_mask]).value_counts()
            if proto_cts.index[0] != ct:
                continue

            # Niche purity within this celltype's cells in the proto
            ct_in_proto = proto_mask & ct_mask
            if ct_in_proto.sum() < 2:
                continue

            niche_counts = pd.Series(niches[ct_in_proto]).value_counts(normalize=True)
            purities.append(niche_counts.iloc[0])

        if purities:
            results[ct] = {
                'mean': np.mean(purities),
                'median': np.median(purities),
                'std': np.std(purities),
                'n_protos': len(purities),
            }

    return results


def niche_separation_chi2(t, ad=None, ct_key='celltype', niche_key='niches_2D',
                          target_ct='Fibroblasts'):
    """
    Test if prototypes can separate niches within a cell type using chi-squared test.

    For a target cell type, tests whether prototype assignments are independent
    of niche labels. Low p-value = prototypes discriminate niches well.
    """
    from scipy.stats import chi2_contingency

    if ad is None:
        ad = t.train_ds.adata

    # Filter to target cell type
    ct_mask = ad.obs[ct_key] == target_ct
    ad_ct = ad[ct_mask]

    hard_assign = get_proto_assignments(t, ad_ct)
    niches = ad_ct.obs[niche_key].values

    # Build contingency table
    tab = pd.crosstab(hard_assign, niches)

    # Chi-squared test
    if tab.shape[0] > 1 and tab.shape[1] > 1:
        chi2, pval, dof, expected = chi2_contingency(tab)
        cramers_v = np.sqrt(chi2 / (tab.sum().sum() * (min(tab.shape) - 1)))
    else:
        chi2, pval, cramers_v = np.nan, np.nan, np.nan

    return {
        'chi2': chi2,
        'pval': pval,
        'cramers_v': cramers_v,  # effect size
        'n_protos': len(np.unique(hard_assign)),
        'n_niches': len(np.unique(niches)),
        'contingency_table': tab,
    }


def niche_silhouette_scores(t, ad=None, ct_key='celltype', niche_key='niches_2D',
                            target_ct='Fibroblasts', use_proto_space=True):
    """
    Compute silhouette scores for niche separation in embedding/proto space.

    Higher score = better niche separation.
    """
    from sklearn.metrics import silhouette_score, silhouette_samples

    if ad is None:
        ad = t.train_ds.adata

    # Filter to target cell type
    ct_mask = ad.obs[ct_key] == target_ct
    ad_ct = ad[ct_mask]

    if use_proto_space:
        # Use soft assignment scores as embedding
        scores = t.encode_adata(ad_ct, t.model, z_idx=2)
        X = scores.detach().cpu().numpy()
    else:
        # Use latent z
        z = t.encode_adata(ad_ct, t.model, z_idx=0)
        X = z.detach().cpu().numpy()

    niches = ad_ct.obs[niche_key].values

    # Filter out 'Excluded' or rare niches
    valid_niches = pd.Series(niches).value_counts()
    valid_niches = valid_niches[valid_niches >= 10].index
    valid_mask = pd.Series(niches).isin(valid_niches).values

    X = X[valid_mask]
    niches = niches[valid_mask]

    if len(np.unique(niches)) < 2:
        return {'overall': np.nan, 'per_niche': {}}

    # Overall silhouette
    overall_sil = silhouette_score(X, niches)

    # Per-niche silhouette
    sample_sils = silhouette_samples(X, niches)
    per_niche = {}
    for niche in np.unique(niches):
        mask = niches == niche
        per_niche[niche] = sample_sils[mask].mean()

    return {
        'overall': overall_sil,
        'per_niche': per_niche,
    }


def niche_dge_overlap(t, ad=None, ct_key='celltype', niche_key='niches_2D',
                      target_ct='Fibroblasts', ground_truth_genes=None, topk=10):
    """
    Compare DGE genes from metacells vs ground truth niche markers.

    For each niche, finds top DE genes between metacells in that niche vs others,
    compares to known marker genes.

    Args:
        ground_truth_genes: dict mapping niche -> list of marker genes
            e.g., {'Tumor surface': ['FN1', 'IGFBP5'], ...}
    """
    import scanpy as sc

    if ad is None:
        ad = t.train_ds.adata

    # Build metacell adata
    mc_ad, cell_ad = build_metacell_adata(t, ad, ct_key=ct_key, niche_key=niche_key)

    # Filter to target cell type
    mc_ct = mc_ad[mc_ad.obs[ct_key] == target_ct].copy()

    if mc_ct.n_obs < 5:
        return {'error': 'Too few metacells for target cell type'}

    # Preprocess for DGE
    sc.pp.normalize_total(mc_ct, target_sum=1e4)
    sc.pp.log1p(mc_ct)

    results = []
    niches = mc_ct.obs[niche_key].unique()

    for niche in niches:
        if niche == 'Excluded':
            continue

        # Binary label: this niche vs rest
        mc_ct.obs['_group'] = (mc_ct.obs[niche_key] == niche).map({True: 'pos', False: 'neg'})

        n_pos = (mc_ct.obs['_group'] == 'pos').sum()
        n_neg = (mc_ct.obs['_group'] == 'neg').sum()

        if n_pos < 2 or n_neg < 2:
            continue

        # Run DGE
        try:
            sc.tl.rank_genes_groups(mc_ct, '_group', groups=['pos'], reference='neg',
                                   method='wilcoxon', use_raw=False)
            dge_df = sc.get.rank_genes_groups_df(mc_ct, group='pos')
            mc_genes = dge_df['names'].head(topk).tolist()
        except Exception as e:
            mc_genes = []

        # Compare to ground truth
        if ground_truth_genes and niche in ground_truth_genes:
            gt_genes = ground_truth_genes[niche][:topk]
            shared = list(set(mc_genes) & set(gt_genes))
            jaccard = len(shared) / len(set(mc_genes) | set(gt_genes)) if mc_genes else 0
        else:
            gt_genes, shared, jaccard = [], [], np.nan

        results.append({
            'niche': niche,
            'n_metacells': n_pos,
            'mc_top_genes': mc_genes,
            'gt_genes': gt_genes,
            'shared_genes': shared,
            'jaccard': jaccard,
            'n_shared': len(shared),
        })

    return pd.DataFrame(results)


def compare_methods_summary(scproto_stats, seacells_stats, method_names=('scproto', 'SEACells')):
    """
    Create a comparison table between two methods.

    Args:
        scproto_stats: dict from niche_purity_stats() or similar
        seacells_stats: dict with same structure

    Returns:
        DataFrame with side-by-side comparison
    """
    rows = []

    metrics = ['mean', 'median', 'std', 'n_protos']

    for metric in metrics:
        if metric in scproto_stats and metric in seacells_stats:
            rows.append({
                'metric': f'niche_purity_{metric}',
                method_names[0]: scproto_stats[metric],
                method_names[1]: seacells_stats[metric],
            })

    return pd.DataFrame(rows)


def full_niche_evaluation(t, ad=None, ct_key='celltype', niche_key='niches_2D',
                          target_ct='Fibroblasts', ground_truth_genes=None):
    """
    Run full niche evaluation suite and return summary.

    Returns dict with all metrics for easy comparison.
    """
    results = {}

    # 1. Overall niche purity
    purity = niche_purity_stats(t, ad, niche_key=niche_key)
    results['niche_purity_mean'] = purity['mean']
    results['niche_purity_median'] = purity['median']
    results['niche_purity_std'] = purity['std']
    results['n_active_protos'] = purity['n_protos']

    # 2. Niche purity per celltype
    ct_purity = niche_purity_per_celltype(t, ad, ct_key=ct_key, niche_key=niche_key)
    if target_ct in ct_purity:
        results[f'{target_ct}_niche_purity_mean'] = ct_purity[target_ct]['mean']
        results[f'{target_ct}_niche_purity_median'] = ct_purity[target_ct]['median']

    # 3. Chi-squared separation test
    chi2_result = niche_separation_chi2(t, ad, ct_key=ct_key, niche_key=niche_key,
                                         target_ct=target_ct)
    results[f'{target_ct}_chi2_pval'] = chi2_result['pval']
    results[f'{target_ct}_cramers_v'] = chi2_result['cramers_v']

    # 4. Silhouette scores
    sil = niche_silhouette_scores(t, ad, ct_key=ct_key, niche_key=niche_key,
                                   target_ct=target_ct)
    results[f'{target_ct}_silhouette'] = sil['overall']

    # 5. DGE overlap (if ground truth provided)
    if ground_truth_genes:
        dge = niche_dge_overlap(t, ad, ct_key=ct_key, niche_key=niche_key,
                                target_ct=target_ct, ground_truth_genes=ground_truth_genes)
        if isinstance(dge, pd.DataFrame) and len(dge) > 0:
            results[f'{target_ct}_mean_jaccard'] = dge['jaccard'].mean()
            results[f'{target_ct}_mean_shared_genes'] = dge['n_shared'].mean()

    return results


# Default ground truth markers for Fibroblasts (from s28f dataset)
FIBROBLAST_NICHE_MARKERS = {
    'Tumor surface': ['FN1', 'IGFBP5', 'COL11A1', 'TAGLN', 'VCAN'],
    'Vascular stroma': ['IGFBP7', 'MGP', 'TIMP1', 'COL6A2', 'COL6A1'],
    'Smooth muscle structures': ['MGP', 'CLU', 'CCL2', 'GPNMB', 'GPX3'],
    'Desmoplastic stroma': ['LUM', 'COL1A1', 'PTGDS', 'COL1A2', 'COL3A1'],
    'T cell aggregates': ['CD74', 'HLA-C', 'CCL19', 'HLA-A', 'CXCL9'],
    'Macrophage islands': ['CD74', 'HLA-DRB1', 'HLA-DRA', 'HLA-DQB1', 'HLA-DPA1'],
    'Tumor core': ['FN1', 'IGFBP5', 'COL6A3', 'LGALS1', 'VCAN'],
    'Airways': ['PTGDS', 'MGP', 'CD74', 'CCL2', 'APOD'],
    'Alveolar spaces': ['IGFBP7', 'MGP', 'MZT2A', 'GPX3', 'CFD'],
}


def eval_niche_metrics(t, target_ct='Fibroblasts', ct_key='celltype', niche_key='niches_2D',
                       gt_markers=None, print_results=True):
    """
    One-liner evaluation for niche separation quality.

    Usage in Colab:
        from interpretable_ssl.evaluation.metrics import eval_niche_metrics
        results = eval_niche_metrics(t)
    """
    if gt_markers is None:
        gt_markers = FIBROBLAST_NICHE_MARKERS

    results = full_niche_evaluation(
        t, ct_key=ct_key, niche_key=niche_key,
        target_ct=target_ct, ground_truth_genes=gt_markers
    )

    if print_results:
        print(f"{'='*50}")
        print(f"Niche Evaluation Results ({target_ct})")
        print(f"{'='*50}")
        print(f"Niche Purity (all):    mean={results['niche_purity_mean']:.3f}, median={results['niche_purity_median']:.3f}")
        if f'{target_ct}_niche_purity_mean' in results:
            print(f"Niche Purity ({target_ct[:8]}): mean={results[f'{target_ct}_niche_purity_mean']:.3f}")
        print(f"Chi2 p-value:          {results[f'{target_ct}_chi2_pval']:.2e}")
        print(f"Cramer's V:            {results[f'{target_ct}_cramers_v']:.3f}")
        print(f"Silhouette:            {results[f'{target_ct}_silhouette']:.3f}")
        if f'{target_ct}_mean_jaccard' in results:
            print(f"DGE Jaccard:           {results[f'{target_ct}_mean_jaccard']:.3f}")
        print(f"Active prototypes:     {results['n_active_protos']}")
        print(f"{'='*50}")

    return results


def save_niche_recovery(t, save_path, method_name='scproto',
                        ct_key='cell_ontology_class', niche_key='niches_2D'):
    """
    Save niche recovery metrics for comparison with SEACells.

    Usage:
        from interpretable_ssl.evaluation.metrics import save_niche_recovery
        save_niche_recovery(t, './results', 'scproto')
    """
    import os
    from interpretable_ssl.evaluation.niche_recovery import (
        niche_purity_within_ct, ct_purity_per_mc, joint_purity
    )

    os.makedirs(save_path, exist_ok=True)
    ad = t.train_ds.adata.copy()

    # Get hard assignments
    hard_assign = get_proto_assignments(t, ad)
    ad.obs['mc_idx'] = hard_assign.astype(str)
    mc_key = 'mc_idx'

    # Compute metrics
    ct_pur = ct_purity_per_mc(ad, ct_key, mc_key)
    j_pur = joint_purity(ad, ct_key, niche_key, mc_key)
    niche_pur = niche_purity_within_ct(ad, ct_key, niche_key, mc_key)

    # Build results dict
    results = {
        'method': method_name,
        'n_metacells': len(np.unique(hard_assign)),
        'n_active': (np.bincount(hard_assign) > 0).sum(),
        'ct_purity_mean': ct_pur['mean'],
        'ct_purity_median': ct_pur['median'],
        'joint_purity_mean': j_pur['mean'],
        'joint_purity_median': j_pur['median'],
    }

    # Add per-celltype niche purity
    for ct, pur_series in niche_pur.items():
        results[f'niche_pur_{ct}_mean'] = pur_series.mean()
        results[f'niche_pur_{ct}_median'] = pur_series.median()

    # Save
    df = pd.DataFrame([results])
    out_path = os.path.join(save_path, f'{method_name}_niche_recovery.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

    return results
