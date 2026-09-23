def get_defaults():
    defaults = {
        "dataset_id": "pbmc-immune",
        "model_name_version": 7,
        "num_prototypes": 300,  # swav specific or 8, 128
        "hidden_dim": 64,
        "latent_dims": 8,  # swav specific
        # "batch_size_version": 2,
        
        # "custom_cross_val": False,
        # "description": "",
        "experiment_name": "",  # swav specific / or swav
        "condition_key": "study",
        "cell_type_key": "cell_type",
        # "epochs": 300,
        "linear_eval": False,
        "only_eval": False,
        "use_early_stopping": False,        
        
        "training_type": 'pretrain_encoder',  # semi_supervised, transfer_learning
        'pretrain_dataset_id': 'hlca',
        'finetune_dataset_id': 'pbmc-immune',
        
        "dump_name_version": 4,  # swav specific
        "augmentation_type": "knn",  # swav specific
        "size_crops": [224],  # swav specific
        "min_scale_crops": [0.14],  # swav specific
        "max_scale_crops": [1],  # swav specific
        "views_for_assign": [0],  # swav specific
        "nmb_views": 2,  # swav specific
        
        "feat_dim": 8,  # swav specific
        
        "cvae_reg": 0,  # swav specific
        "dist_url": "env://",  # swav specific
        "world_size": 1,  # swav specific
        "rank": 0,  # swav specific
        "local_rank": 0,  # swav specific
        
        "checkpoint_freq": 30,  # swav specific
        "umap_checkpoint_freq": 20,
        "scib_freq": 10, 
        "save_scib": 1,
        "use_fp16": False,  # swav specific
        "sync_bn": "pytorch",  # swav specific
        "syncbn_process_group_size": 8,  # swav specific
        "seed": 31,  # swav specific
        
        "optimizer": "",  # swav specific
        "lr_schedule": "",  # swav specific
        "queue": None,  # swav specific
        "train_loader": "",  # swav specific
        "training_stats": "",  # swav specific
        "device": "cuda",  # swav specific
        
        ## TODO: replaced by 2 new reg, to be removed
        "prot_decoding_loss_scaler": 0.0,  # swav specific, 5
        "hidden_mlp": 1024,  # swav specific
        "swav_dim": 64,  # swav specific
        "use_projector": False,  # swav specific
        ## TODO: to be removed, not used except sbatch template
        "model_version": 1,  # swav specific
        "train_decoder": False,  # swav specific
        "longest_path": 1,  # swav specific, maybe 5 would be cool
        "dimensionality_reduction": 'pca',  # swav specific
        'k_neighbors': 50,  # swav specific
        
        'job_name': '',
        'no_data': 'False',
        "freezable_prototypes": 0,  # swav specific (should be true)
        "prot_init": 'kmeans', #can be kmeans, waypoint, or [D22] 'leid'
        'prot_init_gamma': 0.0069,      # [D22] CPM resolution for prot_init='leid'. WHY THIS EXISTS: at K=200 on s28nsc/bk02 a hard optimizer already finds a partition that beats SEACells on EVERY axis -- Leiden-CPM at gamma=0.0069 gives K=197, density 0.0321, joint purity 0.529, 49 states, max/median 2.6, against SEACells' 0.0295 / 0.504 / 46 / 3.6. scProto optimizing the SAME objective reaches only 68% of that density, 83% of the purity and 73% of the states, so the gap is the SEARCH, not the loss. This initialises each prototype at the centroid of one Leiden community, i.e. starts the search at the answer. If training holds it, the loss is fine and only the optimizer was failing; if training destroys it, the soft fixed-K parameterisation cannot represent that partition, which is a far more fundamental result. NOTE gamma is a FLOOR, not a target density: at gamma=0.0069 the achieved density is 0.0321, 4.6x higher.
        
        "loss_type": 'cross_entropy',
        "decodable_prototypes": 0,
        "save_temp_res": 1,
        "temp_res_path": "temp-res",
        
        "n_components": 50,
        "supervised_ratio": 0.1,
        "multi_layer_protos": 0,
        "batch_removal_ratio": 0.0,
        "use_bknn": 0,
        "freeze_batch_embedding": 0,
        "freeze_decoder": 0,
        "batch_sinkhorn": 1,
        "weighted_batch": 0,
        "knn_similarity": 'cosine',
        "recon_loss": 'mse',
        "no_sinkhorn": 0,
        "wandb_sweep": 0,
        "sweep_id": -1,
        "mask_probability": 0.2,
        "default_dispersion": 0.1,
        "fold": 0,
        "spatial": 0,
        "use_the_queue": 0,
        "entropy_reg": 0.0,
        "dataset_cnt": 0,
        "study_id": '',
        
        "use_rbf": 1,
        
        
        
        
        "cvae_loss_scaler": 0.01, # 0.01,  # swav specific, 0.0001
        
        "workers": 1,  # swav specific
        
 
        
        "batch_size": 1024,
        
        
        "freeze_prototypes_nepochs": 1, # used to be 50
        
        "hard_clustering": 0,
        

        
        "epoch_queue_starts": 0, # used to be 5
        "queue_length":  0,
        
        "use_counts": 1,
        "knn_method": 'faiss',
        
        "affinity_type": "uctx",
        "covet_alpha": None,
        'cell_w_mode': 'uniform',
        "model": "swav",  # swav specific
        "lambda_align": 0,
        
        "beta": 0.3, # not used any more
        
        "lambda_l2": 0.0, # 1e-3
        
        "lambda_kl": 0.0,
        "lambda_recon": 0.0, 
        'lambda_balance': 0.0,
        'lambda_swav': 0.0,
        "lambda_proto": 0.0, # 1,
        "lambda_commit": 0.0, # 0.25
        "lambda_p_uncertainty": 0.0,
        'lambda_proto_entropy': 0.0,
        
        "lambda_proto_recon": 0.0,
        "proto_recon_hard": False,
        "lambda_umap": 1,
        'umap_similarity': 'embedding',
        'assignment_metric': 'dotp',
        "model_type": 'gm',
        "l2norm": 1,
        

        "recon_update_target": 'encoder',
        "description": '17: calc affinity on whole datase instead of batch wise',
        'recon_type': 'normal',
        'normalize_loss': 0,
        'div_type': 'ce',
        
        'kl_sched': 1,
        'swav_sched': 0, # not used yet
        'recon_start_epoch': 0,
        'kl_start_epoch': 0,
        "cvae_epochs": 0,
        "pretraining_epochs": 25,
        "ft_epochs": 0,
        
        'umap_metric': 'euclidean',
        'opt': 'adam',
        "wd": 1e-5,  # swav specific
        "base_lr": 3e-4,  # swav specific
        "final_lr": 1e-5,  # swav specific
        "warmup_epochs": 5,  # swav specific
        "start_warmup": 0.0,  # swav specific, 1e-6
        # change to dotp in future
        
        "temperature": 0.1,  # swav specific, lower make sharper assignment of z to protos, swav default 0.1
        "epsilon": 0.05,  # swav specific, swav default: 0.05 
        'weighted_kl': 0, # weighted swav loss
        "sinkhorn_iterations": 0,  # swav specific
        "full_dataset_mode": 1,
        'mode': 'train',
        'adoptive_eps': 0,
        'auto_eps_tau': 0,  # auto-calibrate eps/tau from affinity after init_prototypes
        'p': 0.5,
        'k_pos': 0,
        'softm': 0,
        'lambda_aff': 0.0,
        'lambda_r1r2': 0.0,
        'r1r2_log': 0,               # 0 = linear max, 1 = log(max) (penalizes dead protos more aggressively)
        'lambda_proto_attract': 0.0,  # dead-proto attraction to poorly-represented cells
        'lambda_proto_anchor': 0.0,   # soft MSE pull: prototype (still a free, gradient-trained param) toward soft_assign-column-normalized combination of this batch's cell embeddings — latent-space analogue of SEACells' B matrix, computed fresh per minibatch. Ignored if proto_decoupled=True (mutually exclusive — that already controls position via EMA). See files/proto_anchoring_vs_proto_usage.md (option A).
        'two_sided': 0,
        # Edge-centric UMAP parameters
        'umap_min_dist': 0.5,      # UMAP min_dist (scanpy default=0.5, umap default=0.1)
        'umap_spread': 1.0,        # UMAP spread (default=1.0)
        'umap_neg_rate': 5,        # Negative samples per positive edge
        'umap_edge_epochs': 200,   # Epochs for edge sampling expansion
        'umap_similarity': 'embedding',  # 'embedding' (distance kernel) or 'proto' (soft assignment dot product)
        'calibrate_eps': 0,              # 0=off; 1=p/q matching (E[q_pos]=E[p_pos], falls back to effk if unreachable or effk_mean<3); 2=always use effk alignment
        'umap_proto_effk': 5.0,          # target effective k for proto soft assignments (auto-calibrates temperature)
        'umap_proto_effk_agg': 'mean',   # aggregation for effk calibration: 'mean' or 'median'
        'usage_norm_sim': 0,             # 0 = none; 1 = post-softmax global n_k; 2 = pre-softmax EMA per-batch; 3 = batch-balanced n_k; 4 = coverage w, no renorm, grad through w; 5 = pre-softmax log-corr; 6 = coverage w, renorm, grad through w; 7 = robust coverage (mean above median), renorm, parameter-free; 8 = pre-softmax col-shift+max-norm then usage-norm (each proto gets ≥1 cell=1, penalizes dominant protos); 9 = per-batch Sinkhorn (asymmetric loss: grad through s, Sinkhorn target t detached)
        'sinkhorn_iters': 3,             # Sinkhorn iterations for usage_norm_sim=9
        'usage_norm_corr_clamp': 10.0,   # clamp for log_corr in mode 5 (default 10; raise if corr hits ceiling and dead protos persist)
        'usage_nk_alpha': 0.9,           # EMA smoothing for usage_norm_sim=2 and proto_usage_mode='max' (per mini-batch step)
        'lambda_degree_weight': 0,       # 1 = weight positive loss by A_ij/(k_i*k_j/2m); aligns loss with modularity null model
        'degree_norm_loss': 0,           # 1 = normalize positive loss by 1/sqrt(d_i*d_j); each cell contributes equally
        'lambda_proto_usage': 0.0,       # proto usage loss — prevents dead protos, silent when active
        'proto_usage_mode': 'nk',        # 'nk': log(1+1/n_k) group-level; 'max': -log(max_i S[i,k]) per-batch; 'ema': EMA-smoothed max (usage_nk_alpha controls decay)
        'lambda_nassoc': 0.0,            # normalized association regularizer: forces S.T@W@S/vol toward identity
        'nassoc_alpha': 1.0,             # weight of off-diagonal vs diagonal terms (after per-count normalization)
        'nassoc_agg': 'mean',            # how to aggregate per-batch M matrices: 'mean' (avg loss) or 'max' (element-wise max of M, then loss) or 'pbch' (loss per batch then avg; no cross-batch constraint, use mse diagonal)
        'nassoc_diag_loss': 'mse',       # diagonal loss form: 'mse' = (diag-1)^2, 'nll' = -log(avg_diag), 'nll2' = -log(avg_b[1-(d_b-1)^2]) rewards multi-batch moderate usage
        'nassoc_diag': True,             # whether to include diagonal (purity) term; set False to keep only off-diagonal (redundancy) term
        'nassoc_diag_norm': 'volume',    # [D1] DIAGONAL denominator. 'volume' = edges_inside/volume (current: a containment measure -- volume is additive under merging, so merging two protos always raises it and hierarchical splits are penalized). 'pair' = edges_inside/possible_pairs, i.e. the induced-subgraph edge density = mean pairwise affinity between members; possible_pairs is superadditive so merging LOWERS it, which is what lets the loss rank a split above a merge. Off-diagonal keeps volume normalization either way. See files/notes/nassoc_redesign.md
        'nassoc_gamma': 0.0,             # [D2] resolution parameter, used only by nassoc_diag_loss='hinge'/'cpm'. The density a prototype must reach to be "worth existing"; equivalently the price of a missing link relative to a present one (CPM, Traag et al. 2011). Higher = tighter/smaller prototypes. Its absolute scale depends on the affinity scale and minibatch sampling rate, so sweep it and read off the resulting mean metacell size rather than picking a value a priori.
        'nassoc_offdiag_thresh': 0.0,   # [D19] HINGE threshold for nassoc_offdiag_agg='hnge'. 0 (default) disables the hinge. Set it to the leakage a GOOD partition normally shows, so ordinary adjacency between neighbouring niches is FREE and only excess -- the signature of two prototypes occupying the same cells -- is charged. Measure it per graph as the p95 of a reference Leiden partition's off-diagonal entries; on s28nsc/bk02 at K=200 that is 0.0343. Why it matters: on the constructed overlap pathology (split every community in half at random so the halves share a region) detection power is sum-of-squares 4.08x, max 3.55x, top3 1.85x, HINGE 19.63x. And in the combined objective ratio_assoc - w*offdiag, scored against joint purity at FIXED K=200, the hinge keeps a perfect ranking up to w=5 where sum-of-squares breaks at w=2 -- so it has a 5x wider safe operating range (decisions.md 21).
        'nassoc_offdiag_agg': 'pair',    # [D16] how the squared off-diagonal entries are AGGREGATED. 'pair' (default, behaviour-preserving) = mean over all K*K-K prototype PAIRS. 'hnge' = same per-cluster sum but of max(0, leakage - nassoc_offdiag_thresh)^2, so normal adjacency is free. 'hnlv' = [D24] the same hinge but averaged over LIVE prototypes only -- dead slots contribute a hard 0 and would otherwise drag the mean down for free. Measured on bk02 with min(vol) normalisation: overlap-pathology detection 4.32x (vs 2.57x for plain squares and 1.28x for the published mean-over-pairs) and K-transfer 0.93 going K=200->573 (vs 0.10 for the published form, i.e. its weight drifts 10x and never transfers). Use with nassoc_offdiag_norm='min': under min(vol) killing prototypes RAISES the term rather than diluting it, so the 65x dead-slot dilution seen with the 'sqrt' denominator disappears. 'pclu' = for each prototype sum the squared leakages to every other, THEN mean over the K prototypes -- the form normalized cut itself uses (K terms, never K^2). This is a NORMALIZER choice, independent of nassoc_offdiag_norm which picks the per-entry DENOMINATOR. Why it matters: nonzero prototype pairs grow roughly LINEARLY in K (a prototype only borders its spatial neighbours -- measured 14 per prototype at K=16 rising to only 168 at K=1774) while K^2 grows quadratically, so dividing by K^2 divides by something growing far faster than the signal. Measured on bk02, the nassoc_alpha needed for diagonal/off-diagonal parity drifts 77 -> 87,009 across K=16..1774 under 'pair' (1130x) but only 5.2 -> 49 under 'pclu' (9.5x). Under 'pclu' the term stays bounded by 1 because the leakages out of one prototype sum to 1 - containment[k] <= 1.
        'nassoc_offdiag_norm': 'sqrt',   # [D4] OFF-DIAGONAL denominator. 'sqrt' = edges_between/sqrt(vol_p*vol_q) (current: geometric mean is dominated by the larger volume, so a small prototype leaking into a big one is nearly invisible). 'min' = edges_between/min(vol_p,vol_q) -- bills the leakage against the SMALLER prototype's own budget instead (conductance-of-the-smaller-side, Kannan/Vempala/Vetta 2004), so a rare prototype's leakage registers. Independent of nassoc_diag_norm -- diagonal and off-diagonal denominators don't need to match. See literature/spatial/nassoc_redesign.md Fix D4.
        'nassoc_size_scale': 'btch',     # [D5] scale of size/possible_pairs used by the DIAGONAL. 'btch' (current) uses the raw minibatch soft counts -- but possible_pairs is QUADRATIC in size, so any prototype whose soft minibatch size is < 1 gets pairs clamped to exactly 0 and density = edges/eps EXPLODES (measured: mean diag 371-510 against a target of 1). Under (density-1)^2 an EMPTY prototype then costs 1 while a merely SMALL one costs ~1e16, so the loss actively pays the model to kill prototypes -- the observed 15/200-survivor collapse. 'full' rescales both counts to full-dataset scale (size*n_cells/n_batch, edges*n_edges/n_edges_batch) before forming density, so sizes are rarely <1, density lands in [0,1], and nassoc_gamma becomes interpretable as a real density (~mean_weighted_degree/(target_size-1)).
        'nassoc_cpm_norm': 'pair',       # [D2] denominator of the 'cpm' diagonal. 'pair' (current) divides by sum(possible_pairs) -- the SAME quantity gamma multiplies, so gamma algebraically cancels to an additive constant and has EXACTLY ZERO GRADIENT (confirmed: a 10x gamma sweep moved largest_share only 0.168->0.159). 'graf' divides by the summed affinity weight of THIS STEP's sampled edges: assignment-independent, so gamma survives -- but minibatch-scaled, which mismatches a full-dataset-scale numerator when nassoc_size_scale='full' and inflates the term by ~n_edges_total/n_edges_batch (measured: nassoc=2874 vs umap=2.5, i.e. 99.5% of the total loss). 'gful' divides by the FULL-graph total edge weight instead -- this is DMoN's 1/(2m) exactly -- and is the one to use with nassoc_size_scale='full' so numerator and denominator are on the same scale. 'dpar' divides by DETACHED sum(possible_pairs): since score[p] in [-gamma*pairs, (1-gamma)*pairs], this puts the whole term in [-(1-gamma), +gamma] for any dataset/batch size -- the natural bound implied by the objective itself, and the right partner for nassoc_size_scale='btch' (everything batch-native, no rescaling factors that can silently mismatch). Detaching is what keeps gamma's gradient alive: undetached, gamma*sum(pairs)/sum(pairs) collapses to a constant, which is the 'pair' bug. 'cpar' [D8] divides by the EXPECTED sum(possible_pairs) of a balanced hard partition, N*(N-K)/(2K), computed ONCE from N and K: it scales like possible_pairs (so the term keeps the same interval when K or N changes, making lambda_nassoc portable across K) but is a FIXED CONSTANT (so unlike 'dpar' it never drifts as the assignment evolves during training). Measured on s28nsc/bk02/K=200: cpar=8,503,906 vs gful=2m=1,540,839, a fixed 5.52x -- so at FIXED N and K, cpar and gful are EQUIVALENT up to rescaling lambda_nassoc (cpar@lambda=5520 == gful@lambda=1000, identical gradients); cpar differs from gful only once K or N changes, since 2m is blind to K and N*(N-K)/(2K) is not. This normalization convention is DMoN's (Tsitsulin et al. JMLR 2023); applying it to CPM, and the detached-pairs and constant-pairs variants, are our own adaptations, not published results. 'btsz' [D9] divides by the NUMBER OF EDGES IN THE BATCH -- the partner normalizer for nassoc_ht=1, where both halves are batch-scale. reward/B is then literally the mean membership overlap over the sampled edges (the soft internal fraction, bounded in [0,1]), independent of N, K and 2m. Needs lambda_nassoc ~1e2 smaller than gful's and gamma ~1.13x the full-graph value.
        'nassoc_cpm_free': 0,            # [D2] 1 = compute the 'cpm' diagonal in its published DIVISION-FREE form, score[p] = edges_inside[p] - gamma*possible_pairs[p] (Traag/Van Dooren/Nesterov PRE 2011), instead of the algebraically-identical but numerically fragile (density-gamma)*possible_pairs. At possible_pairs=0 the subtraction gives edges-0 (finite) while the ratio form gives inf*0. This is why Leiden never hits the degenerate-cluster problem: CPM as published is a subtraction, never a ratio.
        'nassoc_exact': 0,               # [D6] 1 = compute the CPM terms exactly. Two corrections, both verified against the data: (a) possible_pairs = (size^2 - sum_i assign^2)/2, the EXACT soft pair count (identity for sum_{i<j} assign_i*assign_j), which reduces to n(n-1)/2 for hard assignments and is always >=0 -- unlike size*(size-1)/2, which is the hard formula misapplied to soft counts, goes negative when size<1, gets clamped to 0 and causes the divide-by-zero behind the prototype collapse; (b) edges_inside was 2x inflated and contaminated by self-loops, because the affinity is symmetric AND stores aff[i,i], so EdgeDataset lists each undirected edge twice plus one self-loop per cell (bk02: nnz=4,648,883 = 2*2,295,230 + 58,423). Inflating edges_inside but not gamma*possible_pairs HALVES the effective gamma, which is why gamma never acted as the density threshold it is defined to be. [D7] When combined with nassoc_size_scale='full', sum_i assign^2 must be rescaled LINEARLY by cell_scale (same as size), not by cell_scale**2 -- it is a sum over cells, same extensive quantity as size, and size_eff**2 already supplies the quadratic term by squaring the linearly-rescaled size. The c^2 version silently reintroduces the exact collapse this option exists to fix for genuinely rare prototypes (verified on real s28nsc data: a 3-cell-total prototype with 1 cell landing in the minibatch, cell_scale=57, gives pairs_eff=0.0 under c^2 vs 43.3 under linear c). See scproto.py [D7].
        'nassoc_ht': 0,                  # [D9] 1 = compute the CPM diagonal LITERALLY ON THIS BATCH, with both sampling biases removed. Measured by simulation in modal_scripts/measure_estimator_bias.py. A batch is B EDGES drawn WITH REPLACEMENT PROPORTIONAL TO w, and the batch's cells are their deduplicated endpoints -- so nothing in a batch is a uniform sample. The batch-vs-full SCALE factors (reward ~ 2B/W, pairs ~ (n_b/N)^2, ratio 1.13) are CONSTANTS and a constant only shifts gamma, so they are deliberately NOT carried around; gamma absorbs them. What gamma cannot absorb is anything that differs PER PROTOTYPE, and there are exactly two such distortions. (a) edges_inside multiplies by w a SECOND time (A_part = S_h.T @ (w_b*S_t)) on top of the sampler's own w, so the objective is effectively weighted by w^2, not w; the UMAP term does not do this, so the two losses weighted the same batch differently. Measured bias 1.046 overall but 1.347 for prototypes under 25 cells vs 1.049 at 500-1k -- size-dependent. Fix: reward = sum over sampled edges of the membership overlap, with NO w factor. Measured 1.001, flat at every size. (b) cells in sparse regions are under-represented because they have fewer edges to be drawn by: inclusion probability p_i = 1-(1-2*d_i/W)^B ranges 0.021..0.210 on s28nsc/bk02, a 10x spread, and a uniform N/n_b rescale cannot correct a non-uniform sample. Neither the [D7] linear-c form (bias 1.124 overall, 2.386 under 25 cells) nor the pre-[D7] c^2 form (1.111, 0.757) is unbiased. Fix: divide each batch cell by the RELATIVE propensity q_i = p_i/mean(p_i) -- a survey weight, mean 1, so batch scale is preserved -- then the same closed form 0.5*((sum u)^2 - sum u^2). Measured 0.999 overall, 0.98-1.11 at every size, with LOWER spread than the current form (p90 1.281 vs 4.157). In the 200-900 cell band where every current prototype sits the net effect is only ~8% (effective gamma 0.92x), so this does NOT explain existing results; it matters below ~100 cells, i.e. exactly the regime finer granularity needs and exactly where SEACells' uneven sizes (median 258, max 925) give it small metacells our loss would over-charge. Pair with nassoc_cpm_norm='btsz'. Makes nassoc_size_scale irrelevant (nothing is rescaled). Requires nassoc_exact=1. Default 0 = behaviour-preserving.
        'nassoc_gamma_scale': 'none',    # [D10] how nassoc_gamma is interpreted. 'none' (default, behaviour-preserving) = gamma is used as given. 'soft' = gamma is given on the HARD/graph density scale -- the scale we can actually MEASURE, e.g. measure_gamma_from_graph.py derived 0.026-0.030 for s28nsc/bk02 -- and the code converts it ONCE to the soft scale the loss really compares against. WHY THIS EXISTS: the loss compares gamma against a density built from SOFT memberships, but every gamma we ever set was derived from HARD partition statistics, and the two are not the same number. Measured directly in the ht_L16 run at gamma=0.028: the logged soft density (nassoc diag) plateaued at 0.011 while the final hard aggregate density of the very same run was 0.02695 -- a factor of 2.45, against effk 2.7. So gamma sat 2.5x above anything the loss could reach and EVERY prototype scored negative for the whole run: the saturated, non-discriminating regime EXPERIMENTS.md 3c blames for gamma=0.15. This mis-scaling affects every gamma sweep on record, including the ones labelled 'the provably-correct window', and explains why gamma=0.01 -- the lowest ever tried, and the only one near the reachable scale -- produced the highest internal_fraction of any scProto run (0.3788). The conversion is measured on the real data from the pretrained encoder (soft density of S vs hard density of argmax(S), same cells, same graph), so it is a units conversion, not a tuning knob, and it is NOT circular -- it never looks at the answer. CAVEAT: it is computed ONCE, and effk drifts during training (5.0 -> 2.7 measured), so the conversion is exact at the start and approximate later; the ratio is printed loudly so drift is visible. Related root cause, not fixed here: epsilon calibration FAILS on both datasets tried (s28nsc: E[p_pos]=0.3357 unreachable, max E[q_pos]=0.2051; pancreas: 0.3397 vs 0.2992) and silently falls back to umap_proto_effk=5.0, a hard-coded default nothing chose.
        'lambda_modularity': 0.0,        # [D11] weight of the SOFT MODULARITY term (DMoN, Tsitsulin et al. JMLR 24 2023). A complete objective on its own -- ONE scalar, no diagonal/off-diagonal split, because modularity scores every cell pair exactly once and pairs in DIFFERENT prototypes contribute zero, so 'keep communities separate' is already in it. Set lambda_nassoc=0 when using this. WHY IT EXISTS: with the SAME optimizer, graph and K, Leiden-modularity reaches DGE 0.856 / 18 pairs / jointP 0.534 while Leiden-CPM reaches 0.630 / 23 / 0.530, and scProto at the K-matched gamma reproduced Leiden-CPM almost exactly (0.605) -- so our CPM implementation is faithful and CPM's OPTIMUM is simply a poor partition for DGE. Two measured mechanisms: (a) CPM charges a flat gamma per pair while modularity charges resolution*deg_i*deg_j/total_degree, and bk02 spans 17.8x in degree -- community purity vs mean degree has spearman +0.025 under modularity vs +0.118 (Leiden-CPM) and +0.304 (scProto-CPM); (b) CPM permits 1-cell communities where modularity's minimum was 15, and DGE is computed from pseudobulk. See decisions.md 18 for the full derivation and my-tasks/todo/modularity-loss-plan.txt for the plan. Default 0 = off, behaviour-preserving.
        'modularity_res': 1.0,           # [D11] resolution for lambda_modularity. Scales the expected-by-chance charge: higher -> harder to justify grouping -> more, smaller communities. Because the loss uses DMoN's total_degree normalizer, this is EXACTLY igraph/Leiden's resolution parameter and transfers directly: on s28nsc/bk02, resolution=19.43 gives Leiden K=200 (measured, modal_scripts/leiden_cpm_control.py). Cross-checked independently by the resolution limit: min resolvable size = n_cells*sqrt(mean_edge_weight)/sqrt(total_degree*resolution), which at 19.43 gives 6.2 cells -- below all but 5 of the 175 celltype x niche states, so the limit does not bind. NOT portable across graphs; re-derive per dataset with a Leiden bisection.
        'modularity_agg': 'glob',        # [D12] how the modularity term handles MULTIPLE BIOLOGICAL BATCHES. 'glob' (default, behaviour-preserving) computes one global modularity over the whole graph. 'pbch' computes it PER BATCH and takes, FOR EACH PROTOTYPE, the max across batches, then sums over prototypes -- the direct analogue of the published nassoc's element-wise max on its diagonal (sections/method.tex). WHY 'pbch' EXISTS: the affinity is approximately block-diagonal by batch (method.tex says so explicitly -- cross-batch similarities are negligible), and on a block-diagonal graph GLOBAL modularity's optimum is BATCH-SEPARATED communities, because putting each technology in its own prototypes maximises within-community edges against the null. Measured on pancreas: the global version collapsed batch entropy to 0.2471 against the published loss's 0.4220 -- close to SEACells' 0.19, which is exactly the behaviour the paper criticises. Per batch, W_b keeps only edges touching b, so degree_b is ~0 for outside cells and contribution_b[k] asks only 'is prototype k dense within batch b's view?' -- never looking at cross-batch pairs. A prototype spanning batches is judged once per batch and scores if it is coherent in AT LEAST ONE, never charged for cross-batch affinity it does not have. NOTE each batch keeps its own total_degree_b in BOTH the edge rescaling and the normalizer, and the volume^2 EMA is maintained PER (batch, prototype) -- with 9 technologies a batch sees only ~1024/9 = 114 sampled edges, so a shared EMA would reinstate the variance bias that measure_modularity_estimator.py showed reaches 1.9x at 25-50 cells. On single-batch data (s28nsc) the max is over one element and this is byte-identical to 'glob'.
        'modularity_res_pbch': 0.0,      # [D12] resolution used when modularity_agg='pbch'. 0 = fall back to modularity_res. MUST be re-derived: the per-batch contributions sit against total_degree_b (~1/9 of global on pancreas), so a globally-bisected resolution is calibrated to the wrong denominator -- measured 36.52 global vs a 10.65-22.71 per-batch range on pancreas/arbf. Use ONE SHARED value (the median of the per-batch Leiden bisections, 16.58 on pancreas), NOT per-batch values: with max aggregation a lower resolution_b means a smaller charge and therefore a higher contribution, so per-batch resolutions would make the max systematically select whichever batch is graded most leniently (fluidigmc1 at 10.65) rather than the batch where a prototype is genuinely best. A shared bar also keeps 'evidence required to group two cells' a single scientific standard rather than one that varies by sequencing technology.
        'lambda_collapse': 0.0,          # DMoN collapse regularization (Tsitsulin et al. JMLR 2023, eq. 5): (sqrt(K)/n_cells)*||soft_sizes||_2 - 1. Equals sqrt(K/effective_K)-1, i.e. a smooth monotone function of how many prototypes are actually in use: 0 when sizes are balanced, sqrt(K)-1 when everything collapses into one. A SOFT balance prior -- unlike SwAV/Sinkhorn it never forces equal sizes, which is what we want (prototype sizes should be set by biology). Scale-free in the minibatch, so no sampling-rate bias. Set >0 to use it alongside or instead of lambda_proto_usage.
        'proto_reinit_every': 0,         # dead-prototype restart (VQ-VAE codebook restarts, van den Oord et al. 2017; the random-restart trick in Jukebox, Dhariwal et al. 2020). 0 = off. If >0, every N epochs any prototype whose EMA usage is below proto_reinit_thresh is teleported onto a currently worst-served cell (the cell with the lowest max assignment probability, i.e. farthest from every existing prototype). Gradient-based penalties cannot revive a prototype that already has zero assigned mass -- there is no gradient path back -- so a direct re-teleport is the only mechanism that acts on an already-dead slot.
        'proto_reinit_thresh': 0.01,     # EMA-usage level below which a prototype counts as dead for proto_reinit_every. Only read when proto_reinit_every > 0.
        # --- Cell-cell similarity reconstruction (resolution pressure nassoc can't provide) ---
        'proto_reinit_target': 'wrst',  # [D20] WHERE a revived prototype is teleported. 'wrst' (default, behaviour-preserving) = the worst-served cells, i.e. lowest max assignment = farthest from every existing prototype -- the VQ-VAE 'highest reconstruction error' analogue. That target is an OUTLIER by construction: it has almost no mass to claim, so the revived prototype starves again within two epochs. Measured across every configuration tried on 2026-09-21 -- nassoc, cohesion, ratio association, with and without an off-diagonal -- the revival count never falls (161->173, 159->171, 174->167): pure churn, never healing. 'splt' instead samples cells from the prototypes holding the MOST mass, i.e. it splits an overloaded prototype rather than chasing an outlier. That is the k-means restart rule, and the splitting property guarantees the objective improves when a non-degenerate group is split, so the revived prototype lands where there is real mass to claim.
        'proto_reinit_rel': 0.1,         # [D15] RELATIVE dead-prototype cut, used only when reinit reads the n_k (soft-mass) EMA -- i.e. whenever lambda_proto_usage == 0. proto_reinit_thresh is an ABSOLUTE 0.01, which is meaningful for _c_k_ema (a max-assignment PROBABILITY, naturally in [0,1]) but not for _n_k_ema, whose natural scale is batch_size/K (about 5 at 1024/200). Measured on the first working-reinit run: the absolute 0.01 flagged 10 prototypes while 145 were actually unused. This cut is a fraction of MEAN usage, so it is scale-free in batch_size and K. 0.1 = 'holding under a tenth of what an average prototype holds'.
        'lambda_cohesion': 0.0,          # [D14] off by default. Within-prototype AFFINITY-VECTOR cohesion: cohesion[k] = sum_ij assign[i,k]*assign[j,k]*cos(affinity_row_i, affinity_row_j) / size[k]^2, loss = 1 - sum_k size[k]*cohesion[k]/n_cells, bounded in [0,1]. Unlike CPM/modularity/DMoN, whose charges are functions of the assignment COLUMN SUMS only (size/volume) and therefore cannot move a cell for similarity reasons at all, this term's gradient d(loss)/d assign[i,k] = -(mean similarity of i to k's members - cohesion[k]) contains NO size, volume or degree -- a cell joins k iff it would IMPROVE k. Being a ratio of two quadratics it is intensive/scale-free, so a 20-cell state is as viable as a 2000-cell one, and the k-means splitting property (splitting any non-degenerate group strictly lowers the objective) means its optimum uses every prototype with no anti-collapse term and no size-uniformity side effect. Weighted kernel k-means; Dhillon/Guan/Kulis KDD 2004 prove that family equals normalized cut / normalized association -- i.e. the same family as the published nassoc loss.
        'cohesion_form': 'perc',        # [D21] how the cohesion term aggregates. 'perc' (default) = sum_k within[k]/size[k]. 'pcex' = the same numerator but with cell c EXCLUDED from its own divisor: for each cell, sum_k assign[c][k] * (sum_{i!=c} assign[i][k]*sim[c][i]) / (size[k] - assign[c][k]), averaged over cells. Conceptually 'how similar is cell c to a RANDOM OTHER member of k', which is what cohesion is supposed to mean. 'perc' divides by size[k] INCLUDING c, which is a 0.3% error for a 290-cell prototype but a 50% error for a 2-cell one -- so 'perc' systematically under-scores small prototypes, i.e. exactly the rare states the term exists to protect. Note the size behaviour is unchanged and is correct either way: the quantity is an AVERAGE, so a prototype growing by absorbing SIMILAR cells pays nothing while one absorbing DISSIMILAR cells sees its average drop -- it penalises mismatch, never size. That is also why it supplies no size control and needs modularity as a partner. Its size-control strength is ~1/m, which VANISHES for large prototypes -- measured consequence: median metacell size 12-136 against a mean of 292, and DGE 0.39-0.73 against modularity's 0.822. 'qwcw' = sum_k (size[k]^2 - within[k]) / n^2, the MISSING internal connections, i.e. possible pairs minus the affinity actually present. Its gradient is 2*(size[k] - similarity_of_i_to_k), so a perfectly-fitting cell pays 0 to join a group of any size while a poorly-fitting one pays 2*size[k] -- O(m) size control, waived exactly when the cell belongs. Motivation: density = within/possible_pairs is the quantity that actually separates good partitions from bad (SEACells 0.0295, modularity 0.0256, every scale-free arm 0.0062-0.0087) and it is naturally size-controlled, whereas int_frac is NOT (our arms beat SEACells on int_frac while being far worse).
        'cohesion_floor': 0.0,          # [D25] target for cohesion_form='flor'. The FLOOR every prototype must clear, measured as the p25 of within[k]/size[k] across a good reference partition (8.93 on s28nsc/bk02 at K=200). Why a floor and not a target to chase: prototypes already above it should pay NOTHING, so the term applies pressure only where it is needed. Why this exists at all: the per-cell cohesion forms ('perc'/'pcex') give a dead prototype a contribution of exactly 0, so they never PENALISE collapse, they merely fail to reward it -- there is no anti-collapse signal. A one-sided hinge over ALL K slots does penalise it (a dead slot pays the full floor^2) while keeping the size denominator, so it has no merge bias, and unlike the published (containment-1)^2 its target is REACHABLE (the ceiling of within/size is degree/m, so a target of 1 would be unreachable by ~11x on bk02). Measured discrimination: a sliver partition pays 44x what a good one does.
        'cohesion_sim': 'gram',         # [D18] what plays the role of similarity[i][j] in the cohesion term. 'gram' (default) = dot(affinity_row[i], affinity_row[j]) -- the 2-HOP kernel A^2, 'do i and j connect to the same cells'. 'aff' = affinity[i][j] itself -- 1-HOP, 'are i and j directly connected'. With 'aff' the term collapses to sum_k affinity_within[k]/size[k], i.e. RATIO ASSOCIATION, which is the published normalized-association diagonal with the denominator swapped from volume to a cell COUNT. That one swap removes the merge bias entirely: measured on the six most strongly-coupled Leiden communities on bk02, /size penalises every merge while /volume rewards one of them, because volume already contains the between-affinity that a merge hands to the numerator whereas a count does not. It is also LESS degree-coupled, not more: spearman(mean member degree, score) is -0.190 for /size against -0.400 for /volume, so the normalized-cut argument for volume does not hold on this graph -- volume over-corrects. And unlike the off-diagonal alone, it actually REDUCES total leakage rather than merely redistributing it (decisions.md 21). Note 'aff' makes cohesion_self moot: the affinity has a zero diagonal, so there are no self-pairs to keep or drop.
        'cohesion_self': 1,             # [D17]/[D30] WEIGHT of the i==j self-pairs in within[k] -- now a float, not only 0/1. The self-pair's value IS the founder gate: a singleton prototype scores exactly cohesion_self, so a cell joining it gains 2*sim*a against a dilution of cohesion_self*a, i.e. it joins iff sim > cohesion_self/2. At 1.0 the gate is 0.5 and the MEDIAN cell's best-neighbour gram cosine on bk02 is 0.472 -- most founders can never recruit their first follower, which is why d27/d29 kept prototypes alive but tiny (median 69-84). 0.5 puts the gate at 0.25, below the p5 of best-neighbour cosine (0.312), so ~97% of cells can follow their best-matching founder while unrelated cells (typical cosine ~0.08) are still rejected 3x over; a live singleton is still worth w > 0 against an empty slot's 0, so the anti-collapse value survives. Derived from the measured graph, not swept (CLAUDE.md 9b). 1 and 0 reproduce the historical branches exactly (verified). Old text: keep the i==j SELF-pairs in within_similarity. 1 (default) is SEACells-EXACT: minimizing ||M - M B A||_F^2 equals maximizing sum_k within_dot[k]/size[k] with similarity = dot product of kernel rows and self-pairs INCLUDED (verified numerically to 1.8e-12). 0 drops them. The trade-off, measured on bk02 at batch 1024 / K=200: the self part equals K_live/n_batch = 0.1875, which is 71% of the Leiden value (0.2627) and 97% of the random value (0.1937). For HARD assignments it is a constant with no gradient, but for SOFT ones sum_i assign^2 / sum_i assign is a SHARPNESS measure, so it has a live gradient rewarding 'sharpen assignments and keep more slots alive' rather than 'group similar cells'. Useful pressure with K fixed, but it can swamp the signal -- hence the toggle.
        'coh_diag_every': 50,           # [D26] cadence of the ZERO-WEIGHT diagnostics (prototype redundancy + modularity value). They need an n_batch x n_batch DENSE affinity block -- ~196MB at 7000 unique cells, the same order as the Gram path's own densify -- so computing them every step roughly DOUBLES the per-step dense cost of an arm that has no 1-hop cohesion term at all. They summarise an epoch, so sampling every 50 steps loses nothing real. 0 disables them. The epoch mean is rescaled by the actual sample count before the per-batch division, so the reported value is a true mean and not ~50x too small.
        'cohesion_global': 0,           # [D28]/[D29] how the cohesion is estimated. 0 (default) = batch only, the old behaviour. 1 = [D28] divide by a DETACHED global size (EMA of batch_size/p) and rescale each part to its full-graph rate (self/p, cross/p^2); cheap (200 floats) and removes both batch artefacts, but when a prototype is invisible the numerator is still 0, so the cell simply earns nothing. 2 = [D29] REMEMBER every cell's last assignment (N x K, ~47MB) and swap in this batch's rows each step; then numer[c][k] = sum over ALL j of sim[c][j]*assign[j][k] is exact, size[k] is exact rather than an EMA, and an invisible prototype still has a real profile to compare against, so the cell gets a genuine similarity instead of silence. Mode 2 needs no p-rescaling at all because nothing was subsampled away. Non-batch rows are detached: they are CONTEXT, so a cell moves under its own gradient while its prototype-mates hold still -- the classic Lloyd/EM assignment step. Cost: ~47MB plus one extra sparse matmul per step; the risk is staleness, since a non-batch row is as old as the last batch that contained it (about 8.6 steps on average here). A step sees only p = n_batch/N of the cells (0.116 on s28nsc at batch 1024), which breaks the estimate twice. (1) A prototype with S cells is INVISIBLE in a batch with probability (1-p)^S -- 29% at S=10, 54% at S=5 -- and the loss cannot tell invisible from empty, so a cell joining it is scored as if FOUNDING one: within=a*a over size=a gives gradient 1.0, the MAXIMUM, where the truth is its mean similarity to members it cannot see (~0.08). A 12x overpayment aimed precisely at small prototypes; measured on a real run, 11.1 of 200 prototypes invisible per batch and 35 invisible in >10% of batches, while the even-sized modularity baseline never hits it (P=4e-16 at S=288). (2) within[k] mixes a PAIR count (cross terms, needing BOTH cells, surviving at p*p) with a CELL count (self terms, one per cell, surviving at p), so dividing by size[k] leaves cross unbiased but inflates self by 1/p = 8.6x -- measured, the self term is 4.3% of the full-graph objective but 24.9% of the batch one. Same class as the possible_pairs bug in CLAUDE.md 9b. Setting 1 divides by a DETACHED global size (EMA of batch_size/p, reusing the usage_nk_alpha smoothing) and rescales each part to its full-graph rate (self/p, cross/p^2). Verified on s28nsc/bk02 at n_batch=6757: raw estimate 1.25x the full-graph truth -> 1.00x, and the invisible-prototype gradient 1.000 -> 2.0e-07.
        'coh_mem_init': 'unif',         # [D32] how the [D29] assignment memory starts, cohesion_global=2 only. 'unif' (default, behaviour-preserving) = uniform 1/K rows: NEUTRAL (identical memory contribution to every prototype, spread ~1e-16 measured, so no false preference) but DILUTED -- real differential signal accumulates only as cells get visited over the first epoch; rows carry last_seen=-1 so under [D31] the prior fades by age. 'full' = ONE no-grad inference pass over all N cells at first use fills the memory with real assignments softmax(z@protos/eps) -- the encoder is pretrained at finetune start, so step-0 profiles and sizes are honest immediately; costs one forward-only pass (seconds on GPU). Uses the UNCORRECTED canonical softmax because the usage/log-correction variants need running EMAs that do not exist before step 0; each row is refreshed with the loop's own value on the cell's first batch visit. Values 'unif'/'full' are 4-char-distinct.
        'coh_mem_hlife': 0.0,           # [D31] age half-life (in steps) for the [D29] assignment memory, cohesion_global=2 only. 0 (default, behaviour-preserving) = off, bit-identical to [D29]: every remembered row counts at full weight regardless of age. >0 = a remembered row is discounted by 0.5**(steps_since_last_visit/half_life), applied to BOTH the similarity numerators and size_g (both-sides rule, CLAUDE.md 9b), so stale assignments fade smoothly instead of counting as fresh; the current batch's rows are always live at weight 1, and a revisit still REPLACES the old row outright (memory-bank semantics -- decay is for the unobserved, replacement for the observed). Never-visited cells age from step -1, so the uniform init prior fades too. Anchor value: half-life of one epoch (~500 steps at umap_steps_per_epoch=500) = 'information a full epoch old counts half'; derive from the coh_mem_drift diagnostic (logged every step alongside coh_mem_age), not by sweeping -- drift ~0 means leave this off. Unit-tested against a triple-loop brute force in interpretable_ssl/tests/test_cohesion_memory.py.
        'cohesion_diag': 0.0,           # [D26] CLOSED-neighbourhood self-affinity used when building the 'gram' descriptor, applied BEFORE the row norm. 0.0 (default) is behaviour-preserving = the open neighbourhood. Why a non-zero value is the conceptually right one: the affinity IS an RBF kernel, so K[c][c] = exp(0) = 1 by construction -- setdiag(0) is the artificial step, made only because every graph-cut consumer (degree / sampling / UMAP / nassoc) needs a no-self-loop convention, and the same file already restores 1 for the sim_recon reconstruction target ('SEACells RSS convention'). cohesion-with-gram is a reconstruction objective (kernel k-means on kernel rows), so it belongs on that side. With delta = 0, two DIRECTLY ADJACENT cells in a locally triangle-free region get Gram[c][j] = 0 exactly -- the open-neighbourhood Gram calls two neighbours completely dissimilar, which is the pathology closing the neighbourhood fixes. Algebraically setdiag(delta) then row-norm makes the Gram the genuine cosine between closed vectors v[c] = aff_row[c] + delta*e_c, and on the un-normalised scale adds exactly 2*delta*aff[c][j] to the off-diagonal -- so delta is ALSO the 1-hop/2-hop mixture knob, and delta=1 is co-dominant, not a small correction (measured: the added part is ~1/sqrt(n_neighbours) ~ 0.25 on an edge against a cosine of order 0.1-0.5). Note this is a DIFFERENT diagonal from the Gram's own diagonal: cohesion_form='pcex' still removes Gram[c][c] from the loss, because a cell's similarity to itself must not count toward its own prototype's cohesion.
        'lambda_cohesion_aff': 0.0,     # [D26] weight of a SECOND cohesion term computed with similarity = the raw 1-hop affinity, added ALONGSIDE the term that lambda_cohesion drives (which keeps following cohesion_sim, so every existing run reproduces unchanged). 0.0 = off. Why both: 'gram' measures STRUCTURAL EQUIVALENCE (do c and j have the same neighbours) and 'aff' measures PROXIMITY (are c and j connected) -- the classic non-redundant pair, and neither sees the other's failure case. Two cells hanging off the same hub are structurally equivalent but not near each other, so pure-gram wrongly merges them; two adjacent cells straddling a type boundary are proximate but embedded differently, so pure-aff wrongly merges them. Requiring both is a real conjunction. The 'aff' term uses the RAW, un-row-normalised affinity on purpose: row-normalising it would make the sliced block ASYMMETRIC (row c divided by ||row c||, column j not), which is not a similarity at all. CAUTION when pairing with cohesion_diag > 0: the gram descriptor then already carries 2*delta*aff[c][j], so the total 1-hop weight is lambda_cohesion*(built-in part) + lambda_cohesion_aff and lambda_cohesion_aff = 0 is NOT 'no first order'. For a clean separable ablation run the arms at cohesion_diag = 0.
        'cohesion_row_norm': 'l2',       # [D14] row normalisation of the affinity before the Gram. 'glob' = divide the whole Gram by the mean of its diagonal -- keeps the RELATIVE row magnitudes (SEACells' geometry, so local density still carries weight) while making the scale dataset-independent. 'l2' = cosine similarity (Salton index on neighbourhoods), bounded [0,1], removes the 17.8x degree spread. 'none' = raw inner product, which lets hub cells dominate. 4-char-distinct.
        'lambda_sim_recon': 0.0,         # off by default; decodes prototypes -> per-cell similarity target, reconstructed within each minibatch (S and prototypes both get gradient)
        'sim_recon_hidden_dim': 256,     # hidden width of the similarity decoder trunk
        'sim_recon_target': 'full',      # 'full': reconstruct each cell's actual aff_raw row (most literal match to SEACells); 'diffusion': regress to a precomputed per-cell diffusion-map coordinate instead (cheaper, compare both)
        'sim_recon_n_eigs': 10,          # diffusion-map dimensionality when sim_recon_target='diffusion'; unused for 'full'
        'sim_recon_diffusion_t': 0.0,    # 'diffusion' only: eigenvector weighting eigenvalue**t before the batch-size rescale. 0 (default) = unweighted, every eigen-direction equal — protects fine/local (rare sub-community) resolution. 0.5 makes the per-cell MSE mathematically equivalent (Eckart-Young) to MSE on a rank-n_eigs reconstruction of the affinity matrix itself — i.e. the closest 'diffusion' gets to behaving like sim_recon_target='full' (or SEACells' own RSS), at the cost of the same fine/rare-pattern sensitivity 0.0 protects. A real dial, not a bug fix — see files/sim_recon_global_vs_local_compaction.md.
        'sim_recon_neg_sample': 0,       # 'full' only: if >0, decode/reconstruct against each row's true-neighbor columns plus a random sample of this many zero columns (fresh sample each step) instead of every column in the batch — same loss, cheaper per step. 0 = original behavior (all columns)
        # --- Decoupled prototype learning (prevents proto collapse) ---
        'proto_decoupled': False,        # True: protos updated via online GMM EM; detached from all losses
        'gmm_eta': 0.1,                  # initial forgetting factor η (schedule: gmm_eta → gmm_eta_end over training)
        'gmm_eta_end': 0.5,              # final forgetting factor η
        'gmm_beta_start': 1.0,           # initial β for annealing (1=sharp; set <1 for softer early assignments)
        'gmm_beta_end': 1.0,             # final β
        'gmm_resurrect': False,          # True: split dominant proto into most unused one (disabled by default)
        'gmm_resurrect_thresh': 3.0,     # resurrect when π_k > thresh/K (multiple of uniform weight 1/K)
        'umap_proto_metric': 'cosine',     # 'dotp' | 'cosine' | 'bhattacharyya' | 'jsd' | 'bhatt_dist' | 'hellinger' | 'idot' (idot=1-dotp in [0,1], fed through a,b kernel like hellinger)
        'jsd_min_dist': 0.1,             # min_dist for JSD kernel a,b calibration (in JSD units, max ~0.693); only used when umap_proto_metric='jsd'
        'dist_min_dist': 0.1,            # min_dist for distribution-distance kernel (bhatt_dist/jsd/hellinger). Auto-calibrated from data when calibrate_eps=1.
        'dist_spread': 0.3,              # spread for distribution-distance kernel. Auto-calibrated from data when calibrate_eps=1; smaller = faster q decay.
        'temperature_min': 0.04,
        'epsilon_min': 0.02,
        'sched_temp_eps': 0,
        'lsim': 'normal',
        # proto labeling version
        'pl_version': 1,
        'graph_mode': 'knn',
        'recon_v': 2,
        
        # in this version we pass self.epsilon instead of self.tempreture for kl and proto recon
        "version": 31 # changed proto recon, changed kl not to leak pos pair info
    }
    return defaults
