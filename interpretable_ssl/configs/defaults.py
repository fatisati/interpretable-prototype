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
        "prot_init": 'kmeans', #can be kmeans or waypoint
        
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
        'nassoc_offdiag_norm': 'sqrt',   # [D4] OFF-DIAGONAL denominator. 'sqrt' = edges_between/sqrt(vol_p*vol_q) (current: geometric mean is dominated by the larger volume, so a small prototype leaking into a big one is nearly invisible). 'min' = edges_between/min(vol_p,vol_q) -- bills the leakage against the SMALLER prototype's own budget instead (conductance-of-the-smaller-side, Kannan/Vempala/Vetta 2004), so a rare prototype's leakage registers. Independent of nassoc_diag_norm -- diagonal and off-diagonal denominators don't need to match. See literature/spatial/nassoc_redesign.md Fix D4.
        'nassoc_size_scale': 'btch',     # [D5] scale of size/possible_pairs used by the DIAGONAL. 'btch' (current) uses the raw minibatch soft counts -- but possible_pairs is QUADRATIC in size, so any prototype whose soft minibatch size is < 1 gets pairs clamped to exactly 0 and density = edges/eps EXPLODES (measured: mean diag 371-510 against a target of 1). Under (density-1)^2 an EMPTY prototype then costs 1 while a merely SMALL one costs ~1e16, so the loss actively pays the model to kill prototypes -- the observed 15/200-survivor collapse. 'full' rescales both counts to full-dataset scale (size*n_cells/n_batch, edges*n_edges/n_edges_batch) before forming density, so sizes are rarely <1, density lands in [0,1], and nassoc_gamma becomes interpretable as a real density (~mean_weighted_degree/(target_size-1)).
        'nassoc_cpm_norm': 'pair',       # [D2] denominator of the 'cpm' diagonal. 'pair' (current) divides by sum(possible_pairs) -- the SAME quantity gamma multiplies, so gamma algebraically cancels to an additive constant and has EXACTLY ZERO GRADIENT (confirmed: a 10x gamma sweep moved largest_share only 0.168->0.159). 'graf' divides by the summed affinity weight of THIS STEP's sampled edges: assignment-independent, so gamma survives -- but minibatch-scaled, which mismatches a full-dataset-scale numerator when nassoc_size_scale='full' and inflates the term by ~n_edges_total/n_edges_batch (measured: nassoc=2874 vs umap=2.5, i.e. 99.5% of the total loss). 'gful' divides by the FULL-graph total edge weight instead -- this is DMoN's 1/(2m) exactly -- and is the one to use with nassoc_size_scale='full' so numerator and denominator are on the same scale. 'dpar' divides by DETACHED sum(possible_pairs): since score[p] in [-gamma*pairs, (1-gamma)*pairs], this puts the whole term in [-(1-gamma), +gamma] for any dataset/batch size -- the natural bound implied by the objective itself, and the right partner for nassoc_size_scale='btch' (everything batch-native, no rescaling factors that can silently mismatch). Detaching is what keeps gamma's gradient alive: undetached, gamma*sum(pairs)/sum(pairs) collapses to a constant, which is the 'pair' bug. This normalization convention is DMoN's (Tsitsulin et al. JMLR 2023); applying it to CPM, and the detached-pairs variant, are our own adaptations, not published results.
        'nassoc_cpm_free': 0,            # [D2] 1 = compute the 'cpm' diagonal in its published DIVISION-FREE form, score[p] = edges_inside[p] - gamma*possible_pairs[p] (Traag/Van Dooren/Nesterov PRE 2011), instead of the algebraically-identical but numerically fragile (density-gamma)*possible_pairs. At possible_pairs=0 the subtraction gives edges-0 (finite) while the ratio form gives inf*0. This is why Leiden never hits the degenerate-cluster problem: CPM as published is a subtraction, never a ratio.
        'nassoc_exact': 0,               # [D6] 1 = compute the CPM terms exactly. Two corrections, both verified against the data: (a) possible_pairs = (size^2 - sum_i assign^2)/2, the EXACT soft pair count (identity for sum_{i<j} assign_i*assign_j), which reduces to n(n-1)/2 for hard assignments and is always >=0 -- unlike size*(size-1)/2, which is the hard formula misapplied to soft counts, goes negative when size<1, gets clamped to 0 and causes the divide-by-zero behind the prototype collapse; (b) edges_inside was 2x inflated and contaminated by self-loops, because the affinity is symmetric AND stores aff[i,i], so EdgeDataset lists each undirected edge twice plus one self-loop per cell (bk02: nnz=4,648,883 = 2*2,295,230 + 58,423). Inflating edges_inside but not gamma*possible_pairs HALVES the effective gamma, which is why gamma never acted as the density threshold it is defined to be.
        'lambda_collapse': 0.0,          # DMoN collapse regularization (Tsitsulin et al. JMLR 2023, eq. 5): (sqrt(K)/n_cells)*||soft_sizes||_2 - 1. Equals sqrt(K/effective_K)-1, i.e. a smooth monotone function of how many prototypes are actually in use: 0 when sizes are balanced, sqrt(K)-1 when everything collapses into one. A SOFT balance prior -- unlike SwAV/Sinkhorn it never forces equal sizes, which is what we want (prototype sizes should be set by biology). Scale-free in the minibatch, so no sampling-rate bias. Set >0 to use it alongside or instead of lambda_proto_usage.
        'proto_reinit_every': 0,         # dead-prototype restart (VQ-VAE codebook restarts, van den Oord et al. 2017; the random-restart trick in Jukebox, Dhariwal et al. 2020). 0 = off. If >0, every N epochs any prototype whose EMA usage is below proto_reinit_thresh is teleported onto a currently worst-served cell (the cell with the lowest max assignment probability, i.e. farthest from every existing prototype). Gradient-based penalties cannot revive a prototype that already has zero assigned mass -- there is no gradient path back -- so a direct re-teleport is the only mechanism that acts on an already-dead slot.
        'proto_reinit_thresh': 0.01,     # EMA-usage level below which a prototype counts as dead for proto_reinit_every. Only read when proto_reinit_every > 0.
        # --- Cell-cell similarity reconstruction (resolution pressure nassoc can't provide) ---
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
