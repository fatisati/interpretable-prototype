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
        "prot_init": 'kmeans', #can be kmeans
        
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
        
        "affinity_type": "arbf",
        'cell_w_mode': 'uniform',
        "model": "swav",  # swav specific
        "lambda_align": 0,
        
        "beta": 0.3, # not used any more
        
        "lambda_l2": 0.0, # 1e-3
        
        "lambda_kl": 0.1,
        "lambda_recon": 1.0, 
        'lambda_balance': 0.0,
        'lambda_swav': 1.0,
        "lambda_proto": 0.0, # 1,
        "lambda_commit": 0.0, # 0.25
        "lambda_p_uncertainty": 0.0,
        'lambda_proto_entropy': 0.0,
        
        "lambda_proto_recon": 0.0,
        'assignment_metric': 'sneuc',
        "model_type": 'gm',
        "l2norm": 0,
        

        "recon_update_target": 'encoder',
        "description": '17: calc affinity on whole datase instead of batch wise',
        'recon_type': 'normal',
        'normalize_loss': 0,
        'div_type': 'ce',
        
        'kl_sched': 1,
        'swav_sched': 0, # not used yet
        'recon_start_epoch': 0,
        'kl_start_epoch': 10,
        "cvae_epochs": 25,
        "pretraining_epochs": 50,
        "ft_epochs": 10,
        
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
        'p': 0.5,
        'k_pos': 0,
        'softm': 0,
        'lambda_aff': 0.0,
        'two_sided': 0,
        'temperature_min': 0.04,
        'epsilon_min': 0.02,
        'sched_temp_eps': 0,
        'lsim': 'normal',
        # proto labeling version
        'pl_version': 1,
        'graph_mode': 'knn',
        'recon_v': 2,
        
        # in this version we pass self.epsilon instead of self.tempreture for kl and proto recon
        "version": 30 # changed proto recon, changed kl not to leak pos pair info
    }
    return defaults
