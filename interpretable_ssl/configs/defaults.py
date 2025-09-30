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
        "views_for_assign": [0, 1],  # swav specific
        "nmb_views": [4],  # swav specific
        
        "sinkhorn_iterations": 3,  # swav specific
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
        "model": "",  # swav specific
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
        
        "prot_emb_sim_reg": 0.0,
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
        "recon_loss": 'nb',
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
        
        "l2norm": 1,
        "use_rbf": 1,
        'assignment_metric': 'dotp',
        
        "cvae_epochs": 50,
        "pretraining_epochs": 100,
        "ft_epochs": 20,
        
        "cvae_loss_scaler": 0.01, # 0.01,  # swav specific, 0.0001
        "propagation_reg": 1, # 1,
       
        "model_type": 'gm',
        
        "workers": 1,  # swav specific
        
        "wd": 1e-6,  # swav specific
        "base_lr": 4.8,  # swav specific
        "final_lr": 0.0048,  # swav specific
        "warmup_epochs": 10,  # swav specific
        "start_warmup": 0.3,  # swav specific
        
        "batch_size": 1024,
        
        
        "freeze_prototypes_nepochs": 1, # used to be 50
        
        "hard_clustering": 0,
        
        "temperature": 0.1,  # swav specific, lower make sharper assignment of z to protos, swav default 0.1
        "epsilon": 0.05,  # swav specific, swav default: 0.05 
        
        "epoch_queue_starts": 0, # used to be 5
        "queue_length":  0,
        
        "use_counts": 1,
        "knn_method": 'faiss',
        "version": 5,
        "affinity_type": "arbf",
        'cell_w_mode': 'uniform',
        'assignment_mode': 'gm' # TODO: remove addignment metric
        # change to dotp in future
    }
    return defaults
