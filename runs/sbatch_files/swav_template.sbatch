#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={memory}
#SBATCH --output={output_file}
#SBATCH --error={error_file}
#SBATCH --nice={nice_value}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --gres={gres}

export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR

source ~/.bashrc
conda activate shiny-apex
python main.py {model} --num_prototypes {num_prototypes} --latent_dims {latent_dims} --batch_size {batch_size} \
--augmentation_type {augmentation_type} --epsilon {epsilon} --cvae_loss_scaler {cvae_loss_scaler} \
--prot_decoding_loss_scaler {prot_decoding_loss_scaler} --model_version {model_version} \
--dimensionality_reduction {dimensionality_reduction} --k_neighbors {k_neighbors} --experiment_name {experiment_name} \
--ft_epochs {ft_epochs} --pretraining_epochs {pretraining_epochs} --training_type {training_type} \
--dataset_id {dataset_id} --workers {workers} --job_name {job_name} --decodable_prototypes {decodable_prototypes} \
--cvae_epochs {cvae_epochs} --prot_init {prot_init} --checkpoint_freq {checkpoint_freq} --umap_checkpoint_freq {umap_checkpoint_freq} \
--sinkhorn_iterations {sinkhorn_iterations} --freeze_prototypes_nepochs {freeze_prototypes_nepochs} \
--loss_type {loss_type} --hard_clustering {hard_clustering} \
--n_components {n_components} --longest_path {longest_path} --supervised_ratio {supervised_ratio} --multi_layer_protos {multi_layer_protos} \
--batch_removal_ratio {batch_removal_ratio} --use_bknn {use_bknn} --freeze_batch_embedding {freeze_batch_embedding} \
--batch_sinkhorn {batch_sinkhorn} --weighted_batch {weighted_batch} --knn_similarity {knn_similarity} \
--prot_emb_sim_reg {prot_emb_sim_reg} --propagation_reg {propagation_reg} --recon_loss {recon_loss} --no_sinkhorn {no_sinkhorn} \
--mask_probability {mask_probability} --default_dispersion {default_dispersion} --fold {fold} --spatial {spatial} --entropy_reg {entropy_reg} \
--temperature {temperature} --dataset_cnt {dataset_cnt} --queue_length {queue_length} --model_type {model_type} --l2norm {l2norm} --use_rbf {use_rbf} \
--assignment_metric {assignment_metric} --study_id {study_id} --warmup_epochs {warmup_epochs} --base_lr {base_lr} --use_counts {use_counts} --knn_method {knn_method}