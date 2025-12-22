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
#SBATCH --exclude=gpusrv36

echo "Running on node(s): $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR

# clean environment
unset PYTHONPATH
unset LD_LIBRARY_PATH

# activate conda properly
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate swav_knn

# source ~/.bashrc
# conda activate swav_knn

# export TMPDIR=$HOME/tmp
# mkdir -p $TMPDIR

# # 🔥 THE IMPORTANT CHANGE:
# source /etc/profile
# export PATH="$HOME/miniconda3/bin:$PATH"
# conda deactivate 2>/dev/null
# conda activate swav_knn

python main.py {model} {dataset_id} {mode} --mode {mode} --num_prototypes {num_prototypes} --latent_dims {latent_dims} --batch_size {batch_size} \
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
--lambda_commit {lambda_commit} --lambda_proto {lambda_proto} --recon_loss {recon_loss} --no_sinkhorn {no_sinkhorn} \
--mask_probability {mask_probability} --default_dispersion {default_dispersion} --fold {fold} --spatial {spatial} --entropy_reg {entropy_reg} \
--temperature {temperature} --dataset_cnt {dataset_cnt} --queue_length {queue_length} --model_type {model_type} --l2norm {l2norm} --use_rbf {use_rbf} \
--assignment_metric {assignment_metric} --study_id {study_id} --warmup_epochs {warmup_epochs} --base_lr {base_lr} --use_counts {use_counts} --knn_method {knn_method} \
--epoch_queue_starts {epoch_queue_starts} --affinity_type {affinity_type} --cell_w_mode {cell_w_mode} --beta {beta} --lambda_align {lambda_align} \
--lambda_l2 {lambda_l2} --lambda_recon {lambda_recon} --lambda_kl {lambda_kl} --lambda_balance {lambda_balance} --lambda_swav {lambda_swav} \
--recon_update_target {recon_update_target} --recon_type {recon_type} --normalize_loss {normalize_loss} --div_type {div_type} --weighted_kl {weighted_kl} \
--kl_sched {kl_sched} --swav_sched {swav_sched} --recon_start_epoch {recon_start_epoch} --kl_start_epoch {kl_start_epoch} --opt {opt} --start_warmup {start_warmup} \
--wd {wd} --final_lr {final_lr} --umap_metric {umap_metric} --lambda_p_uncertainty {lambda_p_uncertainty} --lambda_proto_entropy {lambda_proto_entropy} \
--full_dataset_mode {full_dataset_mode} --adoptive_eps {adoptive_eps} --p {p} --k_pos {k_pos} --softm {softm} --lambda_aff {lambda_aff} \
--two_sided {two_sided} --lsim {lsim}