#!/bin/bash
#SBATCH --job-name=seqlabelvae_gpu
#SBATCH --output=slurm-%j.out
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1               # request one A100 GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1                 # single process
#SBATCH --cpus-per-task=18         # threads for data-pipeline
#SBATCH --mem=120G
#SBATCH --time=12:00:00

#–– Environment setup ––#
module purge
module load 2023
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1

# Threads & affinity
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTRAOP_THREADS=$SLURM_CPUS_PER_TASK
export TF_NUM_INTEROP_THREADS=2

# Mixed-precision and XLA left to train.py
# (Remove TF_ENABLE_AUTO_MIXED_PRECISION and TF_XLA_FLAGS overrides)


# Activate your venv
source $HOME/venvs/seqlabelvae-gpu/bin/activate

#–– Stage data onto local NVMe scratch ––#
mkdir -p "$TMPDIR"/data
rsync -a $HOME/football-tracking-transfer-learning/idsse-data/seqlabelvae/ "$TMPDIR"/data/

cd "$TMPDIR"

#–– Launch training ––#
python $HOME/football-tracking-transfer-learning/seqlabelvae/train2.py \
  --timesteps 32 \
  --epochs 30 \
  --unlabeled_batch_size 64 \
  --labeled_batch_size 4 \
  --channels 3 \
  --feature_dim 300 \
  --intermediate_dim 96 \
  --hidden_dim 8 \
  --no_classes 44 \
  --cost_annealing False \
  --labeled_sequences "$TMPDIR"/data/split/train_sequences.npy \
  --labels            "$TMPDIR"/data/split/train_labels.npy \
  --unlabeled_frames  "$TMPDIR"/data/whole/unlabeled_frames.npy \
  --weight_path       "$TMPDIR"/training_weights.h5 \
  --sample_rate 0.15

#–– Copy back weights ––#
cp "$TMPDIR"/training_weights.h5 \
   $HOME/football-tracking-transfer-learning/seqlabelvae/
