#!/bin/bash
#SBATCH --job-name=seqlabelvae_gpu
#SBATCH --output=slurm-%j.out
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=12:00:00

# Load modules
module purge
module load 2023
module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1

export TF_XLA_FLAGS=--tf_xla_auto_jit=0
export TF_ENABLE_AUTO_MIXED_PRECISION=0
export TF_ENABLE_ONEDNN_OPTS=0

# Optionally restrict GPU to deterministic algorithms
export TF_DETERMINISTIC_OPS=1

source $HOME/venvs/seqlabelvae-gpu/bin/activate

# Copy data to scratch
mkdir -p "$TMPDIR"/data
cp -r $HOME/football-tracking-transfer-learning/idsse-data/seqlabelvae/* "$TMPDIR"/data/

# Change dir
cd "$TMPDIR"

# Run training
python $HOME/football-tracking-transfer-learning/seqlabelvae/train.py \
  --feature-dim 300 \
  --intermediate-dim 128 \
  --hidden-dim 8 \
  --timesteps 32 \
  --epochs 30 \
  --unlabeled-batch-size 128 \
  --labeled-batch-size 4 \
  --channels 3 \
  --no-classes 44 \
  --cost-annealing False \
  --labeled-sequences "$TMPDIR"/data/split/train_sequences.npy \
  --labels "$TMPDIR"/data/split/train_labels.npy \
  --unlabeled-frames "$TMPDIR"/data/whole/unlabeled_frames.npy \
  --weight_path "$TMPDIR"/training_weights.h5

# Copy back results
cp training_weights.h5 $HOME/football-tracking-transfer-learning/seqlabelvae/
