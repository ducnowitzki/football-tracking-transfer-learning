#!/usr/bin/env python3
"""
Optimized training script for SeqLabelVAE with mixed precision,
fixed input signatures, GPU memory growth, sampled unlabeled windows,
and optional KL cost annealing.
"""
import os
# Disable deterministic ops to allow fused batch-norm backprop on GPU
os.environ['TF_DETERMINISTIC_OPS'] = '0'

"""
Optimized training script for SeqLabelVAE with mixed precision,
fixed input signatures, GPU memory growth, sampled unlabeled windows,
and optional KL cost annealing.
"""
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
import model_utils  # utilities for feature extractors and encoders

# Enable mixed precision for GPU throughput
mixed_precision.set_global_policy('mixed_float16')

# GPU configuration: memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Enabled memory growth for {len(gpus)} GPU(s).")
    except Exception as e:
        print(f"Could not set memory growth: {e}")
else:
    print("No GPU found, running on CPU.")


class SeqLabelVAE(model_utils.keras.Model):
    def __init__(self, recurrent_encoder, timewise_MLP_z, timewise_MLP_a,
                 recurrent_decoder, classifier, **kwargs):
        super(SeqLabelVAE, self).__init__(**kwargs)
        self.recurrent_encoder = recurrent_encoder
        self.recurrent_decoder = recurrent_decoder
        self.timewise_MLP_z    = timewise_MLP_z
        self.timewise_MLP_a    = timewise_MLP_a
        self.classifier        = classifier

    def encode(self, x):
        h_t = self.recurrent_encoder(x)
        a_mean, a_log_sigma, a_t = self.timewise_MLP_a(h_t)
        z_mean, z_log_sigma, z_t = self.timewise_MLP_z([h_t, a_t])
        return a_mean, a_log_sigma, a_t, z_mean, z_log_sigma, z_t

    def decode(self, a_t, z_t):
        return self.recurrent_decoder([a_t, z_t])

    def classify(self, a_t):
        return self.classifier(a_t)


def parse_args():
    parser = argparse.ArgumentParser(description='Train SeqLabelVAE efficiently')
    parser.add_argument('--unlabeled_frames', type=str, required=True,
                        help='Path to npy file of unlabeled frames')
    parser.add_argument('--labeled_sequences', type=str, required=True,
                        help='Path to npy file of labeled sequences')
    parser.add_argument('--labels', type=str, required=True,
                        help='Path to npy file of labels')
    parser.add_argument('--timesteps', type=int, default=20,
                        help='Number of frames per window')
    parser.add_argument('--channels', type=int, default=9,
                        help='Number of channels in input representation')
    parser.add_argument('--labeled_batch_size', type=int, default=32,
                        help='Batch size for labeled data')
    parser.add_argument('--unlabeled_batch_size', type=int, default=64,
                        help='Batch size for unlabeled data')
    parser.add_argument('--sample_rate', type=float, default=0.05,
                        help='Fraction of total windows to sample for unlabeled data')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--weight_path', type=str, default='weights.h5',
                        help='Path to save model weights')
    parser.add_argument('--feature_dim', type=int, default=300,
                        help='Latent feature dimension')
    parser.add_argument('--intermediate_dim', type=int, default=128,
                        help='Intermediate embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=8,
                        help='Hidden state dimension for RNNs')
    parser.add_argument('--pitch_x_axis', type=int, default=105,
                        help='Pitch length for spatial discretization')
    parser.add_argument('--pitch_y_axis', type=int, default=68,
                        help='Pitch width for spatial discretization')
    parser.add_argument('--no_classes', type=int, default=5,
                        help='Number of event classes')
    parser.add_argument('--cost_annealing', type=bool, default=False,
                        help='Enable linear KL cost annealing over epochs')
    return parser.parse_args()


def build_architecture(opt):
    # Based on original train.py implementation
    # f: (105, 68, channels) -> (feature_dim,)
    input_feature_extractor = model_utils.get_DCGAN_feature_extractor(
        opt.feature_dim, opt.pitch_x_axis, opt.pitch_y_axis, opt.channels)
    # f: (timesteps, 105, 68, channels) -> (timesteps, intermediate_dim)
    recurrent_encoder = model_utils.get_recurrent_encoder_timesteps(
        opt.hidden_dim, opt.intermediate_dim,
        opt.pitch_x_axis, opt.pitch_y_axis,
        opt.channels, opt.timesteps,
        input_feature_extractor)
    # f: (timesteps, intermediate_dim) -> (timesteps, hidden_dim)
    MLP_a = model_utils.get_timewise_MLP_a(
        opt.timesteps, opt.intermediate_dim, opt.hidden_dim)
    # f: (timesteps, intermediate_dim + hidden_dim) -> (timesteps, hidden_dim)
    MLP_z = model_utils.get_timewise_MLP_z(
        opt.timesteps, opt.intermediate_dim, opt.hidden_dim)
    # f: (feature_dim,) -> (105, 68, channels)
    reverse_feature_extractor = model_utils.get_DCGAN_reverse_feature_extractor(
        opt.feature_dim, opt.pitch_x_axis, opt.pitch_y_axis, opt.channels)
    # f: (timesteps, 2*hidden_dim) -> (timesteps, 105, 68, channels)
    recurrent_decoder = model_utils.get_recurrent_decoder_timesteps_flat_latents(
        opt.timesteps, opt.hidden_dim, opt.intermediate_dim,
        opt.feature_dim, reverse_feature_extractor)
    # f: (timesteps, hidden_dim) -> (timesteps, no_classes)
    classifier = model_utils.get_MLP_classifier_timewise(
        opt.timesteps, opt.hidden_dim, opt.no_classes)


    recurrent_encoder = tf.recompute_grad(recurrent_encoder)
    recurrent_decoder = tf.recompute_grad(recurrent_decoder)

    return SeqLabelVAE(
        recurrent_encoder, MLP_z, MLP_a, recurrent_decoder, classifier)


def compute_losses(x, y=None, training=False):
    """
    Compute VAE losses: reconstruction, KL divergence, and optional classification loss.
    Returns (rec_loss, kl_loss, cls_loss).
    """
    # Encode and decode
    a_mean, a_log_sigma, a_t, z_mean, z_log_sigma, z_t = model.encode(x)
    x_recon = model.decode(a_t, z_t)
    # Reconstruction loss (MSE)
    rec_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(x, x_recon))
    # KL divergence for 'a' and 'z'
    kl_a = -0.5 * tf.reduce_sum(1 + a_log_sigma - tf.square(a_mean) - tf.exp(a_log_sigma), axis=[1,2])
    kl_z = -0.5 * tf.reduce_sum(1 + z_log_sigma - tf.square(z_mean) - tf.exp(z_log_sigma), axis=[1,2])
    kl_loss = tf.reduce_mean(kl_a + kl_z)
    # Classification loss (match label to final timestep prediction)
    cls_loss = 0.0
    if y is not None:
        # a_t shape: (batch, timesteps, hidden_dim)
        y_pred_seq = model.classify(a_t)  # (batch, timesteps, no_classes)
        y_pred_last = y_pred_seq[:, -1, :]  # take logits at final timestep
        cls_loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(y, y_pred_last)
        )
    return rec_loss, kl_loss, cls_loss


@tf.function(input_signature=[
    tf.TensorSpec([None, None, 105, 68, None], tf.float32),
    tf.TensorSpec([None, None, 105, 68, None], tf.float32),
    tf.TensorSpec([None], tf.int32),
    tf.TensorSpec([], tf.float32),
])
def train_step(x_unlabeled, x_labeled, y_true, kl_weight):
    """
    Single training step for both labeled and unlabeled data with dtype consistency.
    """
    # Ensure kl_weight matches model dtype for mixed precision
    kl_weight = tf.cast(kl_weight, tf.float16)
    with tf.GradientTape() as tape:
        rec_unl, kl_unl, _   = compute_losses(x_unlabeled, training=True)
        rec_lab, kl_lab, cls_lab = compute_losses(x_labeled, y_true, training=True)
        unlabeled_loss = rec_unl + kl_weight * kl_unl
        labeled_loss   = rec_lab + kl_weight * kl_lab + cls_lab
        total_loss     = unlabeled_loss + labeled_loss
        scaled_loss    = optimizer.get_scaled_loss(total_loss)
    grads = tape.gradient(scaled_loss, model.trainable_variables)
    grads = optimizer.get_unscaled_gradients(grads)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return {'loss': total_loss,
            'rec_loss': rec_lab + rec_unl,
            'kl_loss': kl_unl + kl_lab,
            'cls_loss': cls_lab}


def make_unlabeled_ds(frames, timesteps, channels, batch_size, sample_rate):
    total_windows = frames.shape[0] - timesteps + 1
    num_to_sample = int(sample_rate * total_windows)
    print(f"Sampling {num_to_sample}/{total_windows} windows ({sample_rate*100:.1f}%)")
    indices = np.random.choice(total_windows, num_to_sample, replace=False)
    ds = tf.data.Dataset.from_tensor_slices(indices)
    def _slice_window(idx):
        start = int(idx)
        return frames[start:start+timesteps].astype(np.float32)
    ds = ds.map(lambda idx: tf.numpy_function(_slice_window, [idx], tf.float32),
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x: tf.ensure_shape(x, [timesteps, 105, 68, channels]))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def make_labeled_ds(seqs, labels, batch_size):
    print(f"Building labeled dataset: {seqs.shape[0]} sequences")
    ds = tf.data.Dataset.from_tensor_slices((seqs, labels))
    ds = ds.shuffle(10_000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds.repeat()


def main():
    opt = parse_args()
    print("Parsed arguments:", opt)
    tf.random.set_seed(42)
    print("Loading data...")
    unlabeled_frames = np.load(opt.unlabeled_frames, mmap_mode='r')
    labeled_seqs     = np.load(opt.labeled_sequences, mmap_mode='r')
    labels           = np.load(opt.labels, mmap_mode='r')
    print(f"Data shapes - unlabeled: {unlabeled_frames.shape}, labeled: {labeled_seqs.shape}, labels: {labels.shape}")

    model_ds = make_unlabeled_ds(unlabeled_frames, opt.timesteps,
                                 opt.channels, opt.unlabeled_batch_size,
                                 opt.sample_rate)
    label_ds = make_labeled_ds(labeled_seqs, labels, opt.labeled_batch_size)
    train_ds = tf.data.Dataset.zip((model_ds, label_ds))

    base_opt = tf.keras.optimizers.Adam(learning_rate=opt.learning_rate)
    global optimizer; optimizer = mixed_precision.LossScaleOptimizer(base_opt)
    print("Optimizer configured.")

    global model; model = build_architecture(opt)
    print("Model built.")

    total_windows = unlabeled_frames.shape[0] - opt.timesteps + 1
    steps_per_epoch = (int(opt.sample_rate * total_windows) // opt.unlabeled_batch_size)
    print(f"Training for {opt.epochs} epochs, {steps_per_epoch} steps/epoch")
    print(f"KL cost annealing: {opt.cost_annealing}")

    for epoch in range(1, opt.epochs+1):
        print(f"\nEpoch {epoch}/{opt.epochs}")
        kl_weight = (epoch/opt.epochs) if opt.cost_annealing else 1.0
        print(f"KL weight: {kl_weight:.4f}")
        epoch_loss = 0.0
        for step, (x_unl, (x_lab, y_lab)) in enumerate(train_ds.take(steps_per_epoch), 1):
            logs = train_step(x_unl, x_lab, y_lab, kl_weight)
            epoch_loss += logs['loss']
            if step==1 or step%50==0:
                print(f"Step {step}/{steps_per_epoch} - loss={logs['loss']:.4f}")
        print(f"Avg loss: {epoch_loss/steps_per_epoch:.4f}")
        ckpt = f"{opt.weight_path}.epoch{epoch}.h5"
        model.save_weights(ckpt)
        print(f"Saved {ckpt}")

if __name__ == '__main__':
    main()
