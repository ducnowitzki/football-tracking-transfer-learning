import os
import argparse
import tensorflow as tf
import numpy as np
from model import SeqLabelVAE
from datetime import datetime
import json
from collections import Counter
from sklearn.utils import resample

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--feature_dim', type=int, default=300)
parser.add_argument('--intermediate_dim', type=int, default=128)
parser.add_argument('--hidden_dim', type=int, default=16)
parser.add_argument('--pitch_x_axis', type=int, default=68)  # Updated to match preprocessing W_g
parser.add_argument('--pitch_y_axis', type=int, default=105)  # Updated to match preprocessing H_g
parser.add_argument('--channels', type=int, default=9)  # Updated to match new channel count
parser.add_argument('--timesteps', type=int, default=32)
parser.add_argument('--no_classes', type=int, default=3)  # Updated to match actual classes (pass, shot, dribble)
parser.add_argument('--unlabeled_batch_size', type=int, default=64)
parser.add_argument('--labeled_batch_size', type=int, default=4)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--sample_rate', type=float, default=0.05)  # Updated to match preprocessing
parser.add_argument('--labeled_sequences', type=str)
parser.add_argument('--labels', type=str)
parser.add_argument('--unlabeled_frames', type=str)
parser.add_argument('--weights_dir', type=str)
parser.add_argument('--logs_dir', type=str)
parser.add_argument('--changes', type=str)
parser.add_argument('--alpha', type=float, default=0.1, help='Weight for classification loss')
parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma parameter')
args = parser.parse_args()

# Create weights and logs directories
os.makedirs(args.weights_dir, exist_ok=True)
os.makedirs(args.logs_dir, exist_ok=True)

# Create timestamp for log file
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(args.logs_dir, f'training_log_{timestamp}_{args.changes}.json')

# Initialize logging
def log_metrics(epoch, batch, logs):
    """Log metrics to file."""
    log_entry = {
        'epoch': epoch + 1,
        'batch': batch,
        'metrics': {k: float(v) for k, v in logs.items()}
    }
    
    # Append to log file
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

# Load data
print("Loading data...")
unlabeled_sequences = np.load(args.unlabeled_frames)
train_sequences = np.load(args.labeled_sequences)
train_labels = np.load(args.labels)

print(f"Unlabeled sequences shape: {unlabeled_sequences.shape}, dtype: {unlabeled_sequences.dtype}")
print(f"Train sequences shape: {train_sequences.shape}, dtype: {train_sequences.dtype}")
print(f"Train labels shape: {train_labels.shape}, dtype: {train_labels.dtype}")

# Print a sample of the data for sanity check
print("Sample labeled sequence min/max:", np.min(train_sequences), np.max(train_sequences))
print("Sample unlabeled sequence min/max:", np.min(unlabeled_sequences), np.max(unlabeled_sequences))
print("Sample labels:", np.unique(train_labels, return_counts=True))

# Initialize model
print("Initializing model...")
model = SeqLabelVAE(
    feature_dim=args.feature_dim,
    intermediate_dim=args.intermediate_dim,
    hidden_dim=args.hidden_dim,
    pitch_x_axis=args.pitch_x_axis,
    pitch_y_axis=args.pitch_y_axis,
    channels=args.channels,
    timesteps=args.timesteps,
    no_classes=args.no_classes
)

# Print model input spec if possible
try:
    print("Model input spec:", model.input_shape)
except Exception as e:
    print("Could not print model input shape:", e)

# Initialize optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

# Initialize metrics
total_loss_metric = tf.keras.metrics.Mean()
reconstruction_loss_metric = tf.keras.metrics.Mean()
kl_loss_metric = tf.keras.metrics.Mean()
kl_loss_z_metric = tf.keras.metrics.Mean()
kl_loss_a_metric = tf.keras.metrics.Mean()
classification_loss_metric = tf.keras.metrics.Mean()

# Initialize focal loss function with class-specific alpha values
# Alpha values for: [pass, shot, dribble] - adjust based on your actual class order
focal_loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(
    gamma=args.focal_gamma,
    alpha=[0.01, 0.41, 0.58],  # order = [pass, shot, dribble]
    from_logits=False
)

@tf.function
def train_step(x_labeled, y, x_unlabeled, labeled_batch_size, unlabeled_batch_size, kl_weight):
    """Single training step."""
    with tf.GradientTape() as tape:
        # Forward pass for unlabeled data
        a_mean_t, a_log_sigma_t, a_t, z_mean_t, z_log_sigma_t, z_t = model.encode(x_unlabeled)
        reconstruction = model.decode(a_t, z_t)
        
        # Reconstruction loss
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(x_unlabeled, reconstruction),
                axis=(1, 2, 3)
            )
        )
        
        # KL losses with minimum threshold
        kl_loss_z = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_sigma_t - tf.square(z_mean_t) - tf.exp(z_log_sigma_t),
                axis=(1, 2)
            )
        )
        kl_loss_a = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + a_log_sigma_t - tf.square(a_mean_t) - tf.exp(a_log_sigma_t),
                axis=(1, 2)
            )
        )
        
        # Apply minimum threshold to KL losses
        kl_min = 2.0
        kl_loss = kl_weight * (
            tf.keras.backend.maximum(kl_loss_z, kl_min) + 
            tf.keras.backend.maximum(kl_loss_a, kl_min)
        )
        
        # Classification loss for labeled data using TensorFlow's focal loss
        a_mean_t_labeled, _, a_t_labeled, _, _, _ = model.encode(x_labeled)
        y_pred = model.classify(a_t_labeled)
        
        # Convert sparse labels to one-hot encoding for focal loss
        y_reshaped = tf.reshape(tf.repeat(y, args.timesteps), (labeled_batch_size, args.timesteps))
        y_one_hot = tf.one_hot(tf.cast(y_reshaped, tf.int32), depth=args.no_classes)
        
        # Apply focal loss - CategoricalFocalCrossentropy already returns per-sample losses
        classification_loss_un = tf.reduce_mean(
            focal_loss_fn(y_one_hot, y_pred)
        )
        # classification_loss = args.alpha * ((unlabeled_batch_size + labeled_batch_size) / labeled_batch_size) * classification_loss_un
        classification_loss = args.alpha * classification_loss_un
        
        # Total loss
        total_loss = reconstruction_loss + kl_loss + classification_loss
    
    # Compute and apply gradients
    gradients = tape.gradient(total_loss, model.trainable_variables)
    # clip gradients
    # gradients = [tf.clip_by_norm(g, 1.0) for g in gradients]
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Update metrics
    total_loss_metric.update_state(total_loss)
    reconstruction_loss_metric.update_state(reconstruction_loss)
    kl_loss_metric.update_state(kl_loss)
    kl_loss_z_metric.update_state(kl_loss_z)
    kl_loss_a_metric.update_state(kl_loss_a)
    classification_loss_metric.update_state(classification_loss_un)
    
    return {
        'total_loss': total_loss_metric.result(),
        'reconstruction_loss': reconstruction_loss_metric.result(),
        'kl_loss_z (unweighted)': kl_loss_z_metric.result(),
        'kl_loss_a (unweighted)': kl_loss_a_metric.result(),
        'kl_loss (weighted)': kl_loss_metric.result(),
        'classification_loss (unweighted)': classification_loss_metric.result()
    }

def reset_metrics():
    """Reset all metrics."""
    total_loss_metric.reset_states()
    reconstruction_loss_metric.reset_states()
    kl_loss_metric.reset_states()
    kl_loss_z_metric.reset_states()
    kl_loss_a_metric.reset_states()
    classification_loss_metric.reset_states()

def train():
    """Main training loop."""
    print("Starting training...")
    print(f"Logging to: {log_file}")
    
    # Unlabeled data is now already sequences, so we can sample directly
    total_unlabeled_sequences = unlabeled_sequences.shape[0]
    sequences_per_epoch = int(args.sample_rate * total_unlabeled_sequences)
    
    print(f"Total unlabeled sequences: {total_unlabeled_sequences}")
    print(f"Sequences per epoch: {sequences_per_epoch}")
    
    for epoch in range(args.epochs):
        reset_metrics()
        
        # KL weight annealing
        if epoch < 10:
            kl_weight = 0.001 * (float(epoch) ** 3)
        else:
            kl_weight = 1.0

        print(f"KL weight: {kl_weight}")
        
        # Shuffle indices for this epoch
        labeled_indices = np.random.permutation(len(train_sequences))
        unlabeled_indices = np.random.permutation(total_unlabeled_sequences)
        
        # Number of batches per epoch for unlabeled data
        n_batches = sequences_per_epoch // args.unlabeled_batch_size
        
        for batch in range(n_batches):
            # Get current batch indices
            labeled_batch_start = (batch * args.labeled_batch_size) % len(train_sequences)
            unlabeled_batch_start = batch * args.unlabeled_batch_size
            
            # Sample labeled data (wrap around if needed)
            if labeled_batch_start + args.labeled_batch_size > len(train_sequences):
                # Handle wrap-around for labeled data
                remaining = len(train_sequences) - labeled_batch_start
                first_part = labeled_indices[labeled_batch_start:]
                second_part = labeled_indices[:args.labeled_batch_size - remaining]
                current_labeled_indices = np.concatenate([first_part, second_part])
            else:
                current_labeled_indices = labeled_indices[labeled_batch_start:labeled_batch_start + args.labeled_batch_size]
            
            x_labeled = train_sequences[current_labeled_indices]
            y = train_labels[current_labeled_indices]
            
            # Sample unlabeled sequences directly (no need to create sequences from frames)
            current_unlabeled_indices = unlabeled_indices[unlabeled_batch_start:unlabeled_batch_start + args.unlabeled_batch_size]
            x_unlabeled = unlabeled_sequences[current_unlabeled_indices]
            
            # Log batch shapes and dtypes
            print(f"Batch {batch}: x_labeled shape: {x_labeled.shape}, dtype: {x_labeled.dtype}")
            print(f"Batch {batch}: y shape: {y.shape}, dtype: {y.dtype}")
            print(f"Batch {batch}: x_unlabeled shape: {x_unlabeled.shape}, dtype: {x_unlabeled.dtype}")

            # Training step
            logs = train_step(
                x_labeled, y, x_unlabeled,
                args.labeled_batch_size,
                args.unlabeled_batch_size,
                kl_weight
            )
            
            # Log every 50 steps
            if batch % 50 == 0:
                print(f"Epoch {epoch + 1}/{args.epochs}, Batch {batch}/{n_batches}")
                log_metrics(epoch, batch, logs)
                print("Metrics logged")
        
        # Save weights after each epoch
        model.save_weights(
            os.path.join(args.weights_dir, f'weights_epoch_{epoch + 1}.h5')
        )
        
        # Print epoch summary and log final metrics
        print(f"\nEpoch {epoch + 1}/{args.epochs} completed")
        for metric_name, value in logs.items():
            print(f"{metric_name}: {value:.4f}")
        log_metrics(epoch, n_batches - 1, logs)

if __name__ == '__main__':
    train()
