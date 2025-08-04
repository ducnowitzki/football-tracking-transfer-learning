import os
import argparse
import tensorflow as tf
import numpy as np
from model import SeqLabelVAE

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--feature_dim', type=int, default=300)
parser.add_argument('--intermediate_dim', type=int, default=128)
parser.add_argument('--hidden_dim', type=int, default=16)
parser.add_argument('--pitch_x_axis', type=int, default=105)
parser.add_argument('--pitch_y_axis', type=int, default=68)
parser.add_argument('--channels', type=int, default=3)
parser.add_argument('--timesteps', type=int, default=32)
parser.add_argument('--no_classes', type=int, default=9)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--sample_rate', type=float, default=0.15)
parser.add_argument('--labeled_sequences', type=str)
parser.add_argument('--labels', type=str)
parser.add_argument('--unlabeled_frames', type=str)
parser.add_argument('--weights_dir', type=str)
args = parser.parse_args()

# Create weights directory
os.makedirs(args.weights_dir, exist_ok=True)

# Load data
print("Loading data...")
unlabeled_frames = np.load(args.unlabeled_frames)
train_sequences = np.load(args.labeled_sequences)
train_labels = np.load(args.labels)

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

# Initialize optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

# Initialize metrics
total_loss_metric = tf.keras.metrics.Mean()
reconstruction_loss_metric = tf.keras.metrics.Mean()
kl_loss_metric = tf.keras.metrics.Mean()
classification_loss_metric = tf.keras.metrics.Mean()

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
        
        # KL losses
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
        kl_loss = kl_weight * (kl_loss_z + kl_loss_a)
        
        # Classification loss for labeled data
        a_mean_t_labeled, _, a_t_labeled, _, _, _ = model.encode(x_labeled)
        y_pred = model.classify(a_t_labeled)
        classification_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.sparse_categorical_crossentropy(
                    tf.repeat(y, args.timesteps),
                    y_pred
                ),
                axis=-1
            )
        )
        
        # Total loss
        total_loss = reconstruction_loss + kl_loss + classification_loss
    
    # Compute and apply gradients
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    # Update metrics
    total_loss_metric.update_state(total_loss)
    reconstruction_loss_metric.update_state(reconstruction_loss)
    kl_loss_metric.update_state(kl_loss)
    classification_loss_metric.update_state(classification_loss)
    
    return {
        'total_loss': total_loss_metric.result(),
        'reconstruction_loss': reconstruction_loss_metric.result(),
        'kl_loss': kl_loss_metric.result(),
        'classification_loss': classification_loss_metric.result()
    }

def reset_metrics():
    """Reset all metrics."""
    total_loss_metric.reset_states()
    reconstruction_loss_metric.reset_states()
    kl_loss_metric.reset_states()
    classification_loss_metric.reset_states()

def train():
    """Main training loop."""
    print("Starting training...")
    
    # Training loop
    for epoch in range(args.epochs):
        reset_metrics()
        
        # KL weight annealing
        if epoch < 10:
            kl_weight = (epoch ** 3) * 0.001
        else:
            kl_weight = 1.0
        
        # Number of batches per epoch
        n_batches = len(train_sequences) // args.batch_size
        
        for batch in range(n_batches):
            # Sample labeled data
            indices = np.random.choice(len(train_sequences), args.batch_size)
            x_labeled = train_sequences[indices]
            y = train_labels[indices]
            
            # Sample unlabeled data
            unlabeled_indices = np.random.choice(
                len(unlabeled_frames) - args.timesteps,
                int(args.batch_size * args.sample_rate)
            )
            x_unlabeled = np.array([
                unlabeled_frames[i:i + args.timesteps]
                for i in unlabeled_indices
            ])
            
            # Training step
            logs = train_step(
                x_labeled, y, x_unlabeled,
                args.batch_size,
                int(args.batch_size * args.sample_rate),
                kl_weight
            )
            
            # Log every 50 steps
            if batch % 50 == 0:
                print(f"Epoch {epoch + 1}/{args.epochs}, Batch {batch}/{n_batches}")
                for metric_name, value in logs.items():
                    print(f"{metric_name}: {value:.4f}")
        
        # Save weights after each epoch
        model.save_weights(
            os.path.join(args.weights_dir, f'weights_epoch_{epoch + 1}.h5')
        )
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1}/{args.epochs} completed")
        for metric_name, value in logs.items():
            print(f"{metric_name}: {value:.4f}")

if __name__ == '__main__':
    train()
