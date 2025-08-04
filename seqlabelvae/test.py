import os
import argparse
import numpy as np
import tensorflow as tf
from model import SeqLabelVAE
from datetime import datetime
import json
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dim', type=int, default=300)
    parser.add_argument('--intermediate_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--pitch_x_axis', type=int, default=68)
    parser.add_argument('--pitch_y_axis', type=int, default=105)
    parser.add_argument('--channels', type=int, default=9)
    parser.add_argument('--timesteps', type=int, default=32)
    parser.add_argument('--no_classes', type=int, default=3)
    parser.add_argument('--test_sequences', type=str, required=True)
    parser.add_argument('--test_labels', type=str, required=True)
    parser.add_argument('--weights_file', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--alpha', type=float, default=0.1, help='Weight for classification loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma parameter')
    parser.add_argument('--save_plots', action='store_true', help='Save confusion matrix plot')
    parser.add_argument('--plots_dir', type=str, default='plots', help='Directory to save plots')
    parser.add_argument('--show_plots', action='store_true', default=True, help='Show confusion matrix plot')
    return parser.parse_args()

def normalize_data(data):
    """Normalize data to [0, 1] range."""
    return data / 255.0

def log_metrics(log_file, logs):
    """Log metrics to file."""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'metrics': {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                   for k, v in logs.items()}
    }
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

def calculate_classification_metrics(y_true, y_pred_probs):
    """Calculate classification metrics."""
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Get unique classes in test set
    unique_classes = np.unique(y_true)
    print(f"\nClasses present in test set: {unique_classes}")
    
    # Initialize metrics dictionary
    metrics = {}
    
    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=unique_classes
    )
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=unique_classes)
    
    # Calculate AUC for each class (one-vs-rest)
    auc_scores = []
    for i in unique_classes:
        try:
            auc = roc_auc_score(
                (y_true == i).astype(int),
                y_pred_probs[:, i]
            )
            auc_scores.append(auc)
        except ValueError:
            auc_scores.append(np.nan)
    
    # Store per-class metrics
    for i, class_idx in enumerate(unique_classes):
        metrics[f'class_{class_idx}'] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i]),
            'auc': float(auc_scores[i])
        }
    
    # Calculate accuracy
    accuracy = np.mean(y_true == y_pred)
    
    # Calculate macro-averaged metrics
    metrics['macro_avg'] = {
        'precision': float(np.mean(precision)),
        'recall': float(np.mean(recall)),
        'f1': float(np.mean(f1)),
        'auc': float(np.nanmean(auc_scores)),
        'accuracy': float(accuracy)
    }
    
    # Add confusion matrix
    metrics['confusion_matrix'] = conf_matrix.tolist()
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None, show_plot=True):
    """Plot confusion matrix with proper formatting and labels."""
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Default class names if not provided
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Create heatmap using seaborn
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    # Show plot
    if show_plot:
        plt.show()
    
    plt.close()
    
    return cm

def main():
    # Parse arguments
    args = parse_args()
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join('logs', f'test_log_{timestamp}.json')
    
    # Load and normalize test data
    print("Loading test data...")
    try:
        test_sequences = np.load(args.test_sequences)
        test_labels = np.load(args.test_labels)
        print(f"Test sequences shape: {test_sequences.shape}")
        print(f"Test labels shape: {test_labels.shape}")
        # test_sequences = normalize_data(test_sequences)
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
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
    
    # Build model with a sample batch
    print("Building model...")
    sample_batch = test_sequences[:args.batch_size]
    _ = model(sample_batch)  # This creates the model's variables
    
    # Load weights
    print(f"Loading weights from {args.weights_file}...")
    try:
        model.load_weights(args.weights_file)
    except Exception as e:
        print(f"Error loading weights: {e}")
        return
    
    # Initialize focal loss function to match train.py
    focal_loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(
        gamma=args.focal_gamma,
        alpha=[0.01, 0.41, 0.58],  # order = [pass, shot, dribble] - match train.py exactly
        from_logits=False
    )
    
    # Evaluation loop
    print("Evaluating...")
    num_batches = int(np.ceil(test_sequences.shape[0] / args.batch_size))
    total_loss = 0.0
    reconstruction_loss = 0.0
    classification_loss = 0.0
    kl_loss_z = 0.0
    kl_loss_a = 0.0
    
    # Lists to store predictions and true labels for classification metrics
    all_predictions = []
    all_true_labels = []
    
    for i in range(num_batches):
        batch_start = i * args.batch_size
        batch_end = min((i + 1) * args.batch_size, test_sequences.shape[0])
        x_batch = test_sequences[batch_start:batch_end]
        y_batch = test_labels[batch_start:batch_end]
        
        # Forward pass
        a_mean_t, a_log_sigma_t, a_t, z_mean_t, z_log_sigma_t, z_t = model.encode(x_batch)
        reconstruction = model.decode(a_t, z_t)
        y_pred = model.classify(a_t)
        
        # Store predictions and true labels
        all_predictions.append(y_pred.numpy())
        all_true_labels.append(y_batch)
        
        # Losses (match train.py exactly)
        rec_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(x_batch, reconstruction),
                axis=(1, 2, 3)
            )
        ).numpy()
        
        kl_z = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_sigma_t - tf.square(z_mean_t) - tf.exp(z_log_sigma_t),
                axis=(1, 2)
            )
        ).numpy()
        
        kl_a = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + a_log_sigma_t - tf.square(a_mean_t) - tf.exp(a_log_sigma_t),
                axis=(1, 2)
            )
        ).numpy()
        
        # Use the same minimum threshold as in train.py
        kl_min = 2.0
        kl_loss = max(kl_z, kl_min) + max(kl_a, kl_min)
        
        # Classification loss using focal loss to match train.py
        y_true = tf.reshape(tf.repeat(y_batch, args.timesteps), (y_batch.shape[0], args.timesteps))
        y_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=args.no_classes)
        class_loss = tf.reduce_mean(focal_loss_fn(y_one_hot, y_pred)).numpy()
        
        # Total loss (use same alpha scaling as in train.py)
        class_loss_scaled = args.alpha * class_loss
        
        total = rec_loss + kl_loss + class_loss_scaled
        
        # Accumulate
        total_loss += total * args.batch_size
        reconstruction_loss += rec_loss * args.batch_size
        classification_loss += class_loss * args.batch_size
        kl_loss_z += kl_z * args.batch_size
        kl_loss_a += kl_a * args.batch_size
        
        # Log batch metrics
        log_metrics(log_file, {
            'batch': i,
            'total_loss': total,
            'reconstruction_loss': rec_loss,
            'kl_loss_z (unweighted)': kl_z,
            'kl_loss_a (unweighted)': kl_a,
            'kl_loss (sum, weighted)': kl_loss,
            'classification_loss (unweighted)': class_loss,
            'classification_loss (scaled)': class_loss_scaled
        })
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{num_batches} batches")
    
    # Concatenate all predictions and true labels
    all_predictions = np.concatenate(all_predictions, axis=0)  # (N, T, no_classes) or (N*T, no_classes)
    all_true_labels = np.concatenate(all_true_labels, axis=0)  # (N,)

    # Aggregate per-sequence predictions (mean over timesteps)
    all_predictions = all_predictions.reshape(-1, args.timesteps, args.no_classes)  # (N, T, no_classes)
    sequence_pred_probs = np.mean(all_predictions, axis=1)  # (N, no_classes)
    sequence_pred_labels = np.argmax(sequence_pred_probs, axis=1)  # (N,)
    sequence_true_labels = all_true_labels  # (N,)

    # Calculate classification metrics
    print("\nCalculating classification metrics...")
    classification_metrics = calculate_classification_metrics(sequence_true_labels, sequence_pred_probs)

    # Compute averages for loss metrics
    n = test_sequences.shape[0]
    print(f"\nTest results on {n} samples:")
    print(f"Total loss: {total_loss / n:.4f}")
    print(f"Reconstruction loss: {reconstruction_loss / n:.4f}")
    print(f"KL loss z (unweighted): {kl_loss_z / n:.4f}")
    print(f"KL loss a (unweighted): {kl_loss_a / n:.4f}")
    print(f"Classification loss (unweighted): {classification_loss / n:.4f}")

    # Print classification metrics
    print("\nClassification Metrics:")
    print("\nPer-class metrics:")
    for class_idx in np.unique(sequence_true_labels):
        metrics = classification_metrics[f'class_{class_idx}']
        print(f"\nClass {class_idx}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall (TP-Rate): {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  Support: {metrics['support']}")

    print("\nOverall metrics:")
    macro_metrics = classification_metrics['macro_avg']
    print(f"  Accuracy: {macro_metrics['accuracy']:.4f}")
    print(f"  Macro Precision: {macro_metrics['precision']:.4f}")
    print(f"  Macro Recall: {macro_metrics['recall']:.4f}")
    print(f"  Macro F1 Score: {macro_metrics['f1']:.4f}")
    print(f"  Macro AUC: {macro_metrics['auc']:.4f}")

    # Plot confusion matrix
    print("\nPlotting confusion matrix...")
    y_pred_classes = sequence_pred_labels
    class_names = ['Pass', 'Shot', 'Dribble']  # Based on LABEL_ENCODING order

    # Create plots directory if saving
    save_path = 'seqlabelvae_v2/g1/plots/g1_dutch_eval.png'
    # save_path = None
    # if args.save_plots:
    #     os.makedirs(args.plots_dir, exist_ok=True)
    #     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    #     save_path = os.path.join(args.plots_dir, f'confusion_matrix_{timestamp}.png')

    # Plot confusion matrix
    cm = plot_confusion_matrix(
        sequence_true_labels, 
        y_pred_classes, 
        class_names=class_names,
        save_path=save_path,
        show_plot=args.show_plots
    )

    # Print confusion matrix
    print("\nConfusion Matrix:")
    print("True\\Pred\t" + "\t".join(class_names))
    for i, true_class in enumerate(class_names):
        row = f"{true_class}\t"
        for j in range(len(class_names)):
            row += f"{cm[i, j]}\t"
        print(row)

    # Log final metrics
    log_metrics(log_file, {
        'final_metrics': {
            'loss_metrics': {
                'total_loss': float(total_loss / n),
                'reconstruction_loss': float(reconstruction_loss / n),
                'kl_loss_z': float(kl_loss_z / n),
                'kl_loss_a': float(kl_loss_a / n),
                'classification_loss': float(classification_loss / n)
            },
            'classification_metrics': classification_metrics
        }
    })

if __name__ == '__main__':
    main() 