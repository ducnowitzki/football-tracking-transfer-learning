import os
import argparse
import numpy as np
import tensorflow as tf
from model import SeqLabelVAE
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='Create UMAP visualization of encoded sequences')
    parser.add_argument('--feature_dim', type=int, default=300)
    parser.add_argument('--intermediate_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--pitch_x_axis', type=int, default=105)
    parser.add_argument('--pitch_y_axis', type=int, default=68)
    parser.add_argument('--channels', type=int, default=9)
    parser.add_argument('--timesteps', type=int, default=32)
    parser.add_argument('--no_classes', type=int, default=3)
    parser.add_argument('--test_sequences', type=str, required=True, help='Path to test sequences')
    parser.add_argument('--test_labels', type=str, required=True, help='Path to test labels')
    parser.add_argument('--weights_file', type=str, required=True, help='Path to model weights')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for encoding')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples to use for UMAP (max)')
    parser.add_argument('--n_neighbors', type=int, default=15, help='UMAP n_neighbors parameter')
    parser.add_argument('--min_dist', type=float, default=0.1, help='UMAP min_dist parameter')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
    parser.add_argument('--output_dir', type=str, default='umap_plots', help='Output directory for plots')
    parser.add_argument('--class_names', type=str, nargs='+', default=None, help='Class names for plotting')
    return parser.parse_args()

def load_data_and_model(args):
    """Load test data and initialize model with weights."""
    print("Loading test data...")
    try:
        test_sequences = np.load(args.test_sequences)
        test_labels = np.load(args.test_labels)
        print(f"Test sequences shape: {test_sequences.shape}")
        print(f"Test labels shape: {test_labels.shape}")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return None, None, None
    
    # Subsample if too many samples
    if test_sequences.shape[0] > args.n_samples:
        indices = np.random.choice(test_sequences.shape[0], args.n_samples, replace=False)
        test_sequences = test_sequences[indices]
        test_labels = test_labels[indices]
        print(f"Subsampled to {args.n_samples} samples")
    
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
    _ = model(sample_batch)
    
    # Load weights
    print(f"Loading weights from {args.weights_file}...")
    try:
        model.load_weights(args.weights_file)
        print("Weights loaded successfully!")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return None, None, None
    
    return model, test_sequences, test_labels

def encode_sequences(model, sequences, batch_size=32):
    """Encode sequences using VAE encoder and extract latent representations."""
    print("Encoding sequences...")
    
    num_sequences = sequences.shape[0]
    all_a_representations = []
    all_z_representations = []
    all_h_representations = []
    
    for i in range(0, num_sequences, batch_size):
        batch_end = min(i + batch_size, num_sequences)
        batch_sequences = sequences[i:batch_end]
        
        # Encode batch: returns (a_mean_t, a_log_sigma_t, a_t, z_mean_t, z_log_sigma_t, z_t)
        a_mean_t, a_log_sigma_t, a_t, z_mean_t, z_log_sigma_t, z_t = model.encode(batch_sequences)
        
        # Get intermediate representations (h_t) from recurrent encoder
        h_t = model.recurrent_encoder(batch_sequences)
        
        # Store representations
        all_a_representations.append(a_t.numpy())
        all_z_representations.append(z_t.numpy())
        all_h_representations.append(h_t.numpy())
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"Encoded {batch_end}/{num_sequences} sequences")
    
    # Concatenate all batches
    a_representations = np.concatenate(all_a_representations, axis=0)
    z_representations = np.concatenate(all_z_representations, axis=0)
    h_representations = np.concatenate(all_h_representations, axis=0)
    
    print(f"A representations shape: {a_representations.shape}")
    print(f"Z representations shape: {z_representations.shape}")
    print(f"H representations shape: {h_representations.shape}")
    
    return a_representations, z_representations, h_representations

def prepare_data_for_umap(representations, labels, representation_type="a"):
    """Prepare data for UMAP visualization."""
    print(f"Preparing {representation_type} representations for UMAP...")
    
    # Flatten temporal dimension: (N_sequences, T, D) -> (N_sequences, T*D)
    if len(representations.shape) == 3:
        X_flat = representations.reshape(representations.shape[0], -1)
    else:
        X_flat = representations
    
    print(f"Flattened {representation_type} representations shape: {X_flat.shape}")
    
    # Get class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Class distribution:")
    for class_idx, count in zip(unique, counts):
        print(f"  Class {class_idx}: {count} sequences")
    
    return X_flat, labels

def apply_umap(X, labels, args, representation_type="a"):
    """Apply UMAP dimensionality reduction."""
    print(f"Applying UMAP to {representation_type} representations...")
    
    # Apply UMAP
    reducer = umap.UMAP(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state,
        verbose=True
    )
    
    X_umap = reducer.fit_transform(X)
    print(f"UMAP completed. Output shape: {X_umap.shape}")
    
    return X_umap, labels

def create_umap_plot(X_umap, labels, args, representation_type="a", class_names=None):
    """Create UMAP visualization plot."""
    print(f"Creating UMAP plot for {representation_type} representations...")
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'UMAP 1': X_umap[:, 0],
        'UMAP 2': X_umap[:, 1],
        'Class': labels
    })
    
    # Set up class names
    if class_names is None:
        unique_classes = np.unique(labels)
        class_names = [f'Class {cls}' for cls in unique_classes]
    
    # Create color palette
    colors = sns.color_palette("husl", len(class_names))
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Plot each class
    for i, class_idx in enumerate(np.unique(labels)):
        class_data = df[df['Class'] == class_idx]
        class_name = class_names[class_idx] if class_idx < len(class_names) else f'Class {class_idx}'
        
        plt.scatter(
            class_data['UMAP 1'], 
            class_data['UMAP 2'],
            c=[colors[i]],
            label=f'{class_name} (n={len(class_data)})',
            alpha=0.7,
            s=50
        )
    
    plt.title(f'UMAP Visualization of {representation_type.upper()} Latent Representations\n'
              f'n_neighbors: {args.n_neighbors}, min_dist: {args.min_dist}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('UMAP Component 1', fontsize=12)
    plt.ylabel('UMAP Component 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()

def save_plots(fig, args, representation_type="a"):
    """Save UMAP plots."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'umap_{representation_type}_neighbors_{args.n_neighbors}_mindist_{args.min_dist}_{timestamp}.png'
    filepath = os.path.join(args.output_dir, filename)
    
    # Save plot
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"UMAP plot saved to: {filepath}")
    
    # Also save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'representation_type': representation_type,
        'n_neighbors': args.n_neighbors,
        'min_dist': args.min_dist,
        'n_samples': len(fig.axes[0].collections[0].get_offsets()),
        'model_params': {
            'feature_dim': args.feature_dim,
            'intermediate_dim': args.intermediate_dim,
            'hidden_dim': args.hidden_dim,
            'timesteps': args.timesteps,
            'no_classes': args.no_classes
        }
    }
    
    metadata_file = filepath.replace('.png', '_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_file}")

def main():
    # Parse arguments
    args = parse_args()
    
    print("="*60)
    print("UMAP VISUALIZATION OF LATENT REPRESENTATIONS")
    print("="*60)
    
    # Load data and model
    model, test_sequences, test_labels = load_data_and_model(args)
    if model is None:
        return
    
    # Encode sequences
    print("\n" + "="*40)
    print("ENCODING SEQUENCES")
    print("="*40)
    a_representations, z_representations, h_representations = encode_sequences(
        model, test_sequences, args.batch_size
    )
    
    # Set up class names
    if args.class_names is None:
        unique_classes = np.unique(test_labels)
        class_names = [f'Class {cls}' for cls in unique_classes]
    else:
        class_names = args.class_names
    
    # Create UMAP visualizations for different representations
    representations_to_plot = [
        ("a", a_representations, "Label-specific latent (a_t)"),
        ("z", z_representations, "Content-specific latent (z_t)"),
        ("h", h_representations, "Intermediate representations (h_t)")
    ]
    
    for rep_type, representations, description in representations_to_plot:
        print(f"\n" + "="*40)
        print(f"PROCESSING {description.upper()}")
        print("="*40)
        
        # Prepare data for UMAP
        X_flat, labels = prepare_data_for_umap(representations, test_labels, rep_type)
        
        # Apply UMAP
        X_umap, labels = apply_umap(X_flat, labels, args, rep_type)
        
        # Create and save plot
        fig = create_umap_plot(X_umap, labels, args, rep_type, class_names)
        save_plots(fig, args, rep_type)
        
        # Show plot
        plt.show()
        plt.close()
    
    print("\n" + "="*60)
    print("UMAP VISUALIZATION COMPLETED!")
    print("="*60)
    print(f"Plots saved to: {args.output_dir}")

if __name__ == '__main__':
    main() 