import os
import argparse
import numpy as np
import tensorflow as tf
from model import SeqLabelVAE
from datetime import datetime
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import joblib
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Generate balanced latent training data using VAE encoder and SMOTE')
    parser.add_argument('--feature_dim', type=int, default=300)
    parser.add_argument('--intermediate_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--pitch_x_axis', type=int, default=68)
    parser.add_argument('--pitch_y_axis', type=int, default=105)
    parser.add_argument('--channels', type=int, default=9)
    parser.add_argument('--timesteps', type=int, default=32)
    parser.add_argument('--no_classes', type=int, default=3)
    parser.add_argument('--train_sequences', type=str, required=True, help='Path to training sequences')
    parser.add_argument('--train_labels', type=str, required=True, help='Path to training labels')
    parser.add_argument('--weights_file', type=str, required=True, help='Path to VAE weights')
    parser.add_argument('--output_dir', type=str, default='balanced_latent_data', help='Output directory for balanced data')
    parser.add_argument('--k_neighbors', type=int, default=5, help='Number of neighbors for SMOTE')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
    return parser.parse_args()

def load_data_and_model(args):
    """Load training data and initialize VAE model with weights."""
    print("Loading training data...")
    
    train_sequences = np.load(args.train_sequences)
    train_labels = np.load(args.train_labels)
    
    print(f"Training sequences shape: {train_sequences.shape}")
    print(f"Training labels shape: {train_labels.shape}")
    
    print("Initializing VAE model...")
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

    print("Building model...")
    sample_batch = train_sequences[:1]  # Just need one sample to build
    _ = model(sample_batch)
    
    # Load pre-trained weights
    print(f"Loading weights from {args.weights_file}...")
    try:
        model.load_weights(args.weights_file)
        print("Weights loaded successfully!")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return None, None, None
    
    return model, train_sequences, train_labels

def encode_sequences(model, sequences, batch_size=32):
    print("Encoding sequences...")
    
    num_sequences = sequences.shape[0]
    all_a_representations = []
    
    for i in range(0, num_sequences, batch_size):
        batch_end = min(i + batch_size, num_sequences)
        batch_sequences = sequences[i:batch_end]
        
        # Encode batch: returns (a_mean_t, a_log_sigma_t, a_t, z_mean_t, z_log_sigma_t, z_t)
        _, _, a_t, _, _, _ = model.encode(batch_sequences)
        
        # Extract label-specific representation (a_t)
        all_a_representations.append(a_t.numpy())
        
        if (i // batch_size + 1) % 10 == 0:
            print(f"Encoded {batch_end}/{num_sequences} sequences")
    
    # Concatenate all batches
    a_representations = np.concatenate(all_a_representations, axis=0)
    print(f"A representations shape: {a_representations.shape}")
    
    return a_representations

def prepare_sequence_wise_data(a_representations, labels):
    """Prepare sequence-wise data for SMOTE (treat each sequence as one point in R^(TxD))."""
    print("Preparing sequence-wise data for SMOTE...")
    
    # A_representations shape: (N_sequences, T, D)
    # Labels shape: (N_sequences,)
    
    # Flatten each sequence: (N_sequences, T, D) -> (N_sequences, T*D)
    # Each sequence becomes one point in R^(TxD)
    X_sequences = a_representations.reshape(a_representations.shape[0], -1)
    y_sequences = labels  # Keep original labels (one per sequence)
    
    print(f"Sequence-wise X_sequences shape: {X_sequences.shape}")
    print(f"Sequence-wise y_sequences shape: {y_sequences.shape}")
    
    # Print class distribution
    unique, counts = np.unique(y_sequences, return_counts=True)
    print("Class distribution (sequence-wise):")
    for class_idx, count in zip(unique, counts):
        print(f"  Class {class_idx}: {count} sequences")
    
    return X_sequences, y_sequences

def standardize_latent_space(X_sequences):
    """Standardize latent space for proper SMOTE neighbor selection."""
    print("Standardizing latent space...")
    
    # StandardScaler: zero-mean, unit-variance
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X_sequences)
    
    print(f"Standardized X_sequences shape: {X_standardized.shape}")
    print(f"Mean: {np.mean(X_standardized, axis=0)[:5]}...")  # Show first 5 dimensions
    print(f"Std: {np.std(X_standardized, axis=0)[:5]}...")   # Show first 5 dimensions
    
    return X_standardized, scaler

def apply_smote_balancing(X_standardized, y_sequences, k_neighbors=5, random_state=42):
    """Apply SMOTE with explicit sampling strategy to balance classes 1 and 2 to match class 0."""
    print("Applying SMOTE balancing...")
    
    # Get class distribution
    unique, counts = np.unique(y_sequences, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    print("Original class distribution:")
    for class_idx in sorted(class_counts.keys()):
        print(f"  Class {class_idx}: {class_counts[class_idx]} sequences")
    
    # Find the majority class (class 0)
    majority_class = max(class_counts, key=class_counts.get)
    majority_count = class_counts[majority_class]
    
    print(f"Majority class: {majority_class} with {majority_count} sequences")
    
    # Explicit sampling strategy: balance classes 1 and 2 to match class 0
    sampling_strategy = {1: majority_count, 2: majority_count}
    
    # Apply SMOTE with explicit sampling strategy
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state
    )
    
    X_resampled, y_resampled = smote.fit_resample(X_standardized, y_sequences)
    
    # Verify the balancing worked
    unique_res, counts_res = np.unique(y_resampled, return_counts=True)
    class_counts_res = dict(zip(unique_res, counts_res))
    
    print("Resampled class distribution:")
    for class_idx in sorted(class_counts_res.keys()):
        print(f"  Class {class_idx}: {class_counts_res[class_idx]} sequences")
    
    return X_resampled, y_resampled

def reshape_synthetic_sequences(X_resampled, y_resampled, timesteps, hidden_dim):
    """Reshape synthetic sequences back to (N_sequences, T, D) format."""
    print("Reshaping synthetic sequences...")
    
    # X_resampled shape: (N_balanced, T*hidden_dim) - each row is one synthetic sequence
    # y_resampled shape: (N_balanced,) - one label per synthetic sequence
    
    print(f"X_resampled shape: {X_resampled.shape}")
    print(f"Expected latent dimension per sample: {timesteps * hidden_dim}")
    
    # Verify the array has the expected dimensions
    if X_resampled.shape[1] != timesteps * hidden_dim:
        raise ValueError(f"Expected X_resampled to have {timesteps * hidden_dim} features per sample, but got {X_resampled.shape[1]}")
    
    # Each row in X_resampled is already one complete sequence flattened
    # Just reshape each row back to (timesteps, hidden_dim)
    n_seqs = X_resampled.shape[0]  # Number of synthetic sequences
    
    print(f"Number of synthetic sequences: {n_seqs}")
    print(f"Each sequence has {timesteps} timesteps with {hidden_dim} features each")
    
    # Reshape each synthetic sequence: (N_balanced, T*hidden_dim) -> (N_balanced, T, hidden_dim)
    X_sequences = X_resampled.reshape(n_seqs, timesteps, hidden_dim)
    y_final = y_resampled  # Labels are already one per sequence
    
    print(f"Final X_sequences shape: {X_sequences.shape}")
    print(f"Final y_final shape: {y_final.shape}")
    
    unique, counts = np.unique(y_final, return_counts=True)
    print("Final class distribution:")
    for class_idx, count in zip(unique, counts):
        print(f"  Class {class_idx}: {count} sequences")
    
    return X_sequences, y_final

def save_balanced_data(X_sequences, y_final, scaler, output_dir):
    """Save the balanced latent dataset and scaler."""
    print(f"Saving balanced data to {output_dir}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'balanced_latent_sequences.npy'), X_sequences)
    np.save(os.path.join(output_dir, 'balanced_latent_labels.npy'), y_final)
    
    joblib.dump(scaler, os.path.join(output_dir, 'latent_scaler.pkl'))
    
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'X_sequences_shape': X_sequences.shape,
        'y_final_shape': y_final.shape,
        'class_distribution': {f'class_{i}': int(count) for i, count in zip(*np.unique(y_final, return_counts=True))},
        'scaler_info': {
            'mean_shape': scaler.mean_.shape,
            'scale_shape': scaler.scale_.shape,
            'feature_names_in': scaler.feature_names_in_.tolist() if hasattr(scaler, 'feature_names_in_') else None
        }
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Balanced data saved successfully!")
    print(f"  Sequences: {os.path.join(output_dir, 'balanced_latent_sequences.npy')}")
    print(f"  Labels: {os.path.join(output_dir, 'balanced_latent_labels.npy')}")
    print(f"  Scaler: {os.path.join(output_dir, 'latent_scaler.pkl')}")
    print(f"  Metadata: {os.path.join(output_dir, 'metadata.json')}")

def main():
    args = parse_args()
    
    print("="*60)
    print("GENERATING BALANCED LATENT TRAINING DATA")
    print("="*60)
    
    # Load data and model
    model, train_sequences, train_labels = load_data_and_model(args)
    if model is None:
        return
    
    # Encode sequences to get label-specific representations
    print("\n" + "="*40)
    print("STEP 1: ENCODING SEQUENCES")
    print("="*40)
    a_representations = encode_sequences(model, train_sequences)
    # Shape: (N_sequences, T, feature_dim)
    
    # Prepare sequence-wise data for SMOTE
    print("\n" + "="*40)
    print("STEP 2: PREPARING SEQUENCE-WISE DATA")
    print("="*40)
    X_sequences, y_sequences = prepare_sequence_wise_data(a_representations, train_labels)
    # Shapes: X_sequences (N_sequences, T*feature_dim), y_sequences (N_sequences,)
    
    # Standardize latent space
    print("\n" + "="*40)
    print("STEP 3: STANDARDIZING LATENT SPACE")
    print("="*40)
    X_standardized, scaler = standardize_latent_space(X_sequences)
    # Shape: X_standardized (N_sequences, T*feature_dim) - zero-mean, unit-variance
    
    # Apply SMOTE balancing
    print("\n" + "="*40)
    print("STEP 4: APPLYING SMOTE BALANCING")
    print("="*40)
    X_resampled, y_resampled = apply_smote_balancing(
        X_standardized, y_sequences, args.k_neighbors, args.random_state
    )
    # Shapes: X_resampled (N_balanced, T*feature_dim), y_resampled (N_balanced,)
    
    # Reshape synthetic sequences
    print("\n" + "="*40)
    print("STEP 5: RESHAPING SYNTHETIC SEQUENCES")
    print("="*40)
    X_final_sequences, y_final = reshape_synthetic_sequences(
        X_resampled, y_resampled, args.timesteps, args.hidden_dim
    )
    # Shapes: X_final_sequences (N_sequences, T, hidden_dim), y_final (N_sequences,)
    
    # Save balanced data
    print("\n" + "="*40)
    print("STEP 6: SAVING BALANCED DATA")
    print("="*40)
    save_balanced_data(X_final_sequences, y_final, scaler, args.output_dir)
    
    print("\n" + "="*60)
    print("BALANCED LATENT DATA GENERATION COMPLETED!")
    print("="*60)

if __name__ == '__main__':
    main() 