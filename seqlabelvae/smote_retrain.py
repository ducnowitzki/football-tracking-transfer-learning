import os
import argparse
import numpy as np
import tensorflow as tf
from model import SeqLabelVAE
from datetime import datetime
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Retrain classifier head using balanced latent data')
    parser.add_argument('--feature_dim', type=int, default=300)
    parser.add_argument('--intermediate_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--pitch_x_axis', type=int, default=68)
    parser.add_argument('--pitch_y_axis', type=int, default=105)
    parser.add_argument('--channels', type=int, default=9)
    parser.add_argument('--timesteps', type=int, default=32)
    parser.add_argument('--no_classes', type=int, default=3)
    parser.add_argument('--vae_weights_file', type=str, required=True, help='Path to VAE weights')
    parser.add_argument('--balanced_sequences', type=str, required=True, help='Path to balanced latent sequences')
    parser.add_argument('--balanced_labels', type=str, required=True, help='Path to balanced latent labels')
    parser.add_argument('--output_dir', type=str, default='retrained_classifier', help='Output directory for retrained weights')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for classifier training')
    parser.add_argument('--validation_split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
    return parser.parse_args()

def load_balanced_data(balanced_sequences_path, balanced_labels_path):
    """Load the balanced latent training data."""
    print("Loading balanced latent data...")
    
    # Load balanced sequences and labels
    balanced_sequences = np.load(balanced_sequences_path)
    balanced_labels = np.load(balanced_labels_path)
    
    print(f"Balanced sequences shape: {balanced_sequences.shape}")
    print(f"Balanced labels shape: {balanced_labels.shape}")
    
    # Convert labels to one-hot encoding
    balanced_labels_onehot = tf.keras.utils.to_categorical(balanced_labels, num_classes=3)
    
    print(f"Balanced labels one-hot shape: {balanced_labels_onehot.shape}")
    
    # Print class distribution
    unique, counts = np.unique(balanced_labels, return_counts=True)
    print("Class distribution in balanced data:")
    for class_idx, count in zip(unique, counts):
        print(f"  Class {class_idx}: {count} sequences")
    
    return balanced_sequences, balanced_labels_onehot

def create_classifier_only_model(hidden_dim, timesteps, no_classes):
    """Create a model that only contains the classifier head."""
    print("Creating classifier-only model...")
    
    latent_input = tf.keras.layers.Input(shape=(timesteps, hidden_dim))
    # Pool across timesteps (mean or max)
    pooled = tf.keras.layers.GlobalAveragePooling1D()(latent_input)
    # Sequence-level prediction
    classifier_output = tf.keras.layers.Dense(no_classes, activation='softmax')(pooled)
    
    classifier_model = tf.keras.Model(latent_input, classifier_output, name="classifier_only")
    classifier_model.summary()
    return classifier_model

def train_classifier(classifier_model, balanced_sequences, balanced_labels_onehot, args):
    """Train the classifier head using balanced data."""
    print("Training classifier head...")
    
    # Compile the model
    classifier_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Split data into train and validation
    from sklearn.model_selection import train_test_split
    
    X_train, X_val, y_train, y_val = train_test_split(
        balanced_sequences, 
        balanced_labels_onehot, 
        test_size=args.validation_split, 
        random_state=args.random_state,
        stratify=balanced_labels_onehot
    )
    
    print(f"Training set: {X_train.shape[0]} sequences")
    print(f"Validation set: {X_val.shape[0]} sequences")
    
    # Training callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train the classifier
    history = classifier_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history, classifier_model

def save_retrained_weights(classifier_model, output_dir, args):
    """Save the retrained classifier weights."""
    print(f"Saving retrained weights to {output_dir}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the classifier model
    classifier_model.save(os.path.join(output_dir, '1_retrained_classifier.h5'))
    
    # Save training metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_architecture': {
            'hidden_dim': args.hidden_dim,
            'timesteps': args.timesteps,
            'no_classes': args.no_classes
        },
        'training_params': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'validation_split': args.validation_split,
            'random_state': args.random_state
        },
        'input_data': {
            'balanced_sequences': args.balanced_sequences,
            'balanced_labels': args.balanced_labels,
            'vae_weights': args.vae_weights_file
        }
    }
    
    with open(os.path.join(output_dir, '1_training_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Retrained classifier saved successfully!")
    print(f"  Model: {os.path.join(output_dir, '1_retrained_classifier.h5')}")
    print(f"  Metadata: {os.path.join(output_dir, '1_training_metadata.json')}")

def evaluate_classifier(classifier_model, balanced_sequences, balanced_labels_onehot):
    """Evaluate the trained classifier."""
    print("Evaluating classifier...")
    
    # Make predictions
    predictions = classifier_model.predict(balanced_sequences, verbose=0)
    predicted_classes = np.argmax(predictions, axis=-1)
    true_classes = np.argmax(balanced_labels_onehot, axis=-1)
    
    # Calculate accuracy
    accuracy = np.mean(predicted_classes == true_classes)
    print(f"Overall accuracy: {accuracy:.4f}")
    
    # Per-class accuracy
    for class_idx in range(3):
        class_mask = true_classes == class_idx
        if np.any(class_mask):
            class_accuracy = np.mean(predicted_classes[class_mask] == true_classes[class_mask])
            print(f"Class {class_idx} accuracy: {class_accuracy:.4f}")
    
    return accuracy, predictions

def main():
    args = parse_args()
    
    print("="*60)
    print("RETRAINING CLASSIFIER HEAD WITH BALANCED DATA")
    print("="*60)
    
    # Set random seeds for reproducibility
    np.random.seed(args.random_state)
    tf.random.set_seed(args.random_state)
    
    # Step Load balanced data
    print("\n" + "="*40)
    print("STEP 1: LOADING BALANCED DATA")
    print("="*40)
    balanced_sequences, balanced_labels_onehot = load_balanced_data(
        args.balanced_sequences, args.balanced_labels
    )
    
    # Step Create classifier-only model
    print("\n" + "="*40)
    print("STEP 2: CREATING CLASSIFIER MODEL")
    print("="*40)
    classifier_model = create_classifier_only_model(
        args.hidden_dim, args.timesteps, args.no_classes
    )
    
    # Train classifier
    print("\n" + "="*40)
    print("STEP 3: TRAINING CLASSIFIER")
    print("="*40)
    history, trained_classifier = train_classifier(
        classifier_model, balanced_sequences, balanced_labels_onehot, args
    )

    # Save training log after each epoch (JSON lines, like train.py)
    log_file = os.path.join(args.output_dir, f'training_log_transfer_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(log_file, 'w') as f:
        for epoch in range(len(history.history['loss'])):
            log_entry = {
                'epoch': epoch + 1,
                'batch': 0,
                'metrics': {
                    'loss': float(history.history['loss'][epoch]),
                    'accuracy': float(history.history['accuracy'][epoch]),
                    'val_loss': float(history.history['val_loss'][epoch]),
                    'val_accuracy': float(history.history['val_accuracy'][epoch])
                }
            }
            f.write(json.dumps(log_entry) + '\n')
    print(f"Training log saved to {log_file}")
    
    # Evaluate classifier
    print("\n" + "="*40)
    print("STEP 4: EVALUATING CLASSIFIER")
    print("="*40)
    accuracy, predictions = evaluate_classifier(
        trained_classifier, balanced_sequences, balanced_labels_onehot
    )
    
    # Save retrained weights
    print("\n" + "="*40)
    print("STEP 5: SAVING RETRAINED WEIGHTS")
    print("="*40)
    save_retrained_weights(trained_classifier, args.output_dir, args)
    
    print("\n" + "="*60)
    print("CLASSIFIER RETRAINING COMPLETED!")
    print("="*60)
    print(f"Final accuracy: {accuracy:.4f}")
    print(f"Output directory: {args.output_dir}")

if __name__ == '__main__':
    main() 