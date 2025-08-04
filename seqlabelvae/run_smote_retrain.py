import subprocess
import os
from datetime import datetime

# Define your parameters here
args_dict = {
    '--feature_dim': 300,
    '--intermediate_dim': 512,
    '--hidden_dim': 128,
    '--pitch_x_axis': 105,
    '--pitch_y_axis': 68,
    '--channels': 9,
    '--timesteps': 32,
    '--no_classes': 3,
    '--vae_weights_file': './seqlabelvae_v2/training_weights_g1_fixed/weights_epoch_37.h5',
    # '--vae_weights_file': './seqlabelvae_v2/training_weights_dutch/weights_epoch_50.h5',
    '--balanced_sequences': './dutch_data/split/balanced_latent_data/balanced_latent_sequences.npy',  # Update with your actual path
    '--balanced_labels': './dutch_data/split/balanced_latent_data/balanced_latent_labels.npy',  # Update with your actual path
    '--output_dir': './seqlabelvae_v2/training_weights_dutch/',
    '--epochs': 50,
    '--batch_size': 32,
    '--learning_rate': 0.001,
    '--validation_split': 0.2,
    '--random_state': 42
}

# # Create timestamp for unique output directory
# timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# args_dict['--output_dir'] = f'./seqlabelvae_v2/g1/smote_retrain.py{timestamp}/'

# Build the command
cmd = ['python', './seqlabelvae_v2/g1/smote_retrain.py']
for k, v in args_dict.items():
    if isinstance(v, bool):
        if v:
            cmd.append(k)
    else:
        cmd.extend([k, str(v)])

# Print for checking
print("="*60)
print("RUNNING CLASSIFIER RETRAINING")
print("="*60)
print("Command:", ' '.join(cmd))
print("\nParameters:")
for k, v in args_dict.items():
    print(f"  {k}: {v}")
print("\n" + "="*60)

# Check if input files exist
input_files = [
    args_dict['--vae_weights_file'],
    args_dict['--balanced_sequences'],
    args_dict['--balanced_labels']
]

print("Checking input files...")
for file_path in input_files:
    if os.path.exists(file_path):
        print(f"  ✓ {file_path}")
    else:
        print(f"  ✗ {file_path} - NOT FOUND!")
        print("Please check the file paths and run again.")
        exit(1)

print("\nAll input files found. Starting classifier retraining...")

# Run the command
try:
    result = subprocess.run(cmd, check=True)
    print("\n" + "="*60)
    print("CLASSIFIER RETRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Output directory: {args_dict['--output_dir']}")
    print("Files generated:")
    print(f"  - retrained_classifier.h5")
    print(f"  - training_metadata.json")
    
except subprocess.CalledProcessError as e:
    print(f"\nError running classifier retraining: {e}")
    print("Please check the error messages above and fix any issues.")
except FileNotFoundError:
    print("\nError: retrain_classifier.py not found!")
    print("Make sure you're running this script from the correct directory.")
except KeyboardInterrupt:
    print("\nClassifier retraining interrupted by user.") 