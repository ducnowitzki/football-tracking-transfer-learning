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
    # '--test_sequences': './dfl_data/split/test_labeled_sequences.npy',  # Update with your actual path
    # '--test_labels': './dfl_data/split/test_labels.npy',        # Update with your actual path
    '--vae_weights_file': './seqlabelvae_v2/training_weights_g1_fixed/weights_epoch_37.h5',
    # '--classifier_weights_file': './seqlabelvae_v2/training_weights_g1_fixed/1_retrained_classifier.h5',
    '--test_sequences': './dutch_data/split/test_labeled_sequences.npy',  # Update with your actual path
    '--test_labels': './dutch_data/split/test_labels.npy',        # Update with your actual path
    # '--vae_weights_file': './seqlabelvae_v2/training_weights_dutch/weights_epoch_50.h5',
    '--classifier_weights_file': './seqlabelvae_v2/training_weights_dutch/1_retrained_classifier.h5',
    '--batch_size': 32,
    '--save_plots': True,
    '--plots_dir': './plots/',
    '--show_plots': False
}

# Build the command
cmd = ['python', 'seqlabelvae_v2/g1/test_smote.py']
for k, v in args_dict.items():
    if isinstance(v, bool):
        if v:
            cmd.append(k)
    else:
        cmd.extend([k, str(v)])

# Print for checking
print("="*60)
print("RUNNING TEST_SMOTE")
print("="*60)
print("Command:", ' '.join(cmd))
print("\nParameters:")
for k, v in args_dict.items():
    print(f"  {k}: {v}")
print("\n" + "="*60)

# Check if input files exist
input_files = [
    args_dict['--test_sequences'],
    args_dict['--test_labels'],
    args_dict['--vae_weights_file'],
    args_dict['--classifier_weights_file']
]

print("Checking input files...")
for file_path in input_files:
    if os.path.exists(file_path):
        print(f"  ✓ {file_path}")
    else:
        print(f"  ✗ {file_path} - NOT FOUND!")
        print("Please check the file paths and run again.")
        exit(1)

print("\nAll input files found. Starting test_smote...")

# Run the command
try:
    result = subprocess.run(cmd, check=True)
    print("\n" + "="*60)
    print("TEST_SMOTE COMPLETED SUCCESSFULLY!")
    print("="*60)
except subprocess.CalledProcessError as e:
    print(f"\nError running test_smote: {e}")
    print("Please check the error messages above and fix any issues.")
except FileNotFoundError:
    print("\nError: test_smote.py not found!")
    print("Make sure you're running this script from the correct directory.")
except KeyboardInterrupt:
    print("\ntest_smote interrupted by user.") 