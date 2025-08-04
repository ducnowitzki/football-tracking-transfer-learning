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
    # '--train_sequences': './dfl_data/split/train_labeled_sequences.npy',
    # '--train_labels': './dfl_data/split/train_labels.npy',
    # '--weights_file': './seqlabelvae_v2/training_weights_g1_fixed/weights_epoch_37.h5',  # Adjust epoch as needed
    '--train_sequences': './dutch_data/split/train_labeled_sequences.npy',
    '--train_labels': './dutch_data/split/train_labels.npy',
    '--weights_file': './seqlabelvae_v2/training_weights_dutch/weights_epoch_50.h5',  # Adjust epoch as needed
    '--output_dir': './dutch_data/split/balanced_latent_data/',
    '--k_neighbors': 3,
    '--random_state': 42
}

# Create timestamp for unique output directory
# timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# args_dict['--output_dir'] = f'./dfl_data/split/balanced_latent_data_{timestamp}/'

# Build the command
cmd = ['python', 'seqlabelvae_v2/g1/smote.py']
for k, v in args_dict.items():
    if isinstance(v, bool):
        if v:
            cmd.append(k)
    else:
        cmd.extend([k, str(v)])


print("="*60)
print("RUNNING SMOTE DATA GENERATION")
print("="*60)
print("Command:", ' '.join(cmd))
print("\nParameters:")
for k, v in args_dict.items():
    print(f"  {k}: {v}")
print("\n" + "="*60)

input_files = [
    args_dict['--train_sequences'],
    args_dict['--train_labels'],
    args_dict['--weights_file']
]

print("Checking input files...")
for file_path in input_files:
    if os.path.exists(file_path):
        print(f"  ✓ {file_path}")
    else:
        print(f"  ✗ {file_path} - NOT FOUND!")
        print("Please check the file paths and run again.")
        exit(1)

print("\nAll input files found. Starting SMOTE generation...")


try:
    result = subprocess.run(cmd, check=True)
    print("\n" + "="*60)
    print("SMOTE DATA GENERATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Output directory: {args_dict['--output_dir']}")
    print("Files generated:")
    print(f"  - balanced_latent_sequences.npy")
    print(f"  - balanced_latent_labels.npy")
    print(f"  - metadata.json")
    
except subprocess.CalledProcessError as e:
    print(f"\nError running SMOTE generation: {e}")
    print("Please check the error messages above and fix any issues.")
except FileNotFoundError:
    print("\nError: generate_balanced_latent_data.py not found!")
    print("Make sure you're running this script from the correct directory.")
except KeyboardInterrupt:
    print("\nSMOTE generation interrupted by user.") 