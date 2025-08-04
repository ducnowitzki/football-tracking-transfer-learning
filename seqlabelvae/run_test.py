import subprocess

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
    '--weights_file': './seqlabelvae_v2/training_weights_dutch/weights_epoch_50.h5',
    # '--test_sequences': './idsse-data/seqlabelvae/split/test_sequences.npy',
    # '--test_labels': './idsse-data/seqlabelvae/split/test_labels.npy',
    '--test_sequences': './dfl_data/split/test_labeled_sequences.npy',
    '--test_labels': './dfl_data/split/test_labels.npy',
}

# Build the command
cmd = ['python', 'seqlabelvae_v2/g1/test.py']
for k, v in args_dict.items():
    cmd.extend([k, str(v)])

# Print for checking
print("Running command:", ' '.join(cmd))

# Run the command
subprocess.run(cmd)
