import subprocess

# Define your parameters here
args_dict = {
    '--feature_dim': 300,
    '--intermediate_dim': 512,
    '--hidden_dim': 128,
    '--pitch_x_axis': 105,
    '--pitch_y_axis': 68,
    # '--channels': 3,
    '--channels': 9,
    '--timesteps': 32,
    '--no_classes': 3,
    '--unlabeled_batch_size': 32,
    '--labeled_batch_size': 4,
    '--epochs': 50,
    '--sample_rate': 1,
    '--alpha': 0.1,
    # '--data_dir': './idsse-data/seqlabelvae/split/',
    # '--labeled-batch-size': 4,
    # '--cost-annealing': False,
    # '--labeled_sequences': './idsse-data/seqlabelvae/split/train_sequences.npy',
    # '--labels': './idsse-data/seqlabelvae/split/train_labels.npy',
    # '--unlabeled_frames': './idsse-data/seqlabelvae/whole/unlabeled_frames.npy',
    '--labeled_sequences': './dutch_data/split/train_labeled_sequences.npy',
    '--labels': './dutch_data/split/train_labels.npy',
    '--unlabeled_frames': './dutch_data/split/unlabeled_frames.npy',
    '--weights_dir': './seqlabelvae_v2/training_weights/',
    '--logs_dir': './seqlabelvae_v2/logs/',
    '--changes': 'dutch_g1'
}

# Build the command
cmd = ['python', 'seqlabelvae_v2/g1/train_v4_correct_split_unlabeled_sequences.py']
for k, v in args_dict.items():
    if isinstance(v, bool):
        if v:
            cmd.append(k)
    else:
        cmd.extend([k, str(v)])

# Print for checking
print("Running command:", ' '.join(cmd))

# Run the command
subprocess.run(cmd)
