import subprocess

# Define your parameters here
args_dict = {
    '--feature-dim': 300,
    '--intermediate-dim': 128,
    '--hidden-dim': 8,
    '--pitch-x-axis': 105,
    '--pitch-y-axis': 68,
    '--channels': 3,
    '--timesteps': 32,
    '--epochs': 30,
    '--unlabeled-batch-size': 64,
    '--labeled-batch-size': 4,
    '--no-classes': 44,
    '--cost-annealing': False,
    '--labeled-sequences': './idsse-data/seqlabelvae/split/train_sequences.npy',
    '--labels': './idsse-data/seqlabelvae/split/train_labels.npy',
    '--unlabeled-frames': './idsse-data/seqlabelvae/whole/unlabeled_frames.npy',
    # '--labeled-sequences': './data/idsse-data/seqlabelvae/labeled_sequences.npy',
    # '--labels': './data/idsse-data/seqlabelvae/labels.npy',
    # '--unlabeled-frames': './data/idsse-data/seqlabelvae/unlabeled_frames.npy',
    '--weight_path': './seqlabelvae/training_weights.h5'
}

# Build the command
cmd = ['python', 'seqlabelvae/train.py']
# cmd = ['python', 'seqlabelvae/train.py']
for k, v in args_dict.items():
    if isinstance(v, bool):
        # For boolean flags, only include if True
        if v:
            cmd.append(k)
    else:
        cmd.extend([k, str(v)])

# Print for checking
print("Running command:", ' '.join(cmd))

# Run the command
subprocess.run(cmd)
