import subprocess

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
    '--labeled_batch_size': 8,
    '--epochs': 50,
    '--sample_rate': 1,
    '--alpha': 0.1,
    # '--data_dir': './idsse-data/seqlabelvae/split/',
    # '--labeled-batch-size': 4,
    # '--cost-annealing': False,
    # '--labeled_sequences': './idsse-data/seqlabelvae/split/train_sequences.npy',
    # '--labels': './idsse-data/seqlabelvae/split/train_labels.npy',
    # '--unlabeled_frames': './idsse-data/seqlabelvae/whole/unlabeled_frames.npy',
    '--labeled_sequences': './dfl_data/split/train_labeled_sequences.npy',
    '--labels': './dfl_data/split/train_labels.npy',
    '--unlabeled_frames': './dfl_data/split/unlabeled_frames.npy',
    '--weights_dir': './seqlabelvae_v2/training_weights/',
    '--logs_dir': './seqlabelvae_v2/logs/',
    '--changes': 'dfl_g1_fixed_mapping_and_unlabeled_sequences'
}

cmd = ['python', 'seqlabelvae_v2/g1/trains.py']
for k, v in args_dict.items():
    if isinstance(v, bool):
        if v:
            cmd.append(k)
    else:
        cmd.extend([k, str(v)])

print("Running command:", ' '.join(cmd))

subprocess.run(cmd)
