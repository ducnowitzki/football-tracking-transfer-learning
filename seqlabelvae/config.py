import os

# Data parameters
PITCH_HEIGHT = 105
PITCH_WIDTH = 68
CHANNELS = 3  # home team, away team, ball
NUM_CLASSES = 9  # number of event classes
WINDOW_SIZE = 32  # sequence length

# Model architecture
FEATURE_DIM = 300
INTERMEDIATE_DIM = 128
HIDDEN_DIM = 16

# Training parameters
BATCH_SIZE = 32
LABELED_BATCH_SIZE = 16
UNLABELED_BATCH_SIZE = 16
LEARNING_RATE = 0.0001
EPOCHS = 100
SAMPLE_RATE = 0.15  # rate of unlabeled data to sample each epoch
ALPHA = 0.1  # classification loss weight

# KL annealing
KL_MIN = 2
KL_ANNEALING_EPOCHS = 10

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
WEIGHTS_DIR = os.path.join(OUTPUT_DIR, 'training_weights')
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Data files
TRAIN_SEQUENCES = os.path.join(DATA_DIR, 'train_sequences.npy')
TRAIN_LABELS = os.path.join(DATA_DIR, 'train_labels.npy')
VAL_SEQUENCES = os.path.join(DATA_DIR, 'val_sequences.npy')
VAL_LABELS = os.path.join(DATA_DIR, 'val_labels.npy')
TEST_SEQUENCES = os.path.join(DATA_DIR, 'test_sequences.npy')
TEST_LABELS = os.path.join(DATA_DIR, 'test_labels.npy')
UNLABELED_FRAMES = os.path.join(DATA_DIR, 'unlabeled_frames.npy') 