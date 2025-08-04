# Football Tracking Transfer Learning for Semi-Supervised Event Detection

## Project Overview

This repository contains the implementation of a Master's thesis project with the topic **Transfer Learning for Semi-Supervised Football Event Detection Models using Tracking Data**. The project explores how to leverage professional-level tracking data to improve event detection models on amateur-level data through transfer learning techniques.


## Setup Instructions
## 0. Dependencies
Python 3.12.8

Install required packages:
```bash
pip install -r requirements.txt
```

### 1. Download Professional DFL Data

Download the professional DFL dataset from the official source:
- **Source**: [An integrated dataset of spatiotemporal and event data in elite soccer](https://springernature.figshare.com/articles/dataset/An_integrated_dataset_of_spatiotemporal_and_event_data_in_elite_soccer/28196177)

**Instructions:**
1. Visit the Figshare link above
2. Download the complete dataset (2.45 GB)
3. Extract the files to `prof_data/raw/`

### 2. Data Preprocessing

#### Professional Data Preprocessing
```bash
python prof_data/data_processing_dfl.py
```

**Configuration Options:**
- **GRANULARITY**: Controls grid resolution (default: 1)
- **WINDOW_SIZE**: Sequence length for training (default: 32)
- **SAMPLE_RATE**: Rate of unlabeled data sampling (default: 0.05)

#### Amateur Data Preprocessing
```bash
python amateur_data/data_processing_dutch.py
```

**Configuration Options:**
- **GRANULARITY**: Controls grid resolution (default: 1)
- **WINDOW_SIZE**: Sequence length for training (default: 32)
- **SAMPLE_RATE**: Rate of unlabeled data sampling (default: 0.08)

### 3. Model Training and Evaluation

#### Basic Training
```bash
python seqlabelvae/run_training.py
```

#### Training with Dutch Data (Transfer Learning)
```bash
python seqlabelvae/run_training_dutch.py
```

#### Testing Trained Models
```bash
python seqlabelvae/run_test.py
```

#### SMOTE Upsampling
```bash
python seqlabelvae/run_smote.py
```

#### SMOTE with Retraining
```bash
python seqlabelvae/run_smote_retrain.py
```

#### Testing SMOTE Models
```bash
python seqlabelvae/run_test_smote.py
```

#### Retrain Classifier
```bash
python seqlabelvae/run_retrain_classifier.py
```

## Model Architecture

### SeqLabelVAE
The main model is a **Sequence Labeling Variational Autoencoder (SeqLabelVAE)** that combines:
- **Encoder**: Processes tracking sequences into latent representations
- **Decoder**: Reconstructs tracking sequences
- **Classifier**: Predicts event types (pass, shot, dribble)
- **Semi-supervised learning**: Uses both labeled and unlabeled data

### Key Features
- **Grid-based representation**: Tracking data converted to 2D grids
- **Multi-channel input**: Home team, away team, and ball positions
- **Velocity information**: Includes player velocity channels
- **Sequence modeling**: Processes temporal sequences of frames
- **Transfer learning**: Pre-trained on professional data, fine-tuned on amateur data

## Configuration

### Model Parameters (`seqlabelvae/config.py`)
- **PITCH_HEIGHT**: 105 (meters)
- **PITCH_WIDTH**: 68 (meters)
- **CHANNELS**: 3 (home team, away team, ball)
- **NUM_CLASSES**: 9 (event classes)
- **WINDOW_SIZE**: 32 (sequence length)
- **BATCH_SIZE**: 32
- **LEARNING_RATE**: 0.0001
- **EPOCHS**: 100

### Data Processing Parameters
- **GRANULARITY**: Controls grid resolution (1 = 105x68, 2 = 210x136, etc.)
- **SAMPLE_RATE**: Rate of unlabeled data sampling per epoch
- **ALPHA**: Classification loss weight in semi-supervised learning

## Results and Logging

Training results are saved in `seqlabelvae/results/`:
- **Logs**: JSON files with training metrics
- **Plots**: Loss curves, confusion matrices, reconstruction examples
- **Weights**: Model checkpoints and trained weights


