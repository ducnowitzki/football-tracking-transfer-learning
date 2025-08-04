# Football Tracking Transfer Learning for Semi-Supervised Event Detection

## Project Overview

This repository contains the implementation of a Master's thesis project focused on **Transfer Learning for Semi-Supervised Football Event Detection Models using Tracking Data**. The project explores how to leverage professional-level tracking data to improve event detection models on amateur-level data through transfer learning techniques.

### Research Focus
- **Semi-supervised learning** for football event detection
- **Transfer learning** from professional to amateur data
- **Sequence-based models** using tracking data
- **Event classification**: pass, shot, dribble detection

## Project Structure

```
football-tracking-transfer-learning/
├── prof_data/                    # Professional DFL data processing
│   ├── raw/                      # Raw DFL data (to be downloaded)
│   ├── data_processing_dfl.py   # Professional data preprocessing
│   └── data_processing_dfl_unlabeled.py
├── amateur_data/                 # Amateur Dutch data processing
│   ├── raw/                      # Raw Dutch data
│   └── data_processing_dutch.py # Amateur data preprocessing
├── seqlabelvae/                  # Main model implementation
│   ├── model.py                  # SeqLabelVAE model architecture
│   ├── config.py                 # Model configuration
│   ├── train.py                  # Training script
│   ├── test.py                   # Testing script
│   ├── run_training.py           # Training execution script
│   ├── run_test.py               # Testing execution script
│   ├── smote.py                  # SMOTE upsampling implementation
│   ├── run_smote.py              # SMOTE execution script
│   └── results/                  # Training results and logs
├── idsse-data/                   # IDSSE dataset processing
├── dutch_data/                   # Dutch dataset processing
└── VAE_GNN/                      # Alternative VAE-GNN implementation
```

## Setup Instructions

### 1. Download Professional DFL Data

Download the professional DFL dataset from the official source:
- **Source**: [An integrated dataset of spatiotemporal and event data in elite soccer](https://springernature.figshare.com/articles/dataset/An_integrated_dataset_of_spatiotemporal_and_event_data_in_elite_soccer/28196177)
- **Size**: 2.45 GB
- **License**: CC BY

**Instructions:**
1. Visit the Figshare link above
2. Download the complete dataset (2.45 GB)
3. Extract the files to `prof_data/raw/`
4. Ensure the following files are present in `prof_data/raw/`:
   - `tracking_df_J03WMX.csv`
   - `tracking_df_J03WN1.csv`
   - `tracking_df_J03WPY.csv`
   - `tracking_df_J03WOH.csv`
   - `tracking_df_J03WQQ.csv`
   - `tracking_df_J03WOY.csv`
   - `tracking_df_J03WR9.csv`

### 2. Data Preprocessing

#### Professional Data Preprocessing
```bash
cd prof_data/
python data_processing_dfl.py
```

**Configuration Options:**
- **GRANULARITY**: Controls grid resolution (default: 1)
- **WINDOW_SIZE**: Sequence length for training (default: 32)
- **SAMPLE_RATE**: Rate of unlabeled data sampling (default: 0.05)

#### Amateur Data Preprocessing
```bash
cd amateur_data/
python data_processing_dutch.py
```

**Configuration Options:**
- **GRANULARITY**: Controls grid resolution (default: 1)
- **WINDOW_SIZE**: Sequence length for training (default: 32)
- **SAMPLE_RATE**: Rate of unlabeled data sampling (default: 0.08)

### 3. Model Training and Evaluation

#### Basic Training
```bash
cd seqlabelvae/
python run_training.py
```

#### Training with Dutch Data (Transfer Learning)
```bash
cd seqlabelvae/
python run_training_dutch.py
```

#### Testing Trained Models
```bash
cd seqlabelvae/
python run_test.py
```

#### SMOTE Upsampling
```bash
cd seqlabelvae/
python run_smote.py
```

#### SMOTE with Retraining
```bash
cd seqlabelvae/
python run_smote_retrain.py
```

#### Testing SMOTE Models
```bash
cd seqlabelvae/
python run_test_smote.py
```

#### Retrain Classifier
```bash
cd seqlabelvae/
python run_retrain_classifier.py
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

## Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Usage Examples

### 1. Professional Data Processing with Different Granularity
```python
# In prof_data/data_processing_dfl.py
GRANULARITY = 2  # Higher resolution grid
```

### 2. Amateur Data Processing with Different Granularity
```python
# In amateur_data/data_processing_dutch.py
GRANULARITY = 1  # Standard resolution grid
```

### 3. Training with Custom Parameters
```python
# In seqlabelvae/config.py
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 200
```

## Research Contributions

1. **Semi-supervised learning** for football event detection
2. **Transfer learning** from professional to amateur data
3. **Grid-based representation** of tracking data
4. **Sequence modeling** with variational autoencoders
5. **Multi-class event detection** (pass, shot, dribble)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{football_tracking_transfer_learning,
  title={Transfer Learning for Semi-Supervised Football Event Detection using Tracking Data},
  author={[Your Name]},
  year={2024},
  institution={[Your Institution]}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please contact [your email] or create an issue in this repository.
