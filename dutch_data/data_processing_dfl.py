import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from numpy.lib.format import open_memmap

# Configuration
GRANULARITY = 1  # n in the specification
H, W = 105, 68  # Output grid dimensions
H_g, W_g = int(105 * GRANULARITY), int(68 * GRANULARITY)  # Output grid dimensions
WINDOW_SIZE = 32
SAMPLE_RATE = 0.05

# Input/Output paths
INPUT_DIR = "dfl_data/raw/"
OUTPUT_DIR = "dfl_data/split/"

# List of CSV files to process
CSV_FILES = [
    "tracking_df_J03WMX.csv",
    "tracking_df_J03WN1.csv", 
    "tracking_df_J03WPY.csv",
    "tracking_df_J03WOH.csv",
    "tracking_df_J03WQQ.csv",
    "tracking_df_J03WOY.csv",
    "tracking_df_J03WR9.csv"
]

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Label encoding for databallpy events
LABEL_ENCODING = {
    'pass': 0,
    'shot': 1,
    'dribble': 2
}

# Precomputed velocity normalization values
VEL_MIN = np.array([-1023.25, -1326.25, -855.75, -1092.75, -1377.0, -573.75])
VEL_MAX = np.array([1247.0, 804.75, 1602.25, 1263.25, 1349.25, 958.75])

def round_and_clip(x, y):
    """Round and clip coordinates to grid dimensions"""
    xg = int(np.round(x))
    yg = int(np.round(y))
    return np.clip(xg, 0, H_g-1), np.clip(yg, 0, W_g-1)

def map_coordinates(x, y):
    """Map coordinates directly into grid space [0, H_g), [0, W_g)"""
    # Check if x, y are within valid pitch bounds
    # x in [-W/2, W/2], y in [-H/2, H/2]
    if not (-W/2 <= x <= W/2) or not (-H/2 <= y <= H/2):
        return None, None
    # Map x from [-W/2, W/2] to [0, W_g)
    x_mapped = ((x + W/2) / W) * W_g
    # Map y from [-H/2, H/2] to [0, H_g)
    y_mapped = ((y + H/2) / H) * H_g
    return round_and_clip(y_mapped, x_mapped)  # Note: y is row, x is col

def normalize_velocity(v, channel_idx):
    """Normalize velocity using precomputed min/max values"""
    min_v = VEL_MIN[channel_idx]
    max_v = VEL_MAX[channel_idx]
    if max_v > min_v:
        return (v - min_v) / (max_v - min_v)
    else:
        return 0.0

def process_tracking_frame(row, prev_row, dt):
    """Process a single tracking frame to create 9-channel grid with correct channel mapping"""
    grid = np.zeros((H_g, W_g, 9), dtype=np.float32)
    
    # --- Home team ---
    home_cols = [col for col in row.index if col.startswith('home_') and col.endswith('_x')]
    for col in home_cols:
        player_id = col.split('_x')[0]
        x_col = f"{player_id}_x"
        y_col = f"{player_id}_y"
        
        if not pd.isna(row[x_col]) and not pd.isna(row[y_col]):
            i, j = map_coordinates(row[x_col], row[y_col])
            if i is not None and j is not None:
                grid[i, j, 0] = 1  # Channel 0: home player exists
                
                # Velocity channels 3, 4: home_vx, home_vy
                if prev_row is not None and not pd.isna(prev_row[x_col]) and not pd.isna(prev_row[y_col]):
                    vx = (row[x_col] - prev_row[x_col]) / dt
                    vy = (row[y_col] - prev_row[y_col]) / dt
                    grid[i, j, 3] = normalize_velocity(vx, 0)  # home_vx
                    grid[i, j, 4] = normalize_velocity(vy, 1)  # home_vy
    
    # --- Away team ---
    away_cols = [col for col in row.index if col.startswith('away_') and col.endswith('_x')]
    for col in away_cols:
        player_id = col.split('_x')[0]
        x_col = f"{player_id}_x"
        y_col = f"{player_id}_y"
        
        if not pd.isna(row[x_col]) and not pd.isna(row[y_col]):
            i, j = map_coordinates(row[x_col], row[y_col])
            if i is not None and j is not None:
                grid[i, j, 1] = 1  # Channel 1: away player exists
                
                # Velocity channels 5, 6: away_vx, away_vy
                if prev_row is not None and not pd.isna(prev_row[x_col]) and not pd.isna(prev_row[y_col]):
                    vx = (row[x_col] - prev_row[x_col]) / dt
                    vy = (row[y_col] - prev_row[y_col]) / dt
                    grid[i, j, 5] = normalize_velocity(vx, 2)  # away_vx
                    grid[i, j, 6] = normalize_velocity(vy, 3)  # away_vy
    
    # --- Ball ---
    if not pd.isna(row['ball_x']) and not pd.isna(row['ball_y']):
        i, j = map_coordinates(row['ball_x'], row['ball_y'])
        if i is not None and j is not None:
            grid[i, j, 2] = 1  # Channel 2: ball exists
            
            # Velocity channels 7, 8: ball_vx, ball_vy
            if prev_row is not None and not pd.isna(prev_row['ball_x']) and not pd.isna(prev_row['ball_y']):
                vx = (row['ball_x'] - prev_row['ball_x']) / dt
                vy = (row['ball_y'] - prev_row['ball_y']) / dt
                grid[i, j, 7] = normalize_velocity(vx, 4)  # ball_vx
                grid[i, j, 8] = normalize_velocity(vy, 5)  # ball_vy
    
    return grid

def process_game_labeled_data(df):
    """Process labeled data for a single game"""
    labeled_sequences = []
    labeled_labels = []
    
    # Find events (where databallpy_event is not 'no_event')
    event_mask = df['databallpy_event'].notna()
    event_indices = df[event_mask].index.tolist()
    
    half_window = WINDOW_SIZE // 2
    dt = 0.04  # 25Hz sampling rate
    
    for event_idx in event_indices:
        # Check if we have enough frames around the event
        if event_idx - half_window >= 0 and event_idx + half_window < len(df):
            # Create sequence around event
            sequence = np.zeros((WINDOW_SIZE, H_g, W_g, 9), dtype=np.float32)
            
            for seq_idx, frame_idx in enumerate(range(event_idx - half_window, event_idx + half_window)):
                row = df.iloc[frame_idx]
                prev_row = df.iloc[frame_idx-1] if frame_idx > 0 else None
                sequence[seq_idx] = process_tracking_frame(row, prev_row, dt)
            
            # Get event type and encode label
            event_type = df.iloc[event_idx]['databallpy_event']
            if event_type in LABEL_ENCODING:
                label = LABEL_ENCODING[event_type]
                labeled_sequences.append(sequence)
                labeled_labels.append(label)
    
    return labeled_sequences, labeled_labels

def select_unlabeled_sequences(df, target_count):
    """Select unlabeled sequences from a game, ensuring they don't cross game boundaries"""
    total_frames = len(df)
    available_sequences = total_frames - WINDOW_SIZE + 1
    
    # Calculate how many sequences to select (15% of available)
    sequences_to_select = min(int(available_sequences * SAMPLE_RATE), target_count)
    
    if sequences_to_select <= 0:
        return []
    
    # Select starting indices for sequences
    # Ensure sequences are spaced properly and don't overlap
    step = max(1, available_sequences // sequences_to_select)
    selected_indices = []
    
    for i in range(sequences_to_select):
        start_idx = i * step
        if start_idx + WINDOW_SIZE <= total_frames:
            selected_indices.append(start_idx)
    
    # Process selected sequences
    dt = 0.04
    sequences = []
    
    for start_idx in selected_indices:
        sequence = np.zeros((WINDOW_SIZE, H_g, W_g, 9), dtype=np.float32)
        
        for seq_idx, frame_idx in enumerate(range(start_idx, start_idx + WINDOW_SIZE)):
            row = df.iloc[frame_idx]
            prev_row = df.iloc[frame_idx-1] if frame_idx > 0 else None
            sequence[seq_idx] = process_tracking_frame(row, prev_row, dt)
        
        sequences.append(sequence)
    
    return sequences

def main():
    print("Starting DFL data preprocessing for seqlabelvae...")
    print("Processing game by game to optimize memory usage...")
    
    # Process labeled data first
    print("Processing labeled data...")
    all_labeled_sequences = []
    all_labeled_labels = []
    
    for csv_file in CSV_FILES:
        path = os.path.join(INPUT_DIR, csv_file)
        print(f"Processing labeled data from {csv_file}...")
        
        # Load game data
        df = pd.read_csv(path, low_memory=False)
        
        # Process labeled sequences for this game
        labeled_sequences, labeled_labels = process_game_labeled_data(df)
        all_labeled_sequences.extend(labeled_sequences)
        all_labeled_labels.extend(labeled_labels)
        
        # Clear memory
        del df
    
    # Convert to numpy arrays and save labeled data
    if all_labeled_sequences:
        all_labeled_sequences = np.array(all_labeled_sequences)
        all_labeled_labels = np.array(all_labeled_labels)
        
        # Split into train/test
        train_seqs, test_seqs, train_labels, test_labels = train_test_split(
            all_labeled_sequences, 
            all_labeled_labels, 
            test_size=0.2, 
            random_state=42,
            stratify=all_labeled_labels
        )
        
        # Save labeled data
        print("Saving labeled data...")
        np.save(os.path.join(OUTPUT_DIR, 'train_labeled_sequences.npy'), train_seqs)
        np.save(os.path.join(OUTPUT_DIR, 'train_labels.npy'), train_labels)
        np.save(os.path.join(OUTPUT_DIR, 'test_labeled_sequences.npy'), test_seqs)
        np.save(os.path.join(OUTPUT_DIR, 'test_labels.npy'), test_labels)
        
        print(f"Labeled sequences - Train: {len(train_seqs)}, Test: {len(test_seqs)}")
        print(f"Label distribution: {np.bincount(all_labeled_labels)}")
        
        # Clear memory
        del all_labeled_sequences, all_labeled_labels, train_seqs, test_seqs, train_labels, test_labels
    else:
        print("No labeled sequences found!")
    
    # Process unlabeled data
    print("Processing unlabeled data...")
    all_unlabeled_sequences = []
    
    # Calculate target number of unlabeled sequences (15% of total possible)
    total_possible_sequences = 0
    for csv_file in CSV_FILES:
        path = os.path.join(INPUT_DIR, csv_file)
        df = pd.read_csv(path, low_memory=False)
        total_possible_sequences += len(df) - WINDOW_SIZE + 1
        del df
    
    target_unlabeled_sequences = int(total_possible_sequences * SAMPLE_RATE)
    print(f"Target unlabeled sequences: {target_unlabeled_sequences}")
    
    # Process each game for unlabeled sequences
    sequences_per_game = target_unlabeled_sequences // len(CSV_FILES)
    
    for csv_file in CSV_FILES:
        path = os.path.join(INPUT_DIR, csv_file)
        print(f"Processing unlabeled data from {csv_file}...")
        
        # Load game data
        df = pd.read_csv(path, low_memory=False)
        
        # Select unlabeled sequences for this game
        unlabeled_sequences = select_unlabeled_sequences(df, sequences_per_game)
        all_unlabeled_sequences.extend(unlabeled_sequences)
        
        # Clear memory
        del df
    
    # Convert to numpy array and save unlabeled data
    if all_unlabeled_sequences:
        all_unlabeled_sequences = np.array(all_unlabeled_sequences)
        
        # Save unlabeled data
        print("Saving unlabeled data...")
        np.save(os.path.join(OUTPUT_DIR, 'unlabeled_frames.npy'), all_unlabeled_sequences)
        
        print(f"Unlabeled sequences: {len(all_unlabeled_sequences)}")
        print(f"Unlabeled data shape: {all_unlabeled_sequences.shape}")
        
        # Clear memory
        del all_unlabeled_sequences
    else:
        print("No unlabeled sequences found!")
    
    print("Data preprocessing completed!")
    print(f"Output saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 