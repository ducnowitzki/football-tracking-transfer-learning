import os
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

# Configuration
GRANULARITY = 1  # n in the specification
H, W = 105, 68  # Output grid dimensions
H_g, W_g = int(105 * GRANULARITY), int(68 * GRANULARITY)  # Output grid dimensions
WINDOW_SIZE = 32
SAMPLE_RATE = 0.008

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

# Precomputed velocity normalization values
VEL_MIN = np.array([-1023.25, -1326.25, -855.75, -1092.75, -1377.0, -573.75])
VEL_MAX = np.array([1247.0, 804.75, 1602.25, 1263.25, 1349.25, 958.75])

def round_and_clip(x, y):
    xg = int(np.round(x))
    yg = int(np.round(y))
    return np.clip(xg, 0, H_g-1), np.clip(yg, 0, W_g-1)

def map_coordinates(x, y):
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
    min_v = VEL_MIN[channel_idx]
    max_v = VEL_MAX[channel_idx]
    if max_v > min_v:
        return (v - min_v) / (max_v - min_v)
    else:
        return 0.0

def process_tracking_frame(row, prev_row, dt):
    grid = np.zeros((H_g, W_g, 9), dtype=np.float32)
    
    # Home team
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

    #Away team
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
    
    # Ball
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

def select_unlabeled_sequences(df, target_count):
    total_frames = len(df)
    available_sequences = total_frames - WINDOW_SIZE + 1
    
    # Calculate how many sequences to select
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
    print("Starting DFL unlabeled data preprocessing...")
    print("Processing game by game to optimize memory usage...")
    
    # First pass: calculate total possible sequences and target count
    print("First pass: calculating total sequences...")
    total_possible_sequences = 0
    for csv_file in CSV_FILES:
        path = os.path.join(INPUT_DIR, csv_file)
        df = pd.read_csv(path, low_memory=False)
        total_possible_sequences += len(df) - WINDOW_SIZE + 1
        del df
    
    target_unlabeled_sequences = int(total_possible_sequences * SAMPLE_RATE)
    print(f"Total possible sequences: {total_possible_sequences}")
    print(f"Target unlabeled sequences: {target_unlabeled_sequences}")
    
    # Create memmap for unlabeled sequences
    unlabeled_path = os.path.join(OUTPUT_DIR, 'unlabeled_frames.npy')
    print(f"Creating memmap at: {unlabeled_path}")
    unlabeled_frames = open_memmap(
        unlabeled_path, mode='w+',
        dtype='float32',
        shape=(target_unlabeled_sequences, WINDOW_SIZE, H_g, W_g, 9),
    )
    
    # Second pass: process each game and write to memmap
    print("Second pass: processing games and writing to disk...")
    sequences_per_game = target_unlabeled_sequences // len(CSV_FILES)
    current_idx = 0
    
    for game_idx, csv_file in enumerate(CSV_FILES):
        path = os.path.join(INPUT_DIR, csv_file)
        print(f"Processing game {game_idx + 1}/{len(CSV_FILES)}: {csv_file}")
        
        # Load game data
        df = pd.read_csv(path, low_memory=False)
        
        # Select unlabeled sequences for this game
        unlabeled_sequences = select_unlabeled_sequences(df, sequences_per_game)
        
        # Write sequences to memmap
        if unlabeled_sequences:
            sequences_array = np.array(unlabeled_sequences)
            end_idx = current_idx + len(sequences_array)
            unlabeled_frames[current_idx:end_idx] = sequences_array
            current_idx = end_idx
            
            print(f"  Added {len(sequences_array)} sequences (total: {current_idx})")
        
        # Clear memory
        del df, unlabeled_sequences
        if 'sequences_array' in locals():
            del sequences_array
    
    unlabeled_frames.flush()
    
    # Trim memmap to actual size if needed
    if current_idx < target_unlabeled_sequences:
        print(f"Trimming memmap from {target_unlabeled_sequences} to {current_idx} sequences")
        # Create new memmap with correct size
        temp_path = unlabeled_path + '.tmp'
        os.rename(unlabeled_path, temp_path)
        
        # Read from temp and write to new file
        temp_data = np.load(temp_path, mmap_mode='r')
        actual_data = temp_data[:current_idx]
        np.save(unlabeled_path, actual_data)
        
        # Clean up
        del temp_data, actual_data
        os.remove(temp_path)
    
    print(f"Unlabeled data preprocessing completed!")
    print(f"Final unlabeled sequences: {current_idx}")
    print(f"Output saved to: {unlabeled_path}")

if __name__ == "__main__":
    main() 