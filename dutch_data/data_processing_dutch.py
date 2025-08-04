import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Configuration
GRANULARITY = 1  # n in the specification
H, W = 105, 68  # Output grid dimensions
H_g, W_g = int(105 * GRANULARITY), int(68 * GRANULARITY)  # Output grid dimensions
WINDOW_SIZE = 32
SAMPLE_RATE = 0.08

# Input/Output paths
RAW_DIR = 'dutch_data/raw/'
OUTPUT_DIR = "dutch_data/split/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Precomputed velocity normalization values from the Dutch dataset
VEL_MIN = np.array([-200.1000061, -292.8999939, -200.80000305, -253.01269531, -201.6499939, -217.94999695])
VEL_MAX = np.array([837.8125, 749.20001221, 839.375, 836.25, 755.79998779, 670.])

# Tracking and event files
TRACKING_FILES = [
    ('tracking1.csv'),
    ('tracking2.csv'),
]
EVENT_FILES = [
    ('events1.csv'),
    ('events2.csv'),
]

# Construct tracking2 columns (superset)
tracking2_cols = ['timestamp', 'ballId', 'ballposX', 'ballposY']
for i in range(35):
    tracking2_cols += [f'playerId{i}', f'posX{i}', f'posY{i}']

# Helper: get tracking1 columns (players 0-31)
tracking1_cols = ['timestamp', 'ballId', 'ballposX', 'ballposY']
for i in range(32):
    tracking1_cols += [f'playerId{i}', f'posX{i}', f'posY{i}']

# Read and align tracking data
def read_and_align_tracking():
    dfs = []
    for fname in TRACKING_FILES:
        path = os.path.join(RAW_DIR, fname)
        if fname == 'tracking1.csv':
            df = pd.read_csv(path, usecols=lambda c: c in tracking1_cols)
            # Add missing columns for players 32-34
            for i in range(32, 35):
                for col in [f'playerId{i}', f'posX{i}', f'posY{i}']:
                    df[col] = np.nan
            # Reorder columns to match tracking2
            df = df[tracking2_cols]
        else:
            # tracking2.csv may have an extra 'Unnamed: 0' column
            df = pd.read_csv(path, usecols=lambda c: c in tracking2_cols)
            # Ensure all columns present
            for col in tracking2_cols:
                if col not in df.columns:
                    df[col] = np.nan
            df = df[tracking2_cols]
        dfs.append(df)
    tracking_df = pd.concat(dfs, ignore_index=True)
    return tracking_df

# Read events data
def read_events():
    dfs = []
    for fname in EVENT_FILES:
        path = os.path.join(RAW_DIR, fname)
        df = pd.read_csv(path, usecols=['Type', 'timestamp'])
        dfs.append(df)
    events_df = pd.concat(dfs, ignore_index=True)
    return events_df

def map_coordinates(x, y):
    # Map coordinates directly into grid space [0, H_g), [0, W_g)
    if not (0 <= x <= W) or not (0 <= y <= H):
        return None, None
    x_mapped = (x / W) * W_g
    y_mapped = (y / H) * H_g
    return int(np.clip(y_mapped, 0, H_g-1)), int(np.clip(x_mapped, 0, W_g-1))  # y=row, x=col

def normalize_velocity(v, channel_idx):
    """Normalize velocity using precomputed min/max values"""
    min_v = VEL_MIN[channel_idx]
    max_v = VEL_MAX[channel_idx]
    if max_v > min_v:
        return (v - min_v) / (max_v - min_v)
    else:
        return 0.0

def process_tracking_frame(row, prev_row, dt, home_idx, away_idx):
    grid = np.zeros((H_g, W_g, 9), dtype=np.float32)
    # Home players
    for i in home_idx:
        x = row.get(f'posX{i}', np.nan)
        y = row.get(f'posY{i}', np.nan)
        if not pd.isna(x) and not pd.isna(y):
            ii, jj = map_coordinates(x, y)
            if ii is not None and jj is not None:
                grid[ii, jj, 0] = 1
                # Velocity
                if prev_row is not None:
                    x_prev = prev_row.get(f'posX{i}', np.nan)
                    y_prev = prev_row.get(f'posY{i}', np.nan)
                    if not pd.isna(x_prev) and not pd.isna(y_prev):
                        vx = (x - x_prev) / dt
                        vy = (y - y_prev) / dt
                        grid[ii, jj, 3] = normalize_velocity(vx, 0) # home_vx
                        grid[ii, jj, 4] = normalize_velocity(vy, 1) # home_vy
    # Away players
    for i in away_idx:
        x = row.get(f'posX{i}', np.nan)
        y = row.get(f'posY{i}', np.nan)
        if not pd.isna(x) and not pd.isna(y):
            ii, jj = map_coordinates(x, y)
            if ii is not None and jj is not None:
                grid[ii, jj, 1] = 1
                # Velocity
                if prev_row is not None:
                    x_prev = prev_row.get(f'posX{i}', np.nan)
                    y_prev = prev_row.get(f'posY{i}', np.nan)
                    if not pd.isna(x_prev) and not pd.isna(y_prev):
                        vx = (x - x_prev) / dt
                        vy = (y - y_prev) / dt
                        grid[ii, jj, 5] = normalize_velocity(vx, 2) # away_vx
                        grid[ii, jj, 6] = normalize_velocity(vy, 3) # away_vy
    # Ball
    x = row.get('ballposX', np.nan)
    y = row.get('ballposY', np.nan)
    if not pd.isna(x) and not pd.isna(y):
        ii, jj = map_coordinates(x, y)
        if ii is not None and jj is not None:
            grid[ii, jj, 2] = 1
            if prev_row is not None:
                x_prev = prev_row.get('ballposX', np.nan)
                y_prev = prev_row.get('ballposY', np.nan)
                if not pd.isna(x_prev) and not pd.isna(y_prev):
                    vx = (x - x_prev) / dt
                    vy = (y - y_prev) / dt
                    grid[ii, jj, 7] = normalize_velocity(vx, 4) # ball_vx
                    grid[ii, jj, 8] = normalize_velocity(vy, 5) # ball_vy
    return grid

def get_home_away_indices(row_idx, split_idx):
    # tracking1: 0-15 home, 16-31 away; tracking2: 0-17 home, 18-34 away
    if row_idx < split_idx:
        return list(range(16)), list(range(16, 32))
    else:
        return list(range(18)), list(range(18, 35))

def map_events_to_tracking(events_df, tracking_df):
    # Convert timestamps to datetime for matching using the specified format
    events_df['timestamp'] = pd.to_datetime(events_df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
    tracking_df['timestamp'] = pd.to_datetime(tracking_df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
    # Map each event to the closest tracking row (forward fill)
    event_to_tracking = []
    skipped_events = 0
    skipped_by_type = {}
    mapped_distances = []
    mapped_by_type = {}
    for idx, event in events_df.iterrows():
        ts = event['timestamp']
        event_type = event['Type']
        tracking_idx = tracking_df['timestamp'].searchsorted(ts)
        if tracking_idx == len(tracking_df):
            tracking_idx -= 1
        # Compute time distance
        closest_time = tracking_df.iloc[tracking_idx]['timestamp']
        time_dist = abs((closest_time - ts).total_seconds())
        if time_dist > 10:  # 1 hour in seconds
            skipped_events += 1
            skipped_by_type[event_type] = skipped_by_type.get(event_type, 0) + 1
            event_to_tracking.append(None)
            continue
        # Print mapping distance
        print(f"Event '{event_type}' at {ts} mapped to tracking at {closest_time} (distance: {time_dist:.2f} seconds)")
        mapped_distances.append(time_dist)
        mapped_by_type[event_type] = mapped_by_type.get(event_type, 0) + 1
        event_to_tracking.append(tracking_idx)
    print(f"\nSkipped {skipped_events} events (>{3600} seconds from any tracking row)")
    print("Breakdown of skipped events by type:")
    for k, v in skipped_by_type.items():
        print(f"  {k}: {v}")
    print("\nMapped event distances (seconds): min={:.2f}, max={:.2f}, mean={:.2f}".format(
        min(mapped_distances) if mapped_distances else 0,
        max(mapped_distances) if mapped_distances else 0,
        np.mean(mapped_distances) if mapped_distances else 0))
    print("Breakdown of mapped events by type:")
    for k, v in mapped_by_type.items():
        print(f"  {k}: {v}")
    return event_to_tracking

def encode_event_type(event_type):
    # Map event types to integer labels (customize as needed)
    event_type = event_type.lower()
    if event_type == 'Pass'.lower():
        return 0
    elif event_type == 'TacklesAndInterceptions'.lower():
        return 1
    elif event_type == 'Shot'.lower():
        return 2
    else:
        return -1  # ignore other types

def process_game_labeled_data(tracking_df, events_df, event_to_tracking, start_idx, end_idx, split_idx):
    labeled_sequences = []
    labeled_labels = []
    half_window = WINDOW_SIZE // 2
    dt = 0.2
    for event_idx, event_row in events_df.iterrows():
        tracking_idx = event_to_tracking[event_idx]
        if not tracking_idx: continue

        if not (start_idx <= tracking_idx < end_idx):
            continue  # skip events not in this game
        # Check if we have enough frames around the event
        if tracking_idx - half_window >= start_idx and tracking_idx + half_window < end_idx:
            sequence = np.zeros((WINDOW_SIZE, H_g, W_g, 9), dtype=np.float32)
            for seq_idx, frame_idx in enumerate(range(tracking_idx - half_window, tracking_idx + half_window)):
                row = tracking_df.iloc[frame_idx]
                prev_row = tracking_df.iloc[frame_idx-1] if frame_idx > 0 else None
                home_idx, away_idx = get_home_away_indices(frame_idx, split_idx)
                sequence[seq_idx] = process_tracking_frame(row, prev_row, dt, home_idx, away_idx)
            label = encode_event_type(event_row['Type'])
            if label >= 0:
                labeled_sequences.append(sequence)
                labeled_labels.append(label)
    return labeled_sequences, labeled_labels

def select_unlabeled_sequences(tracking_df, start_idx, end_idx, target_count, split_idx):
    total_frames = end_idx - start_idx
    available_sequences = total_frames - WINDOW_SIZE + 1
    sequences_to_select = min(int(available_sequences * SAMPLE_RATE), target_count)
    if sequences_to_select <= 0:
        return []
    step = max(1, available_sequences // sequences_to_select)
    selected_indices = []
    for i in range(sequences_to_select):
        start = start_idx + i * step
        if start + WINDOW_SIZE <= end_idx:
            selected_indices.append(start)
    dt = 0.2
    sequences = []
    for start in selected_indices:
        sequence = np.zeros((WINDOW_SIZE, H_g, W_g, 9), dtype=np.float32)
        for seq_idx, frame_idx in enumerate(range(start, start + WINDOW_SIZE)):
            row = tracking_df.iloc[frame_idx]
            prev_row = tracking_df.iloc[frame_idx-1] if frame_idx > 0 else None
            home_idx, away_idx = get_home_away_indices(frame_idx, split_idx)
            sequence[seq_idx] = process_tracking_frame(row, prev_row, dt, home_idx, away_idx)
        sequences.append(sequence)
    return sequences

def main():
    print('Reading and aligning tracking data...')
    tracking_df = read_and_align_tracking()
    print('Tracking data shape:', tracking_df.shape)
    print('Reading events data...')
    events_df = read_events()
    print('Events data shape:', events_df.shape)

    # Find split index between tracking1 and tracking2
    split_idx = 0
    game_indices = []
    for fname in TRACKING_FILES:
        path = os.path.join(RAW_DIR, fname)
        nrows = sum(1 for _ in open(path)) - 1
        game_indices.append((split_idx, split_idx + nrows))
        split_idx += nrows
    print('Game indices:', game_indices)

    # Map events to tracking
    event_to_tracking = map_events_to_tracking(events_df, tracking_df)

    all_labeled_sequences = []
    all_labeled_labels = []
    all_unlabeled_sequences = []
    total_possible_sequences = 0
    for game_idx, (start_idx, end_idx) in enumerate(game_indices):
        print(f'Processing game {game_idx+1}...')
        # Labeled data
        labeled_sequences, labeled_labels = process_game_labeled_data(
            tracking_df, events_df, event_to_tracking, start_idx, end_idx, game_indices[0][1])
        all_labeled_sequences.extend(labeled_sequences)
        all_labeled_labels.extend(labeled_labels)
        # Unlabeled data
        total_frames = end_idx - start_idx
        total_possible_sequences += total_frames - WINDOW_SIZE + 1
    # Calculate target number of unlabeled sequences (15% of total possible)
    target_unlabeled_sequences = int(total_possible_sequences * SAMPLE_RATE)
    print(f'Target unlabeled sequences: {target_unlabeled_sequences}')
    sequences_per_game = target_unlabeled_sequences // len(game_indices)
    for game_idx, (start_idx, end_idx) in enumerate(game_indices):
        print(f'Processing unlabeled data for game {game_idx+1}...')
        unlabeled_sequences = select_unlabeled_sequences(
            tracking_df, start_idx, end_idx, sequences_per_game, game_indices[0][1])
        all_unlabeled_sequences.extend(unlabeled_sequences)

    # Save labeled data
    if all_labeled_sequences:
        all_labeled_sequences = np.array(all_labeled_sequences)
        all_labeled_labels = np.array(all_labeled_labels)
        train_seqs, test_seqs, train_labels, test_labels = train_test_split(
            all_labeled_sequences, all_labeled_labels, test_size=0.2, random_state=42, stratify=all_labeled_labels)
        print('Saving labeled data...')
        np.save(os.path.join(OUTPUT_DIR, 'train_labeled_sequences.npy'), train_seqs)
        np.save(os.path.join(OUTPUT_DIR, 'train_labels.npy'), train_labels)
        np.save(os.path.join(OUTPUT_DIR, 'test_labeled_sequences.npy'), test_seqs)
        np.save(os.path.join(OUTPUT_DIR, 'test_labels.npy'), test_labels)
        print(f'Labeled sequences - Train: {len(train_seqs)}, Test: {len(test_seqs)}')
        print(f'Label distribution: {np.bincount(all_labeled_labels)}')
    else:
        print('No labeled sequences found!')
    # Save unlabeled data
    if all_unlabeled_sequences:
        all_unlabeled_sequences = np.array(all_unlabeled_sequences)
        print('Saving unlabeled data...')
        np.save(os.path.join(OUTPUT_DIR, 'unlabeled_frames.npy'), all_unlabeled_sequences)
        print(f'Unlabeled sequences: {len(all_unlabeled_sequences)}')
        print(f'Unlabeled data shape: {all_unlabeled_sequences.shape}')
    else:
        print('No unlabeled sequences found!')

    # After mapping events to tracking
    # Print resulting event distribution after filtering
    mapped_types = []
    for i, idx in enumerate(event_to_tracking):
        if idx is not None:
            mapped_types.append(events_df.iloc[i]['Type'])
    if mapped_types:
        from collections import Counter
        print("\nResulting event distribution after mapping:")
        for k, v in Counter(mapped_types).items():
            print(f"  {k}: {v}")
    else:
        print("No events mapped!")
    print('Data preprocessing completed!')
    print(f'Output saved to: {OUTPUT_DIR}')

if __name__ == '__main__':
    main()
