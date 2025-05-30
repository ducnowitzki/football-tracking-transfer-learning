import math
import os
import numpy as np
from idsse_helper import load_data
import pickle

DATA_DIR = './data/idsse-data/data/'
OUTPUT_DIR = './data/idsse-data/seqlabelvae/whole'
H, W = 105, 68  # pitch grid
WINDOW_SIZE = 32
FRAME_RATE = 25

# unpickle events_labels.pkl
events_labels_path = os.path.abspath('./data/idsse-data/events_labels.pkl')
with open(events_labels_path, 'rb') as f:
    EVENTS_LABELS = pickle.load(f)

def round_and_clip(x, y):
    xg = int(np.round(x))
    yg = int(np.round(y))
    xg = np.clip(xg, 0, H-1)
    yg = np.clip(yg, 0, W-1)
    return xg, yg

# maybe not necessary as we ignore NaNs later on
def pad_positions(arr, target=22):
    # arr: [n,] (x1, y1, x2, y2, ...)
    # Remove NaNs, reshape to (-1,2)
    arr = arr[~np.isnan(arr)]
    arr = arr[:target*2]  # at most 22 values
    arr = arr.reshape(-1, 2)
    # if arr.shape[0] < target:
    #     padding = np.full((target - arr.shape[0], 2), np.nan)
    #     arr = np.vstack([arr, padding])
    return arr

def build_data(xy_objects: dict, events: dict, window_size: int = 16, output_dir: str = './processed'):
    os.makedirs(output_dir, exist_ok=True)
    halves = ['firstHalf', 'secondHalf']
    all_grids, all_aux = [], []

    # num_frames_first_half = 0

    # For each half, process positions and events
    for half in halves:
        xy_home = xy_objects[half]['Home'].xy  # [T, ?]
        xy_away = xy_objects[half]['Away'].xy  # [T, ?]
        xy_ball = xy_objects[half]['Ball'].xy  # [T, ?]
        framerate = xy_objects[half]['Home'].framerate or FRAME_RATE

        T = xy_home.shape[0]

        # Grid channels: home, away, ball
        grids = np.zeros((T, H, W, 3), dtype=np.float32)
        
        # num_frames

        # Per frame, convert available player positions into grid, pad as needed
        for t in range(T):
            # Home team
            pos_home = pad_positions(xy_home[t, :])
            for x, y in pos_home:
                # check if x and y are numbers
                xg, yg = round_and_clip(x, y)
                grids[t, xg, yg, 0] = 1
            # Away team
            pos_away = pad_positions(xy_away[t, :])
            for x, y in pos_away:
                xg, yg = round_and_clip(x, y)
                grids[t, xg, yg, 1] = 1
            # Ball
            pos_ball = pad_positions(xy_ball[t, :], target=1)
            x, y = pos_ball[0]
            xg, yg = round_and_clip(x, y)
            grids[t, xg, yg, 2] = 1

            # num_frames_first_half += 1 if half == 'firstHalf' else 0

        # Build auxiliary labels using event gameclock
        aux = np.zeros(T, dtype=np.int32)
        for side in ['Home', 'Away']:
            ev_obj = events[half][side]
            print(type(ev_obj.events))
            for _, row in ev_obj.events.iterrows():
                # todo: for second half, add T_first_half * framerate to idx
                 # Use gameclock (seconds since half start) to frame idx
                idx = math.floor(row.gameclock * framerate)
                if 0 <= idx < T:
                    # idx += num_frames_first_half if half == 'secondHalf' else 0
                    aux[idx] = EVENTS_LABELS[row.eID]
        all_grids.append(grids)
        all_aux.append(aux)

    # Merge halves
    grids_all = np.vstack(all_grids)
    aux_all = np.concatenate(all_aux)
    np.save(os.path.join(output_dir, 'unlabeled_frames.npy'), grids_all)

    # Build labeled sequences (sliding window around each event)
    half_w = window_size // 2
    seqs, labels = [], []
    for i, eid in enumerate(aux_all):
        if eid > 0 and i - half_w >= 0 and i + half_w < len(aux_all):
            seqs.append(grids_all[i - half_w:i + half_w])
            labels.append(eid)

    # Sample negatives
    neg_count = len(labels)
    T_all = grids_all.shape[0]
    sampled = 0
    while sampled < neg_count:
        i = np.random.randint(half_w, T_all - half_w)
        if aux_all[i] == 0:
            seqs.append(grids_all[i - half_w:i + half_w])
            labels.append(0)
            sampled += 1

    np.save(os.path.join(output_dir, 'labeled_sequences.npy'), np.stack(seqs))
    np.save(os.path.join(output_dir, 'labels.npy'), np.array(labels, dtype=np.int32))

if __name__ == "__main__":
    # Example usage
    fpos = "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOH.xml"
    finfo = "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOH.xml"
    fevents = "DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WOH.xml"
    xy_objects, events, pitch = load_data(DATA_DIR, fpos, finfo, fevents)
    build_data(xy_objects, events, WINDOW_SIZE, OUTPUT_DIR)
