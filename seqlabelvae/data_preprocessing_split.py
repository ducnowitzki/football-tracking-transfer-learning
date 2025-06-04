import os
import math
import pickle
import numpy as np
from idsse_helper import load_data
from numpy.lib.format import open_memmap
from configs import DATA_DIR, OUTPUT_DIR_SPLIT, OUTPUT_DIR_WHOLE, EVENTS_LABELS_PATH


# ----- Configuration -----
H, W        = 105, 68
WINDOW_SIZE = 32
FRAME_RATE  = 25
HALF_W      = WINDOW_SIZE // 2

FILE_GROUPS = [
     # (pos_xml, info_xml, events_xml),
     ("DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WMX.xml","DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WMX.xml","DFL_03_02_events_raw_DFL-COM-000001_DFL-MAT-J03WMX.xml"),
     ("DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WN1.xml","DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WN1.xml","DFL_03_02_events_raw_DFL-COM-000001_DFL-MAT-J03WN1.xml"),
     ("DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOH.xml","DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOH.xml","DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WOH.xml"),
     ("DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOY.xml","DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOY.xml","DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WOY.xml"),
     ("DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WPY.xml","DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WPY.xml","DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WPY.xml"),
     ("DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WQQ.xml","DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WQQ.xml","DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WQQ.xml"),
     ("DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WR9.xml","DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WR9.xml","DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WR9.xml"),
   ]


# Load event-to-index map
with open(os.path.join(EVENTS_LABELS_PATH), 'rb') as f:
    EVENTS_LABELS = pickle.load(f)

def round_and_clip(x, y):
    xg = int(np.round(x)); yg = int(np.round(y))
    return np.clip(xg, 0, H-1), np.clip(yg, 0, W-1)

def pad_positions(arr, target=22):
    vals = arr[~np.isnan(arr)]
    vals = vals[:target*2].reshape(-1,2)
    return vals

# 1) First pass: compute total frames
print("Computing total frames for all matches...")
frames_per_match = []
for fpos, finfo, fevents in FILE_GROUPS:
    xy, ev, _ = load_data(DATA_DIR, fpos, finfo, fevents)
    T1 = xy['firstHalf']['Home'].xy.shape[0]
    T2 = xy['secondHalf']['Home'].xy.shape[0]
    frames_per_match.append(T1 + T2)
total_frames = sum(frames_per_match)

# 2) Create memmap-backed .npy for all unlabeled frames
print("Creating memmap for unlabeled frames...")
os.makedirs(OUTPUT_DIR_WHOLE, exist_ok=True)
unlabeled_path = os.path.join(OUTPUT_DIR_WHOLE, 'unlabeled_frames.npy')
unlabeled_frames = open_memmap(
    unlabeled_path, mode='w+',
    dtype='float32',
    shape=(total_frames, H, W, 3),
)

labeled_seqs = []
labeled_labels = []

cursor = 0
for (fpos, finfo, fevents), match_frames in zip(FILE_GROUPS, frames_per_match):
    xy, ev, _ = load_data(DATA_DIR, fpos, finfo, fevents)
    # build per-match aux array
    aux = np.zeros(match_frames, dtype=np.int32)
    offset = 0

    # process both halves
    for half in ['firstHalf','secondHalf']:
        home = xy[half]['Home'].xy
        away = xy[half]['Away'].xy
        ball = xy[half]['Ball'].xy
        fr = xy[half]['Home'].framerate or FRAME_RATE
        T = home.shape[0]

        # fill unlabeled memmap
        for t in range(T):
            grid = np.zeros((H, W, 3), dtype=np.float32)
            for x,y in pad_positions(home[t]):
                i,j = round_and_clip(x,y); grid[i,j,0] = 1
            for x,y in pad_positions(away[t]):
                i,j = round_and_clip(x,y); grid[i,j,1] = 1
            bx,by = pad_positions(ball[t], target=1)[0]
            i,j  = round_and_clip(bx,by); grid[i,j,2] = 1

            unlabeled_frames[cursor] = grid
            cursor += 1

        # assign aux labels (skip 'Delete' and negative clock)
        for team in ['Home','Away']:
            df = ev[half][team].events
            for _, row in df.iterrows():
                if row.eID == 'Delete' or row.gameclock < 0:
                    continue
                idx = math.floor(row.gameclock * fr) + offset
                if 0 <= idx < match_frames:
                    aux[idx] = EVENTS_LABELS.get(row.eID, 0)

        offset += T

    # collect positive sequences
    for i, label in enumerate(aux):
        if label > 0 and (i - HALF_W >= 0) and (i + HALF_W < match_frames):
            start = cursor - match_frames + i - HALF_W
            seq = unlabeled_frames[start:start + WINDOW_SIZE]
            labeled_seqs.append(seq)
            labeled_labels.append(label)

# ensure data is flushed
unlabeled_frames.flush()

# stack labeled arrays
labeled_seqs  = np.stack(labeled_seqs)                # (N, WINDOW_SIZE, H, W, 3)
labeled_labels= np.array(labeled_labels, dtype=np.int32)  # (N,)

# 3) Split labeled into train/val/test (80/10/10)
print("Splitting labeled data into train/val/test sets...")
N = labeled_labels.shape[0]
perm = np.random.permutation(N)
n_train = int(0.8 * N)
n_val   = int(0.1 * N)

idx_train = perm[:n_train]
idx_val   = perm[n_train:n_train+n_val]
idx_test  = perm[n_train+n_val:]

# save merged unlabeled (memmap .npy already has it)
# save labeled and splits
np.save(os.path.join(OUTPUT_DIR_WHOLE, 'labeled_sequences.npy'), labeled_seqs)
np.save(os.path.join(OUTPUT_DIR_WHOLE, 'labels.npy'), labeled_labels)

np.save(os.path.join(OUTPUT_DIR_SPLIT, 'train_sequences.npy'), labeled_seqs[idx_train])
np.save(os.path.join(OUTPUT_DIR_SPLIT, 'train_labels.npy'), labeled_labels[idx_train])
np.save(os.path.join(OUTPUT_DIR_SPLIT, 'val_sequences.npy'),   labeled_seqs[idx_val])
np.save(os.path.join(OUTPUT_DIR_SPLIT, 'val_labels.npy'),      labeled_labels[idx_val])
np.save(os.path.join(OUTPUT_DIR_SPLIT, 'test_sequences.npy'),  labeled_seqs[idx_test])
np.save(os.path.join(OUTPUT_DIR_SPLIT, 'test_labels.npy'),     labeled_labels[idx_test])


