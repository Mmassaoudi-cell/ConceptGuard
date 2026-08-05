"""
Real benign-distribution-shift split for Edge-IIoTset, used by TTA-GraphRL's candidate-specific
experiment (Experiment 11 in the master spec). Not synthetic: source and shift pools come from
genuinely different physical sensor-device captures (raw per-device files), not a random split of
the same distribution.

Source domain (train/val/in-distribution-test): attacks (all types) + Normal traffic from 7 of the
10 device folders.
Shift domain (target, adaptation-time): Normal traffic from the 3 held-out device folders + a
held-out slice of attacks (so attack detection remains meaningful in the shifted evaluation).

Memory note: the raw per-device/per-attack CSVs total ~12M+6M rows and this machine has limited
free RAM alongside other running processes (~16GB free of 68GB observed) — loading full files
before subsampling caused a C-parser out-of-memory error in practice. Row caps are therefore
applied PER FILE, at read time, via chunked reservoir-style sampling, so peak memory is bounded by
chunk size, never by full-file size.
"""
import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from dataset_configs import DATASETS

EDGE_ROOT = os.path.join(DATASETS["edge_iiot"]["files"][0].rsplit(os.sep, 2)[0])  # .../Edge-IIoTset dataset
ATTACK_DIR = os.path.join(EDGE_ROOT, "Attack traffic")
NORMAL_DIR = os.path.join(EDGE_ROOT, "Normal traffic")
LABEL_COL = "Attack_type"
ID_LIKE_COLS = ["ip.src_host", "ip.dst_host", "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.protoname"]


def _read_csv_capped(path, max_rows_per_file, chunksize=100_000, seed=0):
    """Reads a (possibly huge) CSV with bounded peak memory: streams in chunks, keeps every
    chunk fully until the running total would exceed max_rows_per_file, then keeps a random
    subsample of the final chunk to hit the cap exactly. If max_rows_per_file is None, reads the
    whole file (only safe for the small files)."""
    if max_rows_per_file is None:
        return pd.read_csv(path, low_memory=False)
    rng = np.random.default_rng(seed)
    kept = []
    total = 0
    for chunk in pd.read_csv(path, low_memory=False, chunksize=chunksize):
        if total >= max_rows_per_file:
            break
        remaining = max_rows_per_file - total
        if len(chunk) <= remaining:
            kept.append(chunk)
            total += len(chunk)
        else:
            idx = rng.choice(len(chunk), size=remaining, replace=False)
            kept.append(chunk.iloc[idx])
            total += remaining
            break
    return pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()


def _read_many_capped(paths, max_rows_per_file, tag_col=None, seed=0):
    frames = []
    for i, p in enumerate(paths):
        d = _read_csv_capped(p, max_rows_per_file, seed=seed + i)
        if tag_col is not None:
            d[tag_col] = os.path.splitext(os.path.basename(p))[0]
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_device_shift_split(seed=42, n_holdout_devices=3, attack_sample_frac=0.3,
                             max_rows_per_file=40_000):
    """max_rows_per_file: per-source-file cap applied AT READ TIME (chunked, memory-safe). With
    7 source devices + 14 attack files this bounds the source pool to roughly
    7*max_rows_per_file (normal) + 14*attack_sample_frac*max_rows_per_file (attack) rows —
    for the default 40,000 that's on the order of 4-5*10^5 rows, comfortably within RAM and still
    far larger than anything IDSPOMDPEnv samples per episode (window_size ~256)."""
    devices = sorted(DATASETS["edge_iiot"]["normal_device_dirs"])
    rng = np.random.default_rng(seed)
    held_out = list(rng.choice(devices, size=n_holdout_devices, replace=False))
    source_devices = [d for d in devices if d not in held_out]

    normal_source = _read_many_capped(
        [os.path.join(NORMAL_DIR, d, f"{d}.csv") for d in source_devices],
        max_rows_per_file, tag_col="_device", seed=seed,
    )
    normal_shift = _read_many_capped(
        [os.path.join(NORMAL_DIR, d, f"{d}.csv") for d in held_out],
        max_rows_per_file, tag_col="_device", seed=seed + 100,
    )

    attack_files = sorted(glob.glob(os.path.join(ATTACK_DIR, "*.csv")))
    attack_df = _read_many_capped(attack_files, max_rows_per_file, tag_col="_device", seed=seed + 200)
    # sample a slice for the shift-domain evaluation, rest for source training, no overlap
    attack_shift, attack_source = train_test_split(attack_df, test_size=(1 - attack_sample_frac), random_state=seed)

    source_df = pd.concat([normal_source, attack_source], ignore_index=True).drop_duplicates().reset_index(drop=True)
    shift_df = pd.concat([normal_shift, attack_shift], ignore_index=True).drop_duplicates().reset_index(drop=True)

    vc = source_df[LABEL_COL].value_counts()
    stratify_col = source_df[LABEL_COL] if vc.min() >= 2 else None
    train_df, val_df = train_test_split(source_df, test_size=0.15, random_state=seed, stratify=stratify_col)

    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test_in_distribution": source_df.sample(frac=0.15, random_state=seed + 1).reset_index(drop=True),
        "test_shift": shift_df.reset_index(drop=True),
        "label_col": LABEL_COL,
        "id_like_cols": ID_LIKE_COLS,
        "held_out_devices": held_out,
        "source_devices": source_devices,
    }
