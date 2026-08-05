"""
Leakage-safe loading, splitting, and preprocessing shared by every candidate/baseline.

Rules enforced here (per master spec Part 9):
 - split BEFORE any fitting (scalers/encoders/imputers fit on train only)
 - dedup exact duplicates BEFORE splitting, never after
 - official splits (UNSW-NB15, CICIoT23) are respected as-is, not re-shuffled
 - every split is written to a manifest file (row-count + label distribution + content hash)
   so re-running with the same seed is verifiably reproducible
"""
import os
import glob
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from dataset_configs import DATASETS

MANIFEST_DIR = r"C:\Users\MMASSAOUDI\Desktop\research work\Alienware\Innovative paper 4\project\results\split_manifests"
os.makedirs(MANIFEST_DIR, exist_ok=True)


def _hash_df(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return h.hexdigest()[:16]


def _write_manifest(dataset_name, seed, splits: dict, label_col):
    manifest = {"dataset": dataset_name, "seed": seed, "splits": {}}
    for split_name, df in splits.items():
        info = {"n_rows": int(len(df)), "content_hash": _hash_df(df)}
        if label_col in df.columns:
            info["label_distribution"] = {str(k): int(v) for k, v in df[label_col].value_counts().items()}
        manifest["splits"][split_name] = info
    path = os.path.join(MANIFEST_DIR, f"{dataset_name}_seed{seed}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path


def _dedup_then_split(df: pd.DataFrame, label_col: str, seed: int, val_frac=0.15, test_frac=0.15):
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    # stratify only if every class has >= 2 members (required by sklearn)
    vc = df[label_col].value_counts()
    stratify_col = df[label_col] if (vc.min() >= 2) else None
    train_df, temp_df = train_test_split(
        df, test_size=(val_frac + test_frac), random_state=seed, stratify=stratify_col
    )
    vc2 = temp_df[label_col].value_counts()
    stratify_col2 = temp_df[label_col] if (vc2.min() >= 2) else None
    rel_test = test_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        temp_df, test_size=rel_test, random_state=seed, stratify=stratify_col2
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True), removed


def load_rt_iot2022(seed=42):
    cfg = DATASETS["rt_iot2022"]
    df = pd.read_csv(cfg["files"][0], low_memory=False)
    for c in cfg["drop_cols"]:
        if c in df.columns:
            df = df.drop(columns=[c])
    train_df, val_df, test_df, removed = _dedup_then_split(df, cfg["label_col"], seed)
    manifest_path = _write_manifest("rt_iot2022", seed, {"train": train_df, "val": val_df, "test": test_df}, cfg["label_col"])
    return {"train": train_df, "val": val_df, "test": test_df, "label_col": cfg["label_col"],
            "id_like_cols": cfg["id_like_cols"], "duplicates_removed": removed, "manifest": manifest_path}


def load_unsw_nb15(seed=42):
    cfg = DATASETS["unsw_nb15"]
    train_full = pd.read_csv(cfg["files"]["train"], low_memory=False)
    test_df = pd.read_csv(cfg["files"]["test"], low_memory=False)
    for c in cfg["drop_cols"]:
        for d in (train_full, test_df):
            if c in d.columns:
                d.drop(columns=[c], inplace=True)
    # official split has no validation partition -> carve one out of train only (never touches test)
    vc = train_full[cfg["label_col"]].value_counts()
    stratify_col = train_full[cfg["label_col"]] if vc.min() >= 2 else None
    train_df, val_df = train_test_split(train_full, test_size=0.15, random_state=seed, stratify=stratify_col)
    train_df, val_df, test_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
    manifest_path = _write_manifest("unsw_nb15", seed, {"train": train_df, "val": val_df, "test": test_df}, cfg["label_col"])
    return {"train": train_df, "val": val_df, "test": test_df, "label_col": cfg["label_col"],
            "binary_label_col": cfg.get("binary_label_col"),
            "id_like_cols": cfg["id_like_cols"], "duplicates_removed": None, "manifest": manifest_path}


def load_edge_iiot(seed=42):
    cfg = DATASETS["edge_iiot"]
    df = pd.read_csv(cfg["files"][0], low_memory=False)
    train_df, val_df, test_df, removed = _dedup_then_split(df, cfg["label_col"], seed)
    manifest_path = _write_manifest("edge_iiot", seed, {"train": train_df, "val": val_df, "test": test_df}, cfg["label_col"])
    return {"train": train_df, "val": val_df, "test": test_df, "label_col": cfg["label_col"],
            "id_like_cols": cfg["id_like_cols"], "duplicates_removed": removed, "manifest": manifest_path}


def _read_csv_random_sample(path, row_cap, chunksize=100_000, seed=0):
    """Random (not positional-prefix) sample of up to row_cap rows, memory-bounded via chunked
    reservoir-style sampling. Bot_IoT's per-file captures are chronological, and attack bursts vs.
    benign periods are unevenly distributed across a file -- pd.read_csv(nrows=row_cap) would
    silently bias toward whatever traffic happened first, which materially changed the observed
    class balance between two otherwise-identical runs that only differed in row_cap_per_file."""
    if row_cap is None:
        return pd.read_csv(path, low_memory=False)
    rng = np.random.default_rng(seed)
    kept, total = [], 0
    for chunk in pd.read_csv(path, low_memory=False, chunksize=chunksize):
        if total >= row_cap:
            break
        remaining = row_cap - total
        if len(chunk) <= remaining:
            kept.append(chunk)
            total += len(chunk)
        else:
            idx = rng.choice(len(chunk), size=remaining, replace=False)
            kept.append(chunk.iloc[idx])
            total += remaining
            break
    return pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()


def load_bot_iot(seed=42, max_files=None, row_cap_per_file=None):
    """Bot_IoT is 75 files / 73.4M rows. For dev/screening runs, cap files and/or rows per file;
    the final headline-run should raise/remove both caps and note wall-clock cost explicitly."""
    cfg = DATASETS["bot_iot"]
    files = sorted(glob.glob(cfg["files_glob"]))
    files = [f for f in files if not f.endswith("data_names.csv")]
    if max_files is not None:
        files = files[:max_files]
    frames = []
    for i, f in enumerate(files):
        d = _read_csv_random_sample(f, row_cap_per_file, seed=seed + i)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    for c in cfg["drop_cols"]:
        if c in df.columns:
            df = df.drop(columns=[c])
    train_df, val_df, test_df, removed = _dedup_then_split(df, cfg["label_col"], seed)
    manifest_path = _write_manifest("bot_iot", seed, {"train": train_df, "val": val_df, "test": test_df}, cfg["label_col"])
    return {"train": train_df, "val": val_df, "test": test_df, "label_col": cfg["label_col"],
            "id_like_cols": cfg["id_like_cols"], "duplicates_removed": removed, "manifest": manifest_path,
            "n_files_used": len(files)}


def load_cicIot23(seed=42):
    cfg = DATASETS["cicIot23"]
    train_df = pd.read_csv(cfg["files"]["train"], low_memory=False)
    val_df = pd.read_csv(cfg["files"]["validation"], low_memory=False)
    test_df = pd.read_csv(cfg["files"]["test"], low_memory=False)
    manifest_path = _write_manifest("cicIot23", seed, {"train": train_df, "val": val_df, "test": test_df}, cfg["label_col"])
    return {"train": train_df, "val": val_df, "test": test_df, "label_col": cfg["label_col"],
            "id_like_cols": cfg["id_like_cols"], "duplicates_removed": None, "manifest": manifest_path}


LOADERS = {
    "rt_iot2022": load_rt_iot2022,
    "unsw_nb15": load_unsw_nb15,
    "edge_iiot": load_edge_iiot,
    "bot_iot": load_bot_iot,
    "cicIot23": load_cicIot23,
}


def get_binary_labels(df: pd.DataFrame, dataset_name: str) -> np.ndarray:
    """Returns a 0/1 numpy array, 1 = attack, using each dataset's explicit benign definition
    (never inferred by 'smallest class = benign' or similar guesses)."""
    cfg = DATASETS[dataset_name]
    if "benign_values" in cfg:
        benign_set = set(cfg["benign_values"])
        return (~df[cfg["label_col"]].astype(str).isin(benign_set)).astype(int).values
    if "benign_via_binary_label" in cfg:
        bcol = cfg.get("binary_label_col", cfg["label_col"])
        benign_code = cfg["benign_via_binary_label"]
        return (df[bcol].astype(int) != benign_code).astype(int).values
    raise ValueError(f"No benign definition configured for dataset {dataset_name}")


def fit_preprocessor(train_df: pd.DataFrame, label_col: str, exclude_cols=None):
    """Fit-on-train-only: numeric -> StandardScaler, categorical -> OrdinalEncoder(unknown->-1).
    Returns (numeric_cols, categorical_cols, scaler, encoders dict)."""
    exclude_cols = set(exclude_cols or []) | {label_col}
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_df[c])]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    scaler = StandardScaler()
    if numeric_cols:
        scaler.fit(train_df[numeric_cols].fillna(0.0))

    encoders = {}
    for c in categorical_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(train_df[[c]].astype(str).fillna("__MISSING__"))
        encoders[c] = enc

    return {"numeric_cols": numeric_cols, "categorical_cols": categorical_cols,
            "scaler": scaler, "encoders": encoders}


def transform_with_preprocessor(df: pd.DataFrame, prep: dict) -> np.ndarray:
    parts = []
    if prep["numeric_cols"]:
        num = df[prep["numeric_cols"]].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        parts.append(prep["scaler"].transform(num))
    for c in prep["categorical_cols"]:
        enc = prep["encoders"][c]
        col = df[[c]].astype(str).fillna("__MISSING__")
        parts.append(enc.transform(col))
    if not parts:
        return np.zeros((len(df), 0))
    return np.concatenate(parts, axis=1)
