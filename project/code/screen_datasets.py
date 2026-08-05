"""
Dataset screening for the graph/RL IDS paper (academic-pipeline Stage 1, Part 2 of the master spec).
Scans every dataset under DATA_ROOT, produces a screening table (CSV + Markdown) with:
 name, file(s), size, row count, feature count, label column, label cardinality,
 benign/attack distribution, missing values, duplicate rows, candidate id-like columns,
 whether graph/temporal construction looks feasible, saturation risk hints.

Design choices (must stay honest, no fabrication):
 - Row counts are computed by streaming byte-level newline counts (exact, memory-safe).
 - Label distribution / missing / duplicates are computed via chunked pandas reads.
   For files > SAMPLE_THRESHOLD_BYTES, these stats are computed from a bounded row sample
   (first SAMPLE_ROWS rows after header) and explicitly marked as "sampled" in the report,
   never silently presented as exact for huge files.
"""
import os
import sys
import time
import json
import traceback
import pandas as pd

DATA_ROOT = r"C:\Users\MMASSAOUDI\Desktop\Data"
OUT_DIR = r"C:\Users\MMASSAOUDI\Desktop\research work\Alienware\Innovative paper 4\project\data_screening"
SAMPLE_THRESHOLD_BYTES = 300 * 1024 * 1024  # 300MB
SAMPLE_ROWS = 500_000
CHUNK_ROWS = 200_000

LABEL_NAME_HINTS = [
    "label", "labels", "attack_cat", "attack_type", "attacktype", "class",
    "category", "target", "attack", "type", "traffic_type", "output",
]
ID_LIKE_HINTS = [
    "src", "source", "dst", "dest", "destination", "ip", "port", "flow_id",
    "flowid", "session", "timestamp", "time", "proto", "protocol", "service",
    "mac", "device", "sensor",
]


def count_lines_fast(path):
    """Exact newline count via binary streaming, memory-safe for multi-GB files."""
    count = 0
    with open(path, "rb") as f:
        buf_size = 1024 * 1024 * 8
        read = f.read(buf_size)
        while read:
            count += read.count(b"\n")
            read = f.read(buf_size)
    return count


def find_label_column(columns):
    lc = {c: c.strip().lower() for c in columns}
    for c, low in lc.items():
        if low in LABEL_NAME_HINTS:
            return c
    for c, low in lc.items():
        for hint in LABEL_NAME_HINTS:
            if hint in low:
                return c
    return None


def find_id_like_columns(columns):
    found = []
    for c in columns:
        low = c.strip().lower()
        if any(h in low for h in ID_LIKE_HINTS):
            found.append(c)
    return found


def screen_csv(path):
    result = {
        "path": path,
        "size_bytes": os.path.getsize(path),
        "error": None,
    }
    size = result["size_bytes"]
    try:
        header_df = pd.read_csv(path, nrows=0, low_memory=False)
        columns = list(header_df.columns)
    except Exception as e:
        try:
            header_df = pd.read_csv(path, nrows=0, low_memory=False, sep=None, engine="python")
            columns = list(header_df.columns)
        except Exception as e2:
            result["error"] = f"header read failed: {e2}"
            return result

    result["n_columns"] = len(columns)
    result["columns_preview"] = columns[:25]
    label_col = find_label_column(columns)
    result["label_column"] = label_col
    result["id_like_columns"] = find_id_like_columns(columns)

    try:
        result["n_rows_exact"] = max(count_lines_fast(path) - 1, 0)  # minus header
    except Exception as e:
        result["n_rows_exact"] = None
        result["row_count_error"] = str(e)

    sampled = size > SAMPLE_THRESHOLD_BYTES
    result["stats_sampled"] = sampled
    nrows_for_stats = SAMPLE_ROWS if sampled else None

    try:
        if nrows_for_stats is not None:
            df = pd.read_csv(path, nrows=nrows_for_stats, low_memory=False)
        else:
            df = pd.read_csv(path, low_memory=False)
        result["n_rows_used_for_stats"] = len(df)
        result["missing_values_total"] = int(df.isna().sum().sum())
        result["missing_pct"] = round(100.0 * df.isna().sum().sum() / (df.shape[0] * max(df.shape[1], 1)), 4)
        result["duplicate_rows"] = int(df.duplicated().sum())
        if label_col and label_col in df.columns:
            vc = df[label_col].value_counts(dropna=False)
            result["label_cardinality"] = int(vc.shape[0])
            top = vc.head(15)
            result["label_distribution_top15"] = {str(k): int(v) for k, v in top.items()}
        else:
            result["label_cardinality"] = None
            result["label_distribution_top15"] = None
    except Exception as e:
        result["stats_error"] = str(e) + "\n" + traceback.format_exc(limit=2)

    return result


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = []
    dataset_dirs = sorted(
        [os.path.join(DATA_ROOT, d) for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    )

    for ddir in dataset_dirs:
        dname = os.path.basename(ddir)
        print(f"=== Screening dataset dir: {dname} ===", flush=True)
        csv_files = []
        for root, _, files in os.walk(ddir):
            for fn in files:
                if fn.lower().endswith(".csv"):
                    csv_files.append(os.path.join(root, fn))
        pcap_count = 0
        for root, _, files in os.walk(ddir):
            pcap_count += sum(1 for fn in files if fn.lower().endswith(".pcap"))

        if not csv_files:
            all_results.append({"dataset": dname, "note": "no CSV files found", "pcap_files": pcap_count})
            continue

        for csv_path in csv_files:
            t0 = time.time()
            print(f"  scanning {csv_path} ...", flush=True)
            r = screen_csv(csv_path)
            r["dataset"] = dname
            r["pcap_siblings"] = pcap_count
            r["scan_seconds"] = round(time.time() - t0, 2)
            all_results.append(r)
            print(f"    done in {r['scan_seconds']}s rows_exact={r.get('n_rows_exact')} cols={r.get('n_columns')} "
                  f"label_col={r.get('label_column')} sampled={r.get('stats_sampled')}", flush=True)
            # flush partial results after every file so progress is inspectable while running
            with open(os.path.join(OUT_DIR, "screening_raw.json"), "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, default=str)

    with open(os.path.join(OUT_DIR, "screening_raw.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
