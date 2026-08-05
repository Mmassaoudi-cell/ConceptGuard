"""
TTA-GraphRL candidate-specific experiment (Experiment 11 in the master spec): evaluate the
trained policy under a REAL benign-distribution shift (Edge-IIoTset held-out device folders),
comparing performance WITH vs WITHOUT Tent-style test-time adaptation. This is the "essential
proof" the spec requires for this candidate: false-positive reduction under benign shift without
increasing false negatives under attack.
"""
import sys
import os
import json
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from device_shift_loader import load_device_shift_split  # noqa: E402
from data_utils import fit_preprocessor, transform_with_preprocessor  # noqa: E402
from encoders import pretrain_encoder, adapt_tta  # noqa: E402
from dqn import train_dqn  # noqa: E402
from tta_graphrl import prepare_features, prepare_features_apply, evaluate_policy  # noqa: E402

RESULTS_DIR = r"C:\Users\MMASSAOUDI\Desktop\research work\Alienware\Innovative paper 4\project\results"
BENIGN_LABEL = "Normal"


def get_binary_labels_local(df, label_col):
    return (df[label_col].astype(str) != BENIGN_LABEL).astype(int).values


def run(seed=42, n_pretrain_windows=200, n_dqn_episodes=300, window_size=256,
        max_rows_per_file=40_000, n_adapt_windows=30, device="cpu"):
    t0 = time.time()
    print(f"=== TTA-GraphRL device-shift experiment (seed={seed}) ===", flush=True)
    split = load_device_shift_split(seed=seed, max_rows_per_file=max_rows_per_file)
    label_col = split["label_col"]
    id_like_cols = split["id_like_cols"]
    print(f"held_out_devices={split['held_out_devices']} source_devices={split['source_devices']}", flush=True)

    train_df, val_df = split["train"], split["val"]
    test_id_df, test_shift_df = split["test_in_distribution"], split["test_shift"]

    y_train = get_binary_labels_local(train_df, label_col)
    y_val = get_binary_labels_local(val_df, label_col)
    y_test_id = get_binary_labels_local(test_id_df, label_col)
    y_test_shift = get_binary_labels_local(test_shift_df, label_col)

    X_cheap_tr, cat_ids_tr, X_exp_tr, cheap_prep, exp_prep = prepare_features(train_df, label_col, id_like_cols)
    X_cheap_val, cat_ids_val, X_exp_val = prepare_features_apply(val_df, label_col, id_like_cols, cheap_prep, exp_prep)
    X_cheap_id, cat_ids_id, X_exp_id = prepare_features_apply(test_id_df, label_col, id_like_cols, cheap_prep, exp_prep)
    X_cheap_sh, cat_ids_sh, X_exp_sh = prepare_features_apply(test_shift_df, label_col, id_like_cols, cheap_prep, exp_prep)

    in_dim = X_cheap_tr.shape[1]
    print(f"cheap_dim={in_dim} expensive_dim={X_exp_tr.shape[1]} n_train={len(train_df)} n_shift={len(test_shift_df)}", flush=True)

    model, _ = pretrain_encoder(X_cheap_tr, cat_ids_tr, y_train, in_dim=in_dim, n_classes=2,
                                 window_size=window_size, n_windows=n_pretrain_windows, seed=seed, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def embed_fn(env):
        return model.encoder(env.graph.x.to(device), env.graph.edge_index.to(device))

    qnet, ep_rewards = train_dqn(X_cheap_tr, X_exp_tr, cat_ids_tr, y_train, embed_fn,
                                  n_episodes=n_dqn_episodes, window_size=window_size, seed=seed,
                                  device=device, embed_dim=64)

    val_metrics = evaluate_policy(qnet, model, X_cheap_val, X_exp_val, cat_ids_val, y_val,
                                   window_size=window_size, seed=seed + 1, device=device)
    id_metrics = evaluate_policy(qnet, model, X_cheap_id, X_exp_id, cat_ids_id, y_test_id,
                                  window_size=window_size, seed=seed + 2, device=device)

    shift_no_adapt = evaluate_policy(qnet, model, X_cheap_sh, X_exp_sh, cat_ids_sh, y_test_shift,
                                      window_size=window_size, seed=seed + 3, device=device)

    adapter = adapt_tta(model, X_cheap_sh, cat_ids_sh, window_size=window_size,
                         n_adapt_windows=n_adapt_windows, seed=seed, device=device)
    shift_with_adapt = evaluate_policy(qnet, model, X_cheap_sh, X_exp_sh, cat_ids_sh, y_test_shift,
                                        window_size=window_size, seed=seed + 3, device=device, adapter=adapter)

    results = {
        "candidate": "TTA-GraphRL", "experiment": "device_holdout_shift", "seed": seed,
        "held_out_devices": split["held_out_devices"], "source_devices": split["source_devices"],
        "n_train": len(train_df), "n_shift_test": len(test_shift_df),
        "val_metrics": val_metrics, "in_distribution_test_metrics": id_metrics,
        "shift_test_no_adaptation": shift_no_adapt, "shift_test_with_tta_adaptation": shift_with_adapt,
        "total_seconds": round(time.time() - t0, 2),
    }
    out_path = os.path.join(RESULTS_DIR, f"tta_graphrl_deviceshift_seed{seed}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}", flush=True)
    print(json.dumps({
        "in_distribution_test_metrics": id_metrics,
        "shift_test_no_adaptation": shift_no_adapt,
        "shift_test_with_tta_adaptation": shift_with_adapt,
    }, indent=2), flush=True)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_pretrain_windows", type=int, default=200)
    ap.add_argument("--n_dqn_episodes", type=int, default=300)
    ap.add_argument("--max_rows_per_file", type=int, default=40_000)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run(seed=args.seed, n_pretrain_windows=args.n_pretrain_windows,
        n_dqn_episodes=args.n_dqn_episodes, max_rows_per_file=args.max_rows_per_file, device=args.device)
