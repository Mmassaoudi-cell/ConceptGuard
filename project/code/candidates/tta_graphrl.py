"""
Candidate 5 (spec priority #1): TTA-GraphRL — Security-Preserving Test-Time Adaptation for
Graph-Based Intrusion Response.

Pipeline: pretrain frozen GraphSAGE encoder + linear head (supervised, binary attack/benign) ->
train DQN policy over the 5-action IDSPOMDPEnv using frozen-encoder embeddings -> evaluate
in-distribution -> (Edge-IIoTset only) evaluate under a REAL benign-distribution shift
(held-out device folders) with vs without Tent-style test-time adaptation.
"""
import sys
import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from dataset_configs import DATASETS  # noqa: E402
from data_utils import LOADERS, fit_preprocessor, transform_with_preprocessor, get_binary_labels  # noqa: E402
from graph_env import IDSPOMDPEnv, ACTIONS, build_window_graph, A_BENIGN, A_ATTACK, A_ABSTAIN, A_TELEMETRY, A_ESCALATE  # noqa: E402
from encoders import pretrain_encoder, adapt_tta  # noqa: E402
from dqn import train_dqn, QNetwork  # noqa: E402

RESULTS_DIR = r"C:\Users\MMASSAOUDI\Desktop\research work\Alienware\Innovative paper 4\project\results"
DATA_ROOT = DATASETS["edge_iiot"]["files"][0]  # unused placeholder for clarity


def prepare_features(df, label_col, id_like_cols):
    all_cols = [c for c in df.columns if c != label_col]
    non_id_cols = [c for c in all_cols if c not in id_like_cols]

    cheap_prep = fit_preprocessor(df[id_like_cols + [label_col]], label_col=label_col)
    X_cheap = transform_with_preprocessor(df, cheap_prep)
    n_cat = len(cheap_prep["categorical_cols"])
    cheap_cat_ids = X_cheap[:, X_cheap.shape[1] - n_cat:].astype(int) if n_cat > 0 else np.full((len(df), 1), -1)

    expensive_prep = fit_preprocessor(df[non_id_cols + [label_col]], label_col=label_col) if non_id_cols else None
    X_expensive = transform_with_preprocessor(df, expensive_prep) if expensive_prep else np.zeros((len(df), 0))

    return X_cheap.astype(np.float32), cheap_cat_ids, X_expensive.astype(np.float32), cheap_prep, expensive_prep


def prepare_features_apply(df, label_col, id_like_cols, cheap_prep, expensive_prep):
    """Apply already-fit (train-only) preprocessors to a new split — no re-fitting."""
    non_id_cols = [c for c in df.columns if c not in id_like_cols and c != label_col]
    X_cheap = transform_with_preprocessor(df, cheap_prep).astype(np.float32)
    n_cat = len(cheap_prep["categorical_cols"])
    cheap_cat_ids = X_cheap[:, X_cheap.shape[1] - n_cat:].astype(int) if n_cat > 0 else np.full((len(df), 1), -1)
    X_expensive = transform_with_preprocessor(df, expensive_prep).astype(np.float32) if expensive_prep else np.zeros((len(df), 0))
    return X_cheap, cheap_cat_ids, X_expensive


def evaluate_policy(qnet, encoder_model, X_cheap, X_expensive, cheap_cat_ids, y_binary,
                     n_episodes=20, window_size=256, seed=999, device="cpu", adapter=None):
    """Rolls the trained (encoder[+adapter], policy) out greedily (eps=0) over fresh windows,
    returns IDS metrics (restricted to flows where the agent actually classified) + decision metrics."""
    env = IDSPOMDPEnv(X_cheap, X_expensive, cheap_cat_ids, y_binary, window_size=window_size, seed=seed)
    all_true, all_pred = [], []
    total_reward, n_telemetry, n_escalate, n_abstain, n_classified, n_steps = 0.0, 0, 0, 0, 0, 0

    for ep in range(n_episodes):
        env.reset()
        with torch.no_grad():
            h = encoder_model.encoder(env.graph.x.to(device), env.graph.edge_index.to(device))
            if adapter is not None:
                h = adapter(h)
        obs = env._obs(h)
        done = False
        while not done:
            with torch.no_grad():
                qvals = qnet(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
                action = int(qvals.argmax(dim=1).item())
            obs, reward, done, info = env.step(action, h)
            total_reward += reward
            n_steps += 1
            if action == A_TELEMETRY:
                n_telemetry += 1
            elif action == A_ESCALATE:
                n_escalate += 1
            elif action == A_ABSTAIN:
                n_abstain += 1
            elif action in (A_BENIGN, A_ATTACK):
                n_classified += 1
                all_true.append(info["true_label"])
                all_pred.append(1 if action == A_ATTACK else 0)

    from sklearn.metrics import f1_score, balanced_accuracy_score, precision_score, recall_score
    metrics = {}
    if all_true:
        metrics["macro_f1"] = float(f1_score(all_true, all_pred, average="macro"))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(all_true, all_pred))
        metrics["attack_recall"] = float(recall_score(all_true, all_pred, pos_label=1, zero_division=0))
        metrics["attack_precision"] = float(precision_score(all_true, all_pred, pos_label=1, zero_division=0))
    metrics["cumulative_reward"] = float(total_reward)
    metrics["mean_reward_per_step"] = float(total_reward / max(n_steps, 1))
    metrics["telemetry_rate"] = n_telemetry / max(n_steps, 1)
    metrics["escalation_rate"] = n_escalate / max(n_steps, 1)
    metrics["abstention_rate"] = n_abstain / max(n_steps, 1)
    metrics["classification_rate"] = n_classified / max(n_steps, 1)
    metrics["n_steps"] = n_steps
    return metrics


def run(dataset_name, seed=42, n_pretrain_windows=200, n_dqn_episodes=300, window_size=256, device="cpu"):
    t0 = time.time()
    print(f"=== TTA-GraphRL on {dataset_name} (seed={seed}) ===", flush=True)
    data = LOADERS[dataset_name](seed=seed)
    label_col = data["label_col"]
    id_like_cols = [c for c in data["id_like_cols"] if c in data["train"].columns]
    train_df, val_df, test_df = data["train"], data["val"], data["test"]

    y_train_bin = get_binary_labels(train_df, dataset_name)
    y_val_bin = get_binary_labels(val_df, dataset_name)
    y_test_bin = get_binary_labels(test_df, dataset_name)

    X_cheap_tr, cat_ids_tr, X_exp_tr, cheap_prep, exp_prep = prepare_features(train_df, label_col, id_like_cols)
    X_cheap_val, cat_ids_val, X_exp_val = prepare_features_apply(val_df, label_col, id_like_cols, cheap_prep, exp_prep)
    X_cheap_test, cat_ids_test, X_exp_test = prepare_features_apply(test_df, label_col, id_like_cols, cheap_prep, exp_prep)

    in_dim = X_cheap_tr.shape[1]
    print(f"cheap_dim={in_dim} expensive_dim={X_exp_tr.shape[1]} n_train={len(train_df)}", flush=True)

    model, pretrain_losses = pretrain_encoder(
        X_cheap_tr, cat_ids_tr, y_train_bin, in_dim=in_dim, n_classes=2,
        window_size=window_size, n_windows=n_pretrain_windows, seed=seed, device=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def embed_fn(env):
        h = model.encoder(env.graph.x.to(device), env.graph.edge_index.to(device))
        return h

    qnet, ep_rewards = train_dqn(
        X_cheap_tr, X_exp_tr, cat_ids_tr, y_train_bin, embed_fn,
        n_episodes=n_dqn_episodes, window_size=window_size, seed=seed, device=device, embed_dim=64,
    )

    val_metrics = evaluate_policy(qnet, model, X_cheap_val, X_exp_val, cat_ids_val, y_val_bin,
                                   window_size=window_size, seed=seed + 1, device=device)
    test_metrics = evaluate_policy(qnet, model, X_cheap_test, X_exp_test, cat_ids_test, y_test_bin,
                                    window_size=window_size, seed=seed + 2, device=device)

    results = {
        "candidate": "TTA-GraphRL", "dataset": dataset_name, "seed": seed,
        "n_pretrain_windows": n_pretrain_windows, "n_dqn_episodes": n_dqn_episodes,
        "mean_dqn_episode_reward_last50": float(np.mean(ep_rewards[-50:])),
        "val_metrics": val_metrics, "test_metrics": test_metrics,
        "total_seconds": round(time.time() - t0, 2),
    }
    out_path = os.path.join(RESULTS_DIR, f"tta_graphrl_{dataset_name}_seed{seed}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}", flush=True)
    print(json.dumps({"val_metrics": val_metrics, "test_metrics": test_metrics}, indent=2), flush=True)
    return results, model, qnet, cheap_prep, exp_prep, id_like_cols


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="rt_iot2022")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_pretrain_windows", type=int, default=200)
    ap.add_argument("--n_dqn_episodes", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run(args.dataset, seed=args.seed, n_pretrain_windows=args.n_pretrain_windows,
        n_dqn_episodes=args.n_dqn_episodes, device=args.device)
