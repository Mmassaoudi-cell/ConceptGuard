"""
Candidate 9 (spec priority #4): TailGuard — Distributional Risk-Sensitive Graph RL for Critical
Intrusion Detection.

Uses Edge-IIoTset's two genuinely rare attack classes (MITM: 1,214/157,800 rows; Fingerprinting:
1,001/157,800 — confirmed in dataset_screening_report.md, not downsampled or synthetic) as the
high-severity classes. False negatives on these get a 5x reward penalty (severity_weight), and the
policy is trained with QR-DQN using CVaR-based (not mean-based) action selection, so it is
explicitly optimizing to avoid catastrophic misses on rare classes rather than average reward.
"""
import sys
import os
import json
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from data_utils import LOADERS, get_binary_labels  # noqa: E402
from graph_env import IDSPOMDPEnv, A_BENIGN, A_ATTACK, A_ABSTAIN, A_TELEMETRY, A_ESCALATE  # noqa: E402
from encoders import pretrain_encoder  # noqa: E402
from qr_dqn import train_qrdqn, QRNetwork, cvar_values  # noqa: E402
from tta_graphrl import prepare_features, prepare_features_apply  # noqa: E402

RESULTS_DIR = r"C:\Users\MMASSAOUDI\Desktop\research work\Alienware\Innovative paper 4\project\results"
SEVERE_CLASSES = ["MITM", "Fingerprinting"]
SEVERITY_MULTIPLIER = 5.0


def severity_weights(df, label_col):
    return np.where(df[label_col].astype(str).isin(SEVERE_CLASSES), SEVERITY_MULTIPLIER, 1.0)


def evaluate_tail(qnet, encoder_model, X_cheap, X_expensive, cheap_cat_ids, y_binary, severe_mask,
                   n_episodes=30, window_size=256, seed=999, device="cpu", cvar_alpha=0.25):
    env = IDSPOMDPEnv(X_cheap, X_expensive, cheap_cat_ids, y_binary, window_size=window_size, seed=seed)
    episode_returns = []
    all_true, all_pred = [], []
    severe_total, severe_missed = 0, 0
    n_telemetry, n_escalate, n_abstain, n_classified, n_steps = 0, 0, 0, 0, 0

    for ep in range(n_episodes):
        env.reset()
        with torch.no_grad():
            h = encoder_model.encoder(env.graph.x.to(device), env.graph.edge_index.to(device))
        obs = env._obs(h)
        ep_reward = 0.0
        done = False
        while not done:
            with torch.no_grad():
                q = qnet(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
                cvar = cvar_values(q, alpha=cvar_alpha)
                action = int(cvar.argmax(dim=1).item())
            flow_idx = env.idx[env.pos]
            is_severe = bool(severe_mask[flow_idx])
            obs, reward, done, info = env.step(action, h)
            ep_reward += reward
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
                if is_severe:
                    severe_total += 1
                    if action == A_BENIGN:  # missed a severe/rare attack
                        severe_missed += 1
        episode_returns.append(ep_reward)

    from sklearn.metrics import f1_score, balanced_accuracy_score, recall_score, precision_score
    episode_returns = np.array(episode_returns)
    metrics = {}
    if all_true:
        metrics["macro_f1"] = float(f1_score(all_true, all_pred, average="macro"))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(all_true, all_pred))
        metrics["attack_recall"] = float(recall_score(all_true, all_pred, pos_label=1, zero_division=0))
        metrics["attack_precision"] = float(precision_score(all_true, all_pred, pos_label=1, zero_division=0))
    metrics["expected_return"] = float(episode_returns.mean())
    metrics["worst_decile_return"] = float(np.percentile(episode_returns, 10))
    k = max(1, int(round(0.1 * len(episode_returns))))
    metrics["cvar_10"] = float(np.sort(episode_returns)[:k].mean())
    metrics["policy_variance"] = float(episode_returns.var())
    metrics["severe_class_total_seen"] = severe_total
    metrics["severe_false_negative_rate"] = severe_missed / max(severe_total, 1)
    metrics["telemetry_rate"] = n_telemetry / max(n_steps, 1)
    metrics["escalation_rate"] = n_escalate / max(n_steps, 1)
    metrics["abstention_rate"] = n_abstain / max(n_steps, 1)
    metrics["classification_rate"] = n_classified / max(n_steps, 1)
    return metrics


def run(dataset_name="edge_iiot", seed=42, n_pretrain_windows=200, n_dqn_episodes=300, window_size=256, device="cpu"):
    t0 = time.time()
    print(f"=== TailGuard on {dataset_name} (seed={seed}) ===", flush=True)
    data = LOADERS[dataset_name](seed=seed)
    label_col = data["label_col"]
    id_like_cols = [c for c in data["id_like_cols"] if c in data["train"].columns]
    train_df, val_df, test_df = data["train"], data["val"], data["test"]

    y_train_bin = get_binary_labels(train_df, dataset_name)
    y_val_bin = get_binary_labels(val_df, dataset_name)
    y_test_bin = get_binary_labels(test_df, dataset_name)
    sev_train = severity_weights(train_df, label_col)
    sev_val = severity_weights(val_df, label_col)
    sev_test = severity_weights(test_df, label_col)
    print(f"severe rows: train={int((sev_train>1).sum())} val={int((sev_val>1).sum())} test={int((sev_test>1).sum())}", flush=True)

    X_cheap_tr, cat_ids_tr, X_exp_tr, cheap_prep, exp_prep = prepare_features(train_df, label_col, id_like_cols)
    X_cheap_val, cat_ids_val, X_exp_val = prepare_features_apply(val_df, label_col, id_like_cols, cheap_prep, exp_prep)
    X_cheap_test, cat_ids_test, X_exp_test = prepare_features_apply(test_df, label_col, id_like_cols, cheap_prep, exp_prep)

    in_dim = X_cheap_tr.shape[1]
    model, _ = pretrain_encoder(X_cheap_tr, cat_ids_tr, y_train_bin, in_dim=in_dim, n_classes=2,
                                 window_size=window_size, n_windows=n_pretrain_windows, seed=seed, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def embed_fn(env):
        return model.encoder(env.graph.x.to(device), env.graph.edge_index.to(device))

    qnet, ep_rewards = train_qrdqn(
        X_cheap_tr, X_exp_tr, cat_ids_tr, y_train_bin, embed_fn,
        n_episodes=n_dqn_episodes, window_size=window_size, seed=seed, device=device,
        embed_dim=64, severity_weight=sev_train,
    )

    val_metrics = evaluate_tail(qnet, model, X_cheap_val, X_exp_val, cat_ids_val, y_val_bin, sev_val,
                                 window_size=window_size, seed=seed + 1, device=device)
    test_metrics = evaluate_tail(qnet, model, X_cheap_test, X_exp_test, cat_ids_test, y_test_bin, sev_test,
                                  window_size=window_size, seed=seed + 2, device=device)

    results = {
        "candidate": "TailGuard", "dataset": dataset_name, "seed": seed,
        "severe_classes": SEVERE_CLASSES, "severity_multiplier": SEVERITY_MULTIPLIER,
        "mean_qrdqn_episode_reward_last50": float(np.mean(ep_rewards[-50:])),
        "val_metrics": val_metrics, "test_metrics": test_metrics,
        "total_seconds": round(time.time() - t0, 2),
    }
    out_path = os.path.join(RESULTS_DIR, f"tailguard_{dataset_name}_seed{seed}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}", flush=True)
    print(json.dumps({"val_metrics": val_metrics, "test_metrics": test_metrics}, indent=2), flush=True)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="edge_iiot")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_pretrain_windows", type=int, default=200)
    ap.add_argument("--n_dqn_episodes", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run(args.dataset, seed=args.seed, n_pretrain_windows=args.n_pretrain_windows,
        n_dqn_episodes=args.n_dqn_episodes, device=args.device)
