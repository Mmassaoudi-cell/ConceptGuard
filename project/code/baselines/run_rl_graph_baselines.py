"""
Phase 5: graph baselines (GCN/GraphSAGE) and RL baselines (random policy, vanilla-feature DQN)
on UNSW-NB15 binary attack/benign, for direct comparison against ConceptGuard-RL (the selected
final method) on identical data/splits/action-space.
"""
import sys
import os
import json
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "candidates"))
from data_utils import LOADERS, fit_preprocessor, transform_with_preprocessor, get_binary_labels  # noqa: E402
from graph_env import (IDSPOMDPEnv, GraphEncoder, GCNEncoder, GATEncoder, build_window_graph,  # noqa: E402
                        A_BENIGN, A_ATTACK, A_ABSTAIN, A_TELEMETRY, A_ESCALATE, ACTIONS)
from dqn import train_dqn, QNetwork  # noqa: E402
from encoders import pretrain_encoder  # noqa: E402

RESULTS_DIR = r"C:\Users\MMASSAOUDI\Desktop\research work\Alienware\Innovative paper 4\project\results"
CHEAP_COLS = ["proto", "service"]


def prepare(dataset_name="unsw_nb15", seed=42):
    data = LOADERS[dataset_name](seed=seed)
    label_col = data["binary_label_col"]
    train_df, val_df, test_df = data["train"], data["val"], data["test"]
    non_cheap = [c for c in train_df.columns if c not in CHEAP_COLS and c not in
                 (data["label_col"], label_col)]

    cheap_prep = fit_preprocessor(train_df[CHEAP_COLS + [label_col]], label_col=label_col)
    exp_prep = fit_preprocessor(train_df[non_cheap + [label_col]], label_col=label_col)

    out = {}
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        X_cheap = transform_with_preprocessor(df, cheap_prep).astype(np.float32)
        out[split_name] = {
            "cheap": X_cheap,
            "cat_ids": X_cheap.astype(int),
            "expensive": transform_with_preprocessor(df, exp_prep).astype(np.float32),
            "y_bin": get_binary_labels(df, dataset_name),
        }
    return out


@torch.no_grad()
def eval_graph_classifier(model, head, pack, window_size=256, n_windows=20, seed=999, device="cpu"):
    rng = np.random.default_rng(seed)
    n = len(pack["y_bin"])
    all_true, all_pred = [], []
    for w in range(n_windows):
        idx = rng.choice(n, size=min(window_size, n), replace=False)
        edge_index = build_window_graph(pack["cat_ids"][idx]).to(device)
        x = torch.tensor(pack["cheap"][idx], dtype=torch.float32, device=device)
        h = model(x, edge_index)
        pred = head(h).argmax(dim=1).cpu().numpy()
        all_true.extend(pack["y_bin"][idx].tolist())
        all_pred.extend(pred.tolist())
    from sklearn.metrics import f1_score, balanced_accuracy_score
    return {"macro_f1": float(f1_score(all_true, all_pred, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(all_true, all_pred))}


def run_graph_baselines(packs, seed, device, window_size=256, n_windows=300):
    results = {}
    for arch_name, encoder_cls in [("GCN", GCNEncoder), ("GraphSAGE", GraphEncoder), ("GAT", GATEncoder)]:
        print(f"training {arch_name}...", flush=True)
        model, losses = pretrain_encoder(
            packs["train"]["cheap"], packs["train"]["cat_ids"], packs["train"]["y_bin"],
            in_dim=packs["train"]["cheap"].shape[1], n_classes=2,
            window_size=window_size, n_windows=n_windows, seed=seed, device=device,
            encoder_cls=encoder_cls,
        )
        m = eval_graph_classifier(model.encoder, model.head, packs["test"], window_size=window_size,
                                   seed=seed + 1, device=device)
        print(f"  {arch_name}: {m}", flush=True)
        results[arch_name] = m
    return results


def run_random_policy(packs, seed, window_size=256, n_episodes=20):
    env = IDSPOMDPEnv(packs["test"]["cheap"], packs["test"]["expensive"], packs["test"]["cat_ids"],
                       packs["test"]["y_bin"], window_size=window_size, seed=seed)
    rng = np.random.default_rng(seed)
    all_true, all_pred = [], []
    total_reward, n_steps = 0.0, 0
    for ep in range(n_episodes):
        env.reset()
        done = False
        while not done:
            action = rng.integers(0, len(ACTIONS))
            obs, reward, done, info = env.step(action)
            total_reward += reward
            n_steps += 1
            if action in (A_BENIGN, A_ATTACK):
                all_true.append(info["true_label"])
                all_pred.append(1 if action == A_ATTACK else 0)
    from sklearn.metrics import f1_score, balanced_accuracy_score
    m = {"macro_f1": float(f1_score(all_true, all_pred, average="macro")) if all_true else None,
         "balanced_accuracy": float(balanced_accuracy_score(all_true, all_pred)) if all_true else None,
         "cumulative_reward": float(total_reward), "n_steps": n_steps}
    return m


def run_vanilla_dqn(packs, seed, device, window_size=256, n_episodes=300):
    """No graph, no concepts -- embed is a constant zero vector, so the policy sees only the raw
    cheap+expensive features, same as every other RL candidate's env, isolating the value added by
    ConceptGuard-RL's concept-bottleneck routing specifically."""
    def embed_fn(env):
        return torch.zeros(len(env.idx), 1, device=device)

    qnet, ep_rewards = train_dqn(
        packs["train"]["cheap"], packs["train"]["expensive"], packs["train"]["cat_ids"],
        packs["train"]["y_bin"], embed_fn, n_episodes=n_episodes, window_size=window_size,
        seed=seed, device=device, embed_dim=1,
    )

    env = IDSPOMDPEnv(packs["test"]["cheap"], packs["test"]["expensive"], packs["test"]["cat_ids"],
                       packs["test"]["y_bin"], window_size=window_size, seed=seed + 1, embed_dim=1)
    all_true, all_pred = [], []
    total_reward, n_steps = 0.0, 0
    for ep in range(20):
        env.reset()
        ge = torch.zeros(len(env.idx), 1, device=device)
        obs = env._obs(ge)
        done = False
        while not done:
            with torch.no_grad():
                qvals = qnet(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
                action = int(qvals.argmax(dim=1).item())
            obs, reward, done, info = env.step(action, ge)
            total_reward += reward
            n_steps += 1
            if action in (A_BENIGN, A_ATTACK):
                all_true.append(info["true_label"])
                all_pred.append(1 if action == A_ATTACK else 0)
    from sklearn.metrics import f1_score, balanced_accuracy_score
    m = {"macro_f1": float(f1_score(all_true, all_pred, average="macro")) if all_true else None,
         "balanced_accuracy": float(balanced_accuracy_score(all_true, all_pred)) if all_true else None,
         "cumulative_reward": float(total_reward), "n_steps": n_steps,
         "mean_dqn_episode_reward_last50": float(np.mean(ep_rewards[-50:]))}
    return m


def run(dataset_name="unsw_nb15", seed=42, device="cpu"):
    t0 = time.time()
    print(f"=== Graph/RL baselines on {dataset_name} (seed={seed}) ===", flush=True)
    packs = prepare(dataset_name, seed=seed)

    graph_results = run_graph_baselines(packs, seed, device)
    print("running random policy baseline...", flush=True)
    random_result = run_random_policy(packs, seed)
    print(f"  random policy: {random_result}", flush=True)
    print("training vanilla-feature DQN baseline...", flush=True)
    vanilla_dqn_result = run_vanilla_dqn(packs, seed, device)
    print(f"  vanilla DQN: {vanilla_dqn_result}", flush=True)

    results = {
        "dataset": dataset_name, "seed": seed,
        "graph_baselines": graph_results,
        "random_policy": random_result,
        "vanilla_feature_dqn": vanilla_dqn_result,
        "total_seconds": round(time.time() - t0, 2),
    }
    out_path = os.path.join(RESULTS_DIR, f"baselines_graph_rl_{dataset_name}_seed{seed}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}", flush=True)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="unsw_nb15")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run(args.dataset, seed=args.seed, device=args.device)
