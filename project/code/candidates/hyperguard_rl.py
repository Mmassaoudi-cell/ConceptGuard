"""
Candidate 1 (spec priority #5): HyperGuard-RL — Temporal Hypergraph Reinforcement Learning for
Multi-Entity Intrusion Detection.

Bot_IoT's real 5-tuple + MAC + timestamp fields (saddr, daddr, proto, dport, stime) support genuine
typed hyperedges: e.g. the "same destination host" hyperedge connects EVERY flow in the window
converging on one victim in a single aggregation step -- a higher-order relation a pairwise graph
only approximates via a capped clique. Required ablation (spec "essential proof"): compare against
a pairwise SAGEConv graph encoder with a matched parameter budget on the SAME data, to show any gain
comes from the higher-order hyperedges, not just extra capacity.
"""
import sys
import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from data_utils import LOADERS, fit_preprocessor, transform_with_preprocessor  # noqa: E402
from graph_env import IDSPOMDPEnv, GraphEncoder, build_window_graph, sample_balanced_window  # noqa: E402
from graph_env import A_BENIGN, A_ATTACK, A_ABSTAIN, A_TELEMETRY, A_ESCALATE  # noqa: E402
from hypergraph import HypergraphEncoder, build_hyperedge_incidence  # noqa: E402
from dqn import train_dqn  # noqa: E402

RESULTS_DIR = r"C:\Users\MMASSAOUDI\Desktop\research work\Alienware\Innovative paper 4\project\results"
HYPEREDGE_TYPE_COLS = ["saddr", "daddr", "proto", "dport"]  # + a derived time-bucket, added at runtime
CHEAP_COLS = ["saddr", "daddr", "sport", "dport", "proto", "smac", "dmac"]


def add_time_bucket(df, bucket_seconds=30):
    df = df.copy()
    df["_time_bucket"] = (df["stime"].astype(float) // bucket_seconds).astype(int).astype(str)
    # force dport into the categorical path (fit_preprocessor treats numeric dtypes as
    # StandardScaler features by default; dport is semantically categorical for hyperedge grouping)
    df["dport"] = df["dport"].astype(str)
    return df


def prepare_bot_iot(seed=42, max_files=15, row_cap_per_file=5000):
    data = LOADERS["bot_iot"](seed=seed, max_files=max_files, row_cap_per_file=row_cap_per_file)
    label_col = data["label_col"]
    for split in ["train", "val", "test"]:
        data[split] = add_time_bucket(data[split])
    return data, label_col


def prepare_arrays(df, label_col, cheap_cols, hyperedge_cols, cheap_prep=None, exp_prep=None):
    non_cheap = [c for c in df.columns if c not in cheap_cols and c != label_col and c != "subcategory "
                 and c != "category"]
    if cheap_prep is None:
        cheap_prep = fit_preprocessor(df[cheap_cols + [label_col]], label_col=label_col)
    if exp_prep is None:
        exp_prep = fit_preprocessor(df[non_cheap + [label_col]], label_col=label_col)
    X_cheap = transform_with_preprocessor(df, cheap_prep).astype(np.float32)
    X_exp = transform_with_preprocessor(df, exp_prep).astype(np.float32)

    he_prep = fit_preprocessor(df[hyperedge_cols + [label_col]], label_col=label_col)
    cat_ids = transform_with_preprocessor(df, he_prep)[:, -len(he_prep["categorical_cols"]):].astype(int)
    return X_cheap, X_exp, cat_ids, cheap_prep, exp_prep


def pretrain_hypergraph_classifier(X_cheap, cat_ids, y_bin, window_size=256, n_windows=200,
                                    seed=42, device="cpu", epochs_per_window=1, lr=1e-3,
                                    minority_oversample_frac=0.3):
    torch.manual_seed(seed)
    encoder = HypergraphEncoder(X_cheap.shape[1]).to(device)
    head = nn.Linear(64, 2).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=lr)
    rng = np.random.default_rng(seed)
    n = len(y_bin)
    losses = []
    for w in range(n_windows):
        if minority_oversample_frac is not None:
            idx = sample_balanced_window(y_bin, window_size, rng, minority_oversample_frac)
        else:
            idx = rng.choice(n, size=min(window_size, n), replace=False)
        flow_idx, he_idx, he_type, n_he = build_hyperedge_incidence(cat_ids[idx])
        x = torch.tensor(X_cheap[idx], dtype=torch.float32, device=device)
        y = torch.tensor(y_bin[idx], dtype=torch.long, device=device)
        h = encoder(x, flow_idx.to(device), he_idx.to(device), n_he)
        logits = head(h)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if w % 50 == 0:
            print(f"  [hypergraph pretrain] window {w}/{n_windows} loss={loss.item():.4f}", flush=True)
    return encoder, head, losses


def pretrain_pairwise_graph_classifier(X_cheap, cat_ids_pairwise, y_bin, window_size=256, n_windows=200,
                                        seed=42, device="cpu", lr=1e-3, minority_oversample_frac=0.3):
    """Ablation baseline: same node features, pairwise SAGEConv graph (build_window_graph), matched
    hidden/out dims to HypergraphEncoder (both 64)."""
    torch.manual_seed(seed)
    encoder = GraphEncoder(X_cheap.shape[1], hidden=64, out_dim=64).to(device)
    head = nn.Linear(64, 2).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=lr)
    rng = np.random.default_rng(seed)
    n = len(y_bin)
    losses = []
    for w in range(n_windows):
        if minority_oversample_frac is not None:
            idx = sample_balanced_window(y_bin, window_size, rng, minority_oversample_frac)
        else:
            idx = rng.choice(n, size=min(window_size, n), replace=False)
        edge_index = build_window_graph(cat_ids_pairwise[idx]).to(device)
        x = torch.tensor(X_cheap[idx], dtype=torch.float32, device=device)
        y = torch.tensor(y_bin[idx], dtype=torch.long, device=device)
        h = encoder(x, edge_index)
        logits = head(h)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if w % 50 == 0:
            print(f"  [pairwise-graph pretrain] window {w}/{n_windows} loss={loss.item():.4f}", flush=True)
    return encoder, head, losses


@torch.no_grad()
def eval_classifier_hypergraph(encoder, head, X_cheap, cat_ids, y_bin, window_size=256, n_windows=20, seed=999, device="cpu"):
    rng = np.random.default_rng(seed)
    n = len(y_bin)
    all_true, all_pred = [], []
    for w in range(n_windows):
        idx = rng.choice(n, size=min(window_size, n), replace=False)
        flow_idx, he_idx, he_type, n_he = build_hyperedge_incidence(cat_ids[idx])
        x = torch.tensor(X_cheap[idx], dtype=torch.float32, device=device)
        h = encoder(x, flow_idx.to(device), he_idx.to(device), n_he)
        pred = head(h).argmax(dim=1).cpu().numpy()
        all_true.extend(y_bin[idx].tolist())
        all_pred.extend(pred.tolist())
    from sklearn.metrics import f1_score, balanced_accuracy_score
    return {"macro_f1": float(f1_score(all_true, all_pred, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(all_true, all_pred))}


@torch.no_grad()
def eval_classifier_pairwise(encoder, head, X_cheap, cat_ids, y_bin, window_size=256, n_windows=20, seed=999, device="cpu"):
    rng = np.random.default_rng(seed)
    n = len(y_bin)
    all_true, all_pred = [], []
    for w in range(n_windows):
        idx = rng.choice(n, size=min(window_size, n), replace=False)
        edge_index = build_window_graph(cat_ids[idx]).to(device)
        x = torch.tensor(X_cheap[idx], dtype=torch.float32, device=device)
        h = encoder(x, edge_index)
        pred = head(h).argmax(dim=1).cpu().numpy()
        all_true.extend(y_bin[idx].tolist())
        all_pred.extend(pred.tolist())
    from sklearn.metrics import f1_score, balanced_accuracy_score
    return {"macro_f1": float(f1_score(all_true, all_pred, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(all_true, all_pred))}


def run(seed=42, max_files=15, row_cap_per_file=5000, n_pretrain_windows=200, n_dqn_episodes=300,
        window_size=256, device="cpu"):
    t0 = time.time()
    print(f"=== HyperGuard-RL on bot_iot (seed={seed}) ===", flush=True)
    data, label_col = prepare_bot_iot(seed=seed, max_files=max_files, row_cap_per_file=row_cap_per_file)
    train_df, val_df, test_df = data["train"], data["val"], data["test"]
    from data_utils import get_binary_labels
    y_train = get_binary_labels(train_df, "bot_iot")
    y_val = get_binary_labels(val_df, "bot_iot")
    y_test = get_binary_labels(test_df, "bot_iot")
    print(f"n_train={len(train_df)} n_val={len(val_df)} n_test={len(test_df)} "
          f"attack_frac_train={y_train.mean():.3f}", flush=True)

    he_cols = HYPEREDGE_TYPE_COLS + ["_time_bucket"]
    X_cheap_tr, X_exp_tr, cat_tr, cheap_prep, exp_prep = prepare_arrays(train_df, label_col, CHEAP_COLS, he_cols)
    he_prep_cache = fit_preprocessor(train_df[he_cols + [label_col]], label_col=label_col)

    def apply_split(df):
        X_cheap = transform_with_preprocessor(df, cheap_prep).astype(np.float32)
        X_exp = transform_with_preprocessor(df, exp_prep).astype(np.float32)
        cat_ids = transform_with_preprocessor(df, he_prep_cache)[:, -len(he_prep_cache["categorical_cols"]):].astype(int)
        return X_cheap, X_exp, cat_ids

    X_cheap_val, X_exp_val, cat_val = apply_split(val_df)
    X_cheap_test, X_exp_test, cat_test = apply_split(test_df)

    print("pretraining hypergraph classifier...", flush=True)
    hg_encoder, hg_head, hg_losses = pretrain_hypergraph_classifier(
        X_cheap_tr, cat_tr, y_train, window_size=window_size, n_windows=n_pretrain_windows, seed=seed, device=device)
    print("pretraining pairwise-graph ablation baseline...", flush=True)
    pg_encoder, pg_head, pg_losses = pretrain_pairwise_graph_classifier(
        X_cheap_tr, cat_tr, y_train, window_size=window_size, n_windows=n_pretrain_windows, seed=seed, device=device)

    hg_test_metrics = eval_classifier_hypergraph(hg_encoder, hg_head, X_cheap_test, cat_test, y_test,
                                                  window_size=window_size, seed=seed + 1, device=device)
    pg_test_metrics = eval_classifier_pairwise(pg_encoder, pg_head, X_cheap_test, cat_test, y_test,
                                                window_size=window_size, seed=seed + 1, device=device)
    print(f"ablation -- hypergraph: {hg_test_metrics}  pairwise-graph: {pg_test_metrics}", flush=True)

    for p in hg_encoder.parameters():
        p.requires_grad_(False)
    hg_encoder.eval()

    def embed_fn(env):
        idx = env.idx
        flow_idx, he_idx, he_type, n_he = build_hyperedge_incidence(cat_tr[idx])
        x = torch.tensor(X_cheap_tr[idx], dtype=torch.float32, device=device)
        return hg_encoder(x, flow_idx.to(device), he_idx.to(device), n_he)

    qnet, ep_rewards = train_dqn(X_cheap_tr, X_exp_tr, cat_tr, y_train, embed_fn,
                                  n_episodes=n_dqn_episodes, window_size=window_size, seed=seed,
                                  device=device, embed_dim=64, minority_oversample_frac=0.3)

    def evaluate_policy_local(X_cheap, X_exp, cat_ids, y_bin, n_episodes=20, seed=999):
        env = IDSPOMDPEnv(X_cheap, X_exp, cat_ids, y_bin, window_size=window_size, seed=seed, embed_dim=64)
        all_true, all_pred = [], []
        total_reward, n_steps = 0.0, 0
        n_tel = n_esc = n_abs = n_cls = 0
        for ep in range(n_episodes):
            env.reset()
            idx = env.idx
            flow_idx, he_idx, he_type, n_he = build_hyperedge_incidence(cat_ids[idx])
            x = torch.tensor(X_cheap[idx], dtype=torch.float32, device=device)
            with torch.no_grad():
                h = hg_encoder(x, flow_idx.to(device), he_idx.to(device), n_he)
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
                    n_tel += 1
                elif action == A_ESCALATE:
                    n_esc += 1
                elif action == A_ABSTAIN:
                    n_abs += 1
                elif action in (A_BENIGN, A_ATTACK):
                    n_cls += 1
                    all_true.append(info["true_label"])
                    all_pred.append(1 if action == A_ATTACK else 0)
        from sklearn.metrics import f1_score, balanced_accuracy_score
        m = {}
        if all_true:
            m["macro_f1"] = float(f1_score(all_true, all_pred, average="macro"))
            m["balanced_accuracy"] = float(balanced_accuracy_score(all_true, all_pred))
        m["cumulative_reward"] = float(total_reward)
        m["telemetry_rate"] = n_tel / max(n_steps, 1)
        m["escalation_rate"] = n_esc / max(n_steps, 1)
        m["abstention_rate"] = n_abs / max(n_steps, 1)
        m["classification_rate"] = n_cls / max(n_steps, 1)
        return m

    val_metrics = evaluate_policy_local(X_cheap_val, X_exp_val, cat_val, y_val, seed=seed + 2)
    test_metrics = evaluate_policy_local(X_cheap_test, X_exp_test, cat_test, y_test, seed=seed + 3)

    results = {
        "candidate": "HyperGuard-RL", "dataset": "bot_iot", "seed": seed,
        "n_train": len(train_df), "attack_frac_train": float(y_train.mean()),
        "hyperedge_types": he_cols,
        "ablation_hypergraph_vs_pairwise": {"hypergraph": hg_test_metrics, "pairwise_graph": pg_test_metrics},
        "mean_dqn_episode_reward_last50": float(np.mean(ep_rewards[-50:])),
        "val_metrics": val_metrics, "test_metrics": test_metrics,
        "total_seconds": round(time.time() - t0, 2),
    }
    out_path = os.path.join(RESULTS_DIR, f"hyperguard_rl_bot_iot_seed{seed}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}", flush=True)
    print(json.dumps({"ablation": results["ablation_hypergraph_vs_pairwise"], "test_metrics": test_metrics}, indent=2), flush=True)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_files", type=int, default=15)
    ap.add_argument("--row_cap_per_file", type=int, default=5000)
    ap.add_argument("--n_pretrain_windows", type=int, default=200)
    ap.add_argument("--n_dqn_episodes", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run(seed=args.seed, max_files=args.max_files, row_cap_per_file=args.row_cap_per_file,
        n_pretrain_windows=args.n_pretrain_windows, n_dqn_episodes=args.n_dqn_episodes, device=args.device)
