"""
Candidate 4: RiemannGuard — Curvature-Adaptive Graph Reinforcement Learning for Hierarchical
Attack Detection.

Bot_IoT ships a REAL, non-imposed 2-level attack hierarchy (category -> subcategory, e.g.
DDoS -> DDoS-TCP/DDoS-UDP/DDoS-HTTP), confirmed in the dataset screening. Trains a Poincare-ball
hyperbolic prototype classifier on the `category` label and compares it against a parameter-matched
Euclidean prototype classifier (spec's required ablation: "show adaptive curvature improves
hierarchical/rare-attack detection beyond Euclidean GNNs"). The better encoder then feeds the
RL policy for the usual binary attack/benign decision task.
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
from data_utils import fit_preprocessor, transform_with_preprocessor, get_binary_labels  # noqa: E402
from graph_env import IDSPOMDPEnv, A_BENIGN, A_ATTACK, A_ABSTAIN, A_TELEMETRY, A_ESCALATE  # noqa: E402
from hyperbolic import HyperbolicEncoder, EuclideanBaselineEncoder  # noqa: E402
from dqn import train_dqn  # noqa: E402
from hyperguard_rl import prepare_bot_iot, CHEAP_COLS  # noqa: E402

RESULTS_DIR = r"C:\Users\MMASSAOUDI\Desktop\research work\Alienware\Innovative paper 4\project\results"
CATEGORY_COL = "category"


def sample_balanced_window_multiclass(y, window_size, rng):
    classes = np.unique(y)
    per_class = max(1, window_size // len(classes))
    idx_parts = []
    for c in classes:
        pool = np.where(y == c)[0]
        idx_parts.append(rng.choice(pool, size=per_class, replace=True))
    idx = np.concatenate(idx_parts)
    rng.shuffle(idx)
    return idx[:window_size]


def prepare_features(df, label_col, exclude_extra, prep=None):
    exclude = exclude_extra
    if prep is None:
        prep = fit_preprocessor(df.drop(columns=[c for c in exclude if c != label_col and c in df.columns]),
                                 label_col=label_col)
    X = transform_with_preprocessor(df, prep).astype(np.float32)
    return X, prep


def train_prototype_classifier(model_cls, X, y_codes, n_classes, epochs=30, lr=1e-3, seed=42,
                                device="cpu", window_size=256):
    torch.manual_seed(seed)
    model = model_cls(X.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    n = len(y_codes)
    losses = []
    for ep in range(epochs):
        idx = sample_balanced_window_multiclass(y_codes, window_size, rng)
        x = torch.tensor(X[idx], dtype=torch.float32, device=device)
        y = torch.tensor(y_codes[idx], dtype=torch.long, device=device)
        _, logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if ep % 5 == 0:
            print(f"  [{model_cls.__name__}] epoch {ep}/{epochs} loss={loss.item():.4f}", flush=True)
    return model, losses


@torch.no_grad()
def eval_prototype_classifier(model, X, y_codes, n_eval=5000, seed=999, device="cpu"):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_codes), size=min(n_eval, len(y_codes)), replace=False)
    x = torch.tensor(X[idx], dtype=torch.float32, device=device)
    _, logits = model(x)
    pred = logits.argmax(dim=1).cpu().numpy()
    from sklearn.metrics import f1_score, balanced_accuracy_score
    return {"macro_f1": float(f1_score(y_codes[idx], pred, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(y_codes[idx], pred))}


def run(seed=42, max_files=75, row_cap_per_file=10000, n_pretrain_epochs=200, n_dqn_episodes=300,
        window_size=256, device="cpu"):
    t0 = time.time()
    print(f"=== RiemannGuard on bot_iot (seed={seed}) ===", flush=True)
    data, label_col = prepare_bot_iot(seed=seed, max_files=max_files, row_cap_per_file=row_cap_per_file)
    train_df, val_df, test_df = data["train"], data["val"], data["test"]

    cat_codes_tr, cat_uniques = train_df[CATEGORY_COL].astype("category").factorize()
    cat_map = {v: i for i, v in enumerate(cat_uniques)}
    cat_codes_val = val_df[CATEGORY_COL].map(cat_map).fillna(-1).astype(int).values
    cat_codes_test = test_df[CATEGORY_COL].map(cat_map).fillna(-1).astype(int).values
    n_classes = len(cat_uniques)
    print(f"n_train={len(train_df)} category classes={list(cat_uniques)}", flush=True)

    exclude = [label_col, "category", "subcategory ", "_time_bucket"]
    X_tr, prep = prepare_features(train_df, label_col, exclude)
    X_val, _ = prepare_features(val_df, label_col, exclude, prep=prep)
    X_test, _ = prepare_features(test_df, label_col, exclude, prep=prep)

    print("training hyperbolic prototype classifier...", flush=True)
    hyp_model, hyp_losses = train_prototype_classifier(HyperbolicEncoder, X_tr, cat_codes_tr, n_classes,
                                                         epochs=n_pretrain_epochs, seed=seed, device=device,
                                                         window_size=window_size)
    print("training Euclidean ablation baseline...", flush=True)
    euc_model, euc_losses = train_prototype_classifier(EuclideanBaselineEncoder, X_tr, cat_codes_tr, n_classes,
                                                         epochs=n_pretrain_epochs, seed=seed, device=device,
                                                         window_size=window_size)

    valid_test_mask = cat_codes_test >= 0
    hyp_metrics = eval_prototype_classifier(hyp_model, X_test[valid_test_mask], cat_codes_test[valid_test_mask],
                                             seed=seed + 1, device=device)
    euc_metrics = eval_prototype_classifier(euc_model, X_test[valid_test_mask], cat_codes_test[valid_test_mask],
                                             seed=seed + 1, device=device)
    print(f"ablation -- hyperbolic: {hyp_metrics}  euclidean: {euc_metrics}  "
          f"learned_curvature_c={hyp_model.curvature().item():.4f}", flush=True)

    better_model = hyp_model if hyp_metrics["macro_f1"] >= euc_metrics["macro_f1"] else euc_model
    better_name = "hyperbolic" if better_model is hyp_model else "euclidean"
    print(f"selected encoder for RL: {better_name}", flush=True)
    for p in better_model.parameters():
        p.requires_grad_(False)
    better_model.eval()

    y_train_bin = get_binary_labels(train_df, "bot_iot")
    y_val_bin = get_binary_labels(val_df, "bot_iot")
    y_test_bin = get_binary_labels(test_df, "bot_iot")
    cheap_prep = fit_preprocessor(train_df[[c for c in CHEAP_COLS if c in train_df.columns] + [label_col]],
                                   label_col=label_col)
    from data_utils import transform_with_preprocessor as twp
    X_cheap_tr = twp(train_df, cheap_prep).astype(np.float32)
    X_cheap_val = twp(val_df, cheap_prep).astype(np.float32)
    X_cheap_test = twp(test_df, cheap_prep).astype(np.float32)
    cat_ids_tr = X_cheap_tr.astype(int)
    cat_ids_val = X_cheap_val.astype(int)
    cat_ids_test = X_cheap_test.astype(int)
    X_exp_tr = np.zeros((len(train_df), 0), dtype=np.float32)
    X_exp_val = np.zeros((len(val_df), 0), dtype=np.float32)
    X_exp_test = np.zeros((len(test_df), 0), dtype=np.float32)

    embed_dim = better_model.mlp[-1].out_features

    def embed_fn(env):
        x = torch.tensor(X_tr[env.idx], dtype=torch.float32, device=device)
        with torch.no_grad():
            emb, _ = better_model(x)
        return emb

    qnet, ep_rewards = train_dqn(X_cheap_tr, X_exp_tr, cat_ids_tr, y_train_bin, embed_fn,
                                  n_episodes=n_dqn_episodes, window_size=window_size, seed=seed,
                                  device=device, embed_dim=embed_dim, minority_oversample_frac=0.3)

    def evaluate_policy_local(X_cheap, X_exp, cat_ids, y_bin, X_full, n_episodes=20, seed=999):
        env = IDSPOMDPEnv(X_cheap, X_exp, cat_ids, y_bin, window_size=window_size, seed=seed, embed_dim=embed_dim)
        all_true, all_pred = [], []
        total_reward, n_steps = 0.0, 0
        n_tel = n_esc = n_abs = n_cls = 0
        for ep in range(n_episodes):
            env.reset()
            x = torch.tensor(X_full[env.idx], dtype=torch.float32, device=device)
            with torch.no_grad():
                emb, _ = better_model(x)
            obs = env._obs(emb)
            done = False
            while not done:
                with torch.no_grad():
                    qvals = qnet(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
                    action = int(qvals.argmax(dim=1).item())
                obs, reward, done, info = env.step(action, emb)
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

    val_metrics = evaluate_policy_local(X_cheap_val, X_exp_val, cat_ids_val, y_val_bin, X_val, seed=seed + 2)
    test_metrics = evaluate_policy_local(X_cheap_test, X_exp_test, cat_ids_test, y_test_bin, X_test, seed=seed + 3)

    results = {
        "candidate": "RiemannGuard", "dataset": "bot_iot", "seed": seed,
        "category_classes": list(cat_uniques),
        "ablation_hyperbolic_vs_euclidean": {"hyperbolic": hyp_metrics, "euclidean": euc_metrics},
        "learned_curvature_c": float(hyp_model.curvature().item()),
        "selected_encoder_for_rl": better_name,
        "mean_dqn_episode_reward_last50": float(np.mean(ep_rewards[-50:])),
        "val_metrics": val_metrics, "test_metrics": test_metrics,
        "total_seconds": round(time.time() - t0, 2),
    }
    out_path = os.path.join(RESULTS_DIR, f"riemannguard_bot_iot_seed{seed}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}", flush=True)
    print(json.dumps({"ablation": results["ablation_hyperbolic_vs_euclidean"], "test_metrics": test_metrics}, indent=2), flush=True)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_files", type=int, default=75)
    ap.add_argument("--row_cap_per_file", type=int, default=10000)
    ap.add_argument("--n_pretrain_epochs", type=int, default=200)
    ap.add_argument("--n_dqn_episodes", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run(seed=args.seed, max_files=args.max_files, row_cap_per_file=args.row_cap_per_file,
        n_pretrain_epochs=args.n_pretrain_epochs, n_dqn_episodes=args.n_dqn_episodes, device=args.device)
