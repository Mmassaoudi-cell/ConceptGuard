"""
Candidate 3: CDE-Guard — Continuous-Time (time-decay GRU; see common/timedecay.py for the honest
scope note on why this is not a full Neural CDE) Graph Reinforcement Learning for Asynchronous
Intrusion Detection.

Bot_IoT's real `stime` timestamps give genuine irregular inter-event gaps within a window. Trains a
time-decay GRU vs a naive (dt-blind) GRU ablation baseline on chronologically-sorted windows, then
stress-tests both under increasing timestamp jitter and random event-dropping -- the spec's
required "essential proof": robustness under irregular timing vs a fixed-step baseline.
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
from graph_env import sample_balanced_window  # noqa: E402
from timedecay import TimeDecayGRUCell, NaiveGRUCell, make_dt_sequence  # noqa: E402
from dqn import train_dqn  # noqa: E402
from hyperguard_rl import prepare_bot_iot, CHEAP_COLS  # noqa: E402

RESULTS_DIR = r"C:\Users\MMASSAOUDI\Desktop\research work\Alienware\Innovative paper 4\project\results"
HIDDEN_DIM = 64


class SeqClassifier(nn.Module):
    def __init__(self, cell_cls, in_dim, hidden_dim=HIDDEN_DIM, n_classes=2):
        super().__init__()
        self.rnn = cell_cls(in_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(self, x_seq, dt_seq):
        h_seq = self.rnn(x_seq, dt_seq)  # (T, B, hidden)
        logits = self.head(h_seq)
        return h_seq, logits


def build_sequence(X, stime, y, idx, device="cpu", jitter_std=0.0, drop_frac=0.0, rng=None):
    """idx: flow indices for this window. Returns time-sorted (possibly jittered/dropped)
    x_seq (T,1,D), dt_seq (T,1), y_seq (T,), and the sort permutation (for mapping back)."""
    rng = rng or np.random.default_rng(0)
    order = np.argsort(stime[idx])
    idx_sorted = idx[order]
    if drop_frac > 0:
        keep = rng.random(len(idx_sorted)) >= drop_frac
        if keep.sum() < 2:
            keep[:2] = True
        idx_sorted = idx_sorted[keep]
    st = stime[idx_sorted].astype(np.float64)
    if jitter_std > 0:
        st = st + rng.normal(0, jitter_std, size=len(st))
        resort = np.argsort(st)
        st = st[resort]
        idx_sorted = idx_sorted[resort]
    x_seq = torch.tensor(X[idx_sorted], dtype=torch.float32, device=device).unsqueeze(1)  # (T,1,D)
    dt_seq = make_dt_sequence(torch.tensor(st, dtype=torch.float32, device=device)).unsqueeze(1)  # (T,1)
    y_seq = torch.tensor(y[idx_sorted], dtype=torch.long, device=device)
    return x_seq, dt_seq, y_seq, idx_sorted


def train_seq_classifier(cell_cls, X, stime, y, window_size=256, n_windows=200, seed=42,
                          device="cpu", lr=1e-3, minority_frac=0.3):
    torch.manual_seed(seed)
    model = SeqClassifier(cell_cls, X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    n = len(y)
    losses = []
    for w in range(n_windows):
        idx = sample_balanced_window(y, window_size, rng, minority_frac)
        x_seq, dt_seq, y_seq, _ = build_sequence(X, stime, y, idx, device=device, rng=rng)
        _, logits = model(x_seq, dt_seq)
        loss = F.cross_entropy(logits.squeeze(1), y_seq)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if w % 50 == 0:
            print(f"  [{cell_cls.__name__}] window {w}/{n_windows} loss={loss.item():.4f}", flush=True)
    return model, losses


@torch.no_grad()
def eval_seq_classifier(model, X, stime, y, window_size=256, n_windows=20, seed=999, device="cpu",
                         jitter_std=0.0, drop_frac=0.0):
    rng = np.random.default_rng(seed)
    n = len(y)
    all_true, all_pred = [], []
    for w in range(n_windows):
        idx = rng.choice(n, size=min(window_size, n), replace=False)
        x_seq, dt_seq, y_seq, _ = build_sequence(X, stime, y, idx, device=device,
                                                  jitter_std=jitter_std, drop_frac=drop_frac, rng=rng)
        _, logits = model(x_seq, dt_seq)
        pred = logits.squeeze(1).argmax(dim=1).cpu().numpy()
        all_true.extend(y_seq.cpu().numpy().tolist())
        all_pred.extend(pred.tolist())
    from sklearn.metrics import f1_score, balanced_accuracy_score
    return {"macro_f1": float(f1_score(all_true, all_pred, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(all_true, all_pred))}


def run(seed=42, max_files=75, row_cap_per_file=10000, n_pretrain_windows=300, n_dqn_episodes=300,
        window_size=256, device="cpu"):
    t0 = time.time()
    print(f"=== CDE-Guard on bot_iot (seed={seed}) ===", flush=True)
    data, label_col = prepare_bot_iot(seed=seed, max_files=max_files, row_cap_per_file=row_cap_per_file)
    train_df, val_df, test_df = data["train"], data["val"], data["test"]
    y_train = get_binary_labels(train_df, "bot_iot")
    y_val = get_binary_labels(val_df, "bot_iot")
    y_test = get_binary_labels(test_df, "bot_iot")
    print(f"n_train={len(train_df)} attack_frac_train={y_train.mean():.4f}", flush=True)

    exclude = [label_col, "category", "subcategory ", "_time_bucket", "stime", "ltime"]
    prep = fit_preprocessor(train_df.drop(columns=[c for c in exclude if c != label_col and c in train_df.columns]),
                             label_col=label_col)
    X_tr = transform_with_preprocessor(train_df, prep).astype(np.float32)
    X_val = transform_with_preprocessor(val_df, prep).astype(np.float32)
    X_test = transform_with_preprocessor(test_df, prep).astype(np.float32)
    stime_tr = train_df["stime"].astype(np.float64).values
    stime_val = val_df["stime"].astype(np.float64).values
    stime_test = test_df["stime"].astype(np.float64).values

    print("training time-decay GRU...", flush=True)
    td_model, td_losses = train_seq_classifier(TimeDecayGRUCell, X_tr, stime_tr, y_train,
                                                window_size=window_size, n_windows=n_pretrain_windows,
                                                seed=seed, device=device)
    print("training naive (dt-blind) GRU ablation baseline...", flush=True)
    naive_model, naive_losses = train_seq_classifier(NaiveGRUCell, X_tr, stime_tr, y_train,
                                                      window_size=window_size, n_windows=n_pretrain_windows,
                                                      seed=seed, device=device)

    robustness = {"time_decay": {}, "naive": {}}
    for jitter, drop in [(0.0, 0.0), (5.0, 0.1), (15.0, 0.25), (30.0, 0.4)]:
        key = f"jitter{jitter}_drop{drop}"
        robustness["time_decay"][key] = eval_seq_classifier(td_model, X_test, stime_test, y_test,
                                                              seed=seed + 1, device=device,
                                                              jitter_std=jitter, drop_frac=drop)
        robustness["naive"][key] = eval_seq_classifier(naive_model, X_test, stime_test, y_test,
                                                         seed=seed + 1, device=device,
                                                         jitter_std=jitter, drop_frac=drop)
        print(f"  {key}: time_decay={robustness['time_decay'][key]} naive={robustness['naive'][key]}", flush=True)

    for p in td_model.parameters():
        p.requires_grad_(False)
    td_model.eval()

    cheap_prep = fit_preprocessor(train_df[[c for c in CHEAP_COLS if c in train_df.columns] + [label_col]],
                                   label_col=label_col)
    X_cheap_tr = transform_with_preprocessor(train_df, cheap_prep).astype(np.float32)
    X_cheap_val = transform_with_preprocessor(val_df, cheap_prep).astype(np.float32)
    X_cheap_test = transform_with_preprocessor(test_df, cheap_prep).astype(np.float32)
    cat_ids_tr = X_cheap_tr.astype(int)
    cat_ids_val = X_cheap_val.astype(int)
    cat_ids_test = X_cheap_test.astype(int)
    X_exp_tr = np.zeros((len(train_df), 0), dtype=np.float32)
    X_exp_val = np.zeros((len(val_df), 0), dtype=np.float32)
    X_exp_test = np.zeros((len(test_df), 0), dtype=np.float32)

    def compute_embed_for_idx(X, stime, idx, device):
        """Returns per-flow embeddings ALIGNED to idx order (not chronological order), by
        inverse-permuting the time-decay GRU's chronological output back to idx order."""
        x_seq, dt_seq, y_seq, idx_sorted = build_sequence(X, stime, np.zeros(len(stime), dtype=int), idx, device=device)
        with torch.no_grad():
            h_seq, _ = td_model(x_seq, dt_seq)
        h = h_seq.squeeze(1)  # (T, hidden)
        inv = {flow: pos for pos, flow in enumerate(idx_sorted)}
        order = [inv[i] for i in idx]
        return h[order]

    def embed_fn(env):
        return compute_embed_for_idx(X_tr, stime_tr, env.idx, device)

    qnet, ep_rewards = train_dqn(X_cheap_tr, X_exp_tr, cat_ids_tr, y_train, embed_fn,
                                  n_episodes=n_dqn_episodes, window_size=window_size, seed=seed,
                                  device=device, embed_dim=HIDDEN_DIM, minority_oversample_frac=0.3)

    def evaluate_policy_local(X_cheap, X_exp, cat_ids, y_bin, X_full, stime_full, n_episodes=20, seed=999):
        env = IDSPOMDPEnv(X_cheap, X_exp, cat_ids, y_bin, window_size=window_size, seed=seed, embed_dim=HIDDEN_DIM)
        all_true, all_pred = [], []
        total_reward, n_steps = 0.0, 0
        n_tel = n_esc = n_abs = n_cls = 0
        for ep in range(n_episodes):
            env.reset()
            h = compute_embed_for_idx(X_full, stime_full, env.idx, device)
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

    val_metrics = evaluate_policy_local(X_cheap_val, X_exp_val, cat_ids_val, y_val, X_val, stime_val, seed=seed + 2)
    test_metrics = evaluate_policy_local(X_cheap_test, X_exp_test, cat_ids_test, y_test, X_test, stime_test, seed=seed + 3)

    results = {
        "candidate": "CDE-Guard", "dataset": "bot_iot", "seed": seed,
        "robustness_time_decay_vs_naive": robustness,
        "mean_dqn_episode_reward_last50": float(np.mean(ep_rewards[-50:])),
        "val_metrics": val_metrics, "test_metrics": test_metrics,
        "total_seconds": round(time.time() - t0, 2),
    }
    out_path = os.path.join(RESULTS_DIR, f"cdeguard_bot_iot_seed{seed}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}", flush=True)
    print(json.dumps({"robustness": robustness, "test_metrics": test_metrics}, indent=2), flush=True)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_files", type=int, default=75)
    ap.add_argument("--row_cap_per_file", type=int, default=10000)
    ap.add_argument("--n_pretrain_windows", type=int, default=300)
    ap.add_argument("--n_dqn_episodes", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run(seed=args.seed, max_files=args.max_files, row_cap_per_file=args.row_cap_per_file,
        n_pretrain_windows=args.n_pretrain_windows, n_dqn_episodes=args.n_dqn_episodes, device=args.device)
