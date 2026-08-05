"""
Candidate 6 (spec priority #3): ConceptGuard-RL — Intervenable Graph Concept Bottlenecks for
Explainable Intrusion Response.

UNSW-NB15 ships pre-engineered connection-count features (ct_srv_src, ct_dst_ltm,
ct_src_dport_ltm, ct_dst_sport_ltm, ct_dst_src_ltm, ct_src_ltm, ct_srv_dst, is_sm_ips_ports) that
are genuinely concept-shaped (scan-intensity / fan-out proxies), not label-derived, and not
invented for this experiment. Pipeline: predict these 8 concepts from the OTHER raw features ->
feed PREDICTED concepts (not raw features) as the policy's primary reasoning signal -> train DQN
over the 5-action IDSPOMDPEnv -> at evaluation, run a concept-intervention test: substitute ground-
truth concepts for a subset of flows and measure whether/how often the policy's decision changes
and whether that change is toward the correct action (the spec's required "essential proof").
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
from data_utils import LOADERS, fit_preprocessor, transform_with_preprocessor, get_binary_labels  # noqa: E402
from graph_env import IDSPOMDPEnv, A_BENIGN, A_ATTACK, A_ABSTAIN, A_TELEMETRY, A_ESCALATE  # noqa: E402
from dqn import train_dqn, QNetwork  # noqa: E402

RESULTS_DIR = r"C:\Users\MMASSAOUDI\Desktop\research work\Alienware\Innovative paper 4\project\results"
CONCEPT_COLS = ["ct_srv_src", "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
                 "ct_dst_src_ltm", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports"]
CHEAP_COLS = ["proto", "service"]


class ConceptPredictor(nn.Module):
    def __init__(self, in_dim, n_concepts, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, n_concepts))

    def forward(self, x):
        return self.net(x)


def prepare(dataset_name="unsw_nb15", seed=42, train_frac=1.0):
    data = LOADERS[dataset_name](seed=seed)
    label_col = data["label_col"]
    train_df, val_df, test_df = data["train"], data["val"], data["test"]
    if train_frac < 1.0:
        train_df = train_df.sample(frac=train_frac, random_state=seed).reset_index(drop=True)
    all_cols = [c for c in train_df.columns if c != label_col]
    non_concept_cols = [c for c in all_cols if c not in CONCEPT_COLS and c not in CHEAP_COLS]

    cheap_prep = fit_preprocessor(train_df[CHEAP_COLS + [label_col]], label_col=label_col)
    concept_prep = fit_preprocessor(train_df[CONCEPT_COLS + [label_col]], label_col=label_col)
    exp_prep = fit_preprocessor(train_df[non_concept_cols + [label_col]], label_col=label_col)

    out = {}
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out[split_name] = {
            "cheap": transform_with_preprocessor(df, cheap_prep).astype(np.float32),
            "concept_true": transform_with_preprocessor(df, concept_prep).astype(np.float32),
            "expensive": transform_with_preprocessor(df, exp_prep).astype(np.float32),
            "y_bin": get_binary_labels(df, dataset_name),
        }
    return out, len(non_concept_cols)


def train_concept_predictor(train_pack, n_concepts, epochs=30, lr=1e-3, seed=42, device="cpu"):
    torch.manual_seed(seed)
    model = ConceptPredictor(train_pack["expensive"].shape[1], n_concepts).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.tensor(train_pack["expensive"], dtype=torch.float32, device=device)
    Y = torch.tensor(train_pack["concept_true"], dtype=torch.float32, device=device)
    n = X.shape[0]
    batch_size = 4096
    losses = []
    for ep in range(epochs):
        perm = torch.randperm(n)
        ep_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            pred = model(X[idx])
            loss = F.mse_loss(pred, Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(idx)
        losses.append(ep_loss / n)
        if ep % 5 == 0:
            print(f"  [concept predictor] epoch {ep}/{epochs} mse={losses[-1]:.4f}", flush=True)
    return model, losses


def concept_r2(model, pack, device="cpu"):
    with torch.no_grad():
        X = torch.tensor(pack["expensive"], dtype=torch.float32, device=device)
        pred = model(X).cpu().numpy()
    true = pack["concept_true"]
    ss_res = ((true - pred) ** 2).sum(axis=0)
    ss_tot = ((true - true.mean(axis=0)) ** 2).sum(axis=0) + 1e-8
    r2_per_concept = 1 - ss_res / ss_tot
    mae = np.abs(true - pred).mean()
    return {"mae": float(mae), "r2_per_concept": {c: float(r) for c, r in zip(CONCEPT_COLS, r2_per_concept)},
            "r2_mean": float(np.mean(r2_per_concept))}


def run_policy_with_concepts(qnet, pack, predicted_concepts, window_size=256, n_episodes=20, seed=999, device="cpu"):
    env = IDSPOMDPEnv(pack["cheap"], pack["expensive"], pack["cheap"].astype(int),
                       pack["y_bin"], window_size=window_size, seed=seed, embed_dim=predicted_concepts.shape[1])
    all_true, all_pred = [], []
    total_reward, n_telemetry, n_escalate, n_abstain, n_classified, n_steps = 0.0, 0, 0, 0, 0, 0
    for ep in range(n_episodes):
        env.reset()
        ge = torch.tensor(predicted_concepts[env.idx], dtype=torch.float32)
        obs = env._obs(ge)
        done = False
        while not done:
            with torch.no_grad():
                qvals = qnet(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
                action = int(qvals.argmax(dim=1).item())
            obs, reward, done, info = env.step(action, ge)
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

    from sklearn.metrics import f1_score, balanced_accuracy_score, recall_score, precision_score
    metrics = {}
    if all_true:
        metrics["macro_f1"] = float(f1_score(all_true, all_pred, average="macro"))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(all_true, all_pred))
        metrics["attack_recall"] = float(recall_score(all_true, all_pred, pos_label=1, zero_division=0))
        metrics["attack_precision"] = float(precision_score(all_true, all_pred, pos_label=1, zero_division=0))
    metrics["cumulative_reward"] = float(total_reward)
    metrics["telemetry_rate"] = n_telemetry / max(n_steps, 1)
    metrics["escalation_rate"] = n_escalate / max(n_steps, 1)
    metrics["abstention_rate"] = n_abstain / max(n_steps, 1)
    metrics["classification_rate"] = n_classified / max(n_steps, 1)
    metrics["n_steps"] = n_steps
    return metrics


def concept_intervention_test(qnet, pack, predicted_concepts, window_size=256, n_episodes=20, seed=999, device="cpu"):
    """Compares actions taken with PREDICTED vs TRUE concepts on the identical sampled windows
    (same seed -> same env.idx sequence). Reports how often the action changes, and of those
    changes, how many move from an incorrect classification to a correct one."""
    def rollout(concepts):
        env = IDSPOMDPEnv(pack["cheap"], pack["expensive"], pack["cheap"].astype(int),
                           pack["y_bin"], window_size=window_size, seed=seed, embed_dim=concepts.shape[1])
        actions_log = []
        for ep in range(n_episodes):
            env.reset()
            ge = torch.tensor(concepts[env.idx], dtype=torch.float32)
            obs = env._obs(ge)
            done = False
            while not done:
                with torch.no_grad():
                    qvals = qnet(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
                    action = int(qvals.argmax(dim=1).item())
                obs, reward, done, info = env.step(action, ge)
                actions_log.append((action, info["true_label"]))
        return actions_log

    log_pred = rollout(predicted_concepts)
    log_true = rollout(pack["concept_true"])
    n = min(len(log_pred), len(log_true))
    n_changed, n_changed_to_correct = 0, 0
    for i in range(n):
        a_pred, y = log_pred[i]
        a_true, _ = log_true[i]
        if a_pred != a_true:
            n_changed += 1
            correct_action = A_ATTACK if y == 1 else A_BENIGN
            was_correct = a_pred == correct_action if a_pred in (A_BENIGN, A_ATTACK) else False
            now_correct = a_true == correct_action if a_true in (A_BENIGN, A_ATTACK) else False
            if (not was_correct) and now_correct:
                n_changed_to_correct += 1
    return {"n_steps_compared": n, "action_change_rate": n_changed / max(n, 1),
            "changes_that_became_correct": n_changed_to_correct,
            "intervention_success_rate": n_changed_to_correct / max(n_changed, 1)}


def run(dataset_name="unsw_nb15", seed=42, n_dqn_episodes=400, window_size=256, device="cpu", train_frac=1.0):
    t0 = time.time()
    print(f"=== ConceptGuard-RL on {dataset_name} (seed={seed}) ===", flush=True)
    packs, n_non_concept = prepare(dataset_name, seed=seed, train_frac=train_frac)
    print(f"train_frac={train_frac} n_train={len(packs['train']['y_bin'])}", flush=True)
    concept_model, cp_losses = train_concept_predictor(packs["train"], len(CONCEPT_COLS), seed=seed, device=device)

    with torch.no_grad():
        pred_concepts = {
            split: concept_model(torch.tensor(packs[split]["expensive"], dtype=torch.float32, device=device)).cpu().numpy()
            for split in ["train", "val", "test"]
        }
    r2_train = concept_r2(concept_model, packs["train"], device=device)
    r2_test = concept_r2(concept_model, packs["test"], device=device)
    print(f"concept predictor: train_r2_mean={r2_train['r2_mean']:.3f} test_r2_mean={r2_test['r2_mean']:.3f}", flush=True)

    def embed_fn(env):
        return torch.tensor(pred_concepts["train"][env.idx], dtype=torch.float32)

    qnet, ep_rewards = train_dqn(
        packs["train"]["cheap"], packs["train"]["expensive"], packs["train"]["cheap"].astype(int),
        packs["train"]["y_bin"], embed_fn, n_episodes=n_dqn_episodes, window_size=window_size,
        seed=seed, device=device, embed_dim=len(CONCEPT_COLS),
    )

    val_metrics = run_policy_with_concepts(qnet, packs["val"], pred_concepts["val"], window_size, seed=seed + 1, device=device)
    test_metrics = run_policy_with_concepts(qnet, packs["test"], pred_concepts["test"], window_size, seed=seed + 2, device=device)
    intervention = concept_intervention_test(qnet, packs["test"], pred_concepts["test"], window_size, seed=seed + 3, device=device)

    results = {
        "candidate": "ConceptGuard-RL", "dataset": dataset_name, "seed": seed, "train_frac": train_frac,
        "n_train": len(packs["train"]["y_bin"]),
        "concept_cols": CONCEPT_COLS, "concept_r2_train": r2_train, "concept_r2_test": r2_test,
        "mean_dqn_episode_reward_last50": float(np.mean(ep_rewards[-50:])),
        "val_metrics": val_metrics, "test_metrics": test_metrics,
        "concept_intervention_test": intervention,
        "total_seconds": round(time.time() - t0, 2),
    }
    frac_suffix = "" if train_frac == 1.0 else f"_frac{train_frac}"
    out_path = os.path.join(RESULTS_DIR, f"conceptguard_rl_{dataset_name}_seed{seed}{frac_suffix}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}", flush=True)
    print(json.dumps({"test_metrics": test_metrics, "concept_intervention_test": intervention}, indent=2), flush=True)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="unsw_nb15")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_dqn_episodes", type=int, default=400)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--train_frac", type=float, default=1.0)
    args = ap.parse_args()
    run(args.dataset, seed=args.seed, n_dqn_episodes=args.n_dqn_episodes, device=args.device,
        train_frac=args.train_frac)
