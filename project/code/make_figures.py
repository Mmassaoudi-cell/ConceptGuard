"""Phase 8: generate core figures from real result JSONs. Matplotlib, IEEE-two-column-safe sizes."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = r"C:\Users\MMASSAOUDI\Desktop\research work\Alienware\Innovative paper 4\project\results"
FIG_DIR = r"C:\Users\MMASSAOUDI\Desktop\research work\Alienware\Innovative paper 4\project\figures"
os.makedirs(FIG_DIR, exist_ok=True)


def load(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


def fig_benchmark_comparison():
    cg_f1s = []
    for s in range(1, 11):
        d = load(f"conceptguard_rl_unsw_nb15_seed{s}.json")
        cg_f1s.append(d["test_metrics"]["macro_f1"])
    cg_mean, cg_std = np.mean(cg_f1s), np.std(cg_f1s)

    baseline_names = ["CatBoost", "RandomForest", "GradientBoosting", "XGBoost", "LightGBM",
                       "ExtraTrees", "HistGradientBoosting", "DecisionTree", "MLP", "KNN",
                       "LogisticRegression", "RBF_SVM", "GaussianNB", "LinearSVM"]
    bl_stats = {}
    for name in baseline_names:
        vals = []
        for s in range(1, 11):
            d = load(f"baselines_v2_unsw_nb15_binary_seed{s}.json")
            vals.append(d["models"][name]["test_macro_f1"])
        bl_stats[name] = (np.mean(vals), np.std(vals))

    graph_rl = load("baselines_graph_rl_unsw_nb15_seed42.json")
    extra = {
        "GraphSAGE": (graph_rl["graph_baselines"]["GraphSAGE"]["macro_f1"], 0),
        "GCN": (graph_rl["graph_baselines"]["GCN"]["macro_f1"], 0),
        "GAT": (graph_rl["graph_baselines"]["GAT"]["macro_f1"], 0),
        "Vanilla DQN": (graph_rl["vanilla_feature_dqn"]["macro_f1"], 0),
        "Random policy": (graph_rl["random_policy"]["macro_f1"], 0),
    }
    all_stats = {**bl_stats, **extra}
    sorted_items = sorted(all_stats.items(), key=lambda kv: kv[1][0])

    names = [k for k, v in sorted_items] + ["ConceptGuard-RL"]
    means = [v[0] for k, v in sorted_items] + [cg_mean]
    stds = [v[1] for k, v in sorted_items] + [cg_std]
    colors = ["#8899aa"] * len(sorted_items) + ["#d1495b"]

    fig, ax = plt.subplots(figsize=(7, 8))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, means, xerr=stds, color=colors, height=0.65, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Test macro-F1 (mean ± std, 10 seeds for classical baselines & ConceptGuard-RL;\n"
                   "single seed for graph/RL baselines)", fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.axvline(cg_mean, color="#d1495b", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title("ConceptGuard-RL vs. 19 Baselines (UNSW-NB15, binary attack/benign)", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_benchmark_comparison.png"), dpi=200)
    plt.close()
    print("wrote fig_benchmark_comparison.png")


def fig_lowlabel_curve():
    fracs = [0.01, 0.1, 0.5, 1.0]
    seed1_vals, seed42_vals = [], []
    for frac in fracs[:-1]:
        d1 = load(f"conceptguard_rl_unsw_nb15_seed1_frac{frac}.json")
        d42 = load(f"conceptguard_rl_unsw_nb15_seed42_frac{frac}.json")
        seed1_vals.append(d1["test_metrics"]["macro_f1"])
        seed42_vals.append(d42["test_metrics"]["macro_f1"])
    d1_full = load("conceptguard_rl_unsw_nb15_seed1.json")
    d42_full = load("conceptguard_rl_unsw_nb15_seed42.json")
    seed1_vals.append(d1_full["test_metrics"]["macro_f1"])
    seed42_vals.append(d42_full["test_metrics"]["macro_f1"])

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = [f * 100 for f in fracs]
    ax.plot(x, seed1_vals, marker="o", label="seed 1")
    ax.plot(x, seed42_vals, marker="s", label="seed 42")
    ax.set_xscale("log")
    ax.set_xlabel("% of training data used")
    ax.set_ylabel("Test macro-F1")
    ax.set_ylim(0.9, 1.0)
    ax.set_title("ConceptGuard-RL Low-Label Performance")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_lowlabel_curve.png"), dpi=200)
    plt.close()
    print("wrote fig_lowlabel_curve.png")


def fig_crossguard_robustness():
    d = load("crossguard_rl_edge_iiot_seed42.json")
    rob = d["corruption_robustness"]
    levels = sorted(float(k) for k in rob["robust"].keys())
    robust_f1 = [rob["robust"][str(l)]["macro_f1"] for l in levels]
    naive_f1 = [rob["naive"][str(l)]["macro_f1"] for l in levels]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = [l * 100 for l in levels]
    ax.plot(x, robust_f1, marker="o", label="Modality-dropout + contrastive (robust)")
    ax.plot(x, naive_f1, marker="s", label="Naive fusion (no dropout)")
    ax.set_xlabel("Additional synthetic modality corruption (%)")
    ax.set_ylabel("Test macro-F1")
    ax.set_ylim(0.95, 1.005)
    ax.set_title("CrossGuard-RL: Graceful Degradation Under\nModality Corruption (Edge-IIoTset)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_crossguard_robustness.png"), dpi=200)
    plt.close()
    print("wrote fig_crossguard_robustness.png")


def fig_cdeguard_robustness():
    f1_by_seed = {"time_decay": [], "naive": []}
    for s in [1, 2, 3, 4, 5]:
        d = load(f"cdeguard_bot_iot_seed{s}.json")
        rob = d["robustness_time_decay_vs_naive"]
        f1_by_seed["time_decay"].append(rob["time_decay"]["jitter0.0_drop0.0"]["macro_f1"])
        f1_by_seed["naive"].append(rob["naive"]["jitter0.0_drop0.0"]["macro_f1"])

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    x = np.arange(2)
    means = [np.mean(f1_by_seed["time_decay"]), np.mean(f1_by_seed["naive"])]
    stds = [np.std(f1_by_seed["time_decay"]), np.std(f1_by_seed["naive"])]
    ax.bar(x, means, yerr=stds, capsize=4, color=["#2a9d8f", "#8899aa"])
    ax.set_xticks(x)
    ax.set_xticklabels(["Time-decay GRU", "Naive (dt-blind) GRU"])
    ax.set_ylabel("Test macro-F1 (mean ± std, 5 seeds)")
    ax.set_title("CDE-Guard Ablation (Bot_IoT)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig_cdeguard_ablation.png"), dpi=200)
    plt.close()
    print("wrote fig_cdeguard_ablation.png")


if __name__ == "__main__":
    fig_benchmark_comparison()
    fig_lowlabel_curve()
    fig_crossguard_robustness()
    fig_cdeguard_robustness()
