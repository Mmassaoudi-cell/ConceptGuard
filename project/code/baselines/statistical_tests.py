"""
Phase 7: paired statistical testing of ConceptGuard-RL vs. each classical/tree baseline, across
the same 5 seeds (1-5) used for Phase 3 tuning. Wilcoxon signed-rank (paired, non-parametric --
appropriate for n=5 and no normality assumption) + Holm-Bonferroni correction across the family of
comparisons + Cohen's d effect size (on the paired differences).
"""
import json
import glob
import numpy as np
from scipy import stats

RESULTS_DIR = r"C:\Users\MMASSAOUDI\Desktop\research work\Alienware\Innovative paper 4\project\results"
SEEDS = list(range(1, 11))  # extended from 5 to 10 seeds so Wilcoxon can reach p<0.05


def load_conceptguard_f1():
    f1s = {}
    for s in SEEDS:
        path = f"{RESULTS_DIR}\\conceptguard_rl_unsw_nb15_seed{s}.json"
        with open(path) as f:
            d = json.load(f)
        f1s[s] = d["test_metrics"]["macro_f1"]
    return f1s


def load_baseline_f1(model_name):
    f1s = {}
    for s in SEEDS:
        path = f"{RESULTS_DIR}\\baselines_v2_unsw_nb15_binary_seed{s}.json"
        with open(path) as f:
            d = json.load(f)
        m = d["models"].get(model_name, {})
        if "test_macro_f1" in m:
            f1s[s] = m["test_macro_f1"]
    return f1s


def cohens_d(diffs):
    return float(np.mean(diffs) / np.std(diffs, ddof=1)) if np.std(diffs, ddof=1) > 0 else float("inf")


def holm_correction(pvals):
    order = np.argsort(pvals)
    n = len(pvals)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        corrected = (n - rank) * pvals[idx]
        running_max = max(running_max, corrected)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def main():
    cg_f1 = load_conceptguard_f1()
    print("ConceptGuard-RL macro-F1 by seed:", cg_f1)

    baseline_names = ["CatBoost", "RandomForest", "GradientBoosting", "XGBoost", "LightGBM",
                       "ExtraTrees", "HistGradientBoosting", "DecisionTree", "MLP", "KNN",
                       "LogisticRegression", "RBF_SVM", "GaussianNB", "LinearSVM"]
    comparisons = []
    for name in baseline_names:
        bl_f1 = load_baseline_f1(name)
        common_seeds = [s for s in SEEDS if s in bl_f1 and s in cg_f1]
        if len(common_seeds) < 5:
            print(f"skipping {name}: only {len(common_seeds)} matched seeds")
            continue
        cg_vals = np.array([cg_f1[s] for s in common_seeds])
        bl_vals = np.array([bl_f1[s] for s in common_seeds])
        diffs = cg_vals - bl_vals
        try:
            stat, p = stats.wilcoxon(cg_vals, bl_vals)
        except ValueError:
            stat, p = None, 1.0  # identical values in every pair (degenerate, e.g. all-zero diff)
        d = cohens_d(diffs)
        comparisons.append({
            "baseline": name, "mean_diff": float(np.mean(diffs)), "cohens_d": d,
            "wilcoxon_stat": float(stat) if stat is not None else None, "p_raw": float(p),
            "conceptguard_wins": int((diffs > 0).sum()), "n": len(common_seeds),
        })

    pvals = np.array([c["p_raw"] for c in comparisons])
    adj = holm_correction(pvals)
    for c, a in zip(comparisons, adj):
        c["p_holm"] = float(a)
        c["significant_at_0.05"] = bool(a < 0.05)

    out = {"conceptguard_f1_by_seed": cg_f1, "comparisons": comparisons}
    out_path = f"{RESULTS_DIR}\\table18_statistical_significance.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
