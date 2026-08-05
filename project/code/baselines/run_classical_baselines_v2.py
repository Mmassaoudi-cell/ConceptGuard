"""
Phase 4: expanded classical/tree baseline sweep (13 models) -- Part 8 items 1-13 of the master
spec. No test-set tuning: hyperparameters are fixed, reasonable defaults, not searched on test.
Binary attack/benign task to match ConceptGuard-RL (the selected final method) directly.
"""
import sys
import os
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from data_utils import LOADERS, fit_preprocessor, transform_with_preprocessor  # noqa: E402

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               GradientBoostingClassifier, HistGradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, precision_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

RESULTS_DIR = r"C:\Users\MMASSAOUDI\Desktop\research work\Alienware\Innovative paper 4\project\results"
SVC_SUBSAMPLE_CAP = 20_000  # RBF SVM is O(n^2)-ish; cap training rows for tractability, note it


def evaluate(y_true, y_pred, prefix):
    return {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        f"{prefix}_weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        f"{prefix}_attack_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
        if set(y_true) <= {0, 1} else None,
        f"{prefix}_attack_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        if set(y_true) <= {0, 1} else None,
    }


def run(dataset_name, seed=42, label_mode="binary", **loader_kwargs):
    t0 = time.time()
    data = LOADERS[dataset_name](seed=seed, **loader_kwargs)
    multiclass_label_col = data["label_col"]
    binary_label_col = data.get("binary_label_col")
    if label_mode == "binary" and binary_label_col:
        label_col = binary_label_col
    else:
        label_col = multiclass_label_col
    train_df, val_df, test_df = data["train"], data["val"], data["test"]

    # exclude BOTH label columns from the feature matrix, whichever one is the active target --
    # otherwise the inactive label column leaks in as an ordinary feature (caught during Phase 4:
    # a first run silently fell back to the multiclass target because this dict key was missing,
    # AND would have leaked the binary label as a feature had binary mode activated as intended).
    exclude = {multiclass_label_col}
    if binary_label_col:
        exclude.add(binary_label_col)
    prep = fit_preprocessor(train_df, label_col=label_col, exclude_cols=list(exclude))
    X_train = transform_with_preprocessor(train_df, prep)
    X_val = transform_with_preprocessor(val_df, prep)
    X_test = transform_with_preprocessor(test_df, prep)

    le = LabelEncoder()
    y_train = le.fit_transform(train_df[label_col].astype(str))
    known = set(le.classes_)

    def enc(series):
        arr = series.astype(str).values
        out = np.full(len(arr), -1, dtype=int)
        mask = np.isin(arr, list(known))
        out[mask] = le.transform(arr[mask])
        return out

    y_val = enc(val_df[label_col])
    y_test = enc(test_df[label_col])

    rng = np.random.default_rng(seed)
    if len(X_train) > SVC_SUBSAMPLE_CAP:
        svc_idx = rng.choice(len(X_train), size=SVC_SUBSAMPLE_CAP, replace=False)
    else:
        svc_idx = np.arange(len(X_train))

    models = {
        "LogisticRegression": (LogisticRegression(max_iter=200, n_jobs=-1), False),
        "LinearSVM": (SGDClassifier(loss="hinge", max_iter=1000, random_state=seed), False),
        "RBF_SVM": (SVC(kernel="rbf", random_state=seed), True),  # subsampled, noted
        "GaussianNB": (GaussianNB(), False),
        "KNN": (KNeighborsClassifier(n_neighbors=5, n_jobs=-1), False),
        "DecisionTree": (DecisionTreeClassifier(random_state=seed), False),
        "RandomForest": (RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed), False),
        "ExtraTrees": (ExtraTreesClassifier(n_estimators=200, n_jobs=-1, random_state=seed), False),
        "GradientBoosting": (GradientBoostingClassifier(n_estimators=150, random_state=seed), False),
        "HistGradientBoosting": (HistGradientBoostingClassifier(random_state=seed), False),
        "XGBoost": (xgb.XGBClassifier(n_estimators=300, tree_method="hist", device="cuda",
                                       random_state=seed, eval_metric="logloss", verbosity=0), False),
        "LightGBM": (lgb.LGBMClassifier(n_estimators=300, random_state=seed, verbosity=-1), False),
        "CatBoost": (cb.CatBoostClassifier(n_estimators=300, random_state=seed, verbose=False), False),
        "MLP": (MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=100, random_state=seed), False),
    }

    results = {"dataset": dataset_name, "seed": seed, "label_mode": label_mode,
               "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
               "n_classes": len(le.classes_), "svc_subsample_cap": SVC_SUBSAMPLE_CAP,
               "models": {}}

    for name, (model, subsample) in models.items():
        tm0 = time.time()
        try:
            if subsample:
                model.fit(X_train[svc_idx], y_train[svc_idx])
            else:
                model.fit(X_train, y_train)
            pred_val = model.predict(X_val)
            pred_test = model.predict(X_test)
            m = {}
            m.update(evaluate(y_val, pred_val, "val"))
            m.update(evaluate(y_test, pred_test, "test"))
            m["train_seconds"] = round(time.time() - tm0, 2)
            m["subsampled_train"] = subsample
            results["models"][name] = m
            print(f"[{dataset_name}] {name}: test_macro_f1={m['test_macro_f1']:.4f} "
                  f"test_bal_acc={m['test_balanced_accuracy']:.4f} ({m['train_seconds']}s)", flush=True)
        except Exception as e:
            results["models"][name] = {"error": str(e)}
            print(f"[{dataset_name}] {name}: ERROR {e}", flush=True)

    results["total_seconds"] = round(time.time() - t0, 2)
    out_path = os.path.join(RESULTS_DIR, f"baselines_v2_{dataset_name}_{label_mode}_seed{seed}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}", flush=True)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label_mode", default="binary")
    args = ap.parse_args()
    run(args.dataset, seed=args.seed, label_mode=args.label_mode)
