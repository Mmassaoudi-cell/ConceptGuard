"""
Smallest credible baseline pass: LogisticRegression, RandomForest, XGBoost, LightGBM.
Purpose right now is (a) validate the shared data pipeline end-to-end on a real dataset,
(b) produce a first honest reference point other candidates get compared against later.
No test-set tuning: hyperparameters are fixed defaults / light, not searched on test.
"""
import sys
import os
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from data_utils import LOADERS, fit_preprocessor, transform_with_preprocessor  # noqa: E402

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import xgboost as xgb
import lightgbm as lgb

RESULTS_DIR = r"C:\Users\MMASSAOUDI\Desktop\research work\Alienware\Innovative paper 4\project\results"


def evaluate(y_true, y_pred, prefix):
    return {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        f"{prefix}_weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }


def run(dataset_name, seed=42, label_mode="multiclass", **loader_kwargs):
    t0 = time.time()
    data = LOADERS[dataset_name](seed=seed, **loader_kwargs)
    label_col = data["label_col"]
    train_df, val_df, test_df = data["train"], data["val"], data["test"]

    if label_mode == "binary" and "binary_label_col" in data:
        label_col = data["binary_label_col"]

    prep = fit_preprocessor(train_df, label_col=data["label_col"], exclude_cols=[data["label_col"]])
    X_train = transform_with_preprocessor(train_df, prep)
    X_val = transform_with_preprocessor(val_df, prep)
    X_test = transform_with_preprocessor(test_df, prep)

    le = LabelEncoder()
    y_train = le.fit_transform(train_df[label_col].astype(str))
    # unseen-label safety: map any val/test label not seen in train to a dedicated "unseen" bucket
    known = set(le.classes_)

    def enc(series):
        arr = series.astype(str).values
        out = np.full(len(arr), -1, dtype=int)
        mask = np.isin(arr, list(known))
        out[mask] = le.transform(arr[mask])
        return out

    y_val = enc(val_df[label_col])
    y_test = enc(test_df[label_col])

    models = {
        "LogisticRegression": LogisticRegression(max_iter=200, n_jobs=-1),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, tree_method="hist", device="cuda", random_state=seed,
            eval_metric="mlogloss", verbosity=0,
        ),
        "LightGBM": lgb.LGBMClassifier(n_estimators=300, random_state=seed, verbosity=-1),
    }

    results = {"dataset": dataset_name, "seed": seed, "label_mode": label_mode,
               "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
               "n_classes": len(le.classes_), "duplicates_removed": data.get("duplicates_removed"),
               "models": {}}

    for name, model in models.items():
        tm0 = time.time()
        try:
            model.fit(X_train, y_train)
            pred_val = model.predict(X_val)
            pred_test = model.predict(X_test)
            m = {}
            m.update(evaluate(y_val, pred_val, "val"))
            m.update(evaluate(y_test, pred_test, "test"))
            m["train_seconds"] = round(time.time() - tm0, 2)
            results["models"][name] = m
            print(f"[{dataset_name}] {name}: test_macro_f1={m['test_macro_f1']:.4f} "
                  f"test_bal_acc={m['test_balanced_accuracy']:.4f} ({m['train_seconds']}s)", flush=True)
        except Exception as e:
            results["models"][name] = {"error": str(e)}
            print(f"[{dataset_name}] {name}: ERROR {e}", flush=True)

    results["total_seconds"] = round(time.time() - t0, 2)
    out_path = os.path.join(RESULTS_DIR, f"baselines_{dataset_name}_{label_mode}_seed{seed}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}", flush=True)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label_mode", default="multiclass")
    args = ap.parse_args()
    run(args.dataset, seed=args.seed, label_mode=args.label_mode)
