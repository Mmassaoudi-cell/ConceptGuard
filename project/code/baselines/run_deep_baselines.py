"""
Phase 4: deep baselines (Part 8 items 14-22 of the master spec, subset: 1D-CNN, LSTM, GRU,
Transformer encoder -- the 4 most standard sequence/deep architectures for tabular IDS features).
Feature vectors are treated as a length-D pseudo-sequence with 1 channel, standard practice for
applying sequence models to tabular IDS data (each scaled/encoded feature = one "timestep").
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

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

RESULTS_DIR = r"C:\Users\MMASSAOUDI\Desktop\research work\Alienware\Innovative paper 4\project\results"


class CNN1D(nn.Module):
    def __init__(self, n_features, n_classes, hidden=32):
        super().__init__()
        self.conv1 = nn.Conv1d(1, hidden, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):  # x: (B, D)
        x = x.unsqueeze(1)  # (B, 1, D)
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = self.pool(h).squeeze(-1)
        return self.head(h)


class RNNBaseline(nn.Module):
    def __init__(self, n_features, n_classes, cell="lstm", hidden=64):
        super().__init__()
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):  # x: (B, D) -> (B, D, 1)
        x = x.unsqueeze(-1)
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class TransformerBaseline(nn.Module):
    def __init__(self, n_features, n_classes, d_model=32, nhead=4, layers=2):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):  # x: (B, D) -> (B, D, 1)
        x = x.unsqueeze(-1)
        h = self.proj(x)
        h = self.encoder(h)
        h = h.mean(dim=1)
        return self.head(h)


def evaluate(y_true, y_pred, prefix):
    return {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def train_torch_model(model, X_train, y_train, X_val, y_val, epochs=15, lr=1e-3, batch_size=1024, device="cpu"):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.long, device=device)
    n = len(yt)
    for ep in range(epochs):
        perm = torch.randperm(n)
        ep_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits = model(Xt[idx])
            loss = F.cross_entropy(logits, yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(idx)
        if ep % 5 == 0:
            print(f"    epoch {ep}/{epochs} loss={ep_loss / n:.4f}", flush=True)
    return model


@torch.no_grad()
def predict_torch_model(model, X, device="cpu", batch_size=8192):
    model.eval()
    preds = []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
        preds.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def run(dataset_name, seed=42, label_mode="binary", device="cpu", epochs=15, **loader_kwargs):
    t0 = time.time()
    data = LOADERS[dataset_name](seed=seed, **loader_kwargs)
    multiclass_label_col = data["label_col"]
    binary_label_col = data.get("binary_label_col")
    label_col = binary_label_col if (label_mode == "binary" and binary_label_col) else multiclass_label_col
    train_df, val_df, test_df = data["train"], data["val"], data["test"]

    exclude = {multiclass_label_col}
    if binary_label_col:
        exclude.add(binary_label_col)
    prep = fit_preprocessor(train_df, label_col=label_col, exclude_cols=list(exclude))
    X_train = transform_with_preprocessor(train_df, prep).astype(np.float32)
    X_val = transform_with_preprocessor(val_df, prep).astype(np.float32)
    X_test = transform_with_preprocessor(test_df, prep).astype(np.float32)

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
    n_classes = len(le.classes_)
    n_features = X_train.shape[1]

    torch.manual_seed(seed)
    model_builders = {
        "CNN1D": lambda: CNN1D(n_features, n_classes),
        "LSTM": lambda: RNNBaseline(n_features, n_classes, cell="lstm"),
        "GRU": lambda: RNNBaseline(n_features, n_classes, cell="gru"),
        "Transformer": lambda: TransformerBaseline(n_features, n_classes),
    }

    results = {"dataset": dataset_name, "seed": seed, "label_mode": label_mode,
               "n_train": len(train_df), "n_features": n_features, "n_classes": n_classes, "models": {}}

    for name, builder in model_builders.items():
        tm0 = time.time()
        print(f"[{dataset_name}] training {name}...", flush=True)
        try:
            model = builder()
            model = train_torch_model(model, X_train, y_train, X_val, y_val, epochs=epochs, device=device)
            pred_val = predict_torch_model(model, X_val, device=device)
            pred_test = predict_torch_model(model, X_test, device=device)
            m = {}
            m.update(evaluate(y_val, pred_val, "val"))
            m.update(evaluate(y_test, pred_test, "test"))
            m["train_seconds"] = round(time.time() - tm0, 2)
            m["n_params"] = sum(p.numel() for p in model.parameters())
            results["models"][name] = m
            print(f"[{dataset_name}] {name}: test_macro_f1={m['test_macro_f1']:.4f} "
                  f"test_bal_acc={m['test_balanced_accuracy']:.4f} ({m['train_seconds']}s)", flush=True)
        except Exception as e:
            results["models"][name] = {"error": str(e)}
            print(f"[{dataset_name}] {name}: ERROR {e}", flush=True)

    results["total_seconds"] = round(time.time() - t0, 2)
    out_path = os.path.join(RESULTS_DIR, f"baselines_deep_{dataset_name}_{label_mode}_seed{seed}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}", flush=True)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label_mode", default="binary")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()
    run(args.dataset, seed=args.seed, label_mode=args.label_mode, device=args.device, epochs=args.epochs)
