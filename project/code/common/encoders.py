"""Supervised pretraining of the shared GraphEncoder + a Tent-style entropy-minimization
test-time adapter (Wang et al. 2021 "Tent: Fully Test-Time Adaptation by Entropy Minimization" —
a real, citable, standard TTA method, not invented for this project)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from graph_env import GraphEncoder, build_window_graph


class GraphClassifier(nn.Module):
    def __init__(self, in_dim, hidden=64, embed_dim=64, n_classes=2, encoder_cls=GraphEncoder):
        super().__init__()
        self.encoder = encoder_cls(in_dim, hidden, embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x, edge_index):
        h = self.encoder(x, edge_index)
        logits = self.head(h)
        return h, logits


def pretrain_encoder(X_cheap, cheap_cat_ids, y_binary, in_dim, n_classes=2,
                      window_size=256, n_windows=200, epochs=1, lr=1e-3, seed=0, device="cpu",
                      encoder_cls=GraphEncoder):
    """Supervised warm-up: sample random windows, build the window graph, train encoder+head
    to predict binary attack/benign. This is the 'frozen security encoder' TTA-GraphRL needs.
    encoder_cls: GraphEncoder (SAGEConv, default), GCNEncoder, or GATEncoder -- see graph_env.py."""
    torch.manual_seed(seed)
    model = GraphClassifier(in_dim, n_classes=n_classes, encoder_cls=encoder_cls).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    n = len(y_binary)

    losses = []
    for w in range(n_windows):
        idx = rng.choice(n, size=min(window_size, n), replace=False)
        edge_index = build_window_graph(cheap_cat_ids[idx]).to(device)
        x = torch.tensor(X_cheap[idx], dtype=torch.float32, device=device)
        y = torch.tensor(y_binary[idx], dtype=torch.long, device=device)
        _, logits = model(x, edge_index)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if w % 50 == 0:
            print(f"  [pretrain] window {w}/{n_windows} loss={loss.item():.4f}", flush=True)

    return model, losses


class TentAdapter(nn.Module):
    """Diagonal affine transform on the embedding space, adapted via entropy minimization on
    unlabeled test-time batches. Frozen encoder + frozen head; only this adapter is updated."""
    def __init__(self, embed_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, h):
        return h * self.scale + self.shift


def adapt_tta(model: GraphClassifier, X_cheap, cheap_cat_ids, window_size=256, n_adapt_windows=20,
              lr=1e-2, seed=0, device="cpu"):
    """Runs Tent-style adaptation: minimize prediction entropy on unlabeled windows sampled from
    the (possibly shifted) target data. Returns an adapter module to compose with the frozen model."""
    embed_dim = model.head.in_features
    adapter = TentAdapter(embed_dim).to(device)
    opt = torch.optim.Adam(adapter.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    n = len(X_cheap)

    for m in model.parameters():
        m.requires_grad_(False)

    for w in range(n_adapt_windows):
        idx = rng.choice(n, size=min(window_size, n), replace=False)
        edge_index = build_window_graph(cheap_cat_ids[idx]).to(device)
        x = torch.tensor(X_cheap[idx], dtype=torch.float32, device=device)
        with torch.no_grad():
            h, _ = model(x, edge_index)
        h_adapted = adapter(h)
        logits = model.head(h_adapted)
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
        opt.zero_grad()
        entropy.backward()
        opt.step()

    return adapter
