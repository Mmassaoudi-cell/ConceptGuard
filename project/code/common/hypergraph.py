"""
Typed temporal hypergraph encoder for Candidate 1: HyperGuard-RL. Deliberately implemented in
vanilla PyTorch (index_add_-based scatter-mean) rather than torch_geometric, for two reasons:
(1) torch_geometric has no first-class hyperedge (>2-node) convolution without extra deps
    (torch-scatter/torch-sparse, not installed in this environment), and
(2) it keeps this candidate independent of the torch_geometric import chain that produced a
    transient environment crash earlier in this project (pulls in an unused `transformers` chain).

A hyperedge connects ALL flows in the window sharing one entity value (e.g. every flow with the
same destination host is ONE hyperedge, not a capped pairwise clique) -- a genuine higher-order
structure, not reducible in one layer to the pairwise window-graph used by the other candidates.
Typed by entity kind (source host, destination host, protocol, destination port, time-bucket) so
a message passes flow -> hyperedge -> flow once per type, then combines across types.
"""
import numpy as np
import torch
import torch.nn as nn


def build_hyperedge_incidence(cat_ids: np.ndarray, max_group_size=128):
    """cat_ids: (N, T) ordinal-encoded integer columns, one per entity TYPE (e.g. saddr, daddr,
    proto, dport-bucket, time-bucket). Returns flow_idx, he_idx, he_type, n_hyperedges: parallel
    arrays of (flow, hyperedge) incidences, he_type giving which entity-type column produced each
    hyperedge (so the encoder can keep per-type transforms)."""
    N, T = cat_ids.shape
    flow_idx, he_idx, he_type = [], [], []
    he_counter = 0
    for t in range(T):
        col = cat_ids[:, t]
        for val in np.unique(col):
            if val < 0:
                continue
            members = np.where(col == val)[0]
            if len(members) < 2:
                continue  # a hyperedge of size 1 carries no relational information
            if len(members) > max_group_size:
                members = np.random.default_rng(0).choice(members, size=max_group_size, replace=False)
            flow_idx.extend(members.tolist())
            he_idx.extend([he_counter] * len(members))
            he_type.append(t)
            he_counter += 1
    if he_counter == 0:
        # degenerate fallback: one hyperedge containing every flow, so the model still runs
        flow_idx = list(range(N))
        he_idx = [0] * N
        he_type = [0]
        he_counter = 1
    return (torch.tensor(flow_idx, dtype=torch.long), torch.tensor(he_idx, dtype=torch.long),
            torch.tensor(he_type, dtype=torch.long), he_counter)


class HypergraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=64):
        super().__init__()
        self.node_transform = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.he_transform = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.out_transform = nn.Linear(hidden, out_dim)

    def forward(self, x, flow_idx, he_idx, n_hyperedges):
        device = x.device
        N, hidden = x.shape[0], self.node_transform[0].out_features
        h = self.node_transform(x)  # (N, hidden)

        he_sum = torch.zeros(n_hyperedges, hidden, device=device)
        he_sum.index_add_(0, he_idx, h[flow_idx])
        he_count = torch.zeros(n_hyperedges, 1, device=device)
        he_count.index_add_(0, he_idx, torch.ones(len(flow_idx), 1, device=device))
        he_embed = self.he_transform(he_sum / he_count.clamp(min=1))

        flow_sum = torch.zeros(N, hidden, device=device)
        flow_sum.index_add_(0, flow_idx, he_embed[he_idx])
        flow_count = torch.zeros(N, 1, device=device)
        flow_count.index_add_(0, flow_idx, torch.ones(len(flow_idx), 1, device=device))
        flow_embed = flow_sum / flow_count.clamp(min=1)

        return self.out_transform(flow_embed)
