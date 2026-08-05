"""
Shared POMDP environment + window-graph builder for all RL candidates.

Honest scope note: the master spec lists 14 possible actions. Implementing all 14 with
genuine, non-fabricated semantics (real cost/benefit tradeoffs grounded in the data) is not
tractable in the time available. This environment implements a core 5-action subset that still
makes the task a genuine sequential decision problem rather than a relabeled classifier:
  0 = classify_benign
  1 = classify_attack
  2 = abstain
  3 = request_telemetry  (reveals masked "expensive" features for the current flow, costs budget,
                           does not advance the flow pointer -> agent must still act afterward)
  4 = escalate            (hands off to a human analyst; near-certain correct outcome but expensive)
This is documented explicitly here and must be stated as a scope reduction in the manuscript's
Limitations section, not silently presented as the full 14-action space.

Partial observability is real, not simulated for show: for every dataset, columns are split into
"cheap" (always visible: protocol/service/port/context fields — id_like_cols) and "expensive"
(everything else). The expensive block is zero-masked until the agent pays to reveal it via
request_telemetry. This creates an actual inspect-vs-act-now tradeoff grounded in the dataset's own
feature schema, not an invented one.
"""
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GCNConv, GATConv

ACTIONS = ["classify_benign", "classify_attack", "abstain", "request_telemetry", "escalate"]
A_BENIGN, A_ATTACK, A_ABSTAIN, A_TELEMETRY, A_ESCALATE = range(5)

# reward shaping constants (documented, not tuned on test data)
R_TP = 1.0          # correct attack classification
R_TN = 0.5          # correct benign classification (benign is majority class -> smaller reward)
R_FN = -3.0         # missed attack (costly, matches "penalty for false negatives" requirement)
R_FP = -1.0         # false alarm
R_ABSTAIN = -0.2    # small cost for not deciding
R_TELEMETRY_COST = -0.1
R_ESCALATE_BASE = 0.6   # near-certain correct outcome
R_ESCALATE_COST = -0.4  # but expensive
BUDGET_PER_EPISODE = 32  # telemetry budget for a window of flows


def build_window_graph(cheap_feat_ids: np.ndarray) -> torch.Tensor:
    """cheap_feat_ids: (N, k) integer-coded categorical context columns (proto/service/host/etc,
    already ordinal-encoded). Connect flows that share ANY context value -> edge_index (2, E).
    O(N^2) is fine at window scale (N ~ 256)."""
    n = cheap_feat_ids.shape[0]
    src, dst = [], []
    for j in range(cheap_feat_ids.shape[1]):
        col = cheap_feat_ids[:, j]
        for val in np.unique(col):
            if val == -1:
                continue
            idx = np.where(col == val)[0]
            if 1 < len(idx) <= 64:  # skip singleton and pathologically large hub groups
                # connect within-group as a small clique capped at 8 neighbors each, to bound E
                for a_pos, a in enumerate(idx):
                    for b in idx[a_pos + 1: a_pos + 9]:
                        src += [a, b]
                        dst += [b, a]
    if not src:
        src, dst = list(range(n)), list(range(n))  # fallback: self loops only
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index


def sample_balanced_window(y_binary, window_size, rng, minority_frac, minority_idx=None, majority_idx=None):
    """Draws a class-balanced window index array: minority_frac of window_size from the minority
    class (with replacement), the rest from the majority class. Reusable by IDSPOMDPEnv and by any
    candidate's supervised-pretraining loop that samples its own windows (e.g. HyperGuard-RL on
    Bot_IoT, where the minority/benign class is ~0.1-0.2% of the data). Pass precomputed
    minority_idx/majority_idx to avoid recomputing np.unique on every call."""
    if minority_idx is None or majority_idx is None:
        classes, counts = np.unique(y_binary, return_counts=True)
        minority_class = classes[np.argmin(counts)]
        minority_idx = np.where(y_binary == minority_class)[0]
        majority_idx = np.where(y_binary != minority_class)[0]
    n_min = max(1, int(round(minority_frac * window_size)))
    n_maj = window_size - n_min
    min_pick = rng.choice(minority_idx, size=n_min, replace=True)
    maj_pick = rng.choice(majority_idx, size=min(n_maj, len(majority_idx)), replace=len(majority_idx) < n_maj)
    idx = np.concatenate([min_pick, maj_pick])
    rng.shuffle(idx)
    return idx


class GraphEncoder(torch.nn.Module):
    """GraphSAGE encoder -- separate self/neighbor transforms (Hamilton et al. 2017)."""
    def __init__(self, in_dim, hidden=64, out_dim=64):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, out_dim)

    def forward(self, x, edge_index):
        h = torch.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        return h


class GCNEncoder(torch.nn.Module):
    """Genuine GCN (Kipf & Welling 2017) -- symmetric-normalized neighbor aggregation, distinct
    from GraphSAGE's separate self/neighbor weight matrices."""
    def __init__(self, in_dim, hidden=64, out_dim=64):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, out_dim)

    def forward(self, x, edge_index):
        h = torch.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        return h


class GATEncoder(torch.nn.Module):
    """Genuine GAT (Velickovic et al. 2018) -- learned attention weights over neighbors."""
    def __init__(self, in_dim, hidden=64, out_dim=64, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden // heads, heads=heads)
        self.conv2 = GATConv(hidden, out_dim, heads=1)

    def forward(self, x, edge_index):
        h = torch.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        return h


class IDSPOMDPEnv:
    """One 'episode' = one shuffled window of `window_size` flows from a given split.
    Duck-typed reset()/step() (no gymnasium dependency)."""

    def __init__(self, X_cheap, X_expensive, cheap_cat_ids, y_binary, window_size=256, seed=0, embed_dim=64,
                 severity_weight=None, minority_oversample_frac=None):
        self.X_cheap = X_cheap
        self.X_expensive = X_expensive
        self.cheap_cat_ids = cheap_cat_ids
        self.y_binary = y_binary  # 1 = attack, 0 = benign
        self.window_size = window_size
        self.rng = np.random.default_rng(seed)
        self.n = len(y_binary)
        self.expensive_dim = X_expensive.shape[1]
        self.embed_dim = embed_dim  # dim of the externally-supplied embedding (graph OR concept vector)
        # per-flow false-negative penalty multiplier (>=1), used by TailGuard for rare/high-severity
        # classes; defaults to a scalar 1.0 (no severity weighting, no full-array allocation) for
        # every other candidate -- avoids needlessly allocating an O(n) array on large datasets.
        self.severity_weight = severity_weight if severity_weight is not None else 1.0
        # for datasets with extreme class imbalance (e.g. Bot_IoT: ~0.1-0.2% benign), a uniform
        # window sample of 256 flows contains close to zero minority-class examples on average,
        # making the class effectively unlearnable from any single episode. When set, a fixed
        # fraction of each TRAINING window is drawn (with replacement) from the minority class,
        # oversampling it into every window; the rest is drawn uniformly as before. This changes
        # only what the agent trains on -- evaluation callers should pass None (the default) so
        # reported metrics reflect the true population distribution, not an oversampled one.
        self.minority_oversample_frac = minority_oversample_frac
        if minority_oversample_frac is not None:
            classes, counts = np.unique(y_binary, return_counts=True)
            self._minority_class = classes[np.argmin(counts)]
            self._minority_idx = np.where(y_binary == self._minority_class)[0]
            self._majority_idx = np.where(y_binary != self._minority_class)[0]

    def reset(self):
        if self.minority_oversample_frac is not None and len(self._minority_idx) > 0:
            self.idx = sample_balanced_window(self.y_binary, self.window_size, self.rng,
                                               self.minority_oversample_frac,
                                               self._minority_idx, self._majority_idx)
        else:
            self.idx = self.rng.choice(self.n, size=min(self.window_size, self.n), replace=False)
        self.pos = 0
        self.budget = BUDGET_PER_EPISODE
        self.revealed = np.zeros(len(self.idx), dtype=bool)
        self.prev_action = -1
        edge_index = build_window_graph(self.cheap_cat_ids[self.idx])
        node_x = torch.tensor(self.X_cheap[self.idx], dtype=torch.float32)
        self.graph = Data(x=node_x, edge_index=edge_index)
        self.done = False
        return self._obs()

    def _obs(self, graph_embed=None):
        i = self.idx[self.pos]
        cheap = self.X_cheap[i]
        exp = self.X_expensive[i] if self.revealed[self.pos] else np.zeros(self.expensive_dim)
        mask_flag = np.array([1.0 if self.revealed[self.pos] else 0.0])
        budget_norm = np.array([self.budget / BUDGET_PER_EPISODE])
        prev_action_onehot = np.zeros(len(ACTIONS))
        if self.prev_action >= 0:
            prev_action_onehot[self.prev_action] = 1.0
        if graph_embed is not None:
            ge = graph_embed[self.pos].detach().cpu().numpy() if torch.is_tensor(graph_embed) else np.asarray(graph_embed[self.pos])
        else:
            ge = np.zeros(self.embed_dim)
        return np.concatenate([cheap, exp, mask_flag, budget_norm, prev_action_onehot, ge]).astype(np.float32)

    def step(self, action, graph_embed=None):
        assert not self.done
        i = self.idx[self.pos]
        true_label = self.y_binary[i]
        reward = 0.0
        info = {"telemetry_used": False, "escalated": False, "abstained": False}
        advance = True

        if action == A_TELEMETRY:
            if self.budget > 0 and not self.revealed[self.pos]:
                self.revealed[self.pos] = True
                self.budget -= 1
                reward = R_TELEMETRY_COST
                info["telemetry_used"] = True
                advance = False  # agent gets one more decision now that new info is revealed
            else:
                # Wasted request (no budget left, or already revealed): still costs, but MUST
                # advance -- otherwise a policy that keeps preferring this action never terminates
                # the episode (this caused a real MemoryError from an infinite step() loop).
                reward = R_TELEMETRY_COST
                advance = True
        elif action == A_ABSTAIN:
            reward = R_ABSTAIN
            info["abstained"] = True
        elif action == A_ESCALATE:
            reward = R_ESCALATE_BASE + R_ESCALATE_COST
            info["escalated"] = True
        elif action == A_ATTACK:
            reward = R_TP if true_label == 1 else R_FP
        elif action == A_BENIGN:
            sev = self.severity_weight[i] if isinstance(self.severity_weight, np.ndarray) else self.severity_weight
            reward = (R_FN * sev) if true_label == 1 else R_TN
        else:
            raise ValueError(action)

        info["true_label"] = int(true_label)
        info["pred_action"] = int(action)
        self.prev_action = action

        if advance:
            self.pos += 1
        if self.pos >= len(self.idx):
            self.done = True
            return self._obs(graph_embed) if not self.done else np.zeros_like(self._obs_shape_probe(graph_embed)), reward, True, info

        return self._obs(graph_embed), reward, False, info

    def _obs_shape_probe(self, graph_embed):
        # only used to produce a correctly-shaped terminal observation
        old_pos = self.pos
        self.pos = min(self.pos, len(self.idx) - 1)
        o = self._obs(graph_embed)
        self.pos = old_pos
        return o

    def obs_dim(self):
        return self.X_cheap.shape[1] + self.expensive_dim + 1 + 1 + len(ACTIONS) + self.embed_dim
