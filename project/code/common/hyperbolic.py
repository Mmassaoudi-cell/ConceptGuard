"""
Poincare-ball hyperbolic layer for Candidate 4: RiemannGuard. Standard formulas (Nickel & Kiela
2017; Ganea et al. 2018 "Hyperbolic Neural Networks") -- not a simplification that only LOOKS
hyperbolic; expmap0/mobius_add/distance are the real Poincare-ball operators, with a learnable
curvature c > 0 (via softplus of a free parameter, so gradient descent can't push c negative or
to exactly zero) so the model can shrink toward Euclidean geometry if that fits the data better.
"""
import torch
import torch.nn as nn

EPS = 1e-5


def _clamp_norm(x, c):
    """Keeps points strictly inside the Poincare ball of curvature c (radius 1/sqrt(c))."""
    max_norm = (1 - EPS) / (c.sqrt())
    norm = x.norm(dim=-1, keepdim=True).clamp(min=EPS)
    factor = torch.where(norm > max_norm, max_norm / norm, torch.ones_like(norm))
    return x * factor


def expmap0(v, c):
    """Tangent-space-at-origin -> Poincare ball."""
    sqrt_c = c.sqrt()
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=EPS)
    x = torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)
    return _clamp_norm(x, c)


def mobius_add(x, y, c):
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c * c * x2 * y2
    return num / denom.clamp(min=EPS)


def hyperbolic_distance(x, y, c):
    sqrt_c = c.sqrt()
    diff = mobius_add(-x, y, c)
    diff_norm = diff.norm(dim=-1).clamp(max=(1 - EPS) / sqrt_c.item() if sqrt_c.numel() == 1 else None)
    return (2 / sqrt_c) * torch.atanh((sqrt_c * diff_norm).clamp(max=1 - EPS))


class HyperbolicEncoder(nn.Module):
    """Euclidean MLP feature extractor -> expmap0 into the Poincare ball. Distance-based
    classification: logits = -hyperbolic_distance(embedding, class_prototype) for each class, a
    standard hyperbolic-MLR-style head (Ganea et al. 2018)."""
    def __init__(self, in_dim, n_classes, hidden=64, embed_dim=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, embed_dim))
        self.log_c = nn.Parameter(torch.tensor(0.0))  # c = softplus(log_c) > 0, learnable curvature
        self.prototypes_tangent = nn.Parameter(torch.randn(n_classes, embed_dim) * 0.01)

    def curvature(self):
        return nn.functional.softplus(self.log_c) + 1e-3

    def embed(self, x):
        v = self.mlp(x)
        c = self.curvature()
        return expmap0(v, c), c

    def forward(self, x):
        emb, c = self.embed(x)
        protos = expmap0(self.prototypes_tangent, c)  # (K, D)
        # pairwise hyperbolic distance: (B, K)
        emb_exp = emb.unsqueeze(1).expand(-1, protos.shape[0], -1)
        protos_exp = protos.unsqueeze(0).expand(emb.shape[0], -1, -1)
        dist = hyperbolic_distance(emb_exp, protos_exp, c)
        return emb, -dist  # logits = negative distance (closer = higher score)


class EuclideanBaselineEncoder(nn.Module):
    """Ablation baseline: identical MLP capacity, plain Euclidean prototype-distance classifier
    (no curvature, no expmap) -- isolates whether hyperbolic geometry itself helps."""
    def __init__(self, in_dim, n_classes, hidden=64, embed_dim=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, embed_dim))
        self.prototypes = nn.Parameter(torch.randn(n_classes, embed_dim) * 0.01)

    def forward(self, x):
        emb = self.mlp(x)
        dist = torch.cdist(emb, self.prototypes)
        return emb, -dist
