"""
Quantile-Regression DQN (Dabney et al., 2018, "Distributional RL with Quantile Regression") —
a real, standard distributional-RL method, used here for Candidate 9 (TailGuard). Action selection
uses a CVaR-style objective (mean of the worst alpha-fraction of predicted return quantiles)
instead of plain expected Q-value, so the policy is explicitly risk-averse to catastrophic missed
detections rather than merely reward-maximizing on average.
"""
import numpy as np
import torch
import torch.nn as nn

from graph_env import IDSPOMDPEnv, ACTIONS
from dqn import ReplayBuffer

N_QUANTILES = 21
TAUS = torch.tensor([(2 * i + 1) / (2 * N_QUANTILES) for i in range(N_QUANTILES)], dtype=torch.float32)


class QRNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions=len(ACTIONS), n_quantiles=N_QUANTILES, hidden=128):
        super().__init__()
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions * n_quantiles),
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(-1, self.n_actions, self.n_quantiles)


def cvar_values(quantiles, alpha=0.25):
    """quantiles: (..., n_actions, n_quantiles) sorted ascending along last dim (approx, since
    outputs aren't constrained monotonic in QR-DQN by default -> sort explicitly). Returns mean of
    the worst alpha-fraction of quantiles per action -> (..., n_actions)."""
    k = max(1, int(round(alpha * quantiles.shape[-1])))
    sorted_q, _ = torch.sort(quantiles, dim=-1)
    worst = sorted_q[..., :k]
    return worst.mean(dim=-1)


def quantile_huber_loss(pred_q, target_q, taus, kappa=1.0):
    """pred_q: (B, N) for the taken action. target_q: (B, N) from target net (next-state best action).
    Standard pairwise quantile Huber loss."""
    diff = target_q.unsqueeze(1) - pred_q.unsqueeze(2)  # (B, N_pred, N_target)
    huber = torch.where(diff.abs() <= kappa, 0.5 * diff.pow(2), kappa * (diff.abs() - 0.5 * kappa))
    tau = taus.view(1, -1, 1)
    loss = (tau - (diff.detach() < 0).float()).abs() * huber
    return loss.mean()


def train_qrdqn(X_cheap, X_expensive, cheap_cat_ids, y_binary, embed_fn,
                 n_episodes=300, window_size=256, seed=0, gamma=0.95, lr=1e-3,
                 eps_start=1.0, eps_end=0.05, eps_decay_episodes=200, batch_size=128,
                 target_update_every=20, device="cpu", log_every=50, embed_dim=64,
                 cvar_alpha=0.25, severity_weight=None):
    env = IDSPOMDPEnv(X_cheap, X_expensive, cheap_cat_ids, y_binary, window_size=window_size, seed=seed,
                       embed_dim=embed_dim, severity_weight=severity_weight)
    obs_dim = env.obs_dim()
    qnet = QRNetwork(obs_dim).to(device)
    target = QRNetwork(obs_dim).to(device)
    target.load_state_dict(qnet.state_dict())
    opt = torch.optim.Adam(qnet.parameters(), lr=lr)
    buffer = ReplayBuffer()
    taus = TAUS.to(device)

    rng = np.random.default_rng(seed)
    episode_rewards = []

    for ep in range(n_episodes):
        env.reset()
        with torch.no_grad():
            ge = embed_fn(env)
        obs = env._obs(ge)
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / eps_decay_episodes)
        ep_reward = 0.0
        done = False
        while not done:
            if rng.random() < eps:
                action = rng.integers(0, len(ACTIONS))
            else:
                with torch.no_grad():
                    q = qnet(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
                    cvar = cvar_values(q, alpha=cvar_alpha)  # (1, n_actions)
                    action = int(cvar.argmax(dim=1).item())
            next_obs, reward, done, info = env.step(action, ge)
            buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_reward += reward

            if len(buffer) >= batch_size:
                s, a, r, s2, d = buffer.sample(batch_size)
                s_t = torch.tensor(s, dtype=torch.float32, device=device)
                a_t = torch.tensor(a, dtype=torch.long, device=device)
                r_t = torch.tensor(r, dtype=torch.float32, device=device)
                s2_t = torch.tensor(s2, dtype=torch.float32, device=device)
                d_t = torch.tensor(d, dtype=torch.float32, device=device)

                pred_all = qnet(s_t)  # (B, A, N)
                pred_q = pred_all.gather(1, a_t.view(-1, 1, 1).expand(-1, 1, N_QUANTILES)).squeeze(1)

                with torch.no_grad():
                    next_q = target(s2_t)  # (B, A, N)
                    next_cvar = cvar_values(next_q, alpha=cvar_alpha)  # (B, A) -- risk-averse bootstrap
                    best_a = next_cvar.argmax(dim=1)
                    next_quantiles = next_q.gather(1, best_a.view(-1, 1, 1).expand(-1, 1, N_QUANTILES)).squeeze(1)
                    target_q = r_t.unsqueeze(1) + gamma * (1 - d_t.unsqueeze(1)) * next_quantiles

                loss = quantile_huber_loss(pred_q, target_q, taus)
                opt.zero_grad()
                loss.backward()
                opt.step()

        episode_rewards.append(ep_reward)
        if ep % target_update_every == 0:
            target.load_state_dict(qnet.state_dict())
        if ep % log_every == 0:
            recent = np.mean(episode_rewards[-log_every:])
            print(f"  [QR-DQN] episode {ep}/{n_episodes} eps={eps:.3f} mean_reward(last{log_every})={recent:.3f}", flush=True)

    return qnet, episode_rewards
