"""Minimal, dependency-free (no gymnasium/SB3) DQN trainer for the custom IDSPOMDPEnv.
Kept deliberately small and readable since every RL candidate reuses this loop with a
different encoder/adapter feeding the graph-embedding slice of the observation."""
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_env import IDSPOMDPEnv, ACTIONS


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions=len(ACTIONS), hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, min(batch_size, len(self.buf)))
        s, a, r, s2, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r, dtype=np.float32),
                np.array(s2), np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


def train_dqn(X_cheap, X_expensive, cheap_cat_ids, y_binary, embed_fn,
              n_episodes=300, window_size=256, seed=0, gamma=0.95, lr=1e-3,
              eps_start=1.0, eps_end=0.05, eps_decay_episodes=200, batch_size=128,
              target_update_every=20, device="cpu", log_every=50, embed_dim=64,
              minority_oversample_frac=None):
    """embed_fn(env) -> torch tensor (window_size, embed_dim) for the CURRENT window (env.idx),
    no_grad, frozen encoder/predictor. Works for graph embeddings (GraphSAGE) or concept vectors
    (ConceptGuard-RL) or anything else keyed to env.idx — kept generic on purpose for reuse.
    minority_oversample_frac: for severely imbalanced datasets (e.g. Bot_IoT), oversample the
    minority class into each TRAINING window so it's actually learnable; leave None (population-
    accurate sampling) for evaluation callers."""
    env = IDSPOMDPEnv(X_cheap, X_expensive, cheap_cat_ids, y_binary, window_size=window_size, seed=seed,
                       embed_dim=embed_dim, minority_oversample_frac=minority_oversample_frac)
    obs_dim = env.obs_dim()
    qnet = QNetwork(obs_dim).to(device)
    target = QNetwork(obs_dim).to(device)
    target.load_state_dict(qnet.state_dict())
    opt = torch.optim.Adam(qnet.parameters(), lr=lr)
    buffer = ReplayBuffer()

    rng = np.random.default_rng(seed)
    episode_rewards = []
    step_count = 0

    for ep in range(n_episodes):
        obs0 = env.reset()
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
                    qvals = qnet(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
                    action = int(qvals.argmax(dim=1).item())
            next_obs, reward, done, info = env.step(action, ge)
            buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_reward += reward
            step_count += 1

            if len(buffer) >= batch_size:
                s, a, r, s2, d = buffer.sample(batch_size)
                s_t = torch.tensor(s, dtype=torch.float32, device=device)
                a_t = torch.tensor(a, dtype=torch.long, device=device)
                r_t = torch.tensor(r, dtype=torch.float32, device=device)
                s2_t = torch.tensor(s2, dtype=torch.float32, device=device)
                d_t = torch.tensor(d, dtype=torch.float32, device=device)

                q_sa = qnet(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = target(s2_t).max(dim=1)[0]
                    target_q = r_t + gamma * (1 - d_t) * q_next
                loss = F.smooth_l1_loss(q_sa, target_q)
                opt.zero_grad()
                loss.backward()
                opt.step()

        episode_rewards.append(ep_reward)
        if ep % target_update_every == 0:
            target.load_state_dict(qnet.state_dict())
        if ep % log_every == 0:
            recent = np.mean(episode_rewards[-log_every:])
            print(f"  [DQN] episode {ep}/{n_episodes} eps={eps:.3f} mean_reward(last{log_every})={recent:.3f}", flush=True)

    return qnet, episode_rewards
