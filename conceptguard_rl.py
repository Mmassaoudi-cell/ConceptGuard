"""
ConceptGuard-RL: an intervenable concept-bottleneck policy for network intrusion response.

The policy acts in a partially observable five-action environment (classify benign, classify attack,
abstain, request telemetry at a cost, escalate to an analyst). Flows arrive in windows of N flows with a
telemetry budget of B requests per window. Before a request only a cheap, always-visible block is
observed. After a request a concept predictor maps the revealed telemetry to K concepts, and the policy
sees these predicted concepts only (a hard bottleneck). An analyst can replace the predicted concepts by
verified values at the same decision state.

Learning procedure
  * Squared-error regression of the action values, so each value estimates a conditional mean.
  * Full-feedback targets: the reward of every non-telemetry action is a known function of the label, so
    every replayed flow supplies a target for all four of them; the telemetry target is
    r_telemetry - lambda + max_a Q_target(o'_T, a).
  * Calibrated telemetry price lambda: dual ascent on the telemetry demand of each training episode.
  * Admissible-action masking: telemetry cannot be selected once a flow is unmasked or the budget is spent.
  * The checkpoint with the best reward on validation windows is kept.

Inputs are numeric arrays that are already scaled and encoded (see ``Flows``). The label is used only to
form reward targets during training and is never part of an observation.

Dependencies: numpy, torch, scikit-learn.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score

__all__ = ["ConceptGuardRL", "Flows", "Rewards", "ACTIONS"]

A_B, A_A, A_AB, A_T, A_E = range(5)
N_ACTIONS = 5
ACTIONS = ("classify_benign", "classify_attack", "abstain", "request_telemetry", "escalate")


@dataclass
class Rewards:
    """Reward of the environment. ``escalate`` is the net value of an analyst-resolved case."""
    tp: float = 1.0
    fp: float = -1.0
    tn: float = 0.5
    fn: float = -3.0
    abstain: float = -0.2
    telemetry: float = -0.1
    escalate: float = 0.2


@dataclass
class Flows:
    """A set of flows as numeric arrays.

    cheap      (n, d_cheap)  always-visible block (scaled / one-hot encoded)
    telemetry  (n, d_tel)    block revealed by a telemetry request
    concepts   (n, K)        standardized concept values (training target of the concept predictor and
                             the analyst-verified values used for interventions)
    labels     (n,)          1 = attack, 0 = benign
    """
    cheap: np.ndarray
    telemetry: np.ndarray
    concepts: np.ndarray
    labels: np.ndarray

    def __len__(self):
        return len(self.labels)


# --------------------------------------------------------------------------------------- networks
class _ConceptMLP(nn.Module):
    def __init__(self, d_in, k, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, k))

    def forward(self, x):
        return self.net(x)


class _QNet(nn.Module):
    def __init__(self, d, h=256, depth=2):
        super().__init__()
        layers, di = [], d
        for _ in range(depth):
            layers += [nn.Linear(di, h), nn.ReLU()]
            di = h
        layers.append(nn.Linear(di, N_ACTIONS))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _masked_argmax(q, telemetry_allowed):
    q = q.clone()
    q[~telemetry_allowed, A_T] = -1e9
    return q.argmax(1)


# --------------------------------------------------------------------------------------- environment
class _Window:
    """Windows of ``window`` flows drawn without replacement.

    Observation: [cheap | m | b/B | t/N | one-hot(previous action) | m * concepts].
    """

    def __init__(self, flows, concepts_pred, rewards, window, seed, budget, mix_prob=0.0, observed=False):
        self.f, self.cp, self.R = flows, concepts_pred, rewards
        self.N, self.B = window, budget
        self.rng = np.random.default_rng(seed)
        self.mix_prob, self.observed = mix_prob, observed
        self.K = flows.concepts.shape[1]
        self.i_m = flows.cheap.shape[1]                  # index of the unmask indicator
        self.d_obs = self.i_m + 3 + N_ACTIONS + self.K
        self.records = []

    def reset(self):
        self.idx = self.rng.choice(len(self.f), size=self.N, replace=False)
        self.pos, self.budget, self.prev = 0, self.B, -1
        self.m = np.zeros(self.N, bool)
        if self.observed:
            self.wc = self.f.concepts[self.idx]
        else:
            self.wc = self.cp[self.idx].copy()
            if self.mix_prob > 0:
                sw = self.rng.random(self.N) < self.mix_prob
                self.wc[sw] = self.f.concepts[self.idx][sw]
        self._tel = False
        return self.obs()

    def build(self, i, m, budget, pos, prev, concept_vec):
        onehot = np.zeros(N_ACTIONS, np.float32)
        if prev >= 0:
            onehot[prev] = 1
        parts = [self.f.cheap[i], np.array([m, budget / self.B, pos / self.N], np.float32), onehot, concept_vec * m]
        return np.concatenate(parts).astype(np.float32)

    def obs(self):
        pos = min(self.pos, self.N - 1)
        return self.build(self.idx[pos], float(self.m[pos]), self.budget, pos, self.prev, self.wc[pos])

    def next_obs_after_telemetry(self):
        pos = self.pos
        return self.build(self.idx[pos], 1.0, max(self.budget - 1, 0), pos, A_T, self.wc[pos])

    def telemetry_allowed(self):
        return (not self.m[min(self.pos, self.N - 1)]) and self.budget > 0

    def label(self):
        return int(self.f.labels[self.idx[self.pos]])

    def step(self, a):
        i, pos, R = self.idx[self.pos], self.pos, self.R
        y = int(self.f.labels[i])
        advance = True
        if a == A_T and self.telemetry_allowed():
            r = R.telemetry
            self.m[pos] = True
            self.budget -= 1
            advance = False
            self._tel = True
        elif a == A_T:                                   # not admissible: treated as an abstention
            r, a = R.abstain, A_AB
        elif a == A_AB:
            r = R.abstain
        elif a == A_E:
            r = R.escalate
        elif a == A_A:
            r = R.tp if y == 1 else R.fp
        else:
            r = R.fn if y == 1 else R.tn
        self.prev = a
        if advance:
            self.records.append((y, a, self._tel))
            self._tel = False
            self.pos += 1
        done = self.pos >= self.N
        return (np.zeros(self.d_obs, np.float32) if done else self.obs()), r, done


def _known_rewards(R, y):
    """Reward of (classify benign, classify attack, abstain, escalate) for label y."""
    if y == 1:
        return np.array([R.fn, R.tp, R.abstain, R.escalate], np.float32)
    return np.array([R.tn, R.fp, R.abstain, R.escalate], np.float32)


def _flow_metrics(records, total_reward):
    y = np.array([r[0] for r in records])
    a = np.array([r[1] for r in records])
    tel = np.array([r[2] for r in records])
    n = len(y)
    cls = (a == A_B) | (a == A_A)
    out = {"n_flows": int(n), "coverage": float(cls.mean()), "mean_reward_per_flow": float(total_reward / n),
           "telemetry_rate": float(tel.mean()), "escalation_rate": float((a == A_E).mean()),
           "abstention_rate": float((a == A_AB).mean()), "unresolved_rate": float((a == A_T).mean())}
    if cls.sum() > 1 and len(np.unique(y[cls])) > 1:
        yp = (a[cls] == A_A).astype(int)
        out.update(macro_f1=float(f1_score(y[cls], yp, average="macro")),
                   balanced_accuracy=float(balanced_accuracy_score(y[cls], yp)),
                   attack_recall=float(recall_score(y[cls], yp, pos_label=1, zero_division=0)),
                   attack_precision=float(precision_score(y[cls], yp, pos_label=1, zero_division=0)))
    # full-coverage view: an escalated flow is resolved correctly by the analyst, an abstained or
    # unresolved flow counts as an error
    yp = np.where(a == A_A, 1, np.where(a == A_B, 0, np.where(a == A_E, y, 1 - y)))
    out["resolved_macro_f1"] = float(f1_score(y, yp, average="macro"))
    out["classified_accuracy"] = float((((a == A_A) & (y == 1)) | ((a == A_B) & (y == 0))).sum() / max(cls.sum(), 1))
    return out


# --------------------------------------------------------------------------------------- model
class ConceptGuardRL:
    """Concept-bottleneck policy for intrusion response.

    Parameters
    ----------
    rewards       environment rewards (``Rewards``)
    budget        telemetry requests allowed per window
    window        flows per window
    hidden, depth Q-network width and number of hidden layers
    mix_prob      probability that a training flow's concepts are replaced by the verified values
                  (0 trains on predicted concepts, 1 gives the independent bottleneck)
    seed          seed of initialization, exploration and training / validation windows
    """

    def __init__(self, rewards=None, budget=32, window=256, hidden=256, depth=2, mix_prob=0.0, seed=0):
        self.rewards = rewards or Rewards()
        self.budget, self.window = budget, window
        self.hidden, self.depth, self.mix_prob, self.seed = hidden, depth, mix_prob, seed
        self.concept_net = None
        self.q_net = None
        self.telemetry_price_ = 0.0          # calibrated Lagrange multiplier lambda
        self.best_val_reward_ = None
        self.history_ = {"episode_reward": [], "telemetry_price": []}

    # ------------------------------------------------------------------------------ fitting
    def fit_concepts(self, flows, epochs=30, lr=1e-3, batch_size=4096):
        """Train the concept predictor (telemetry -> concepts) with the mean-squared error."""
        torch.manual_seed(self.seed)
        net = _ConceptMLP(flows.telemetry.shape[1], flows.concepts.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        X, Y = torch.tensor(flows.telemetry), torch.tensor(flows.concepts)
        n = len(X)
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                loss = F.mse_loss(net(X[idx]), Y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.concept_net = net
        return self

    @torch.no_grad()
    def predict_concepts(self, telemetry):
        return self.concept_net(torch.tensor(telemetry)).numpy().astype(np.float32)

    def concept_quality(self, flows):
        """Coefficient of determination per concept, its mean, and the mean absolute error."""
        pred, true = self.predict_concepts(flows.telemetry), flows.concepts
        r2 = 1 - ((true - pred) ** 2).sum(0) / (((true - true.mean(0)) ** 2).sum(0) + 1e-8)
        return {"r2_mean": float(r2.mean()), "mae": float(np.abs(true - pred).mean()),
                "r2_per_concept": [float(v) for v in r2]}

    def fit(self, train, val, episodes=300, eps_decay=150, lr=1e-3, batch_size=256, eps_end=0.05,
            target_every=10, buffer_size=60000, dual_step=0.02, val_every=None, val_windows=20):
        """Fit the concept predictor and the policy.

        The telemetry price lambda is calibrated by dual ascent on the telemetry demand of every training
        episode, and the checkpoint with the best mean reward on ``val_windows`` validation windows is kept.
        """
        self.fit_concepts(train)
        cp_train, cp_val = self.predict_concepts(train.telemetry), self.predict_concepts(val.telemetry)
        env = _Window(train, cp_train, self.rewards, self.window, self.seed, self.budget, self.mix_prob)
        val_every = val_every or max(episodes // 6, 50)

        def validate(q):
            ve = _Window(val, cp_val, self.rewards, self.window, self.seed + 1, self.budget)
            return self._rollout(q, ve, val_windows)[0]["mean_reward_per_flow"]

        self._fit_policy(env, episodes, lr, batch_size, eps_end, eps_decay, target_every, buffer_size,
                         dual_step, validate, val_every)
        return self

    def _fit_policy(self, env, episodes, lr, bs, eps_end, eps_decay, tgt_every, cap, dual_lr, validate, val_every):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed + 77)
        q = _QNet(env.d_obs, self.hidden, self.depth)
        tgt = _QNet(env.d_obs, self.hidden, self.depth)
        tgt.load_state_dict(q.state_dict())
        opt = torch.optim.Adam(q.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=episodes * env.N, eta_min=lr * 0.1)
        R, d = self.rewards, env.d_obs
        S = np.zeros((cap, d), np.float32)
        ST = np.zeros_like(S)
        RW = np.zeros((cap, 4), np.float32)
        VT = np.zeros(cap, np.float32)
        ptr = size = 0
        lam, best = 0.0, (-1e9, None)
        self.history_ = {"episode_reward": [], "telemetry_price": []}
        for ep in range(episodes):
            o = env.reset()
            done = False
            total, demand = 0.0, 0
            eps = max(eps_end, 1.0 - (1.0 - eps_end) * ep / eps_decay)
            while not done:
                allowed = env.telemetry_allowed()
                with torch.no_grad():
                    qa = q(torch.from_numpy(o).unsqueeze(0))
                    if env.m[min(env.pos, env.N - 1)] == 0 and int(qa.argmax(1)) == A_T:
                        demand += 1                       # telemetry wanted, even if the budget is spent
                    if rng.random() >= eps:
                        a = int(_masked_argmax(qa, torch.tensor([allowed])))
                    else:
                        a = int(rng.integers(N_ACTIONS)) if allowed else int(rng.choice([A_B, A_A, A_AB, A_E]))
                S[ptr], RW[ptr], VT[ptr] = o, _known_rewards(R, env.label()), float(allowed)
                ST[ptr] = env.next_obs_after_telemetry() if allowed else o
                ptr = (ptr + 1) % cap
                size = min(size + 1, cap)
                o, r, done = env.step(a)
                total += r
                if size >= bs:
                    b = rng.integers(0, size, bs)
                    s, st = torch.from_numpy(S[b]), torch.from_numpy(ST[b])
                    rw, vtb = torch.from_numpy(RW[b]), torch.from_numpy(VT[b])
                    with torch.no_grad():
                        qn = tgt(st)
                        nxt = _masked_argmax(qn, (st[:, env.i_m] < 0.5) & (st[:, env.i_m + 1] > 1e-6))
                        q_tel = R.telemetry - lam + qn.gather(1, nxt.unsqueeze(1)).squeeze(1)
                    qpred = q(s)
                    targets = torch.zeros_like(qpred)
                    targets[:, [A_B, A_A, A_AB, A_E]] = rw
                    targets[:, A_T] = q_tel
                    mask = torch.ones_like(qpred)
                    mask[:, A_T] = vtb
                    loss = (F.mse_loss(qpred, targets, reduction="none") * mask).sum() / mask.sum()
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                    opt.step()
                    sched.step()
            lam = max(0.0, lam + dual_lr * (demand - env.B) / env.B)
            self.history_["episode_reward"].append(float(total))
            self.history_["telemetry_price"].append(float(lam))
            if ep % tgt_every == 0:
                tgt.load_state_dict(q.state_dict())
            if (ep + 1) % val_every == 0:
                v = validate(q)
                if v > best[0]:
                    best = (v, {k: t.clone() for k, t in q.state_dict().items()})
        if best[1] is not None:
            q.load_state_dict(best[1])
        self.q_net, self.telemetry_price_, self.best_val_reward_ = q, lam, best[0]

    # ------------------------------------------------------------------------------ scoring
    @torch.no_grad()
    def _rollout(self, q, env, n_windows, record_states=False):
        env.records.clear()
        total, states = 0.0, []
        for _ in range(n_windows):
            o = env.reset()
            done = False
            while not done:
                a = int(_masked_argmax(q(torch.from_numpy(o).unsqueeze(0)), torch.tensor([env.telemetry_allowed()])))
                if record_states:
                    states.append((o.copy(), a, env.idx[env.pos], bool(env.m[env.pos]), env.label()))
                o, r, done = env.step(a)
                total += r
        return _flow_metrics(env.records, total), states

    def score(self, flows, n_windows=50, seed=None, concepts="predicted", interventions=False,
              concept_names=None):
        """Roll the greedy policy out on windows drawn from ``flows`` and return the metrics.

        Reported: mean reward per flow, coverage, classified accuracy and macro-F1, resolved macro-F1,
        telemetry / escalation / abstention rates. ``concepts="observed"`` feeds the verified concept
        values instead of the predicted ones. With ``interventions=True`` the key ``"intervention"`` holds
        the state-aligned intervention measures (see ``intervention_scores``).
        """
        seed = self.seed + 2 if seed is None else seed
        env = _Window(flows, self.predict_concepts(flows.telemetry), self.rewards, self.window, seed, self.budget,
                      observed=(concepts == "observed"))
        metrics, states = self._rollout(self.q_net, env, n_windows, record_states=interventions)
        if interventions:
            metrics["intervention"] = self._intervention_scores(states, flows.concepts, env.i_m, concept_names)
        return metrics

    def intervention_scores(self, flows, n_windows=50, seed=None, concept_names=None):
        """State-aligned intervention measures.

        The policy is rolled out with predicted concepts. At every visited decision state the same
        Q-network is queried again on an identical observation whose concept slot holds the verified
        values. Reported on concept-visible states: action-change rate (acr), intervention-success rate
        (isr), error-correction rate, harm rate and the net change in the fraction of correct
        classifications (delta_correct_rate), for all concepts and for each concept alone.
        """
        return self.score(flows, n_windows, seed, interventions=True, concept_names=concept_names)["intervention"]

    @torch.no_grad()
    def _intervention_scores(self, states, verified, i_m, names):
        q = self.q_net
        O = np.stack([s[0] for s in states])
        a_pred = np.array([s[1] for s in states])
        fi = np.array([s[2] for s in states])
        m = np.array([s[3] for s in states])
        y = np.array([s[4] for s in states])
        D, K = O.shape[1], verified.shape[1]
        names = names or [f"concept_{k}" for k in range(K)]

        def actions(obs):
            t = torch.from_numpy(obs)
            return _masked_argmax(q(t), (t[:, i_m] < 0.5) & (t[:, i_m + 1] > 1e-6)).numpy()

        def correct(a):
            return ((a == A_A) & (y == 1)) | ((a == A_B) & (y == 0))

        def stats(a_obs, sel):
            if sel.sum() == 0:
                return None
            ap, ao = a_pred[sel], a_obs[sel]
            kp, ko = correct(a_pred)[sel], correct(a_obs)[sel]
            changed = ap != ao
            wrong = ((ap == A_A) | (ap == A_B)) & ~kp
            return {"n_states": int(sel.sum()), "acr": float(changed.mean()),
                    "isr": float((changed & ~kp & ko).sum() / max(changed.sum(), 1)),
                    "error_correction_rate": float((wrong & ko).sum() / max(wrong.sum(), 1)),
                    "n_wrong_classifications": int(wrong.sum()),
                    "harm_rate": float((kp & ~ko & ((ao == A_A) | (ao == A_B))).sum() / max(kp.sum(), 1)),
                    "delta_correct_rate": float(ko.mean() - kp.mean())}

        O_all = O.copy()
        O_all[:, D - K:] = verified[fi]
        a_obs = actions(O_all)
        out = {"all_states": stats(a_obs, np.ones(len(O), bool)), "concept_visible_states": stats(a_obs, m),
               "pre_telemetry_states": stats(a_obs, ~m), "per_concept": {}}
        for k, name in enumerate(names):
            O_k = O.copy()
            O_k[:, D - K + k] = verified[fi, k]
            out["per_concept"][name] = stats(actions(O_k), m)
        return out

    @torch.no_grad()
    def decide(self, cheap, budget_left, position, previous_action=-1, telemetry=None):
        """Action for one flow of a window: an index into ``ACTIONS``.

        ``telemetry`` is the revealed telemetry row, or None before a request. ``position`` is the flow's
        index in its window and ``previous_action`` the last action taken (-1 at the start of a window).
        """
        revealed = telemetry is not None
        concepts = (self.predict_concepts(np.asarray(telemetry, np.float32)[None])[0] if revealed
                    else np.zeros(self.concept_net.net[-1].out_features, np.float32))
        onehot = np.zeros(N_ACTIONS, np.float32)
        if previous_action >= 0:
            onehot[previous_action] = 1
        o = np.concatenate([np.asarray(cheap, np.float32),
                            np.array([float(revealed), budget_left / self.budget, position / self.window], np.float32),
                            onehot, concepts * float(revealed)]).astype(np.float32)
        allowed = (not revealed) and budget_left > 0
        return int(_masked_argmax(self.q_net(torch.from_numpy(o).unsqueeze(0)), torch.tensor([allowed])))

    @property
    def n_parameters(self):
        return {"concept_net": sum(p.numel() for p in self.concept_net.parameters()),
                "q_net": sum(p.numel() for p in self.q_net.parameters())}

    # ------------------------------------------------------------------------------ persistence
    def state_dict(self):
        return {"concept_net": self.concept_net.state_dict(), "q_net": self.q_net.state_dict(),
                "telemetry_price": self.telemetry_price_,
                "dims": {"d_tel": self.concept_net.net[0].in_features, "k": self.concept_net.net[-1].out_features,
                         "d_obs": self.q_net.net[0].in_features}}

    def load_state_dict(self, state):
        d = state["dims"]
        self.concept_net = _ConceptMLP(d["d_tel"], d["k"])
        self.q_net = _QNet(d["d_obs"], self.hidden, self.depth)
        self.concept_net.load_state_dict(state["concept_net"])
        self.q_net.load_state_dict(state["q_net"])
        self.telemetry_price_ = state["telemetry_price"]
        return self
