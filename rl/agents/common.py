from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class Replay:
    def __init__(self, size, obs_dim, rng):
        self.o = np.zeros((size, obs_dim), np.float32); self.o2 = np.zeros_like(self.o)
        self.a = np.zeros(size, np.int64); self.r = np.zeros(size, np.float32); self.d = np.zeros(size, np.float32)
        self.size, self.n, self.i, self.rng = size, 0, 0, rng

    def add(self, o, a, r, o2, d):
        i = self.i
        self.o[i], self.a[i], self.r[i], self.o2[i], self.d[i] = o, a, r, o2, d
        self.i = (i + 1) % self.size; self.n = min(self.n + 1, self.size)

    def sample(self, b):
        k = self.rng.integers(0, self.n, b)
        t = lambda x: torch.as_tensor(x[k])
        return t(self.o), t(self.a), t(self.r), t(self.o2), t(self.d)


class BaseAgent:
    """Shared DQN training machinery; subclasses supply `_make_net` and exploration."""

    def __init__(self, obs_dim, n_actions, cfg, seed):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.cfg, self.n_actions = cfg, n_actions
        self.q = self._make_net(obs_dim, n_actions, cfg)
        self.q_t = self._make_net(obs_dim, n_actions, cfg)
        self.q_t.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg["lr"])
        self.buf = Replay(cfg["buffer_size"], obs_dim, self.rng)
        self.steps = 0

    def reset(self, seed=0): pass

    def greedy(self, obs):
        with torch.no_grad():
            return int(self.q(torch.as_tensor(obs).unsqueeze(0)).argmax(1).item())

    def act(self, obs):  # evaluation: deterministic
        self.q.eval()
        a = self.greedy(obs)
        self.q.train()
        return a

    def explore(self, obs, total_steps):  # training action; overridden
        return self.greedy(obs)

    def observe(self, o, a, r, o2, done):
        self.buf.add(o, a, r, o2, float(done)); self.steps += 1
        if self.buf.n < self.cfg["learn_start"]:
            return
        o, a, r, o2, d = self.buf.sample(self.cfg["batch_size"])
        with torch.no_grad():
            tgt = r + self.cfg["gamma"] * (1 - d) * self.q_t(o2).max(1).values
        q = self.q(o).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q, tgt)
        self.opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0); self.opt.step()
        tau = self.cfg["tau"]
        for p, pt in zip(self.q.parameters(), self.q_t.parameters()):
            pt.data.mul_(1 - tau).add_(tau * p.data)

    def save(self, path): torch.save(self.q.state_dict(), path)
    def load(self, path): self.q.load_state_dict(torch.load(path)); self.q.eval()
