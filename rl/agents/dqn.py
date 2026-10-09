from __future__ import annotations

import torch.nn as nn

from rl.agents.common import BaseAgent


class DQN(BaseAgent):
    def _make_net(self, obs_dim, n_actions, cfg):
        h = cfg["hidden"]
        return nn.Sequential(nn.Linear(obs_dim, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, n_actions))

    def explore(self, obs, total_steps):
        c = self.cfg
        frac = min(1.0, self.steps / (c["eps_decay_frac"] * total_steps))
        eps = c["eps_start"] + frac * (c["eps_end"] - c["eps_start"])
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_actions))
        return self.greedy(obs)
