"""Noisy Dueling DQN (Wang et al. 2016 dueling heads + Fortunato et al. 2018 factorized noisy layers)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from rl.agents.common import BaseAgent


class NoisyLinear(nn.Module):
    def __init__(self, i, o, sigma_init=0.5):
        super().__init__()
        self.i, self.o = i, o
        self.mu_w = nn.Parameter(torch.empty(o, i)); self.sig_w = nn.Parameter(torch.empty(o, i))
        self.mu_b = nn.Parameter(torch.empty(o)); self.sig_b = nn.Parameter(torch.empty(o))
        self.register_buffer("eps_in", torch.zeros(i)); self.register_buffer("eps_out", torch.zeros(o))
        bound = 1 / math.sqrt(i)
        nn.init.uniform_(self.mu_w, -bound, bound); nn.init.uniform_(self.mu_b, -bound, bound)
        self.sig_w.data.fill_(sigma_init / math.sqrt(i)); self.sig_b.data.fill_(sigma_init / math.sqrt(i))
        self.resample()

    @staticmethod
    def _f(x): return x.sign() * x.abs().sqrt()

    def resample(self):
        self.eps_in = self._f(torch.randn(self.i)); self.eps_out = self._f(torch.randn(self.o))

    def forward(self, x):
        if not self.training:
            return nn.functional.linear(x, self.mu_w, self.mu_b)
        w = self.mu_w + self.sig_w * torch.outer(self.eps_out, self.eps_in)
        b = self.mu_b + self.sig_b * self.eps_out
        return nn.functional.linear(x, w, b)


class NoisyDuelingNet(nn.Module):
    def __init__(self, obs_dim, n_actions, h, sigma):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.v1, self.v2 = NoisyLinear(h, h, sigma), NoisyLinear(h, 1, sigma)
        self.a1, self.a2 = NoisyLinear(h, h, sigma), NoisyLinear(h, n_actions, sigma)

    def noisy(self): return [m for m in self.modules() if isinstance(m, NoisyLinear)]

    def forward(self, x):
        z = self.trunk(x)
        v = self.v2(torch.relu(self.v1(z))); a = self.a2(torch.relu(self.a1(z)))
        return v + a - a.mean(1, keepdim=True)


class NoisyDuelingDQN(BaseAgent):
    def _make_net(self, obs_dim, n_actions, cfg):
        return NoisyDuelingNet(obs_dim, n_actions, cfg["hidden"], cfg["sigma_init"])

    def explore(self, obs, total_steps):  # exploration comes from weight noise, no epsilon
        return self.greedy(obs)

    def observe(self, o, a, r, o2, done):
        for m in self.q.noisy() + self.q_t.noisy(): m.resample()
        super().observe(o, a, r, o2, done)
