"""Non-learning policies: fixed (B0), random (B1), rule-based SON heuristic (B2)."""
from __future__ import annotations

import numpy as np

N = 7


class FixedPolicy:
    def reset(self, seed=0): pass
    def act(self, obs): return 0


class RandomPolicy:
    def reset(self, seed=0): self.rng = np.random.default_rng(seed)
    def act(self, obs): return int(self.rng.integers(2 * N + 1))


class RulePolicy:
    """Most-loaded cell lowers power if load > mean + thr; least-loaded raises it if < mean - thr.
    One action per step: the larger deviation wins. Respects the power bounds."""

    def __init__(self, threshold=0.05, power_min=-6.0, power_max=3.0):
        self.thr, self.lo, self.hi = threshold, power_min, power_max

    def reset(self, seed=0): pass

    def act(self, obs):
        load = obs[:N]
        off = obs[2 * N:3 * N] * (self.hi - self.lo) + self.lo
        mean = load.mean()
        hot = int(load.argmax()); cold = int(load.argmin())
        dev_hot, dev_cold = load[hot] - mean, mean - load[cold]
        options = []
        if dev_hot > self.thr and off[hot] > self.lo + 1e-6:
            options.append((dev_hot, 1 + 2 * hot))
        if dev_cold > self.thr and off[cold] < self.hi - 1e-6:
            options.append((dev_cold, 2 + 2 * cold))
        return max(options)[1] if options else 0
