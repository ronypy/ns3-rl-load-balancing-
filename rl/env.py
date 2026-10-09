"""Gymnasium environment for power-based mobility load balancing.

Observation (4N=28): per cell [load fraction, cell tput / ref, power offset in [0,1], HO-out scaled].
Action (2N+1=15): 0 = no-op; 1+2i = cell i -1 dB; 2+2i = cell i +1 dB.
"""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from rl.reward import reward as reward_fn
from rl.sim_py import N_CELLS, HexLTESim

DEFAULT_CFG = Path(__file__).parent / "configs" / "scenario_default.yaml"


def load_cfg(path=DEFAULT_CFG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class MLBEnv(gym.Env):
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_cfg()
        self.e = self.cfg["env"]
        self.observation_space = spaces.Box(0.0, 1.0, (4 * N_CELLS,), np.float32)
        self.action_space = spaces.Discrete(2 * N_CELLS + 1)
        self.sim: HexLTESim | None = None
        self.offset = np.zeros(N_CELLS)
        self.t = 0

    def _obs(self, kpi):
        e = self.e
        span = e["power_max_db"] - e["power_min_db"]
        n = self.cfg["sim"]["n_ues"]
        o = np.concatenate([
            kpi["ue_count"] / n,
            kpi["cell_tput"] / e["tput_ref_mbps"],
            (self.offset - e["power_min_db"]) / span,
            kpi["ho_out"] / (e["ho_scale"] * n),
        ])
        return np.clip(o, 0.0, 1.0).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim = HexLTESim(self.cfg["sim"], seed if seed is not None else int(self.np_random.integers(1 << 31)))
        self.offset = np.zeros(N_CELLS)
        self.t = 0
        self.kpi = self.sim.step(self.e["step_s"])  # warm-up second, no action
        return self._obs(self.kpi), self._info(self.kpi, 0.0)

    def _info(self, kpi, r):
        return dict(kpi, reward=r, offset=self.offset.copy(), hot_cell=self.sim.hot_cell)

    def step(self, action: int):
        e = self.e
        if action > 0:
            cell, up = (action - 1) // 2, (action - 1) % 2
            self.offset[cell] = np.clip(self.offset[cell] + (1.0 if up else -1.0), e["power_min_db"], e["power_max_db"])
            self.sim.set_offset(self.offset)
        self.kpi = self.sim.step(e["step_s"])
        n = self.cfg["sim"]["n_ues"]
        r = reward_fn(self.kpi["ue_count"], self.kpi["ue_tput"], self.kpi["handovers"], n,
                      self.cfg["sim"]["demand_mbps"], self.cfg["reward"])
        self.t += 1
        return self._obs(self.kpi), r, False, self.t >= e["episode_steps"], self._info(self.kpi, r)
