"""Lightweight pure-Python surrogate of the 7-cell LTE hotspot scenario.

This is NOT ns-3. It reproduces the structure of the plan (hex layout, hotspot,
random-walk users, A3-style handover with hysteresis + TTT, cell breathing via
TX power) so the RL pipeline can be built and debugged quickly. The ns-3
scenario (sim/mlb_scenario.cc) is meant to replace it behind the same Env.
"""
from __future__ import annotations

import numpy as np

N_CELLS = 7


class HexLTESim:
    def __init__(self, cfg: dict, seed: int):
        self.c = cfg
        self.rng = np.random.default_rng(seed)
        isd = cfg["isd_m"]
        ang = np.deg2rad(60.0 * np.arange(6))
        self.enb = np.vstack([[0.0, 0.0], isd * np.c_[np.cos(ang), np.sin(ang)]])
        self.offset = np.zeros(N_CELLS)  # dB relative to base power
        self.noise_mw = 10 ** ((-174 + 10 * np.log10(cfg["bandwidth_mhz"] * 1e6) + 9) / 10)
        self._place_ues()
        self.rsrp = self._rsrp()
        self.serving = self.rsrp.argmax(1)
        self.cand = self.serving.copy()
        self.cnt = np.zeros(cfg["n_ues"], dtype=int)
        self.prev_cell = np.full(cfg["n_ues"], -1)
        self.last_ho_t = np.full(cfg["n_ues"], -1e9)
        self.t = 0.0

    def _place_ues(self):
        c, rng = self.c, self.rng
        n = c["n_ues"]
        n_hot = int(round(c["hotspot_frac"] * n))
        self.hot_cell = int(rng.integers(1, N_CELLS))
        centre = self.enb[self.hot_cell] + self._disk(1, 200.0)[0]
        hot = centre + self._disk(n_hot, c["hotspot_radius_m"])
        rest = self._disk(n - n_hot, c["area_radius_m"])
        self.pos = np.vstack([hot, rest])
        self.heading = rng.uniform(0, 2 * np.pi, n)
        self.speed = rng.uniform(c["speed_min"], c["speed_max"], n)
        self.shadow = rng.normal(0, c["shadow_sigma_db"], (n, N_CELLS))

    def _disk(self, n, radius):
        r = radius * np.sqrt(self.rng.uniform(size=n))
        th = self.rng.uniform(0, 2 * np.pi, n)
        return np.c_[r * np.cos(th), r * np.sin(th)]

    def set_offset(self, offset):
        self.offset = np.asarray(offset, dtype=float)

    def _rsrp(self):
        d = np.linalg.norm(self.pos[:, None, :] - self.enb[None, :, :], axis=2)
        pl = 128.1 + 37.6 * np.log10(np.maximum(d, 10.0) / 1000.0)
        return self.c["base_tx_dbm"] + self.offset[None, :] - pl - self.shadow

    def _move(self, dt):
        c = self.c
        self.heading += self.rng.normal(0, 0.6, len(self.heading))
        self.pos += dt * self.speed[:, None] * np.c_[np.cos(self.heading), np.sin(self.heading)]
        out = np.linalg.norm(self.pos, axis=1) > c["area_radius_m"]
        if out.any():  # turn back toward the centre
            self.heading[out] = np.arctan2(-self.pos[out, 1], -self.pos[out, 0]) + self.rng.normal(0, 0.3, out.sum())

    def _handover(self, ho_out):
        c = self.c
        best = self.rsrp.argmax(1)
        idx = np.arange(len(best))
        trig = (best != self.serving) & (self.rsrp[idx, best] > self.rsrp[idx, self.serving] + c["hysteresis_db"])
        same = trig & (best == self.cand)
        self.cnt = np.where(same, self.cnt + 1, np.where(trig, 1, 0))
        self.cand = np.where(trig, best, self.serving)
        go = self.cnt >= c["ttt_evals"]
        n_pp = 0
        for u in np.flatnonzero(go):
            old, new = self.serving[u], self.cand[u]
            ho_out[old] += 1
            if new == self.prev_cell[u] and self.t - self.last_ho_t[u] <= c["pingpong_window_s"]:
                n_pp += 1
            self.prev_cell[u], self.last_ho_t[u] = old, self.t
            self.serving[u] = new
        self.cnt[go] = 0
        return int(go.sum()), n_pp

    def _throughput(self):
        c = self.c
        lin = 10 ** (self.rsrp / 10.0)
        idx = np.arange(len(self.serving))
        s = lin[idx, self.serving]
        sinr = s / (lin.sum(1) - s + self.noise_mw)  # neighbours assumed fully loaded
        eff = np.clip(0.75 * np.log2(1 + sinr), 0.05, 4.4)
        rate = c["bandwidth_mhz"] * 0.75 * eff  # Mbps
        need = c["demand_mbps"] / rate
        cell_need = np.bincount(self.serving, weights=need, minlength=N_CELLS)
        scale = np.minimum(1.0, 1.0 / np.maximum(cell_need, 1e-9))[self.serving]
        return c["demand_mbps"] * scale, np.minimum(1.0, cell_need)

    def step(self, duration_s: float) -> dict:
        """Advance `duration_s` seconds; return raw KPIs (normalisation lives in the Env)."""
        dt = self.c["substep_s"]
        n_sub = int(round(duration_s / dt))
        ho_out = np.zeros(N_CELLS)
        n_ho = n_pp = 0
        tput = np.zeros(len(self.serving))
        prb = np.zeros(N_CELLS)
        for _ in range(n_sub):
            self._move(dt)
            self.t += dt
            self.rsrp = self._rsrp()
            h, p = self._handover(ho_out)
            n_ho += h
            n_pp += p
            t, u = self._throughput()
            tput += t / n_sub
            prb += u / n_sub
        counts = np.bincount(self.serving, minlength=N_CELLS)
        cell_tput = np.bincount(self.serving, weights=tput, minlength=N_CELLS)
        return dict(ue_count=counts, cell_tput=cell_tput, ho_out=ho_out, ue_tput=tput,
                    prb_util=prb, handovers=n_ho, pingpongs=n_pp)
