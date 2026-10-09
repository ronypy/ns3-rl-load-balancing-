"""Fairness and reward functions (unit tested in tests/test_reward.py)."""
from __future__ import annotations

import numpy as np


def jain_index(x) -> float:
    """Jain's fairness index; 1.0 when all entries are equal (or all zero)."""
    x = np.asarray(x, dtype=float)
    s2 = float((x ** 2).sum())
    if s2 == 0.0:
        return 1.0
    return float(x.sum() ** 2 / (len(x) * s2))


def reward(ue_count, ue_tput, handovers, n_ues, tput_ref, w) -> float:
    """r = w_fair*Jain(loads) + w_mean*mean/ref + w_edge*p5/ref - w_ho*HO/UEs."""
    ue_tput = np.asarray(ue_tput, dtype=float)
    return float(
        w["w_fair"] * jain_index(ue_count)
        + w["w_mean"] * ue_tput.mean() / tput_ref
        + w["w_edge"] * np.percentile(ue_tput, 5) / tput_ref
        - w["w_ho"] * handovers / n_ues
    )
