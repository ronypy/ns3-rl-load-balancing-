import numpy as np
import pytest

from rl.reward import jain_index, reward

W = dict(w_fair=1.0, w_mean=0.5, w_edge=0.5, w_ho=0.5)


def test_jain_equal_is_one():
    assert jain_index([3, 3, 3, 3]) == pytest.approx(1.0)


def test_jain_single_loaded_cell():
    assert jain_index([1, 0, 0, 0]) == pytest.approx(0.25)


def test_jain_all_zero():
    assert jain_index(np.zeros(7)) == 1.0


def test_reward_components():
    tput = np.full(10, 0.5)
    r = reward([1, 1, 1], tput, handovers=2, n_ues=10, tput_ref=1.0, w=W)
    assert r == pytest.approx(1.0 + 0.25 + 0.25 - 0.1)


def test_reward_handover_penalty_monotone():
    t = np.full(10, 0.5)
    assert reward([1, 1], t, 0, 10, 1.0, W) > reward([1, 1], t, 5, 10, 1.0, W)
