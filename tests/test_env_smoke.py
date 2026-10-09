import numpy as np

from rl.env import MLBEnv


def test_short_episode_runs():
    env = MLBEnv()
    o, _ = env.reset(seed=1)
    assert o.shape == (28,) and np.all((o >= 0) & (o <= 1))
    done, steps = False, 0
    while not done:
        o, r, term, trunc, info = env.step(env.action_space.sample())
        done = term or trunc; steps += 1
        assert np.isfinite(r)
    assert steps == env.cfg["env"]["episode_steps"]


def test_power_clipped():
    env = MLBEnv(); env.reset(seed=2)
    for _ in range(12):
        env.step(1)  # cell 0 -1 dB repeatedly
    assert env.offset[0] == env.e["power_min_db"]


def test_reproducible():
    a, b = MLBEnv(), MLBEnv()
    oa, _ = a.reset(seed=7); ob, _ = b.reset(seed=7)
    assert np.allclose(oa, ob)
    assert np.allclose(a.step(3)[0], b.step(3)[0])
