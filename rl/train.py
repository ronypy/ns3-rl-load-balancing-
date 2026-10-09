"""Train one learned agent over several seeds.  Usage: python -m rl.train rl/configs/dqn.yaml [--episodes N]"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from rl.agents import LEARNED
from rl.env import MLBEnv, load_cfg

ROOT = Path(__file__).resolve().parent.parent


def train(cfg: dict, seed: int, scenario: dict, episodes: int):
    env = MLBEnv(scenario)
    name = cfg["agent"]
    agent = LEARNED[name](env.observation_space.shape[0], env.action_space.n, cfg, seed)
    total = episodes * scenario["env"]["episode_steps"]
    rows = []
    for ep in range(episodes):
        o, _ = env.reset(seed=10_000 + 1000 * seed + ep)  # disjoint from eval seeds 1..5
        ret, done = 0.0, False
        while not done:
            a = agent.explore(o, total)
            o2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            agent.observe(o, a, r, o2, term)
            o, ret = o2, ret + r
        rows.append(dict(agent=name, seed=seed, episode=ep, ret=ret))
        if ep % 20 == 0:
            print(f"{name} seed={seed} ep={ep} return={ret:.2f}", flush=True)
    (ROOT / "results/raw").mkdir(parents=True, exist_ok=True); (ROOT / "checkpoints").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(ROOT / f"results/raw/train_{name}_seed{seed}.csv", index=False)
    agent.save(ROOT / f"checkpoints/{name}_seed{seed}.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("--episodes", type=int); ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--scenario", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    scenario = load_cfg(args.scenario) if args.scenario else load_cfg()
    for s in args.seeds or cfg["train_seeds"]:
        train(cfg, s, scenario, args.episodes or cfg["episodes"])
