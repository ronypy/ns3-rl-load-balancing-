"""Evaluate all policies on held-out seeds 1..5.  Usage: python -m rl.evaluate"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from rl.agents import LEARNED, FixedPolicy, RandomPolicy, RulePolicy
from rl.env import MLBEnv, load_cfg
from rl.reward import jain_index

ROOT = Path(__file__).resolve().parent.parent
EVAL_SEEDS = [1, 2, 3, 4, 5]


def run_episode(env, policy, seed):
    o, info = env.reset(seed=seed)
    policy.reset(seed)
    n = env.cfg["sim"]["n_ues"]; steps = env.cfg["env"]["episode_steps"]
    R, tputs, loads, jain, ho, pp = 0.0, [], [], [], 0, 0
    done = False
    while not done:
        o, r, term, trunc, info = env.step(policy.act(o))
        done = term or trunc
        R += r; tputs.append(info["ue_tput"]); loads.append(info["ue_count"] / n)
        jain.append(jain_index(info["ue_count"])); ho += info["handovers"]; pp += info["pingpongs"]
    t = np.concatenate(tputs)
    row = dict(reward=R, mean_tput=t.mean(), edge_tput=np.percentile(t, 5), jain=np.mean(jain),
               ho_per_ue_min=ho / n / (steps * env.cfg["env"]["step_s"] / 60.0),
               pingpong_rate=pp / ho if ho else 0.0)
    row.update({f"load_c{i}": v for i, v in enumerate(np.mean(loads, 0))})
    return row


def policies(scenario, train_seeds):
    e = scenario["env"]
    yield "fixed", None, FixedPolicy()
    yield "random", None, RandomPolicy()
    yield "rule", None, RulePolicy(power_min=e["power_min_db"], power_max=e["power_max_db"])
    for name, cls in LEARNED.items():
        cfg = yaml.safe_load(open(ROOT / f"rl/configs/{name}.yaml"))
        for s in train_seeds:
            ck = ROOT / f"checkpoints/{name}_seed{s}.pt"
            if not ck.exists():
                continue
            ag = cls(28, 15, cfg, s); ag.load(ck)
            yield name, s, ag


def summarize(df):
    cols = ["reward", "mean_tput", "edge_tput", "jain", "ho_per_ue_min", "pingpong_rate"]
    out = []
    for name, g in df.groupby("policy", sort=False):
        d = {"policy": name, "n_runs": len(g)}
        for c in cols:
            m, n = g[c].mean(), len(g)
            ci = stats.t.ppf(0.975, n - 1) * g[c].std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
            d[c] = f"{m:.3f} ± {ci:.3f}"
        out.append(d)
    return pd.DataFrame(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--train-seeds", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--scenario", default=None)
    args = ap.parse_args()
    scenario = load_cfg(args.scenario) if args.scenario else load_cfg()
    env = MLBEnv(scenario)
    rows = []
    for name, ts, pol in policies(scenario, args.train_seeds):
        for es in EVAL_SEEDS:
            rows.append(dict(policy=name, train_seed=ts, eval_seed=es, **run_episode(env, pol, es)))
        print("evaluated", name, ts, flush=True)
    df = pd.DataFrame(rows)
    (ROOT / "results/raw").mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / "results/raw/eval_episodes.csv", index=False)
    s = summarize(df)
    s.to_csv(ROOT / "results/summary.csv", index=False)
    print(s.to_markdown(index=False) if hasattr(s, "to_markdown") else s)
