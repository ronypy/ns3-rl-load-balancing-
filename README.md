# RL-Driven Mobility Load Balancing

Deep RL agent that balances load across a 7-cell LTE hotspot scenario by adjusting per-cell TX power.

**Status: baseline MVP on a pure-Python surrogate simulator. ns-3 / ns3-ai integration is not built yet** (see `sim/README.md`).

## What exists
- `rl/sim_py.py` surrogate sim, `rl/env.py` Gymnasium env (28-dim obs, 15 actions), `rl/reward.py`
- Policies: fixed, random, rule-based, DQN, Noisy Dueling DQN (`rl/agents/`)
- `rl.train`, `rl.evaluate`, `analysis/plots.py`; tests in `tests/`

## Run
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
for c in dqn noisy_dueling_dqn; do .venv/bin/python -m rl.train rl/configs/$c.yaml; done
.venv/bin/python -m rl.evaluate && .venv/bin/python -m analysis.plots
```

## Results (surrogate sim; 5 eval seeds; mean ± 95% CI; learned = 3 train seeds x 5 eval seeds)
Raw data: `results/raw/eval_episodes.csv` (gitignored; regenerate with the commands above), summary: `results/summary.csv`.

| policy | cell-edge tput (Mbps) | mean tput (Mbps) | Jain (load) | HO/UE/min |
|---|---|---|---|---|
| fixed | 0.228 ± 0.064 | 0.371 ± 0.035 | 0.515 ± 0.028 | 0.018 ± 0.014 |
| random | 0.203 ± 0.064 | 0.366 ± 0.033 | 0.518 ± 0.014 | 0.055 ± 0.034 |
| rule | 0.149 ± 0.038 | 0.337 ± 0.028 | 0.562 ± 0.023 | 0.057 ± 0.040 |
| DQN | 0.233 ± 0.026 | 0.370 ± 0.014 | 0.509 ± 0.011 | 0.033 ± 0.008 |
| Noisy Dueling DQN | 0.229 ± 0.027 | 0.367 ± 0.013 | 0.510 ± 0.010 | 0.031 ± 0.007 |

**Note:** no learned agent beats the fixed policy, and the rule-based baseline trades
throughput for fairness. In this surrogate the power knob has little headroom (`docs/design.md`).
Ping-pong rate was 0 everywhere because few handovers occur.

## Work In-progress
Fix scenario headroom (CIO knob / load / interference), then the ns-3 scenario and ns3-ai bridge.
