# sim/

The ns-3 scenario (`mlb_scenario.cc`) is **not written yet** (plan Phase 1). Until then the
environment runs on the pure-Python surrogate in `rl/sim_py.py`, which is not ns-3 and makes
no claim about LTE-level fidelity (no scheduler, no real RRC/X2, interference assumes fully
loaded neighbours).
