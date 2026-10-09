#!/usr/bin/env bash
# UNTESTED: written from plan.md Phase 0. Check the ns3-ai README for the ns-3 version it recommends first.
set -euo pipefail
NS3_TAG="${NS3_TAG:?set NS3_TAG to the ns-3 release ns3-ai recommends, e.g. ns-3.XX}"
PY="${PYTHON:-$(which python)}"
cd ~
[ -d ns-3-dev ] || git clone https://gitlab.com/nsnam/ns-3-dev.git
cd ns-3-dev && git checkout "$NS3_TAG"
[ -d contrib/ai ] || git clone https://github.com/hust-diangroup/ns3-ai.git contrib/ai
./ns3 configure --enable-examples -- -DPython_EXECUTABLE="$PY" -DPython3_EXECUTABLE="$PY"
./ns3 build ai
pip install -e contrib/ai/python_utils
pip install -e contrib/ai/model/gym-interface/py
