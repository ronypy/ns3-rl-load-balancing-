#!/usr/bin/env bash
set -euo pipefail
ln -sfn "$(cd "$(dirname "$0")/.." && pwd)/sim" ~/ns-3-dev/scratch/mlb
