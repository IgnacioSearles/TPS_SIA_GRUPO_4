#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/nsynth_warmup_study
exec 9>results/nsynth_warmup_study/run.lock
flock -n 9 || { echo 'Ya está activa la ablación warmup.'; exit 1; }
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
nice -n 10 .venv/bin/python -u -m src.analysis.nsynth_warmup_protocol --jobs "${1:-2}"
