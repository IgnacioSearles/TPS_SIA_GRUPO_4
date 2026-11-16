#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/nsynth_reconstruction_replication
exec 9>results/nsynth_reconstruction_replication/run.lock
flock -n 9 || { echo 'Ya existe un ejecutor de replicas de reconstrucción.'; exit 1; }
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
nice -n 5 .venv/bin/python -u -m src.analysis.nsynth_reconstruction_replication --jobs "${1:-3}"
