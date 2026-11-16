#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/nsynth_warmup_replication
exec 9>results/nsynth_warmup_replication/run.lock
flock -n 9 || { echo 'La réplica warmup ya está activa.'; exit 1; }
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -u -m src.analysis.nsynth_warmup_replication --jobs "${1:-6}"
