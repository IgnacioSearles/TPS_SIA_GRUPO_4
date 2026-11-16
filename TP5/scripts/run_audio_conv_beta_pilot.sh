#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/audio_conv_beta_pilot
exec 9>results/audio_conv_beta_pilot/run.lock
flock -n 9 || { echo 'El piloto beta ya está activo.'; exit 1; }
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -u -m src.analysis.audio_conv_beta --prepare
.venv/bin/python -u main.py 'configs/audio_conv_beta_pilot/*.json' --jobs "${1:-6}" --threads 1 --checkpoint-every 5
.venv/bin/python -u -m src.analysis.audio_conv_beta
echo 'Piloto beta terminado. Ver reports/audio_conv_beta_pilot/informe.md'
