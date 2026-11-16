#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/nsynth_cnn_study
exec 9>results/nsynth_cnn_study/run.lock
flock -n 9 || { echo 'Ya hay un ejecutor NSynth CNN activo.'; exit 1; }
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -u -m src.analysis.nsynth_cnn_protocol --jobs "${1:-6}"
echo 'Terminados reconstruccion, generacion y denoising. Ver reports/nsynth_cnn_study/informe.md'
