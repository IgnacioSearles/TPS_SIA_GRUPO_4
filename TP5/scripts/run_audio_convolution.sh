#!/usr/bin/env bash
# Repeat this command to resume; a lock prevents concurrent duplicate launchers.
set -euo pipefail
cd "$(dirname "$0")/.."
jobs="${1:-3}"
if [[ ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
  echo 'El número de procesos debe ser positivo.' >&2
  exit 2
fi
mkdir -p results/audio_convolution
exec 9>results/audio_convolution/run.lock
if ! flock -n 9; then
  echo 'Ya hay un ejecutor de esta comparación activo. Consultar results/audio_convolution/run.log.' >&2
  exit 1
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -u -m src.analysis.audio_convolution --audit
.venv/bin/python -u main.py 'configs/audio_conv/*_complete.json' --jobs "$jobs" --threads 1 --checkpoint-every 10
.venv/bin/python -u -m src.analysis.audio_convolution
.venv/bin/python -u main.py 'configs/audio_conv/*_omitted.json' --jobs "$jobs" --threads 1 --checkpoint-every 10
.venv/bin/python -u -m src.analysis.audio_convolution
echo 'Comparación de audio terminada. Ver reports/ej2_audio_convolution/informe.md.'
