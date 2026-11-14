#!/usr/bin/env bash
# Replica el piloto; no abre las referencias finales ni declara éxito automático.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m src.analysis.art_search --promote cnn4k_lowbeta_conv2_50
python main.py configs/ej2_search_art_cnn4k_lowbeta_conv2_50.json --jobs 2 --threads 1 --checkpoint-every 10
python -m src.analysis.art_search --evaluate cnn4k_lowbeta_conv2_50
for seed in 7 17 27; do
    python -m src.analysis.art_search_audit cnn4k_lowbeta_conv2_50 --seed "$seed"
done
python -m src.analysis.art_search_report
