#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# Solo la última tanda autorizada. Reutiliza corridas completas sin --force.
# Las incompletas reinician su afinación desde los pesos de origen guardados.
python main.py \
  configs/ej2_search_art_cnn4k_lowbeta_more50.json \
  configs/ej2_search_art_cnn4k_lowbeta_dense50.json \
  configs/ej2_search_art_cnn8k_lowbeta_more25.json \
  --jobs 3 --threads 1
python -m src.analysis.art_search --evaluate \
  cnn4k_lowbeta_more50 cnn4k_lowbeta_dense50 cnn8k_lowbeta_more25
python -m src.analysis.art_search_audit \
  cnn4k_lowbeta_more50 cnn4k_lowbeta_dense50 cnn8k_lowbeta_more25
python -m src.analysis.art_search_report
