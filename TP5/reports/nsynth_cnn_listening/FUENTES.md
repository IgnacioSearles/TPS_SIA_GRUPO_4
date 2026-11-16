# Comparación de escucha NSynth

Seis notas fijadas en el manifiesto antes de observar los resultados; no son los mejores casos. Instrumentos de prueba, fuera del entrenamiento de estos modelos.

Originales: results/nsynth_cnn_study/data/example_originals.npz. Espectrogramas limpios y reconstrucciones: reports/nsynth_cnn_study/vae_{mlp,conv}/seed={7,17,27}/examples.npz. No se cargan pesos ni se reentrena. Se verifican etiquetas, orden, referencias y hashes antes de generar.

PGHI y fase aleatoria inicializan la reconstrucción de fase, seguida en ambos casos de 60 iteraciones Griffin-Lim; inversión mel 200 iteraciones. El selector se aplica al control de códec y a ambos modelos simultáneamente. Una mejora por cambiar fase no es una mejora de la CNN. La fase cambia el audio, no los espectrogramas ni la MSE. Los WAV históricos no se modifican.

WAV nuevos: audio/{pghi,random}/seed={7,17,27}/{original,codec,dense,cnn}/{label}.wav. Cada WAV es mono PCM16, 16000Hz, un segundo. Las salidas del códec se normalizan a pico1; no hay igualación de sonoridad. audio_manifest.json identifica cada recurso y el códec explícito. index.html incorpora todos los WAV y funciona sin conexión.

Regenerar: OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/build_nsynth_listening.py --jobs 2
