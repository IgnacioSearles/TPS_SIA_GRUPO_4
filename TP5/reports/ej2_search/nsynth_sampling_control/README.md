# Evidencia del control de muestreo NSynth

`comparison.png` muestra los seis primeros índices de cada método, sin escoger
los mejores. `summary.csv` conserva las métricas descriptivas; cada subdirectorio
contiene CSV de los 256 casos y WAV de seis generaciones y sus vecinos pasados
por el códec. `manifest.json` identifica modelos, datos, coordenadas y código.

Las matrices completas y los doce mosaicos sin filtrar se conservan localmente
en `results/nsynth_sampling_control/`, relativo a TP5. No se incluyen todos los
mosaicos grandes en Git; se regeneran con `python -m src.analysis.nsynth_sampling`
desde las corridas completas. La copia de lectura en Windows también los conserva.
La evaluación reutiliza el informe terminado si los hashes coinciden.

Hay cuatro métodos: N(0,I) y gaussiana ajustada, para AE y para VAE. Un AE con densidad ajustada conserva los mismos pesos y objetivo. Estas distancias
no son calidad auditiva ni pruebas de novedad o generalización. Una semilla de
modelo y notas de entrenamiento: ver el informe de controles para el alcance.
