# Control del muestreo de NSynth

## Pregunta

¿Cuánto del contraste AE/VAE al generar desde N(0,I) proviene de que esa
distribución coincide mejor con los códigos del VAE? Se agrega una tercera
alternativa: ajustar una gaussiana global a los códigos de entrenamiento del AE.

Los pesos usados para crear los audios originales de la presentación no estaban
en este checkout. Se reproducen sus configuraciones en dos corridas nuevas, con
otros nombres, sin sobrescribir los assets ni afirmar que son los mismos pesos.

## Entrenamiento y control

Dos modelos, semilla 7, NSynth MIDI 48–72, hasta 150 ejemplos por familia,
1 s, 16 kHz, STFT 1024, salto 256, 128 bandas mel. Capas [512,128], latente 16,
ReLU, float32, MSE, Adam 0,001, batch 64, 200 épocas. El VAE agrega mu/logvar,
reparametrización y KL de peso 0,5 con calentamiento de 50 épocas.

Las evaluaciones originales se desactivan en estos JSON para hacer un único
comparador común después de entrenar. Esto no cambia el entrenamiento. Se
verifica igualdad de datos, orden, arquitectura compartida y presupuesto. Se
comprueba que el hash de las notas coincida con el guardado en los checkpoints.

El evaluador usa 256 vectores normales base, semilla 2026111900:

- AE estándar: z = epsilon.
- AE ajustado: z = media + raíz(covarianza) · epsilon. Media y covarianza muestral
  completa se ajustan exclusivamente a los códigos de entrenamiento del AE.
- VAE estándar: z = epsilon.

El ajuste usa raíz simétrica por autovalores; solo recorta negativos de redondeo.
No se ajustan hiperparámetros sobre las generaciones. Los códigos de cada modelo
tienen su propio sistema de coordenadas; compartir epsilon no iguala su semántica.

## Exportaciones y alcance

`results/nsynth_sampling_control/` conserva matrices y coordenadas de los tres
métodos, media/covarianza, CSV por muestra, mosaicos de los 256 casos, comparación
de los seis primeros índices y WAV de esas seis generaciones y sus vecinos
pasados por el códec. No hay selección de los mejores audios.

Se informa distancia MSE al entrenamiento y diversidad MSE entre pares. Son
diagnósticos descriptivos sobre espectrogramas normalizados, no afinación,
calidad perceptual ni pruebas de novedad. El promedio del espectrograma no se
interpreta como energía acústica. Los WAV comparten el mismo códec con salida
normalizada por pico. No se demuestra generalización: las referencias son las
notas de entrenamiento del protocolo original y hay una sola semilla de modelo.
Una gaussiana única puede atravesar regiones con pocos códigos del AE.

Ajustar esa gaussiana no convierte al AE en VAE: conserva sus pesos y objetivo
determinista. El VAE aprende posteriores por entrada y se regulariza durante
el entrenamiento hacia un prior definido previamente.

## Comandos

Desde WSL, con ningún proceso idéntico activo:

```bash
cd ~/sia/TPS_SIA_GRUPO_4/TP5
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python main.py configs/ej2_nsynth_sampling_control_ae.json configs/ej2_nsynth_sampling_control_vae.json --jobs 2 --threads 1 --checkpoint-every 25
python -m src.analysis.nsynth_sampling
```

Repetir el primer comando después de una interrupción reanuda checkpoints y
reutiliza corridas terminadas. No usar `--force`. La evaluación solo requiere
los modelos completos; no entrena ni modifica sus pesos.
