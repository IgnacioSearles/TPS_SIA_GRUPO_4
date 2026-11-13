# Comparación del tamaño latente: 2, 4 y 16 dimensiones

> **Estudio cerrado (13/11/2026).** Esta guía conserva el protocolo histórico.
> Ver [resultados y decisión de cierre](../reports/ej2_notas_omitidas/cierre.md).
> No se propone continuar las corridas finales.

Primera tanda para comprobar si un latente más pequeño organiza mejor la altura
musical. Se conservan KL=0,5, calentamiento de 50 épocas, MSE, capas `[512,128]`,
Adam 0,001, 200 épocas, batch 64, precisión float32 y el códec actual. No se
implementan todavía otra pérdida ni un decoder armónico: esos serían estudios
separados para poder atribuir cada cambio a una sola decisión.

## Validación y controles

Se usa el dataset microtonal de 1.720 sonidos y la exclusión estricta de ±0,75
semitonos. Las notas de validación son MIDI **51,57,63**. Las notas finales
**55,61,67**, distintas de los objetivos anteriores, se excluyen siempre del
entrenamiento y de la calibración durante la selección.

- Control completo de validación: 1.420 ejemplos; conoce las notas de validación,
  pero excluye todas las variantes y microtonos cercanos de las tres notas finales.
- Omisión de validación: 1.120 ejemplos; excluye las ventanas de ambos grupos.
- Tres dimensiones × tres semillas (7,17,27) × dos condiciones = **18 corridas**.
- Las tres envolventes y los once puntos de interpolación son los mismos del
  estudio anterior. No se consultan las notas finales al elegir dimensión.

Las configuraciones y carpetas nuevas no reemplazan los resultados anteriores.
La grilla sólo cambia `model.latent_size`. Las corridas completadas se reutilizan.
Al reducir el latente también cambia la cantidad de parámetros de sus capas de
entrada/salida; no se intenta igualar ese conteo ajustando otras capas.

Desde WSL:

```bash
cd ~/sia/TPS_SIA_GRUPO_4/TP5 &&
source .venv/bin/activate &&
OPENBLAS_NUM_THREADS=1 python main.py configs/ej2_pitch_latent_validation_*.json --jobs 3 --threads 1 &&
python -m src.analysis.pitch_latent --prepare-final
```

`--jobs 3 --threads 1` ejecuta tres corridas simultáneas. Para ejecutar en serie,
quitar ambos argumentos; no cambia el protocolo. Esta suite es más larga que el
experimento anterior: son 18 modelos, no tres, y no hay checkpoints intermedios.
No usar un comodín que mezcle configuraciones de validación y finales.

## Selección fijada antes de observar resultados

El comparador exige las 18 corridas terminadas y una calibración válida. Verifica
semillas, configuraciones guardadas, separación de datos y los 27 casos por método.
Genera `results/pitch_latent_validation_report/comparison.csv` y `selection.json`.

Se selecciona exclusivamente con la condición de omisión de validación:

1. Mayor número de puntos medios con **afinación ≤50 cents Y cercanía espectral
   al objetivo**, simultáneamente.
2. En empate, menor MSE espectral media contra el objetivo.
3. En otro empate, menor dimensión latente.

Los controles y baselines aparecen en la tabla, pero no se usa el porcentaje de
afinación aislado para elegir: la mezcla de vecinos puede falsear ese indicador.
Elegir un ganador relativo no implica que su resultado sea bueno o que supere
los baselines. Si la calibración falla o falta una corrida, no se preparan finales.

## Prueba final, después de fijar la dimensión

`--prepare-final` genera únicamente dos JSON para la dimensión seleccionada.
No lee los resultados finales ni inicia entrenamientos. Si ya hay una selección
diferente en esos archivos, falla sin sobrescribirla.

Una vez revisada la comparación, se pueden ejecutar las seis corridas finales:

```bash
OPENBLAS_NUM_THREADS=1 python main.py configs/ej2_pitch_latent_final_*.json --jobs 3 --threads 1 &&
python -m src.analysis.pitch_latent --final
```

Se entrenan desde cero con las mismas tres semillas. En esta fase se evalúan
55,61,67 y se mantienen reservadas las antiguas notas de validación 51,57,63.
El control final conoce los objetivos finales; la condición de omisión no los
conoce. Son 1.420 y 1.120 ejemplos respectivamente. La condición de omisión tiene
la misma matriz de entrenamiento que su equivalente de validación; repetirla
desde cero con igual semilla no crea una réplica estadística independiente.

El informe final queda en `results/pitch_latent_final_report`. El comparador
comprueba que se evalúe solamente la dimensión seleccionada. No comparar las
tres dimensiones con las notas finales para luego escoger la mejor: si se usan
para nuevas decisiones, dejan de ser una prueba reservada.

`reserved_targets` es una opción nueva de `dataset.pitch_study`: excluye siempre
toda su ventana, incluso con `omit_targets:false`, y no incluye esas notas en
la calibración ni en las figuras de referencia. El VAE, sus gradientes y el
entrenador conservan su implementación.
