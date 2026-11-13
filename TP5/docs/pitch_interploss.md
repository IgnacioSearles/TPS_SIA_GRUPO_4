# VAE con pérdida auxiliar de interpolación

> **Estudio cerrado (13/11/2026).** Esta guía conserva el protocolo histórico.
> Ver [resultados y decisión de cierre](../reports/ej2_notas_omitidas/cierre.md).
> No se propone continuar las corridas finales.

Esta tanda prueba si entrenar explícitamente la interpolación entre notas
conocidas mejora la generación de alturas omitidas. Es un **VAE con supervisión
auxiliar por altura musical**, no un descubrimiento completamente no supervisado:
las etiquetas MIDI se usan para construir ternas, sin entrar como condición del
encoder ni del decoder.

## Objetivo y gradientes

Para una terna conocida `(p-1, p, p+1)` del mismo instrumento, velocidad y
envolvente:

```text
z_medio = (mu(x_inferior) + mu(x_superior)) / 2
L_interpolacion = error(decoder(z_medio), x_centro)
L_total = L_reconstruccion + beta * KL + lambda * L_interpolacion
```

Se usan las medias posteriores, sin ruido en el camino auxiliar. La derivada se
propaga por el decoder y hacia ambos extremos del encoder, con factor 1/2.
No se agrega otro término KL. La log-varianza no recibe gradiente directo de
esta pérdida; las capas compartidas del encoder sí pueden cambiar.

La pérdida usa la misma escala que la reconstrucción existente: suma sobre las
características y promedio sobre muestras. Con el MSE configurado se comparan
espectrogramas mel normalizados. Un minibatch independiente de 32 ternas, con
reemplazo, se procesa por actualización del optimizador. Su RNG es independiente
del muestreo VAE y del orden de ejemplos. Se suman los dos gradientes y se hace
**una sola actualización**, no dos pasos de optimización distintos.

Con `lambda=0` no hay forward/backward auxiliar ni se sortean ternas: se conserva
exactamente la trayectoria original con igual configuración y semilla. En ese
caso la columna `interpolation` del historial contiene cero por estar desactivada;
no debe interpretarse como error real nulo sobre ternas.

## Datos y comparación

- Latente **4**, KL 0,5 con calentamiento de 50 épocas, capas `[512,128]`, Adam
  0,001, 200 épocas, batch principal 64 y float32, como en la tanda anterior.
- Lambdas **0, 0,1 y 1**, semillas **7,17,27** y controles completos/omitidos:
  **18 corridas de validación** en directorios nuevos.
- Datos microtonales y envolventes sin cambios. Objetivos de validación 51,57,63;
  notas finales 55,61,67 siempre reservadas durante esta etapa. Radio 0,75 semitonos.
- Las ternas se construyen exclusivamente con filas presentes en el entrenamiento,
  no con la lista completa de referencias. No se sintetizan centros ausentes.
- En omisión hay 1.120 sonidos y **280 ternas**; en el control hay 1.420 sonidos y
  **560 ternas**. El control puede usar centros de validación conocidos, pero no
  notas finales. Los microtonos contribuyen al ELBO; las ternas auxiliares usan
  semitonos enteros y la misma envolvente en sus tres componentes.

```json
"training": {
  "interpolation": {"weight": 0.1, "batch_size": 32}
}
```

La sección se omite para mantener el entrenamiento previo. Si se solicita,
requiere un modelo variacional, metadata `pitch_study` y al menos una terna
completa; una configuración imposible falla antes de entrenar.

## Ejecutar validación

```bash
cd ~/sia/TPS_SIA_GRUPO_4/TP5 &&
source .venv/bin/activate &&
OPENBLAS_NUM_THREADS=1 python main.py configs/ej2_pitch_interploss_validation_*.json --jobs 3 --threads 1 &&
python -m src.analysis.pitch_interploss --prepare-final
```

Son 18 modelos y los términos activos requieren más cómputo por actualización.
Para ejecutar en serie, quitar `--jobs 3 --threads 1`. Las corridas terminadas se
reutilizan; no hay checkpoints de épocas intermedias. No se reutilizan ni se
sobrescriben los pesos o resultados de las suites anteriores.

El comparador y el de tamaño latente comparten las verificaciones de protocolo.
Selecciona con la condición omitida: mayor número de aciertos conjuntos de
afinación y cercanía espectral; luego menor MSE espectral y luego menor lambda.
El porcentaje de afinación aislado no decide. Verifica también que ninguna
etiqueta del manifiesto de ternas pertenezca a referencias reservadas.

El informe queda en `results/pitch_interploss_validation_report/comparison.csv`
y `selection.json`. Elegir el mejor lambda no implica que el experimento haya
sido exitoso: comparar su acierto conjunto y MSE espectral con lambda 0 y con
la mezcla de vecinos, y revisar que no empeore la reconstrucción conocida.

## Evidencia por corrida

- `interpolation_triplets.csv`: etiquetas inferior, centro y superior, pitch,
  variante, instrumento y velocidad; permite auditar todos los objetivos usados.
- `history.csv`: `loss` total, `elbo_loss`, reconstrucción, KL, `interpolation`
  sin ponderar y `interpolation_weight`. Son promedios de los minibatches.
- `training.json`: cantidad de ternas y si la supervisión auxiliar estuvo activa.
- Se mantienen los audios, controles, calibración y métricas espectrales anteriores.

Las evaluaciones de inferencia no aplican la pérdida nueva y no entrenan. Se
pueden repetir con `--evaluate-only` usando los JSON de esta misma suite.

## Prueba final opcional

El comparador prepara únicamente las configuraciones del lambda elegido, sin
ejecutarlas ni leer resultados finales. Para correr luego las seis pruebas finales:

```bash
OPENBLAS_NUM_THREADS=1 python main.py configs/ej2_pitch_interploss_final_*.json --jobs 3 --threads 1 &&
python -m src.analysis.pitch_interploss --final
```

Se intercambian objetivos y reservas: se evalúan 55,61,67 y siguen excluidas
51,57,63. Se entrena desde cero; no se deben elegir lambdas usando estas notas.
Si se modifica el protocolo tras verlas, dejan de ser una evaluación reservada.
La intervención usa conocimientos musicales explícitos y sólo se prueba con un
timbre sintético: los resultados no demostrarían aprendizaje musical universal.
