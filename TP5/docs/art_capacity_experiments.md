# Capacidad del VAE RGB: ¿se conserva mejor la geometría?

## Pregunta y alcance

El VAE de cuatro dimensiones produce figuras de colores, pero algunas
reconstrucciones de triángulos y rectángulos pierden sus esquinas. Esta tanda
comprueba si ampliar exclusivamente el código latente mejora la reconstrucción
de la geometría, manteniendo el mismo procedimiento generativo.

| Variante | Latente | Capas ocultas | beta | Semillas |
|---|---:|---|---:|---|
| Base, ya entrenada | 4 | [128,32] | 0,5 | 7,17,27 |
| Nueva | 8 | [128,32] | 0,5 | 7,17,27 |
| Nueva | 16 | [128,32] | 0,5 | 7,17,27 |

Son seis entrenamientos nuevos y tres modelos reutilizados. Se conserva RGB
32×32, las 1.024 imágenes originales de entrenamiento, la distribución del
dataset, MSE, Adam 0,001, batch 64, 200 épocas y calentamiento KL de 50 épocas.
No se cambian beta, capas ocultas, pérdida ni cantidad de datos. Aumentar el
latente cambia también las dimensiones de las conexiones que entran y salen de
ese código; no es un estudio con idéntico número de parámetros.

Solo se usan figuras individuales. Las composiciones quedan fuera de esta
comparación para evitar mezclar aumentos de capacidad y de dificultad.
No se entrenan nuevos modelos de audio. Adaptar la capacidad a cada modalidad
mantiene la coherencia de la presentación: mismo VAE y objetivo generativo,
modelos independientes con presupuestos explicados.

## Validación y prueba final independientes

La prueba anterior ya fue observada. Se conserva como diagnóstico histórico,
pero no elige la nueva capacidad. Antes de entrenar se fijan otras dos particiones
de 512 imágenes, con 128 por familia:

- Validación: semilla de datos 2026111401.
- Prueba final reservada: semilla de datos 2026111402.

Se generan desde la misma distribución y se excluyen duplicados exactos con
entrenamiento, prueba anterior y entre ambas particiones. No se omiten clases
ni atributos completos: se mide generalización a nuevas realizaciones de la
misma distribución. Los parámetros originales solo se usan para evaluar.
Estas particiones se guardan fuera de los datasets enviados al entrenador.

Las configuraciones nuevas conservan la evaluación histórica sobre la prueba
anterior para compatibilidad con el ejecutor. Esas métricas no intervienen en
la elección. Los archivos originales y los pesos de la base no se sobrescriben.

## Métricas nuevas

- **IoU de silueta:** intersección sobre unión, entre 0 y 1. Más alto es mejor.
- **F1 de borde:** coincidencia de contornos con tolerancia fija de un píxel.
- **Distancia de contorno:** promedio simétrico de distancias al borde más
  cercano, en píxeles. Menor es mejor.
- **MSE en primer plano:** error RGB sobre la silueta original, sin premiar el
  acierto sobre el fondo vacío.
- **MSE de color interior:** error RGB en el interior erosionado de la figura,
  para reducir el peso de los bordes. Si no queda interior se usa la silueta.
- MSE global y cantidad de predicciones vacías como diagnósticos adicionales.

La silueta original se obtiene del renderizador a una cobertura de al menos
50 %. La predicha se segmenta por distancia RGB al fondo conocido: el umbral
es la mitad del contraste del color original, fijado antes de comparar modelos.
No se optimizan umbrales por variante. Esto permite cambios de tono con brillo
similar, pero una figura muy tenue puede perder silueta; forma y color no quedan
completamente aislados.

Se admite una tolerancia numérica de 1e-6 en el contraste para no penalizar
coberturas de exactamente 50 % por redondeo del antialias float32. Antes de
evaluar modelos se calibra con las 512 referencias originales contra sí mismas:
todas deben dar IoU/F1 de 1 y distancia de contorno cero. El diagnóstico y su
procedencia quedan en `segmentation_calibration.json`.

Una predicción vacía recibe IoU y F1 cero y una distancia igual a la diagonal de
la imagen. Se conserva en todos los promedios. Los errores no desaparecen por
carecer de contorno.

Se promedia primero dentro de cada familia y después entre las cuatro familias.
Se reportan medias y desvíos entre las tres semillas. Una IoU alta no certifica
por sí sola preservación de esquinas o de la categoría geométrica; se acompañan
los números con reconstrucciones prefijadas y resultados por familia.

## Elección fijada antes del entrenamiento

Se elige **el menor latente cuya IoU macro media de validación quede a no más de
0,01 del mejor**. Un punto porcentual es una tolerancia práctica de este estudio,
no un intervalo de confianza. Contornos y color son diagnósticos secundarios.

La elección se guarda en `selection.json`, junto con los hashes de todos los
modelos/configuraciones, particiones y código de métricas. Después se abre la
prueba final una sola vez para comparar la elección congelada con la base de
latente 4, siempre en las tres semillas. Si se eligió 4, se evalúan solo esos
tres modelos. Cambiar las fuentes después de elegir produce un error en vez
de reutilizar silenciosamente una selección incompatible.

El criterio selecciona **reconstrucción geométrica**, no el mejor generador en
todas las dimensiones de calidad. Se exportan 64 generaciones por modelo para
inspeccionar ruido, formas y diversidad, con vecinos de entrenamiento. Usan
prefijos de una matriz normal común de semilla 1234; eso no establece que los
ejes de modelos distintos tengan el mismo significado. Distancia al entrenamiento
y diversidad no certifican calidad ni novedad.

## Ejecutar y retomar

```bash
cd ~/sia/TPS_SIA_GRUPO_4/TP5 &&
source .venv/bin/activate &&
OPENBLAS_NUM_THREADS=1 python -m src.analysis.art_capacity --prepare &&
OPENBLAS_NUM_THREADS=1 python main.py 'configs/ej2_capacity_art_*.json' --jobs 3 --threads 1 &&
OPENBLAS_NUM_THREADS=1 python -m src.analysis.art_capacity --compare &&
OPENBLAS_NUM_THREADS=1 python -m src.analysis.art_capacity --final &&
OPENBLAS_NUM_THREADS=1 python -m src.analysis.art_capacity_report
```

Repetir el comando reutiliza modelos terminados y evaluaciones congeladas. No
usar `--force` para retomar. Una corrida interrumpida antes de guardar pesos
reinicia el entrenamiento, porque no hay checkpoints por época. La base
`configs/ej2_art_1_single.json` debe estar terminada; no se vuelve a entrenar.

## Salidas

- `results/art_capacity_data/`: protocolo, hashes y particiones independientes.
- `results/ej2_capacity_art_latent8/` y `...latent16/`: modelos e historial nuevos.
- `results/art_capacity_validation/`: CSV por muestra, familia y corrida;
  resumen de tres semillas; gráficos de geometría y familias; reconstrucciones
  y generaciones; selección congelada.
- `results/art_capacity_final/`: confirmación de la elección frente a la base,
  sin reentrenamiento ni nueva selección.
- `results/art_capacity_report/report.md`: informe reproducible de validación,
  elección y confirmación final, con cambios emparejados por semilla.

Las comparaciones compactas muestran dos referencias por familia y los índices
generados 0,8,...,56, iguales para cada variante. Las galerías completas incluyen
todos los casos, también los defectuosos. Los CSV conservan fallos y salidas vacías.

No se usa la prueba final para ajustar beta, número de épocas o arquitectura.
Si la geometría no mejora, un estudio posterior puede probar mayor ancho de red
o regularización distinta, con otra validación y una nueva prueba reservada.
Un resultado negativo sigue generando un informe válido.
