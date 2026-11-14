# AE/VAE y ancho de red en arte RGB

## Objetivo y diseño

Diagnosticar si la pérdida de esquinas y el ruido provienen principalmente de
capacidad de red, del entrenamiento variacional o de ambos. Se mantienen latente
8, dataset RGB, MSE y presupuesto; se cruzan dos factores:

| Condición | Modelo | Capas ocultas | Corridas |
|---|---|---|---|
| vae_small | VAE | [128,32] | 3 ya entrenadas |
| ae_small | AE determinista | [128,32] | 3 nuevas |
| vae_wide | VAE | [256,128] | 3 nuevas |
| ae_wide | AE determinista | [256,128] | 3 nuevas |

Son nueve entrenamientos nuevos y doce modelos evaluados. El decoder refleja
las capas del encoder. Se conserva ReLU/He, float32, salida sigmoide, MSE, Adam
0,001, batch 64, 200 épocas y semillas 7,17,27. Los VAE usan beta 0,5 con rampa
de 50 épocas. Los AE no tienen KL ni muestreo de posterior. No se modifica la
matemática de ninguno de los modelos.

El AE/VAE de igual ancho no tiene idéntica cantidad de parámetros: el VAE incluye
la cabeza de log-varianza. La diferencia de entrenamiento incluye KL, muestreo
y esa cabeza. No puede atribuirse exclusivamente a KL; para eso haría falta un
control posterior de VAE beta=0. El informe conserva el número de parámetros.

## Datos y referencias nuevas

Todos entrenan con las mismas 1.024 imágenes originales de una figura RGB 32×32.
Los parámetros de geometría y color son metadata; el modelo recibe píxeles.

Se fijan antes de entrenar dos particiones nuevas de 512 imágenes, con 128 por
familia: validación con semilla 2026111501 y final reservada con 2026111502. No
comparten imágenes exactas con entrenamiento, prueba histórica, las particiones
del estudio anterior de latentes ni entre sí. Se guardan fuera del dataset que
recibe el entrenador. La confirmación anterior de capacidad no se reutiliza para
elegir ancho o tipo de modelo.

Las configuraciones mantienen las evaluaciones históricas del ejecutor para
compatibilidad. Sus valores no eligen las variantes en este estudio.

## Métricas y contrastes prefijados

Se reutiliza el evaluador geométrico calibrado: IoU de silueta, F1 de borde a un
píxel, distancia simétrica de contorno, MSE en figura y color interior, MSE global
y salidas vacías. Los promedios se hacen por familia y entre familias, con media
y desvío entre tres semillas. Se conserva cada muestra, incluidos fallos.

La segmentación depende del contraste con el fondo; un objeto muy tenue puede
perder silueta. Una IoU alta no prueba esquinas perfectas ni clase correcta.
La calibración sobre originales idénticos debe dar IoU/F1=1 y distancia=0.

Se fijan cuatro contrastes de IoU, emparejados por semilla y referencias:

1. AE ancho − AE pequeño: efecto de ampliar el AE.
2. VAE ancho − VAE pequeño: efecto de ampliar el VAE.
3. AE pequeño − VAE pequeño: diferencia de esquema a igual ancho pequeño.
4. AE ancho − VAE ancho: diferencia de esquema a igual ancho grande.

No son pruebas de significación: tres semillas dan evidencia descriptiva, no
una ley universal. Si ampliar ayuda a ambos, respalda que la capacidad contribuía.
Si el AE reconstruye mejor a igual ancho, respalda un costo del esquema
variacional bajo este presupuesto, no un diagnóstico exclusivo de KL.

Se registra además MSE de reconstrucción de entrenamiento, para contrastarlo
con validación y evitar confundir representación y generalización.

## Generación

Se exportan 64 muestras por modelo y semilla con números normales base de semilla
1234. Son 768 muestras principales:

- VAE: muestreo del prior N(0,I).
- AE: gaussiana de covarianza completa ajustada exclusivamente a sus códigos de
  entrenamiento, sin tocar pesos ni usar referencias reservadas.

Se guardan otras 384 imágenes del AE con N(0,I) como diagnóstico suplementario.
Ese muestreo no es su procedimiento principal porque el AE no fue entrenado para
poner sus códigos en una normal estándar. La gaussiana ajustada puede atravesar
regiones con pocos códigos y tampoco garantiza imágenes válidas.

La comparación enfrenta procedimientos de generación explícitos; no establece
superioridad intrínseca de AE o VAE. Coordenadas numéricas comunes no implican
atributos latentes comunes. Se guardan todos los casos, vecinos de entrenamiento,
coordenadas y parámetros de las gaussianas. Diversidad y cercanía al entrenamiento
son diagnósticos, no certificados de novedad o calidad.

## Validación, elección y confirmación

Dentro de cada tipo se prefiere la red pequeña si su IoU media de validación
queda a no más de 0,01 de la mejor. La elección es sobre reconstrucción geométrica,
no sobre belleza o calidad generativa. Se congela en `selection.json` junto con
las fuentes y hashes antes de abrir la prueba final.

La prueba final confirma **las cuatro condiciones y los cuatro contrastes ya
prefijados**. No se agregan variantes ni se eligen hiperparámetros usando esos
resultados. Se permite confirmar todas porque el objetivo es diagnóstico factorial,
no una búsqueda abierta. Cambios de pesos, configuraciones o evaluador después de
congelar el análisis provocan un error de procedencia.

## Ejecutar y retomar

```bash
cd ~/sia/TPS_SIA_GRUPO_4/TP5 &&
source .venv/bin/activate &&
OPENBLAS_NUM_THREADS=1 python -m src.analysis.art_width --prepare &&
OPENBLAS_NUM_THREADS=1 python main.py 'configs/ej2_width_art_*.json' --jobs 3 --threads 1 &&
OPENBLAS_NUM_THREADS=1 python -m src.analysis.art_width --compare &&
OPENBLAS_NUM_THREADS=1 python -m src.analysis.art_width --final &&
OPENBLAS_NUM_THREADS=1 python -m src.analysis.art_width --report
```

Repetir reutiliza entrenamientos y evaluaciones terminadas. No usar `--force`
para retomar. Los entrenamientos incompletos reinician porque no hay checkpoints
por época. La base de latente 8 debe estar terminada; no se vuelve a entrenar.

## Salidas y presentación

- `results/art_width_data/`: protocolo y particiones nuevas.
- `results/ej2_width_art_*/`: los nueve modelos nuevos.
- `results/art_width_validation/`: métricas, familias, contrastes, galerías,
  gaussianas del AE, vecinos y selección congelada.
- `results/art_width_final/`: confirmación de las cuatro condiciones.
- `results/art_width_report/report.md`: informe reproducible.

Las comparaciones de reconstrucción muestran dos referencias por familia,
prefijadas; las compactas de generación muestran índices 0,8,...,56. Inspeccionar
también las galerías completas. Los modelos entrenados tienen recorridos GIF en
`presentation/` y se pueden abrir con el explorador existente.

La parte de sonidos sigue siendo otra aplicación del mismo esquema variacional.
No se entrenan nuevos modelos de audio, no se comparten pesos ni un latente
conjunto y no se comparan MSE RGB con MSE mel. Adaptar ancho a imágenes conserva
la coherencia conceptual. AE es un control diagnóstico, no un reemplazo automático
del VAE que pide el ejercicio 2.
