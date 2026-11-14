# Diagnóstico de la distribución de muestreo

Mismos tres VAE RGB existentes, sin entrenar ni modificar pesos o pérdidas.
Se declararon 256 puntos normales nuevos por semilla (2026112200), con índices
de entrenamiento elegidos por otra semilla fija (2026112100). Ningún ajuste usa
referencias de validación o finales. Las filas de imágenes comparten ruido base,
pero no necesariamente contenido: no es una comparación de la misma figura.

| Método | Limpias, tres semillas | MSE media al entrenamiento más próximo | Diversidad MSE |
|---|---:|---:|---:|
| Prior N(0,I) | 718/768 (93.5 %) | 0.005439 | 0.033994 |
| Gaussiana con momentos del posterior agregado | 742/768 (96.6 %) | 0.004872 | 0.031304 |
| Mezcla de posteriores de entrenamiento | 767/768 (99.9 %) | 0.000213 | 0.033870 |
| Medias posteriores: reconstrucción | 767/768 (99.9 %) | 0.000184 | 0.033885 |

## Qué significa

- El decoder puede representar figuras limpias alrededor de los códigos que el
  encoder asigna a imágenes conocidas. La mezcla obtiene una IoU media de 0,9704
  respecto de esas imágenes de entrenamiento; las medias, de 0,9737.
- Corregir momentos globales ayuda moderadamente (93,5→96,6 % de limpieza), pero
  no reproduce el resultado de usar la distribución multimodal completa.
- La mezcla y las medias están muchísimo más cerca del entrenamiento. El ruido
  posterior introduce poca variación respecto de reconstruir las mismas fuentes.
  Eso explica parte de su calidad y también limita cualquier afirmación de novedad.
- El resultado apunta a un desacople entre códigos aprendidos y muestreo desde
  el prior, además de los errores geométricos del decoder. No aísla una causa única.

**Los samplers ajustados cambian la distribución de generación en la evaluación.**
No son muestras del prior N(0,I) con el que se entrenó el VAE. No se los acepta
como una mejora del generador estándar ni como aprobación del criterio original.
Las medias son reconstrucciones, no generaciones nuevas. La limpieza no garantiza
formas inequívocas; no se modifica el criterio ni se abre la prueba final.

## Qué se probó después

Dos pilotos de 20 épocas, semilla 7, desde el mismo decoder aprendido y con igual
presupuesto: conservar beta=0,02 o subirla a 0,1, Adam 0,0001. La hipótesis es si
reforzar la regularización ayuda a generar desde el prior original. Una KL mayor
también puede empeorar reconstrucción o nitidez. El resultado se informa aparte;
no se presupone que bajar KL por entrada garantice una distribución agregada normal.

## Evidencia y límites

Se inspeccionaron las comparaciones de ocho índices prefijados por método y
semilla: la mezcla y las medias conservan círculos y polígonos más definidos en
esos ejemplos, pero dependen directamente de fuentes de entrenamiento. Los
3.072 casos completos y matrices están en `results/art_sampling_diagnostic/`.
Aquí se versionan tablas de todos los casos, metadatos y comparaciones compactas.
Son diagnósticos, no una evaluación humana ciega ni una prueba de generalización.
Los 768 ejemplos de cada método no se tratan como 768 modelos independientes.

Comando: `OPENBLAS_NUM_THREADS=1 python -m src.analysis.art_sampling_control`.
Siete pruebas de momentos, selección de componentes y entradas inválidas pasaron.
El evaluador comprueba los datos contra el checkpoint y conserva hashes de los pesos.
La presentación existente permanece intacta; su regla es únicamente agregar.
