# Un VAE, dos aplicaciones: arte geométrico y timbres

## Pregunta que organiza la presentación

¿Puede un VAE aprender de una familia de ejemplos para reconstruir ejemplos
reservados, generar muestras plausibles y recorrer transiciones entre ellos?

Aplicamos el mismo procedimiento a dos modalidades. En imágenes, los ejemplos
varían forma, tamaño, rotación y color. En audio, varían amplitudes de seis
armónicos y envolventes, conservando una fundamental de 220 Hz. Es una analogía
entre variaciones de apariencia y timbre: no hay correspondencia matemática
entre color y armónicos ni conversión de imagen a sonido.

Se entrenan modelos separados por dataset. Comparten implementación y
configuración, no los pesos ni un espacio latente conjunto: una coordenada de
un modelo no corresponde al mismo atributo en el otro.

Los generadores sintéticos crean los datos y registran sus parámetros; esos
parámetros no se entregan al encoder, al decoder ni al entrenador. El decoder
es un MLP libre. No se agrega un renderizador de figuras o un sintetizador
musical dentro del modelo.

## Protocolo compartido

| Elemento | Ambas aplicaciones |
|---|---|
| Datos | 1.024 ejemplos de entrenamiento, 256 de prueba reservados |
| Semillas | Datos: 20261113; modelos: 7,17,27 |
| Modelo | VAE NumPy; posterior gaussiano diagonal; prior N(0,I) |
| Arquitectura | Capas ocultas [128,32], latente 4, ReLU, salida sigmoide, float32 |
| Entrenamiento | Error cuadrático + beta KL; beta=0,5, calentamiento de 50 épocas |
| Optimizador | Adam 0,001, 200 épocas, batch 64 |
| Generación | Prior N(0,I), semilla de generación 1234 |
| Interpolación para presentar | Medias posteriores de referencias reservadas; 11 puntos uniformes |

La imagen es RGB 32×32 (3.072 valores); el audio se representa con el códec mel
existente (64×64, 4.096 valores). Esto cambia las capas de entrada y salida y
el número de parámetros, no la matemática del VAE. El códec introduce pérdidas
de audio; no existe un paso equivalente de inversión mel en las imágenes.

La implementación suma el error cuadrático sobre las características y promedia
sobre el batch. El MSE informado se promedia también sobre características.
Por tanto, beta igual no implica idéntica presión relativa de KL: cambian la
dimensionalidad y la distribución de valores. No se comparan ni se promedian
errores RGB y mel para declarar una modalidad mejor que la otra.

Se conservan 64 generaciones por semilla para arte y las 16 ya exportadas por
semilla para timbres. No constituyen grupos del mismo tamaño para una comparación
estadística entre modalidades. Las tres semillas de entrenamiento son las
réplicas; las muestras generadas dentro de cada semilla no son réplicas de entrenamiento.

## Dos etapas de arte

1. **Principal:** una figura por imagen: círculo, elipse, triángulo o rectángulo.
   Tamaño, rotación y leve desplazamiento variables. Cuatro colores base se
   combinan por mezclas continuas de dos colores; fondo azul oscuro fijo.
2. **Exploratoria:** dos figuras con tamaños, colores y rotaciones independientes,
   posiciones acotadas y superposición opaca. Mayor dificultad con el mismo
   presupuesto; no se presupone que el latente de cuatro dimensiones alcance.

Las figuras se renderizan con supersampling 4x. No hay ruido añadido ni texturas.
Se excluyen duplicados exactos entre entrenamiento y prueba. Los hashes de cada
partición y los parámetros originales quedan en los manifiestos. El conjunto
de prueba se usa para evaluar, no para ajustar pesos o elegir hiperparámetros.

Las figuras anteriores en gris siguen como antecedente independiente. Usaban
BCE y beta=1: no atribuir las diferencias con esta tanda exclusivamente al color.
La comparación AE/VAE y el control del AE con gaussiana ajustada permanecen como
experimentos auxiliares; no se los presenta como pruebas de superioridad universal.

## Comandos

Vista previa de los **datos originales**, sin entrenar:

```bash
cd ~/sia/TPS_SIA_GRUPO_4/TP5 &&
source .venv/bin/activate &&
OPENBLAS_NUM_THREADS=1 python -m src.analysis.art_audio --preview
```

Entrenar la etapa principal: **tres modelos nuevos**. La etapa de timbres ya
terminada se reutiliza al preparar el informe.

```bash
cd ~/sia/TPS_SIA_GRUPO_4/TP5 &&
source .venv/bin/activate &&
OPENBLAS_NUM_THREADS=1 python main.py configs/ej2_art_1_single.json --jobs 3 --threads 1 &&
OPENBLAS_NUM_THREADS=1 python -m src.analysis.art_audio
```

Etapa exploratoria: **otros tres modelos**.

```bash
cd ~/sia/TPS_SIA_GRUPO_4/TP5 &&
source .venv/bin/activate &&
OPENBLAS_NUM_THREADS=1 python main.py configs/ej2_art_2_composition.json --jobs 3 --threads 1 &&
OPENBLAS_NUM_THREADS=1 python -m src.analysis.art_audio --compositions
```

Para correr las dos tandas seguidas:

```bash
cd ~/sia/TPS_SIA_GRUPO_4/TP5 &&
source .venv/bin/activate &&
OPENBLAS_NUM_THREADS=1 python main.py 'configs/ej2_art_*.json' --jobs 3 --threads 1 &&
OPENBLAS_NUM_THREADS=1 python -m src.analysis.art_audio --compositions
```

Repetir estos comandos reutiliza entrenamientos terminados; no usar `--force`
para retomar. Un entrenamiento interrumpido antes de guardar el modelo comienza
de cero porque no hay checkpoints por época. Para regenerar las evaluaciones
de arte con los pesos guardados, usar `main.py ... --evaluate-only`.

El exportador `art_audio` no entrena ni modifica las corridas fuente. Valida
configuración, historial completo, particiones y protocolo compartido. Exporta
recorridos del audio existente y guarda hashes de modelos, configuraciones y
evaluaciones fuente. Reutiliza exportaciones completas con la misma procedencia.

## Archivos para mostrar

En cada corrida de arte:

- `generated.png`: las 64 generaciones; `generated_nearest.png`: vecinos de entrenamiento.
- `test_reconstruction.png`: ejemplos reservados y sus reconstrucciones.
- `presentation/trajectories.png`: tres recorridos completos.
- `presentation/pair_0.gif`, `pair_1.gif`, `pair_2.gif`: recorridos animados.
- `presentation/pair_N_references.png`: originales y reconstrucciones de los extremos.
- Manifiestos, CSV y NPZ: selección de referencias, fracciones y coordenadas auditables.

Las referencias son siempre los índices reservados (0,1), (2,3), (4,5), sin
seleccionar casos según el resultado. Los extremos de los recorridos son
reconstrucciones, no los originales. El GIF recorre once puntos y vuelve por los
mismos puntos; no se generan nuevos puntos en el regreso.

`results/art_audio_presentation/` reúne arte principal y timbres; la variante
con composiciones queda en `results/art_audio_presentation_extended/`. Cada
aplicación tiene tres carpetas de semilla, galerías y reconstrucciones. Audio
incluye WAV de generaciones y recorridos por pasos, una secuencia con pausas y
referencias de los extremos originales/pasadas por el códec.

Explorador interactivo, en dos terminales, después del entrenamiento:

```bash
cd ~/sia/TPS_SIA_GRUPO_4/TP5 && source .venv/bin/activate &&
OPENBLAS_NUM_THREADS=1 python -m src.explorer results/ej2_art_1_single/base__seed=7 --port 8000 --no-browser
```

```bash
cd ~/sia/TPS_SIA_GRUPO_4/TP5 && source .venv/bin/activate &&
OPENBLAS_NUM_THREADS=1 python -m src.explorer results/ej2_new_3_timbres_vae/base__seed=7 --port 8001 --no-browser
```

Abrir `http://localhost:8000` y `http://localhost:8001`. El mismo explorador
permite muestrear del prior e interpolar; RGB se muestra en sus colores reales.
Con latente de cuatro dimensiones el mapa 2D es una proyección y la navegación
usa una aproximación para recuperar coordenadas. Los ejes no son necesariamente
atributos como color, forma o amplitud de un armónico. Los recorridos exportados
usan directamente los códigos de cuatro dimensiones y no dependen de ese mapa.

## Guion y conclusiones permitidas

1. Presentar dos familias de datos y sus variaciones, sin mostrar parámetros como entradas del VAE.
2. Mostrar encoder, distribución latente y decoder como un único procedimiento aplicable a ambas.
3. Mostrar reconstrucciones reservadas para separar capacidad de representación y generación.
4. Mostrar galerías completas, ejemplos difíciles y vecinos; escuchar generaciones y referencias de audio.
5. Mostrar los tres recorridos prefijados. Preguntar qué cambia y qué se degrada.
6. Discutir límites y, como extensión, la dificultad adicional de las composiciones.

La hipótesis es generación plausible y transiciones dentro de familias aprendidas.
No afirmamos descubrimiento de escalas, comprensión musical, pintura artística
general, aprendizaje de atributos independientes ni que interpolar sea inferir
una referencia ausente. No hay una imagen/audio correcto para cada punto intermedio.

Afinación cercana a 220 Hz no mide calidad tímbrica. Suavidad, diversidad MSE y
distancia al entrenamiento no certifican calidad ni novedad. No se define éxito
solo por un MSE bajo: hay que inspeccionar formas, colores y fallos y escuchar
audio contra originales, códec y reconstrucciones. Un resultado negativo en
composiciones sigue siendo un resultado válido del experimento.
