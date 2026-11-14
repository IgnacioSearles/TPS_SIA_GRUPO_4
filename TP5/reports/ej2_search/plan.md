# Búsqueda de generación satisfactoria sin cambiar el error

## Autorización y límites

Continuar experimentando hasta obtener una muestra generativa convincente, sin
modificar las fórmulas del AE/VAE y dentro de la consigna del ejercicio 2.
Se mantiene la implementación de MSE, la reparametrización y KL. La fase
inicial usa beta=0,5 y calentamiento de 50 épocas; las extensiones que siguen
exploran explícitamente beta=0,1, 0,02 y 0,005, sin cambiar la fórmula. No se agregan pérdidas de bordes, perceptuales,
de interpolación o limpieza. No se filtran ni se reemplazan las salidas por un
renderizador. Los hashes de los tres archivos de objetivos están guardados en
`results/art_search/loss_guard.json` y se verifican al evaluar.

La consigna permite construir datos nuevos y elegir arquitectura; no exige que
las capas sean densas ni que imágenes y sonidos compartan hiperparámetros. Se
conserva el VAE como modelo principal, con encoder, posterior normal, decoder y
prior N(0,I). AE continúa como control histórico.

## Pull y revisión de trabajo remoto

Se ejecutó `git fetch origin` y `git pull --ff-only origin main`. Main ya estaba
en `fec6b30`; no había commits nuevos pendientes. Se preservó el trabajo local.

La rama `origin/otras-redes-para-1a`, en `597b555`, contiene las búsquedas de los
ejercicios 1a y 1b y figuras. Se inspeccionaron los informes de arquitectura,
profundidad, optimizadores, batch y denoising. No se mezclaron sus fuentes con
las extensiones locales ni se cambiaron pérdidas para reproducir sus ganadores.

Fuentes del propio repositorio:

- [Búsqueda 1a](https://github.com/IgnacioSearles/TPS_SIA_GRUPO_4/blob/597b555/TP5/reports/ej1_busqueda/resumen.md).
- [Búsqueda 1b](https://github.com/IgnacioSearles/TPS_SIA_GRUPO_4/blob/597b555/TP5/reports/ej1b_busqueda/resumen.md).

Hallazgos útiles y límites de transferencia:

- En 1a, ancho suficiente ayuda y la profundidad puede confundirse con capas
  demasiado angostas. Son resultados sobre memorizar las mismas 32 letras.
- Adam resultó conveniente en esa búsqueda; se conserva Adam en las nuevas.
- En 1b, eta=0,01 perjudicaba algunas redes grandes; eta de 0,001–0,003 permitió
  aprovechar capacidad. Se contempla 0,003 como candidato, sin darlo por ganador.
- Leaky ReLU y ReLU con He son alternativas ya disponibles que permiten estudiar
  activación sin cambiar la pérdida.
- Los resultados de BCE contra MSE en caracteres binarios no justifican cambiar
  nuestro error por otro; esa línea queda excluida por la instrucción del usuario.
- No se reabre el estudio cerrado de notas omitidas ni se presenta el denoising
  bayesiano como objetivo nuevo.

## Por qué una arquitectura espacial

Se añade una red convolucional NumPy con backprop manual. Se conserva la interfaz
de las redes, y las clases de AE y VAE originales calculan exactamente sus mismos
objetivos. El encoder usa dos convoluciones 3×3 de stride 2, canales 8 y 16,
y una capa densa de 128 antes del latente. El decoder aprende mapas 8×8×16 y
dos bloques de duplicación espacial + convolución, hasta emitir logits RGB.

No hay etiquetas geométricas, renderer, máscaras de verdad ni parámetros del
sintetizador dentro de la red. Compartir filtros introduce localidad, no reglas
de triángulos o rectángulos. El decoder usa resize + convolución para evitar
introducir solapamientos desiguales de deconvoluciones; esto no garantiza que
desaparezcan todos los artefactos. [Odena, Dumoulin y Olah](https://distill.pub/2016/deconv-checkerboard/).

La inferencia de monitorización se hace por lotes en modelos espaciales para
controlar memoria; el cálculo de error y los modelos MLP anteriores se conservan.
Se verificaron gradientes de convolución, resize y objetivos completos con
diferencias finitas, persistencia, float32 y equivalencia de la ruta MLP por defecto.

## Datos y búsqueda progresiva

Se generan nuevos conjuntos con semilla de entrenamiento 2026111600: 1.024 o
4.096 figuras RGB individuales. Se conservan las cuatro familias, variación de
color, tamaño, posición y orientación. La representación sigue siendo píxeles.
No se simplifica el dataset en los pilotos iniciales.

Validación externa: 512 referencias, semilla 2026111601. Confirmación final:
otras 512, semilla 2026111602. No se entregan al entrenador. El test interno de
cada configuración solo conserva las evaluaciones estándar del ejecutor; no se
usa para elegir. Al aumentar entrenamiento, cambian las actualizaciones por
época; comparar pilotos con distintas épocas no aísla causalmente solo datos.

Primeros pilotos con semilla de modelo 7:

1. Espacial, 1.024 ejemplos, 200 épocas, Adam 0,001.
2. Espacial, 4.096 ejemplos, 100 épocas, Adam 0,001.

Según los resultados se dispone de más épocas, eta=0,003, Leaky ReLU, latente 16
y control MLP con datos ampliados. No es necesario ejecutar toda la lista si un
candidato cumple; tampoco se declara satisfactorio solo por ser el mejor de una
tanda mala. Los candidatos elegidos se confirman con semillas 7,17,27.

## Criterio fijado antes de ver las generaciones nuevas

Por validación y luego por confirmación final:

- IoU macro media ≥0,92 y ninguna semilla por debajo de 0,90.
- IoU media de triángulos y de rectángulos ≥0,88, sin un caso familia/semilla
  inferior a 0,85.
- ≥90 % de generaciones limpias en promedio, ninguna semilla por debajo de 85 %.
- Revisión visual explícita del conjunto completo: al menos cinco ejemplos limpios y
  reconocibles de cada familia por semilla. Se guardan sus índices; no se eliminan
  las otras muestras de las galerías.

Son criterios del estudio, más exigentes que producir una única muestra plausible
para la consigna. No son teoremas ni certificados universales de validez.

En esta búsqueda la inspección visual la realiza el asistente y se identifica
como tal en la evidencia. No se presenta como una evaluación humana independiente.

La limpieza evalúa contraste/tamaño, conectividad, color interior, residuos
cromáticos, fondo y ancho relativo de los bordes. No participa en el gradiente.
Sus límites se calibran sobre originales válidos, antes de mirar generaciones:
fondo y residuo cromático RMSE≤0,025, interior≤0,05, componente principal≥98 %,
objeto entre 2 % y 45 % del área, contraste≥0,25 y límite de banda de borde
basado en el percentil 99 de los originales, con margen fijado. La calibración
aceptó el 100 % de las 512 referencias.

Esta métrica no distingue por sí sola un rectángulo redondeado de un círculo
válido ni prueba diversidad de clases. Por eso se exige además reconstrucción
con referencia y revisión explícita de las cuatro familias. Los ejemplos usados
como evidencia de diversidad deben pasar también la prueba de limpieza.

La generación de validación usa semilla 2026111603; la final, 2026111604. Cada
galería contiene las 64 salidas, sin filtros ni postprocesado correctivo. La
elección se congela antes de abrir la prueba final. Si la confirmación falla,
se informa el fallo y no se ajusta el candidato sobre esas mismas referencias.

## Comandos

Preparar y ejecutar los pilotos:

```bash
cd ~/sia/TPS_SIA_GRUPO_4/TP5 && source .venv/bin/activate &&
OPENBLAS_NUM_THREADS=1 python -m src.analysis.art_search --prepare &&
OPENBLAS_NUM_THREADS=1 python main.py configs/ej2_search_art_cnn1k_200.json configs/ej2_search_art_cnn4k_100.json --jobs 2 --threads 1 &&
OPENBLAS_NUM_THREADS=1 python -m src.analysis.art_search --evaluate cnn1k_200 cnn4k_100
```

Promover un candidato —por ejemplo, cnn4k_100— a tres semillas:

```bash
python -m src.analysis.art_search --promote cnn4k_100 &&
OPENBLAS_NUM_THREADS=1 python main.py configs/ej2_search_art_cnn4k_100.json --jobs 2 --threads 1 &&
OPENBLAS_NUM_THREADS=1 python -m src.analysis.art_search --evaluate cnn4k_100
```

Después de revisar las galerías y guardar índices por semilla/familia en JSON:

```bash
python -m src.analysis.art_search --freeze cnn4k_100 --coverage cobertura_validacion.json &&
OPENBLAS_NUM_THREADS=1 python -m src.analysis.art_search --final &&
python -m src.analysis.art_search --accept-final --coverage cobertura_final.json
```

`--accept-final` solo produce `satisfactory.json` si la confirmación y la evidencia
visual cumplen. No alcanza con que un candidato pase un piloto de una semilla.
Repetir los comandos reutiliza corridas terminadas. No usar `--force` para retomar.
Los entrenamientos incompletos reinician porque el ejecutor no guarda checkpoints
por época. Pilotos y confirmaciones quedan separados por conjunto de semillas.

## Ampliación adaptativa de la búsqueda

Los primeros pilotos no cumplen el criterio: CNN con 1.024 imágenes/200 épocas
alcanza IoU macro 0,892 y con 4.096/100 épocas 0,915, pero ambos obtienen 0 %
bajo el criterio estricto de limpieza. Eso no significa que todas las salidas
carezcan de forma reconocible: la mayoría falla por bordes demasiado anchos.
En las primeras 64 reconstrucciones, la mediana de banda de borde pasa de 1,97
a 1,67, frente al límite 1,20. El problema ya aparece al reconstruir referencias;
un cambio de muestreador latente por sí solo no lo resolvería.

Se agregan controles de canales/capas más anchos, latente de 16 dimensiones,
Adam a 0,003, resolución nativa de 64×64 y salida lineal. Las fórmulas de error,
beta=0,5 y calentamiento de 50 épocas siguen intactos.

La resolución mayor se renderiza directamente desde las mismas escenas
continuas: no se agrandan las predicciones. Como la MSE implementada suma los
errores de todas las características por muestra, cuadruplicar píxeles cambia
el peso relativo de reconstrucción frente a KL aunque beta siga en 0,5.
Debe declararse ese efecto; no es una comparación de igual presión efectiva
de regularización. IoU no depende de las unidades de píxel, pero la distancia
de contorno en píxeles debe interpretarse dentro de cada resolución.

La salida lineal mantiene MSE y KL. Para este control se evalúan las muestras
crudas y se registra RMS de los valores que exceden [0,1]; no hay recorte para
calcular calidad. Además de los límites comunes, se exige RMS fuera de rango
≤0,005. Las imágenes que se muestran usan el recorte habitual de RGB de la
biblioteca de gráficos, por lo que los números se calculan siempre antes de
dibujar. No se usan filtros ni umbrales para corregir salidas.

Las nuevas configuraciones se guardan con nombres diferentes y no sustituyen
ningún resultado previo. Solo un candidato que pase validación con las tres
semillas puede abrir la confirmación final reservada.

### Extensión del hiperparámetro beta

Tras los pilotos iniciales se agregan `cnn4k_beta01_100` y
`cnn4k_beta002_100`: beta 0,1 y 0,02 respectivamente, sin cambiar los archivos
que implementan el error. Ambos mantienen una KL positiva y calentamiento de
50 épocas. No se usa beta=0, pérdida geométrica ni supervisión de clases.
La fórmula sigue siendo MSE + beta·KL; se modifica el coeficiente configurable
ya existente para medir el compromiso entre nitidez y calidad de generación
desde N(0,I). Los criterios de calidad y las referencias se conservan.

El manifiesto de datos original permanece intacto: su leyenda beta=0,5
describe la fase inicial. Cada nueva evaluación registra explícitamente el
beta usado y su configuración completa. Este apartado documenta la ampliación
antes de observar sus resultados. Los candidatos nuevos usan 4.096 imágenes,
32×32 RGB, 100 épocas y la misma CNN pequeña; así se aíslan los cambios de beta.

### Refinamiento aprendido a resolución completa

`model.conv_refinement=true` agrega una convolución 3×3 con la activación oculta
después del último aumento de resolución y antes de los logits RGB. Se entrena
junto con el resto de la red, con el mismo error. No es un filtro ni un umbral
aplicado a las muestras producidas. El decoder solo recibe z: no tiene conexiones
que salteen el cuello de botella, etiquetas de figuras ni un renderizador analítico.

Se prueban `cnn4k_refine100` (beta=0,5) y `cnn4k_refine_beta01_100` (beta=0,1),
manteniendo los demás parámetros de las CNN pequeñas. Las diferencias finitas
verifican el gradiente de la arquitectura completa tanto para AE como para VAE.
La opción desactivada conserva exactamente las generaciones de los tres pilotos
ya terminados. Se archivan versiones de código y evidencia de compatibilidad
para reutilizar esas evaluaciones sin reescribir resultados ni ignorar cambios
posteriores de métricas, datos, pesos o configuración.

Las convoluciones, aumentos de resolución y gradientes están implementados
directamente en NumPy. La arquitectura aprovecha vecindades espaciales de la
imagen, pero no cambia el fundamento del VAE compartido con la parte de audio.
No se afirma que ambos modelos tengan los mismos pesos o un latente común.

### Inicialización de refinamiento y continuación de pesos

El refinamiento aleatorio se interrumpió tras registrar 50 épocas: MSE de
entrenamiento 0,00693 (beta=0,5) y 0,00671 (beta=0,1), muy por detrás de la CNN
sin esa capa. No se declaran modelos ni generaciones terminadas para esas
dos corridas. El registro está en `results/art_search/pruned_random_refinement.json`.

`cnn4k_identity100` y `cnn4k_identity_beta01_100` inicializan la nueva convolución
con su matriz identidad central y ceros alrededor. Con ReLU, sus entradas son
no negativas y la función inicial coincide exactamente con la CNN sin esa capa.
Después todos sus pesos se entrenan normalmente. Se verificó esa igualdad y
el gradiente de la arquitectura completa por diferencias finitas. Es una
inicialización, no una regla sobre la forma final de los objetos.

`cnn64_tune50` prolonga el piloto `cnn64_4k_50`: 50 épocas iniciales más 50
adicionales con Adam a 0,0003, beta=0,5 y sin repetir el calentamiento ya realizado.
La clave superior `initial_parameters` acepta una ruta con `{seed}`. Se registra
su hash en `training.json`; las salidas y pesos de la fuente quedan intactos.
Se reinician los momentos de Adam y los generadores aleatorios de la fase:
es afinación explícita, no continuación exacta de un checkpoint. Reevaluar el
modelo final no requiere cargar de nuevo el archivo de pesos iniciales.

Para repetir esa fase con tres semillas, hay que terminar primero sus fuentes:

```bash
python -m src.analysis.art_search --promote cnn64_4k_50 &&
OPENBLAS_NUM_THREADS=1 python main.py configs/ej2_search_art_cnn64_4k_50.json --jobs 2 --threads 1 &&
python -m src.analysis.art_search --promote cnn64_tune50 &&
OPENBLAS_NUM_THREADS=1 python main.py configs/ej2_search_art_cnn64_tune50.json --jobs 2 --threads 1 &&
OPENBLAS_NUM_THREADS=1 python -m src.analysis.art_search --evaluate cnn64_tune50
```

También se afina la CNN ancha de 100 épocas con beta=0,1 y Adam a 0,0003
durante 50 épocas adicionales: `cnn4k_wide_tune50`. Su control
`cnn4k_wide_identity_tune50` incorpora antes la convolución de refinamiento
identidad. La transferencia copia todos los parámetros del encoder y del
decoder original y conserva únicamente la capa nueva con sus pesos identidad.
Se verifica que no cambie la función antes de entrenar. No se congela ningún
parámetro; la nueva capa también aprende por el mismo error.

Para esa transferencia explícita, `initial_parameters` admite
`{"path": ".../{seed}/model.npz", "transfer": "identity_refinement"}`. Se rechazan
capas no identidad, activaciones incompatibles o dimensiones distintas.
Ambas variantes consumen 100+50 épocas y guardan sus resultados separados de
la fuente. El informe muestra la secuencia completa de fases.

### Ampliación de datos con presupuesto igualado

La afinación con identidad alcanza 43/64 generaciones limpias frente a 13/64
sin la capa, con IoU macro 0,931 en ambos casos. Los 21 fallos restantes de
limpieza incluyen banda de borde; solo uno incluye además el interior. Se
conserva el límite original de 1,20: no se relaja después de ver el resultado.

Desde ese mismo modelo de 100+50 épocas se preparan cuatro ramas, todas con
Adam reiniciado a 0,0003 y exactamente **3.200 actualizaciones** adicionales:

| Configuración | Imágenes | Épocas adicionales | Beta |
|---|---:|---:|---:|
| `cnn4k_refined_more50` | 4.096 | 50 | 0,1 |
| `cnn4k_refined_beta002_50` | 4.096 | 50 | 0,02 |
| `cnn8k_refined25` | 8.192 | 25 | 0,1 |
| `cnn8k_refined_beta002_25` | 8.192 | 25 | 0,02 |

Batch 64, 64×50=128×25=3.200 pasos. La segunda rama aísla beta frente a la
primera; la tercera contrasta mayor variedad de datos contra repetir los
actuales, con igual cantidad de pasos. No se cambia arquitectura ni error.
La cuarta completa el cruce 2×2 de cantidad de datos y beta, para medir su
combinación sin cambiar el presupuesto. Se agregó antes de observar las
generaciones de las primeras tres ramas.
Los nuevos datos conservan exactamente las 4.096 imágenes originales y agregan
otras 4.096, con 2.048 imágenes por familia en total.

El manifiesto de ampliación es independiente: `results/art_search/expanded_data_manifest.json`.
Se comprobó que ninguna de las 512 referencias de validación ni las 512 finales
entra en las 8.192 imágenes. El antiguo test interno de 64 imágenes del runner
sí queda incluido al extender el prefijo; ese test no se usó para seleccionar
candidatos. La selección usa exclusivamente las particiones externas comunes.
El nuevo test interno se toma después de las 8.192 imágenes.

Preparar y verificar esa ampliación: `python -m src.data.art_search_expanded`.
El evaluador comprueba su hash antes de usar los datos ampliados. La banda de
borde mide ancho en píxeles implícitamente: a mayor resolución, el mismo límite
exige detalle más fino en unidades de la escena. No se interpreta como una
métrica perceptual invariante a resolución.

Desde `cnn4k_refined_beta002_50` (55/64 limpias, IoU 0,933) se agregan dos
afinaciones de 25 épocas, 1.600 actualizaciones y Adam a 0,0003:
`cnn4k_lowbeta_more25` conserva beta=0,02;
`cnn4k_lowbeta0005_25` prueba beta=0,005. Ambas conservan datos y arquitectura,
parten de los mismos pesos y mantienen KL positiva. Comparan más convergencia
contra una reducción adicional del coeficiente existente; no se modifica la
fórmula ni se añade un criterio perceptual a la pérdida.

### Auditoría y última tanda

Una auditoría adicional usa 256 coordenadas de N(0,I), semilla 2026111800,
declaradas en `audit_protocol.json`. No sustituye las 64 muestras principales,
no modifica sus criterios ni utiliza la prueba final. Exporta las 256 salidas
sin filtros, CSV y un intervalo de muestreo condicional al modelo. El intervalo
no incorpora la selección de hiperparámetros ni certifica validez semántica.
Comando: `python -m src.analysis.art_search_audit <candidato>`.

El piloto `cnn4k_lowbeta_more25` pasó la prueba principal con 59/64, pero la
auditoría obtuvo 223/256 (87,1 %). Por eso no se consideró confirmado.
La última tanda compara:

- `cnn4k_lowbeta_more50`: 50 épocas más desde el modelo de 200 épocas,
  con beta=0,02 y la arquitectura existente.
- `cnn4k_lowbeta_dense50`: misma fuente, datos, beta, tasa y épocas; agrega
  una capa densa ReLU 256→256 antes de proyectar a características espaciales.
  Se inserta como identidad y se copian los demás pesos. Se verifica que la
  función y los gradientes de parámetros anteriores coincidan inicialmente.
- `cnn8k_lowbeta_more25`: 25 épocas más sobre 8.192 imágenes desde la rama
  equivalente de 175 épocas. A y C terminan con igual cantidad total de
  actualizaciones, aunque sus épocas y distribución de reinicios difieren.

La capa identidad utiliza el principio de transformación que conserva la función
de [Net2DeeperNet, Chen, Goodfellow y Shlens (ICLR 2016)](https://arxiv.org/abs/1511.05641).
Aquí se aplica al decoder y todos los pesos se entrenan con la misma MSE+beta·KL.
No se reproducen los resultados del artículo. La inserción densa se solicita con
`initial_parameters.transfer="identity_dense"` y `layer_index=2`.

**Historial de ejecución:** el 13 de noviembre se pidió detener la búsqueda
tras esa tanda y luego pausar para reiniciar. El 14 de noviembre el usuario
autorizó reanudar y seguir iterando. Se retomaron las fases interrumpidas desde
sus padres guardados; ese reinicio perdió el avance parcial porque todavía no
había checkpoints. Los resultados anteriores se conservaron. Véase la
[bitácora](bitacora.md) para decisiones y estado actual.


## Segunda capa de refinamiento y checkpoints completos

`cnn4k_lowbeta_conv2_50` parte de `cnn4k_refined_beta002_50` y agrega una
segunda convolución 3×3 de 16 canales antes de RGB, inicializada como identidad.
Frente a `cnn4k_lowbeta_more50` conserva fuente, datos, beta=0,02, tasa 0,0003,
50 épocas y 3.200 pasos. Todos los pesos se entrenan; solo suma 2.320 parámetros.
Se verificó igualdad inicial de salidas y gradientes de los parámetros heredados.
Las semillas 17 y 27 de la fuente de 200 épocas también terminaron de entrenar;
su evaluación conjunta y las réplicas del candidato final aún están pendientes
en la primera entrada de la bitácora.

Para evitar perder fases enteras se agregó `--checkpoint-every 10` al ejecutor.
Guarda pesos, estado del optimizador, generadores aleatorios, historial y época
mediante reemplazo atómico. Repetir el MISMO comando con esa opción reanuda
desde el último checkpoint; no usar `--force`. Como máximo se pierden las
épocas posteriores al último guardado. El checkpoint exige los mismos datos,
configuración, código de entrenamiento, versión de NumPy y opciones de hilos.
Las corridas con `summary.json` se reutilizan. `--evaluate-only` requiere un
modelo terminado; no entrena ni completa checkpoints parciales.

Ejemplo desde WSL, para un candidato ya preparado y con su modelo padre:

```bash
cd ~/sia/TPS_SIA_GRUPO_4/TP5
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python main.py configs/ej2_search_art_cnn4k_lowbeta_conv2_50.json --checkpoint-every 10
```

`initial_parameters` entre fases carga pesos y reinicia Adam/RNG; es una nueva
afinación. Un checkpoint dentro de una fase restaura también Adam/RNG y sí es
una continuación exacta, comprobada contra entrenamientos ininterrumpidos.

Desde un clon nuevo, ejecutar primero `python -m src.analysis.art_search --prepare`
para crear referencias y configuraciones. La protección de pérdidas tiene una
copia versionada en `reports/ej2_search/loss_guard.json`; no depende de que se
copie la carpeta local `results/`. Los padres de una afinación deben entrenarse
en orden: `cnn4k_wide100`, `cnn4k_wide_identity_tune50`,
`cnn4k_refined_beta002_50` y finalmente el candidato. No ejecutar indiscriminadamente
todos los JSON de búsqueda: varios son alternativas preparadas y no ejecutadas.
