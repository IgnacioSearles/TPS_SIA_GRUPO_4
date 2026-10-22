# Desaprendizaje antihebbiano de los cinco QR

## Pregunta e hipótesis

¿Una única red de Hebb con los cinco QR completos puede dejar de colapsar a la
mezcla y recuperar las cinco identidades si debilitamos los atractores espurios?

Se conserva el mismo preprocesamiento, dimensión (1369), diagonal cero,
actualización sincrónica y tope (100) de los experimentos existentes. No se usa
pseudoinversa ni se divide el conjunto en varias redes. Esta extensión no forma
parte de la regla de Hebb del apunte: se informa por separado.

Hipótesis primaria: el desaprendizaje mejora la recuperación exacta frente a Hebb.
Hipótesis mecanística: disminuye la atracción hacia la mezcla. La segunda no basta
para aceptar la primera: podría cambiar la mezcla por otros espurios o ciclos.

## Algoritmo implementado

1. Construir W con Hebb y diagonal cero.
2. Iniciar una recuperación de entrenamiento:
   - `directed`: desde la mezcla mayoritaria ORIGINAL, en cada intento. Es una
     intervención dirigida, no el procedimiento clásico de sueños aleatorios.
   - `dream`: desde un vector de ±1 uniforme generado por una semilla local.
3. Recuperar con la dinámica sincrónica existente. Si termina en un estado
   estable a que no es un recuerdo ni su inverso:

   `W <- W - eta/N * outer(a, a)`; luego anular la diagonal.

4. Si termina en ciclo, tope, recuerdo o inverso, no modificar W. El intento se
   registra igualmente. El método dirigido sigue el atractor actual desde la
   mezcla original; NO resta siempre el mismo vector si la dinámica ya cambió.
5. Evaluar a presupuestos prefijados; esas consultas no cambian el entrenamiento.

Un update mantiene la simetría y aumenta la energía DEL MISMO estado a en
`eta*(N-1)/2`. Penalizar a o -a da la misma actualización. No se renormalizan los
pesos ni se cambia la dinámica para eliminar ciclos. Proteger los recuerdos como
objetivos no impide que otros updates los dañen: hay que medir su estabilidad.

Referencia: Hopfield, Feinstein y Palmer (1983), ['Unlearning' has a stabilizing
effect in collective memories](https://www.its.caltech.edu/~jkenny/nb250c/papers/Hopfield-1983b.pdf).
El rechazo explícito de recuerdos y la modalidad dirigida son decisiones de este
experimento, no una reproducción literal del artículo.

## Ejecución

Desde TP4, limitar BLAS a un hilo evita sobrecostos de múltiples hilos en muchas
multiplicaciones matriz-vector pequeñas. No altera las ecuaciones del método.

```bash
OPENBLAS_NUM_THREADS=1 python3 analysis/hopfield_qr_unlearning.py --config configs/hopfield_unlearning_smoke.json --dry-run
OPENBLAS_NUM_THREADS=1 python3 analysis/hopfield_qr_unlearning.py --config configs/hopfield_unlearning_smoke.json
OPENBLAS_NUM_THREADS=1 python3 analysis/hopfield_qr_unlearning.py --config configs/hopfield_unlearning_pilot.json --dry-run
OPENBLAS_NUM_THREADS=1 python3 analysis/hopfield_qr_unlearning.py --config configs/hopfield_unlearning_pilot.json
```

Cada ejecución crea un directorio nuevo bajo `results/hopfield_unlearning/`, fuera
de Git. `--output RUTA_NUEVA` permite fijarlo; rechaza sobrescribir uno existente.
`--dry-run` valida la grilla y cuenta recuperaciones sin leer los QR ni escribir.
Se guarda progreso entre entrenamientos; no hay reanudación automática. Sólo la
presencia de `COMPLETED` confirma que finalizó toda la ejecución.

## Etapa 0: prueba de funcionamiento

Dos modos, eta=0.05, presupuestos 5 y 20 intentos, semilla de entrenamiento 0.
Consultas: cinco QR x tres niveles (0, 0.2, 0.4) x dos semillas (0, 1).
Cuatro modelos entrenados y dos controles: **180 recuperaciones** en total.
Esta muestra comprueba funcionamiento, no permite elegir un ganador definitivo.

## Etapa 1: exploración prefijada

Archivo: `configs/hopfield_unlearning_pilot.json`.

| Factor | Valores |
| --- | --- |
| Modo | dirigido, sueños aleatorios |
| eta | 0.01, 0.05, 0.1 |
| Presupuesto | 0, 10, 50, 100 intentos acumulativos |
| Semilla de entrenamiento | 0, 1, 2 sólo para sueños; dirigido es determinista |
| Ruido | 0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45 |
| Semillas de consulta | 10–14 |

Hay 36 checkpoints entrenados y dos controles, cada uno con 175 consultas:
**6650 recuperaciones**. El checkpoint 0 se evalúa una sola vez como Hebb.
Todos reciben exactamente las mismas consultas. Los checkpoints de una misma
corrida son acumulativos, no reinicios; las evaluaciones no consumen el RNG de
entrenamiento. Las semillas de ruido se construyen con la utilidad existente
`make_queries`, que incluye índice de patrón e índice de nivel.

Controles: Hebb sin modificar y vecino por Hamming sin red. Hamming selecciona el
primer original si hay empate; no tiene dinámica, energía ni motivo de parada de
Hopfield. Su `stop_reason` figura como `reference`.

## Métricas y salidas

- `config.json`: parámetros efectivos, versiones y hash de los patrones.
- `entrenamiento.csv`: intento, updates acumulados, acción, iteraciones y energía
  del objetivo antes/después. Distingue presupuesto de actualizaciones efectivas.
- `estabilidad.csv`: originales, mezcla e inversa en cada checkpoint; estabilidad,
  campos de signo incorrecto, ceros exactos, margen mínimo y energía.
- `resultados.csv`: resultado individual, origen, nivel y semilla; recuperación
  correcta sólo si es exacta Y estable, categoría, Hamming, ciclos, topes,
  iteraciones, mezcla/inversa y energía. Una coincidencia al llegar al tope NO se
  cuenta como recuperación correcta de Hopfield.
- `resumen.csv`, `resumen_por_qr.csv`: medias por modelo, checkpoint y nivel.
- `pares_hebb.csv`: victorias y derrotas frente a Hebb en consultas emparejadas.

No comparar energías absolutas de matrices distintas para decidir un ganador.
Medir tiempo por separado antes de afirmar eficiencia; aquí no se guarda tiempo
por consulta, porque el objetivo primario es la recuperación.

## Selección y confirmación sin reutilizar ruido de exploración

1. Exigir los cinco originales estables en todas las semillas de entrenamiento
   evaluadas para declarar un candidato viable. Si ninguno cumple, informar ese
   fracaso; no elegir sólo porque ahora cae menos veces en la mezcla.
2. Entre viables, maximizar la media de recuperación en niveles 0.1–0.45, con igual
   peso por nivel, QR y semilla de entrenamiento. Desempatar por menos intentos y
   menor eta. No contar las semillas como nuevos QR independientes.
3. Congelar modo, eta y presupuesto en un nuevo JSON antes de confirmar. Usar el
   mismo esquema de claves, una única combinación, checkpoints `[0, presupuesto]`,
   las mismas semillas de entrenamiento y los mismos niveles de ruido; cambiar
   SÓLO las semillas de consulta por **100–119**. Conservar nombre y hash del
   archivo elegido junto al informe. Este paso es deliberado: el runner no elige
   ni ejecuta automáticamente una confirmación después de ver sus resultados.
4. Comparar candidato, Hebb y Hamming en las mismas consultas nuevas. Informar
   tasas por nivel y QR, pares ganados/perdidos y variación entre semillas de
   entrenamiento. Para intervalos, remuestrear semillas de consulta como bloques
   manteniendo emparejados método y origen; informar la limitación de cinco QR.
5. Sólo si confirma, probar manchas/esquinas como validación adicional separada.
   Reservar nuevos lugares de daño; no cambiar el candidato después de mirarlos.

Éxito: estabilidad de los cinco originales y mejora reproducible de recuperación
con ruido nuevo. Un fracaso también es útil: indicaría que debilitar atractores
espurios con este protocolo no basta para corregir el almacenamiento correlacionado.

## Evidencia de la prueba corta

La [primera ejecución y sus resultados](../reports/ejercicio2/desaprendizaje/README.md) documentan la prueba sobre los QR reales; no equivalen a la exploración completa ni a una confirmación.

## Variante sin módulos compartidos

El runner acepta `remove_shared` (booleano, falso por defecto). Si es verdadero,
calcula la máscara sólo sobre los patrones almacenados, quita los 955 módulos
comunes y construye Hebb y desaprendizaje con N=414. Las consultas siguen
originándose sobre los QR de 1369 módulos, antes del recorte, con las mismas
semillas y niveles de ruido. Las salidas se restauran con los valores comunes
conocidos y el Hamming final se calcula sobre el QR completo.

Mezcla, inversa y clasificación se calculan en el espacio activo. La inversa del
estado activo restaurada conserva los comunes y no es el negativo del QR entero.
La estabilidad se mide sobre las neuronas activas; los comunes se fijan al
restaurar. `query_hamming` conserva el daño completo, `network_query_hamming`
registra el daño que llega a la red. El manifiesto y los CSV identifican la
variante para evitar mezclar sus resultados con los de la red completa.

```bash
OPENBLAS_NUM_THREADS=1 python3 analysis/hopfield_qr_unlearning.py --config configs/hopfield_unlearning_stripped_smoke.json
OPENBLAS_NUM_THREADS=1 python3 analysis/hopfield_qr_unlearning.py --config configs/hopfield_unlearning_stripped_pilot.json --dry-run
OPENBLAS_NUM_THREADS=1 python3 analysis/hopfield_qr_unlearning.py --config configs/hopfield_unlearning_stripped_pilot.json
```

Las configuraciones mantienen los presupuestos originales: 180 recuperaciones
para la prueba corta y 6650 para la exploración. La prueba corta y la exploración completa ya se ejecutaron. Ver [resultados sin compartidos](../reports/ejercicio2/desaprendizaje/SIN_COMPARTIDOS.md).
La confirmación debe conservar también `remove_shared: true`, además de fijar
modo/tasa/presupuesto y reservar semillas de consulta 100–119.

## Ejecución completa del 22 de octubre de 2026

Se completaron ambas exploraciones (6650 predicciones cada una), la confirmación
reducida (3500) y las pruebas espaciales del candidato confirmado (6600). La red
completa no tuvo candidatos elegibles bajo el requisito de estabilidad en todas
las semillas; se registró esa decisión y no se confirmó una mejor semilla aislada.
Ver [informe completo](../reports/ejercicio2/desaprendizaje/PRUEBAS_COMPLETAS.md).

`analysis/select_hopfield_unlearning.py` verifica una exploración finalizada,
ordena los candidatos según el criterio de arriba y guarda `seleccion.json` y un
JSON de confirmación con semillas nuevas. Rechaza sobrescribir una selección ya
congelada. `analysis/confirm_hopfield_unlearning_damage.py` verifica el hash del
candidato y el éxito de la confirmación antes de ejecutar el plan espacial fijo.
La selección y confirmación se ejecutaron secuencialmente, sin reajustar parámetros
tras observar ruido nuevo.

## Reproducción y publicación liviana

Los datos originales siguen en `data/hopfield/` y los pesos y salidas locales en
`results/`, ignorados por Git. Para auditar sin originales ni red se publica una
[copia de métricas comprimidas](../reports/ejercicio2/desaprendizaje/evidencia/README.md).
Los hashes distinguen bytes de origen y publicados; sólo se normalizan rutas en
JSON, nunca se modifican los registros CSV de origen.

Comandos desde TP4, con las dependencias de `requirements.txt` instaladas. Usar
directorios nuevos al repetir una ejecución, porque se rechaza sobrescribir:

```bash
OPENBLAS_NUM_THREADS=1 python3 analysis/hopfield_qr_unlearning.py --config configs/hopfield_unlearning_pilot.json --output results/hopfield_unlearning/repeticion_completa
OPENBLAS_NUM_THREADS=1 python3 analysis/hopfield_qr_unlearning.py --config configs/hopfield_unlearning_stripped_pilot.json --output results/hopfield_unlearning/repeticion_reducida
OPENBLAS_NUM_THREADS=1 python3 analysis/select_hopfield_unlearning.py results/hopfield_unlearning/repeticion_reducida --confirmation configs/hopfield_unlearning_confirmacion_repeticion.json
OPENBLAS_NUM_THREADS=1 python3 analysis/hopfield_qr_unlearning.py --config configs/hopfield_unlearning_confirmacion_repeticion.json --output results/hopfield_unlearning/repeticion_confirmacion
OPENBLAS_NUM_THREADS=1 python3 analysis/confirm_hopfield_unlearning_damage.py --selection results/hopfield_unlearning/repeticion_reducida/seleccion.json --confirmation-results results/hopfield_unlearning/repeticion_confirmacion --plan configs/hopfield_unlearning_damage.json --output results/hopfield_unlearning/repeticion_espacial
OPENBLAS_NUM_THREADS=1 python3 analysis/compare_hopfield_unlearning.py --confirmation-results results/hopfield_unlearning/repeticion_confirmacion --output results/hopfield_unlearning/repeticion_pseudoinversa
```

La validación espacial sólo continúa si la confirmación conserva los originales y
mejora a Hebb. El selector también registra explícitamente ausencia de candidatos.
El publicador de la tanda histórica usa una lista fija de ejecuciones, no copia
indiscriminadamente una repetición ni sus posibles archivos de red.

Para regenerar únicamente la figura e intervalos publicados, sin datos ni red:

```bash
OPENBLAS_NUM_THREADS=1 python3 reports/ejercicio2/desaprendizaje/reproducir_analisis.py --published
```

La [bitácora del ejercicio 2](../reports/ejercicio2/bitacora/README.md) separa
antecedentes, exploración, confirmación y comparación directa. La
[guía de presentación](../presentation/experimentos_hopfield_nuevos.md) fija los
nombres de los métodos y el alcance de los claims.
