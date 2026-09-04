# TP2 — Aproximación de imágenes con algoritmos genéticos

El proyecto contiene un motor genérico de algoritmos genéticos y una
implementación concreta que aproxima una imagen mediante triángulos
semitransparentes.

## Cómo correrlo

Desde `TP2`, crear un entorno con Python compatible e instalar dependencias:

```bash
cd TP2
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Cada simulación se describe en un archivo JSON:

```bash
python main.py configs/default_config.json --image objetivo.png
```

La única clave obligatoria es `image`: todo lo demás tiene un valor por defecto y
puede omitirse. `configs/minimal_config.json` es una corrida completa válida.
También podés editar `image` directamente dentro del JSON.

La línea de comandos solo expone lo que suele cambiar entre corridas del mismo
experimento; el resto vive siempre en el archivo:

```bash
python main.py configs/default_config.json --image otra.png --output otra_out.png
python main.py configs/default_config.json --seed 7      # fija la semilla
python main.py configs/default_config.json --no-preview  # desactiva los previews
python main.py configs/default_config.json --gif evolucion.gif  # arma el GIF de la corrida
python main.py configs/default_config.json --no-gif      # desactiva el GIF
```

Los errores de configuración se detectan antes de empezar a evolucionar e indican
dónde está el problema: tipos incorrectos, valores fuera de rango, combinaciones
imposibles y —sobre todo— claves desconocidas, para que un typo no se convierta en
un parámetro silenciosamente ignorado.

## Configuración

Todos los parámetros son opcionales salvo `image`; una clave ausente y una en
`null` significan lo mismo: usar el valor por defecto.

| Clave            | Default        | Descripción                                            |
| ---------------- | -------------- | ------------------------------------------------------ |
| `image`          | *(obligatoria)* | Imagen objetivo.                                       |
| `output`         | `output.png`   | Dónde guardar el mejor individuo.                       |
| `max_size`       | `128`          | Resolución máxima de trabajo para acelerar el cálculo.  |
| `triangles`      | `50`           | Triángulos por individuo.                               |
| `seed`           | *(sorteada)*   | Semilla; si se omite se sortea una y se informa.         |
| `progress_every` | `10`           | Informa progreso cada N generaciones; `0` lo desactiva. |

Y las secciones `population`, `selection`, `crossover`, `mutation`, `fitness`,
`preview` y `gif`, cada una con sus propios defaults:

```json
{
  "image": "target.png",
  "population": { "size": 100, "parents": 60, "generations": 2000, "survival": "additive" },
  "selection": { "strategy": "tournament", "tournament_size": 5, "win_probability": 0.85 },
  "crossover": { "strategy": "two-point" },
  "mutation": {
    "schedule": "exponential",
    "probability": 0.3, "strength": 0.12, "replacement_probability": 0.08,
    "decay_generations": 1000,
    "final": { "probability": 0.05, "strength": 0.025, "replacement_probability": 0.02 }
  },
  "fitness": { "metric": "mse" }
}
```

- `population.survival`: `additive` o `exclusive` (esta última reemplaza toda la
  generación, así que necesita una cantidad par de padres al menos igual a `size`).
- `selection.strategy`: `elite`, `roulette`, `universal`, `boltzmann`,
  `ranking`, `deterministic-tournament`, `probabilistic-tournament` o
  `tournament` (alias de `probabilistic-tournament`).
  `boltzmann` usa además `boltzmann_temperature`.
- `crossover.strategy`: `one-point`, `two-point`, `uniform` (con
  `uniform_swap_probability`) o `annular`.
- `mutation.schedule`: `constant`, `linear`, `exponential` o `adaptive-reheat`.
  Las tres últimas requieren `decay_generations`; la sección `final` es opcional y
  los parámetros que no declara no cambian durante el decaimiento.
  `adaptive-reheat` acepta además una sección `reheat` (`stagnation_generations`,
  `duration_generations`, `improvement_percent`, `probability_multiplier`,
  `strength_multiplier`, `replacement_multiplier`).
- `mutation.strategy`: `gen` muta un gen, `multigen` varios genes con probabilidad,
  `uniform` reemplaza genes por triángulos aleatorios y `non-uniform` aplica
  perturbaciones cuya magnitud sigue el schedule.
- `termination`: permite elegir `max-generations` (default), `target-fitness` o
  `stagnation`; sus parámetros son `target_fitness`, `stagnation_generations` e
  `improvement`.

Cada corrida escribe `run/config.json`, `run/best.png`, `run/triangles.json`,
`run/history.csv` y `run/summary.json`, junto con la imagen indicada por `output`.

## Experimentos

`experiments.py` ejecuta una matriz declarada en JSON y genera un CSV y un gráfico:

```bash
python experiments.py experiments.json
```

El spec contiene `config`, `output_directory` y una `matrix`; las claves pueden
ser simples o rutas como `mutation.strategy`.

### Experimento exhaustivo de mutación y fitness

Está preparado en `experiments/mutation_fitness_exhaustive.json` y usa una sola
configuración base (`configs/mutation_fitness_base.json`). Recorre las 16
combinaciones de `mutation.strategy` × `mutation.schedule`, las 10 métricas de
fitness (3 controles simples y 7 combinaciones, desde pares hasta una combinación
de 5 métricas). Entre ellas se conserva obligatoriamente el combo
`mse: 0.35 + blur: 0.20 + edge: 0.20 + saliency: 0.25`, con `blur.sigma=1.5`,
`edge.sigma=1.0` y `saliency.weight=3.0`, `saliency.sigma=2.0`. Las tres imágenes
disponibles son (`bandera_100.png`,
`Firefox_logo,_2017.png`, `monalisa.jpg`), siempre con `max_size: 96` y selección
`tournament`. En total son `3 × 4 × 4 × 10 = 480` corridas:

```bash
python experiments.py experiments/mutation_fitness_exhaustive.json
```

Para ejecutar varias corridas independientes en paralelo, por ejemplo dos:

```bash
python experiments.py experiments/mutation_fitness_exhaustive.json --workers 2
```

El coordinador es el único proceso que escribe `results.csv`, `progress.json` y
el gráfico; cada worker escribe exclusivamente su carpeta `run_XXXX`. El valor
por defecto (`--workers 1`) conserva la ejecución secuencial. Conviene empezar
con 2 workers y aumentar solo si CPU y memoria lo permiten.

El resultado queda en `experiments/results/mutation_fitness_exhaustive/`, con un
`results.csv` (incluye `elapsed_seconds` y `cpu_seconds` por corrida), `progress.json` y los artefactos de cada corrida. Las corridas se
ejecutan secuencialmente. Después de cada corrida se actualizan `results.csv`,
`progress.json` y el gráfico; si el proceso se interrumpe, al volver a ejecutar el
comando se omiten las carpetas que ya tienen `run/summary.json` (`resume: true`).
Los overrides anidados se fusionan
con la base: cambiar `mutation.strategy` no elimina los parámetros del schedule.
La semilla 20260903 queda fija para comparar configuraciones bajo el mismo azar;
para reportar variabilidad conviene repetir las configuraciones ganadoras con 3–5
semillas.

Para comparar costos computacionales entre workers concurrentes, usar
`cpu_seconds`: `elapsed_seconds` es el tiempo de pared y puede incluir esperas
por competencia entre procesos.

La cantidad de generaciones depende de la imagen: Italia usa 2000, Firefox 2500
y Mona Lisa 3000. Se representa junto a cada imagen en la matriz para no crear
combinaciones extra.

Se recomienda empezar con 80 triángulos: a 96 px ofrece suficiente capacidad sin
inflar demasiado el costo. Para estudiar sensibilidad, agregar
`"triangles": [40, 80, 120]` como entrada de `matrix` agrega un factor 3 y lleva
el diseño a 1440 corridas; es preferible hacerlo después de identificar las
mejores combinaciones.

### Semilla

Sin `seed` la corrida es aleatoria, pero no irrepetible: el programa sortea una
semilla y la imprime al arrancar, así cualquier resultado interesante se puede
reproducir agregándola al archivo (o pasándola con `--seed`).

### Previews

Los previews son opcionales y se activan declarando la sección; sin ella no se
escribe ninguna imagen intermedia.

```json
"preview": { "directory": "previews/firefox", "every": 100, "full_resolution": false }
```

`directory` es obligatorio dentro de la sección, `every` guarda una imagen cada N
generaciones y `full_resolution` las escribe en la resolución original en vez de
la de trabajo. `--no-preview` los desactiva sin tocar el archivo.

### GIF de la evolución

También opcional y también por presencia de su sección. A diferencia de los
previews produce un único archivo animado y, por defecto, en la resolución
original de la imagen objetivo.

```json
"gif": { "path": "resultados/evolucion.gif", "every": 25 }
```

| Clave                     | Default | Descripción                                                |
| ------------------------- | ------- | ---------------------------------------------------------- |
| `path`                    | *(obligatoria)* | Archivo GIF a escribir; se crean los directorios faltantes. |
| `every`                   | `25`    | Toma un cuadro cada N generaciones.                        |
| `frame_duration_ms`       | `80`    | Duración de cada cuadro.                                   |
| `final_frame_duration_ms` | `1500`  | Duración del último cuadro, para que el resultado se vea.  |
| `loop`                    | `true`  | Repetición infinita; con `false` se reproduce una vez.     |
| `full_resolution`         | `true`  | Usa la resolución original; con `false`, la de trabajo.    |

Los cuadros se acumulan en memoria hasta el final de la corrida, así que `every`
es también el control del costo: a resolución completa conviene un valor que deje
del orden de un centenar de cuadros. `--gif RUTA` arma la animación aunque el
archivo no la declare y `--no-gif` la desactiva aunque la declare.

## Fitness

Además del MSE global (`mse`), hay métricas regionales (`regional`), `ssim`,
`blur`, `multiscale`, `histogram`, bordes Sobel (`edge`), `gradient` (magnitud y
orientación de contornos), `chamfer` (tolerante a pequeños desplazamientos entre
bordes) y `saliency` (pondera más las zonas de alto contraste del objetivo). Cada
una tiene su propia subsección de parámetros dentro de `fitness`.

Todas las métricas devuelven una fitness normalizada en el rango `[0, 1]`: `1`
es una coincidencia perfecta y los valores más altos siempre representan mejores
soluciones. La normalización usa el máximo teórico fijo de cada métrica, por lo
que el valor es comparable entre generaciones y corridas.

Con `metric: "combo"` se optimiza una suma ponderada de varias fitness; cada una
ya está normalizada en `[0, 1]` antes de ponderarse, y los pesos se renormalizan
para sumar 1:

```json
"fitness": {
  "metric": "combo",
  "combo": { "mse": 0.20, "regional": 0.20, "edge": 0.25, "saliency": 0.35 },
  "regional": { "grid_rows": 3, "grid_cols": 1, "detail_weight": 2.0 },
  "edge": { "sigma": 0.8 },
  "saliency": { "weight": 2.0, "sigma": 3.0 }
}
```

## Documentación

- [`FITNESS.md`](FITNESS.md): explicación detallada de cada métrica de fitness,
  qué mide, cómo se interpreta y cuándo conviene usarla.
- [`ARQUITECTURA_IMPLEMENTACION.md`](ARQUITECTURA_IMPLEMENTACION.md): explicación
  de las capas, flujo de ejecución, configuración y estado frente a la consigna.

## Capas

- `genetic_algorithm.domain`: contratos para individuo, imagen objetivo,
  fitness, evaluación/comparación, problema, configuración y estado/resultado.
- `genetic_algorithm.application`: contratos de inicialización de población,
  selección, emparejamiento, supervivencia, cruza, mutación, terminación y motor
  evolutivo. También incluye `OrchestratedGeneticAlgorithm`, que coordina estos
  contratos sin decidir cómo funciona ninguna estrategia.
- `triangle_image`: la implementación concreta (genes, codec, render, métricas de
  fitness, mutadores y políticas de mutación).
- `simulation`: capa de composición. `config` define el esquema declarativo y lo
  valida, `builders` traduce cada opción al operador concreto, `reporting`
  observa la corrida y `runner` la ejecuta de punta a punta.

El proyecto declara Python `>=3.14` en `pyproject.toml`. Además, los contratos
usan `abc.ABC` y sintaxis moderna de parámetros de tipo; no va a parsear con
versiones viejas de Python.

## Cómo extenderlo

1. Implementar `Individual`, `Fitness` e `ImageTarget` con el formato elegido.
2. Implementar `FitnessEvaluator` y `FitnessComparator`.
3. Crear las estrategias concretas necesarias como subclases de los contratos.
4. Registrarlas en `simulation.builders` y agregar sus parámetros al esquema de
   `simulation.config`, de modo que queden disponibles desde los archivos JSON.

`OrchestratedGeneticAlgorithm` ya ofrece el ciclo genérico. Para usarlo se le
inyectan las estrategias, un `EvolutionContext` y una configuración que implemente
`EvolutionConfiguration` (`population_size` y `selected_parent_count`).

## Pruebas

```bash
python -m unittest discover -v
```
