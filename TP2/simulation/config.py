"""Esquema de configuración de una simulación y su carga desde JSON.

Todo parámetro tiene un valor por defecto razonable y puede omitirse; la única
clave obligatoria es la imagen objetivo. Las funcionalidades opcionales (semilla
fija, previews) se activan por presencia de su clave, no por banderas separadas.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from triangle_image import MutationParameters, SCALES

from simulation.section import ConfigSection, ConfigurationError

METRIC_NAMES: tuple[str, ...] = tuple(sorted(SCALES))
FITNESS_CHOICES: tuple[str, ...] = (*METRIC_NAMES, "combo")

SelectionStrategyName = Literal["elite", "tournament"]
SurvivalStrategyName = Literal["additive", "exclusive"]
CrossoverStrategyName = Literal["one-point", "two-point", "uniform", "annular"]
MutationScheduleName = Literal["constant", "linear", "exponential", "adaptive-reheat"]

SELECTION_CHOICES: tuple[str, ...] = ("elite", "tournament")
SURVIVAL_CHOICES: tuple[str, ...] = ("additive", "exclusive")
CROSSOVER_CHOICES: tuple[str, ...] = ("one-point", "two-point", "uniform", "annular")
MUTATION_SCHEDULE_CHOICES: tuple[str, ...] = (
    "constant", "linear", "exponential", "adaptive-reheat",
)


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} debe ser positivo (se recibió {value}).")


def _require_non_negative(value: float, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} debe ser no negativo (se recibió {value}).")


def _require_probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} debe estar entre 0 y 1 (se recibió {value}).")


@dataclass(frozen=True, slots=True)
class PopulationConfig:
    """Tamaño de la población, cuántos padres se seleccionan y cuánto dura la corrida."""

    size: int = 100
    parents: int = 50
    generations: int = 1000
    survival: SurvivalStrategyName = "additive"

    def __post_init__(self) -> None:
        _require_positive(self.size, "population.size")
        _require_positive(self.parents, "population.parents")
        _require_positive(self.generations, "population.generations")

    @property
    def offspring_count(self) -> int:
        """Hijos generados por generación: las parejas consumen padres de a dos."""
        return 2 * (self.parents // 2)

    @classmethod
    def from_section(cls, section: ConfigSection) -> PopulationConfig:
        defaults = cls()
        return cls(
            size=section.integer("size", defaults.size),
            parents=section.integer("parents", defaults.parents),
            generations=section.integer("generations", defaults.generations),
            survival=section.choice("survival", defaults.survival, SURVIVAL_CHOICES),
        )


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    """Estrategia de selección de padres y sus parámetros."""

    strategy: SelectionStrategyName = "elite"
    tournament_size: int = 3
    win_probability: float = 0.85

    def __post_init__(self) -> None:
        if self.tournament_size < 2:
            raise ValueError("selection.tournament_size debe ser al menos 2.")
        _require_probability(self.win_probability, "selection.win_probability")

    @classmethod
    def from_section(cls, section: ConfigSection) -> SelectionConfig:
        defaults = cls()
        return cls(
            strategy=section.choice("strategy", defaults.strategy, SELECTION_CHOICES),
            tournament_size=section.integer("tournament_size", defaults.tournament_size),
            win_probability=section.number("win_probability", defaults.win_probability),
        )


@dataclass(frozen=True, slots=True)
class CrossoverConfig:
    """Estrategia de cruza y sus parámetros."""

    strategy: CrossoverStrategyName = "one-point"
    uniform_swap_probability: float = 0.5

    def __post_init__(self) -> None:
        _require_probability(
            self.uniform_swap_probability, "crossover.uniform_swap_probability"
        )

    @classmethod
    def from_section(cls, section: ConfigSection) -> CrossoverConfig:
        defaults = cls()
        return cls(
            strategy=section.choice("strategy", defaults.strategy, CROSSOVER_CHOICES),
            uniform_swap_probability=section.number(
                "uniform_swap_probability", defaults.uniform_swap_probability
            ),
        )


@dataclass(frozen=True, slots=True)
class ReheatConfig:
    """Pulso de mutación agresiva que dispara `adaptive-reheat` ante estancamiento."""

    stagnation_generations: int = 75
    duration_generations: int = 40
    improvement_percent: float = 0.5
    improvement_delta: float | None = None
    probability_multiplier: float = 2.5
    strength_multiplier: float = 2.5
    replacement_multiplier: float = 6.0

    def __post_init__(self) -> None:
        _require_positive(self.stagnation_generations, "mutation.reheat.stagnation_generations")
        _require_positive(self.duration_generations, "mutation.reheat.duration_generations")
        _require_non_negative(self.improvement_percent, "mutation.reheat.improvement_percent")
        if self.improvement_delta is not None:
            _require_non_negative(self.improvement_delta, "mutation.reheat.improvement_delta")
        for name in ("probability_multiplier", "strength_multiplier", "replacement_multiplier"):
            _require_positive(getattr(self, name), f"mutation.reheat.{name}")

    @classmethod
    def from_section(cls, section: ConfigSection) -> ReheatConfig:
        defaults = cls()
        return cls(
            stagnation_generations=section.integer(
                "stagnation_generations", defaults.stagnation_generations
            ),
            duration_generations=section.integer(
                "duration_generations", defaults.duration_generations
            ),
            improvement_percent=section.number(
                "improvement_percent", defaults.improvement_percent
            ),
            improvement_delta=section.optional_number("improvement_delta"),
            probability_multiplier=section.number(
                "probability_multiplier", defaults.probability_multiplier
            ),
            strength_multiplier=section.number(
                "strength_multiplier", defaults.strength_multiplier
            ),
            replacement_multiplier=section.number(
                "replacement_multiplier", defaults.replacement_multiplier
            ),
        )


@dataclass(frozen=True, slots=True)
class MutationConfig:
    """Política de mutación: valores iniciales, finales y forma del decaimiento."""

    schedule: MutationScheduleName = "constant"
    initial: MutationParameters = MutationParameters(0.1, 0.1, 0.02)
    final: MutationParameters | None = None
    decay_generations: int = 0
    reheat: ReheatConfig = field(default_factory=ReheatConfig)

    def __post_init__(self) -> None:
        for label, parameters in (("", self.initial), (".final", self.final)):
            if parameters is None:
                continue
            _require_probability(parameters.probability, f"mutation{label}.probability")
            _require_non_negative(parameters.strength, f"mutation{label}.strength")
            _require_probability(
                parameters.replacement_probability, f"mutation{label}.replacement_probability"
            )
        _require_non_negative(self.decay_generations, "mutation.decay_generations")
        if self.schedule != "constant" and self.decay_generations <= 0:
            raise ValueError(
                f"la política de mutación '{self.schedule}' requiere "
                "mutation.decay_generations positivo."
            )

    @property
    def final_or_initial(self) -> MutationParameters:
        """Parámetros al terminar el decaimiento; sin sección `final` no cambian."""
        return self.final if self.final is not None else self.initial

    @classmethod
    def from_section(cls, section: ConfigSection) -> MutationConfig:
        defaults = cls()
        initial = cls._read_parameters(section, defaults.initial)
        final_section = section.optional_section("final")
        return cls(
            schedule=section.choice("schedule", defaults.schedule, MUTATION_SCHEDULE_CHOICES),
            initial=initial,
            final=None if final_section is None else cls._read_parameters(final_section, initial),
            decay_generations=section.integer("decay_generations", defaults.decay_generations),
            reheat=ReheatConfig.from_section(section.section("reheat")),
        )

    @staticmethod
    def _read_parameters(
        section: ConfigSection, defaults: MutationParameters
    ) -> MutationParameters:
        return MutationParameters(
            probability=section.number("probability", defaults.probability),
            strength=section.number("strength", defaults.strength),
            replacement_probability=section.number(
                "replacement_probability", defaults.replacement_probability
            ),
        )


@dataclass(frozen=True, slots=True)
class RegionalFitnessConfig:
    """Parámetros del MSE por regiones."""

    grid_rows: int = 8
    grid_cols: int = 8
    detail_weight: float = 1.0

    def __post_init__(self) -> None:
        _require_positive(self.grid_rows, "fitness.regional.grid_rows")
        _require_positive(self.grid_cols, "fitness.regional.grid_cols")
        _require_non_negative(self.detail_weight, "fitness.regional.detail_weight")

    @classmethod
    def from_section(cls, section: ConfigSection) -> RegionalFitnessConfig:
        defaults = cls()
        return cls(
            grid_rows=section.integer("grid_rows", defaults.grid_rows),
            grid_cols=section.integer("grid_cols", defaults.grid_cols),
            detail_weight=section.number("detail_weight", defaults.detail_weight),
        )


@dataclass(frozen=True, slots=True)
class SSIMFitnessConfig:
    """Parámetros de la similitud estructural."""

    window_size: int = 7
    mse_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.window_size < 3 or self.window_size % 2 == 0:
            raise ValueError("fitness.ssim.window_size debe ser impar y al menos 3.")
        _require_probability(self.mse_weight, "fitness.ssim.mse_weight")

    @classmethod
    def from_section(cls, section: ConfigSection) -> SSIMFitnessConfig:
        defaults = cls()
        return cls(
            window_size=section.integer("window_size", defaults.window_size),
            mse_weight=section.number("mse_weight", defaults.mse_weight),
        )


@dataclass(frozen=True, slots=True)
class BlurFitnessConfig:
    """Parámetros del MSE sobre imágenes desenfocadas."""

    sigma: float = 1.5

    def __post_init__(self) -> None:
        _require_non_negative(self.sigma, "fitness.blur.sigma")

    @classmethod
    def from_section(cls, section: ConfigSection) -> BlurFitnessConfig:
        return cls(sigma=section.number("sigma", cls().sigma))


@dataclass(frozen=True, slots=True)
class MultiScaleFitnessConfig:
    """Escalas comparadas por el MSE multiescala."""

    scales: tuple[float, ...] = (1.0, 0.5, 0.25)

    def __post_init__(self) -> None:
        if not self.scales:
            raise ValueError("fitness.multiscale.scales no puede estar vacío.")
        for scale in self.scales:
            _require_positive(scale, "cada valor de fitness.multiscale.scales")

    @classmethod
    def from_section(cls, section: ConfigSection) -> MultiScaleFitnessConfig:
        return cls(scales=section.number_list("scales", cls().scales))


@dataclass(frozen=True, slots=True)
class HistogramFitnessConfig:
    """Parámetros del histograma de color."""

    bins: int = 32
    mse_weight: float = 0.5

    def __post_init__(self) -> None:
        _require_positive(self.bins, "fitness.histogram.bins")
        _require_probability(self.mse_weight, "fitness.histogram.mse_weight")

    @classmethod
    def from_section(cls, section: ConfigSection) -> HistogramFitnessConfig:
        defaults = cls()
        return cls(
            bins=section.integer("bins", defaults.bins),
            mse_weight=section.number("mse_weight", defaults.mse_weight),
        )


@dataclass(frozen=True, slots=True)
class EdgeFitnessConfig:
    """Suavizado previo a Sobel para la métrica de bordes."""

    sigma: float = 1.0

    def __post_init__(self) -> None:
        _require_non_negative(self.sigma, "fitness.edge.sigma")

    @classmethod
    def from_section(cls, section: ConfigSection) -> EdgeFitnessConfig:
        return cls(sigma=section.number("sigma", cls().sigma))


@dataclass(frozen=True, slots=True)
class GradientFitnessConfig:
    """Parámetros de magnitud y orientación de gradientes."""

    sigma: float = 1.0
    orientation_weight: float = 0.5

    def __post_init__(self) -> None:
        _require_non_negative(self.sigma, "fitness.gradient.sigma")
        _require_probability(self.orientation_weight, "fitness.gradient.orientation_weight")

    @classmethod
    def from_section(cls, section: ConfigSection) -> GradientFitnessConfig:
        defaults = cls()
        return cls(
            sigma=section.number("sigma", defaults.sigma),
            orientation_weight=section.number(
                "orientation_weight", defaults.orientation_weight
            ),
        )


@dataclass(frozen=True, slots=True)
class ChamferFitnessConfig:
    """Parámetros de la distancia de chanfle entre mapas de bordes."""

    sigma: float = 1.0
    threshold: float = 20.0

    def __post_init__(self) -> None:
        _require_non_negative(self.sigma, "fitness.chamfer.sigma")
        if not 0.0 <= self.threshold <= 255.0:
            raise ValueError("fitness.chamfer.threshold debe estar entre 0 y 255.")

    @classmethod
    def from_section(cls, section: ConfigSection) -> ChamferFitnessConfig:
        defaults = cls()
        return cls(
            sigma=section.number("sigma", defaults.sigma),
            threshold=section.number("threshold", defaults.threshold),
        )


@dataclass(frozen=True, slots=True)
class SaliencyFitnessConfig:
    """Refuerzo y suavizado de la máscara de saliencia."""

    weight: float = 3.0
    sigma: float = 2.0

    def __post_init__(self) -> None:
        _require_non_negative(self.weight, "fitness.saliency.weight")
        _require_non_negative(self.sigma, "fitness.saliency.sigma")

    @classmethod
    def from_section(cls, section: ConfigSection) -> SaliencyFitnessConfig:
        defaults = cls()
        return cls(
            weight=section.number("weight", defaults.weight),
            sigma=section.number("sigma", defaults.sigma),
        )


@dataclass(frozen=True, slots=True)
class FitnessConfig:
    """Métrica a optimizar y los parámetros de cada evaluador disponible.

    Con `metric: "combo"` se optimiza una suma ponderada de varias métricas
    normalizadas, declaradas en `combo` como pares nombre/peso.
    """

    metric: str = "mse"
    combo: Mapping[str, float] | None = None
    regional: RegionalFitnessConfig = field(default_factory=RegionalFitnessConfig)
    ssim: SSIMFitnessConfig = field(default_factory=SSIMFitnessConfig)
    blur: BlurFitnessConfig = field(default_factory=BlurFitnessConfig)
    multiscale: MultiScaleFitnessConfig = field(default_factory=MultiScaleFitnessConfig)
    histogram: HistogramFitnessConfig = field(default_factory=HistogramFitnessConfig)
    edge: EdgeFitnessConfig = field(default_factory=EdgeFitnessConfig)
    gradient: GradientFitnessConfig = field(default_factory=GradientFitnessConfig)
    chamfer: ChamferFitnessConfig = field(default_factory=ChamferFitnessConfig)
    saliency: SaliencyFitnessConfig = field(default_factory=SaliencyFitnessConfig)

    def __post_init__(self) -> None:
        if self.metric not in FITNESS_CHOICES:
            raise ValueError(
                f"fitness.metric: '{self.metric}' no es válido. "
                f"Opciones: {', '.join(FITNESS_CHOICES)}."
            )
        if self.metric != "combo":
            return
        if not self.combo:
            raise ValueError(
                "fitness.metric 'combo' requiere fitness.combo con al menos una métrica "
                "y su peso."
            )
        unknown = sorted(set(self.combo) - set(METRIC_NAMES))
        if unknown:
            raise ValueError(
                f"fitness.combo: métrica(s) desconocida(s): {', '.join(unknown)}. "
                f"Opciones: {', '.join(METRIC_NAMES)}."
            )
        for name, weight in self.combo.items():
            _require_non_negative(weight, f"fitness.combo.{name}")
        if sum(self.combo.values()) <= 0:
            raise ValueError("fitness.combo: la suma de los pesos debe ser positiva.")

    @classmethod
    def from_section(cls, section: ConfigSection) -> FitnessConfig:
        defaults = cls()
        return cls(
            metric=section.choice("metric", defaults.metric, FITNESS_CHOICES),
            combo=section.number_mapping("combo"),
            regional=RegionalFitnessConfig.from_section(section.section("regional")),
            ssim=SSIMFitnessConfig.from_section(section.section("ssim")),
            blur=BlurFitnessConfig.from_section(section.section("blur")),
            multiscale=MultiScaleFitnessConfig.from_section(section.section("multiscale")),
            histogram=HistogramFitnessConfig.from_section(section.section("histogram")),
            edge=EdgeFitnessConfig.from_section(section.section("edge")),
            gradient=GradientFitnessConfig.from_section(section.section("gradient")),
            chamfer=ChamferFitnessConfig.from_section(section.section("chamfer")),
            saliency=SaliencyFitnessConfig.from_section(section.section("saliency")),
        )


@dataclass(frozen=True, slots=True)
class PreviewConfig:
    """Guardado periódico de imágenes del mejor individuo.

    Es opcional: sin la sección `preview` la simulación no escribe previews.
    """

    directory: Path
    every: int = 1
    full_resolution: bool = False

    def __post_init__(self) -> None:
        _require_positive(self.every, "preview.every")

    @classmethod
    def from_section(cls, section: ConfigSection) -> PreviewConfig:
        defaults_every, defaults_full = 1, False
        return cls(
            directory=Path(section.required_text("directory")),
            every=section.integer("every", defaults_every),
            full_resolution=section.boolean("full_resolution", defaults_full),
        )


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Configuración completa de una corrida, con defaults para todo salvo la imagen."""

    image: Path
    output: Path = Path("output.png")
    max_size: int = 128
    triangles: int = 50
    seed: int | None = None
    progress_every: int = 10
    population: PopulationConfig = field(default_factory=PopulationConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    crossover: CrossoverConfig = field(default_factory=CrossoverConfig)
    mutation: MutationConfig = field(default_factory=MutationConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    preview: PreviewConfig | None = None

    def __post_init__(self) -> None:
        _require_positive(self.max_size, "max_size")
        _require_positive(self.triangles, "triangles")
        _require_non_negative(self.progress_every, "progress_every")
        if (
            self.population.survival == "exclusive"
            and self.population.offspring_count < self.population.size
        ):
            raise ValueError(
                "population.survival 'exclusive' reemplaza toda la generación: requiere una "
                f"cantidad par de padres de al menos {self.population.size} "
                f"(population.parents es {self.population.parents})."
            )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> SimulationConfig:
        """Construye la configuración validando tipos, rangos y claves desconocidas.

        Cualquier problema se reporta como `ConfigurationError`, de modo que quien
        carga configuraciones tiene un único tipo de error del que preocuparse.
        """
        try:
            return cls._parse(ConfigSection(data))
        except ConfigurationError:
            raise
        except ValueError as error:
            raise ConfigurationError(str(error)) from error

    @classmethod
    def _parse(cls, section: ConfigSection) -> SimulationConfig:
        defaults = _TOP_LEVEL_DEFAULTS
        preview_section = section.optional_section("preview")
        config = cls(
            image=Path(section.required_text("image")),
            output=Path(section.text("output", str(defaults.output))),
            max_size=section.integer("max_size", defaults.max_size),
            triangles=section.integer("triangles", defaults.triangles),
            seed=section.optional_integer("seed"),
            progress_every=section.integer("progress_every", defaults.progress_every),
            population=PopulationConfig.from_section(section.section("population")),
            selection=SelectionConfig.from_section(section.section("selection")),
            crossover=CrossoverConfig.from_section(section.section("crossover")),
            mutation=MutationConfig.from_section(section.section("mutation")),
            fitness=FitnessConfig.from_section(section.section("fitness")),
            preview=(
                None if preview_section is None else PreviewConfig.from_section(preview_section)
            ),
        )
        section.ensure_no_unknown_keys()
        return config


_TOP_LEVEL_DEFAULTS = SimulationConfig(image=Path("."))


def load_simulation_config(
    path: str | Path, overrides: Mapping[str, Any] | None = None
) -> SimulationConfig:
    """Carga un archivo JSON de configuración y le aplica overrides de nivel superior.

    Los overrides cubren los pocos parámetros que cambian entre corridas de un
    mismo experimento (imagen, salida, semilla, previews); como en el archivo, un
    valor `None` equivale a no declarar la clave.
    """
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigurationError(
            f"No existe el archivo de configuración: {config_path}"
        ) from error
    except json.JSONDecodeError as error:
        raise ConfigurationError(f"{config_path} no es JSON válido: {error}") from error

    if not isinstance(raw, Mapping):
        raise ConfigurationError(f"{config_path} debe contener un objeto JSON en su raíz.")

    try:
        return SimulationConfig.from_mapping({**raw, **(overrides or {})})
    except ValueError as error:
        raise ConfigurationError(f"{config_path}: {error}") from error
