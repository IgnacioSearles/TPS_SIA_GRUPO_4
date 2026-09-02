"""Imagen objetivo y evaluación de fitness por comparación de píxeles."""

from __future__ import annotations

from typing import Any
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter

from genetic_algorithm.domain.contracts import (EvolutionContext, Fitness,
                                                 FitnessComparator, FitnessEvaluator,
                                                 ImageTarget)
from triangle_image.gene import TriangleIndividual
from triangle_image.rendering import render


class TriangleImageTarget(ImageTarget[np.ndarray]):
    """Imagen objetivo almacenada como array de numpy para cálculos rápidos."""

    def __init__(self, image: Image.Image, max_size: int | None = None) -> None:
        self._orig_width, self._orig_height = image.size
        self._scale_factor = 1.0

        if max_size is not None and max(self._orig_width, self._orig_height) > max_size:
            self._scale_factor = max_size / float(max(self._orig_width, self._orig_height))
            new_width = int(self._orig_width * self._scale_factor)
            new_height = int(self._orig_height * self._scale_factor)
            
            try:
                resample_filter = Image.Resampling.LANCZOS
            except AttributeError:
                resample_filter = Image.LANCZOS
                
            image = image.resize((new_width, new_height), resample_filter)

        self._width, self._height = image.size
        # Usamos int16 para evitar overflow al restar píxeles
        self._image_array = np.array(image.convert("RGB"), dtype=np.int16)

    @property
    def image(self) -> np.ndarray:
        return self._image_array

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def orig_width(self) -> int:
        return self._orig_width

    @property
    def orig_height(self) -> int:
        return self._orig_height

    @property
    def scale_factor(self) -> float:
        return self._scale_factor


class MSEFitness(Fitness[float]):
    """Error cuadrático medio contra la imagen objetivo."""

    def __init__(self, error: float) -> None:
        self._error = error

    @property
    def value(self) -> float:
        return self._error


class MSEComparator(FitnessComparator[MSEFitness]):
    """Compara por menor MSE (problema de minimización)."""

    def is_better(self, left: MSEFitness, right: MSEFitness) -> bool:
        return left.value < right.value


class MSEEvaluator(FitnessEvaluator[TriangleIndividual, TriangleImageTarget, MSEFitness]):
    """Evalúa renderizando los triángulos y calculando el MSE con numpy."""

    def evaluate(
        self,
        individual: TriangleIndividual,
        target: TriangleImageTarget,
        context: EvolutionContext,
    ) -> MSEFitness:
        # Renderizamos el fenotipo (la imagen)
        rendered_image = render(individual, target.width, target.height)

        # Al restar int16, la diferencia está entre -255 y 255.
        # Pero al elevar al cuadrado (hasta 65025), excede el límite de int16 (32767),
        # causando un overflow (números negativos).
        # Convertimos a float32 o int32 antes de elevar al cuadrado.
        target_arr = target.image.astype(np.float32)
        rendered_arr = np.array(rendered_image, dtype=np.float32)

        # Calculamos MSE
        mse = np.mean(np.square(target_arr - rendered_arr))

        return MSEFitness(float(mse))


class RegionalMSEEvaluator(FitnessEvaluator[TriangleIndividual, TriangleImageTarget, MSEFitness]):
    """MSE calculado por regiones de una grilla, promediado con pesos.

    Divide la imagen objetivo en `grid_rows` x `grid_cols` celdas y calcula el MSE
    de cada una por separado. El fitness final es el promedio ponderado de esos MSE
    regionales: las celdas con más detalle/contraste (mayor desvío estándar de píxeles
    en la imagen objetivo) pesan más, para presionar al algoritmo a resolver mejor esas
    zonas en lugar de diluir su error entre regiones lisas y de bajo contraste.
    """

    def __init__(self, grid_rows: int = 8, grid_cols: int = 8, detail_weight: float = 1.0) -> None:
        self._grid_rows = grid_rows
        self._grid_cols = grid_cols
        self._detail_weight = detail_weight

        # Cacheados por objetivo (el objetivo no cambia durante la corrida).
        self._cached_target: TriangleImageTarget | None = None
        self._row_bounds: list[tuple[int, int]] = []
        self._col_bounds: list[tuple[int, int]] = []
        self._weights: np.ndarray | None = None  # (grid_rows, grid_cols), suma 1.0

    def _prepare(self, target: TriangleImageTarget) -> None:
        if self._cached_target is target:
            return

        row_edges = np.linspace(0, target.height, self._grid_rows + 1).astype(int)
        col_edges = np.linspace(0, target.width, self._grid_cols + 1).astype(int)
        self._row_bounds = list(zip(row_edges[:-1], row_edges[1:]))
        self._col_bounds = list(zip(col_edges[:-1], col_edges[1:]))

        target_arr = target.image.astype(np.float32)
        detail = np.zeros((self._grid_rows, self._grid_cols), dtype=np.float32)
        for i, (r0, r1) in enumerate(self._row_bounds):
            for j, (c0, c1) in enumerate(self._col_bounds):
                cell = target_arr[r0:r1, c0:c1]
                detail[i, j] = float(np.std(cell)) if cell.size else 0.0

        # Peso base 1 + contribución proporcional al detalle normalizado (0..1).
        max_detail = float(detail.max())
        normalized = detail / max_detail if max_detail > 0 else np.zeros_like(detail)
        weights = 1.0 + self._detail_weight * normalized
        self._weights = weights / weights.sum()

        self._cached_target = target

    def evaluate(
        self,
        individual: TriangleIndividual,
        target: TriangleImageTarget,
        context: EvolutionContext,
    ) -> MSEFitness:
        self._prepare(target)
        assert self._weights is not None

        rendered_image = render(individual, target.width, target.height)
        target_arr = target.image.astype(np.float32)
        rendered_arr = np.array(rendered_image, dtype=np.float32)
        diff_sq = np.square(target_arr - rendered_arr)

        weighted_sum = 0.0
        for i, (r0, r1) in enumerate(self._row_bounds):
            for j, (c0, c1) in enumerate(self._col_bounds):
                cell = diff_sq[r0:r1, c0:c1]
                if cell.size == 0:
                    continue
                region_mse = float(np.mean(cell))
                weighted_sum += float(self._weights[i, j]) * region_mse

        return MSEFitness(weighted_sum)


def _channel_ssim(img1: np.ndarray, img2: np.ndarray, win_size: int) -> np.ndarray:
    """SSIM local (mapa) entre dos canales 2D, vía medias/varianzas en ventana deslizante."""
    data_range = 255.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu1 = uniform_filter(img1, win_size)
    mu2 = uniform_filter(img2, win_size)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = uniform_filter(img1 * img1, win_size) - mu1_sq
    sigma2_sq = uniform_filter(img2 * img2, win_size) - mu2_sq
    sigma12 = uniform_filter(img1 * img2, win_size) - mu1_mu2

    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    return numerator / denominator


class SSIMEvaluator(FitnessEvaluator[TriangleIndividual, TriangleImageTarget, MSEFitness]):
    """Fitness perceptual basado en SSIM (Structural Similarity Index).

    A diferencia del MSE, que solo mide diferencias exactas de píxel a píxel, SSIM
    compara luminancia, contraste y estructura local usando estadísticas en una
    ventana deslizante, lo que se aproxima mejor a cómo un humano percibe la
    similitud entre dos imágenes (tolera pequeños corrimientos de intensidad,
    penaliza más los cambios estructurales).

    Se calcula el SSIM por canal (R, G, B) y se promedia. Como SSIM ∈ [-1, 1] con
    1 = coincidencia perfecta, el término de error es `1 - SSIM medio` (0 = coincidencia
    perfecta), manteniendo la semántica de minimización del resto del pipeline.

    SSIM puro tiene un punto débil conocido para este problema: en regiones lisas
    (como las franjas de una bandera) el término de contraste/estructura se satura
    cerca de 1 sin importar el color, y el término de luminancia es matemáticamente
    tolerante a diferencias grandes de intensidad. Eso deja un "hueco" que el
    algoritmo genético puede explotar convergiendo a un color plano en vez de
    reproducir la imagen real. Por eso se mezcla con un término de RMSE normalizado
    (`mse_weight`, técnica estándar "SSIM + L1/L2" para evitar ese colapso).
    """

    def __init__(self, win_size: int = 7, mse_weight: float = 0.5) -> None:
        if win_size % 2 == 0:
            raise ValueError("win_size debe ser impar")
        if not 0.0 <= mse_weight <= 1.0:
            raise ValueError("mse_weight debe estar entre 0.0 y 1.0")
        self._win_size = win_size
        self._mse_weight = mse_weight

    def evaluate(
        self,
        individual: TriangleIndividual,
        target: TriangleImageTarget,
        context: EvolutionContext,
    ) -> MSEFitness:
        rendered_image = render(individual, target.width, target.height)
        target_arr = target.image.astype(np.float64)
        rendered_arr = np.array(rendered_image, dtype=np.float64)

        win_size = min(self._win_size, target.height, target.width)
        if win_size % 2 == 0:
            win_size -= 1
        win_size = max(win_size, 1)

        channel_scores = [
            float(np.mean(_channel_ssim(target_arr[..., c], rendered_arr[..., c], win_size)))
            for c in range(target_arr.shape[-1])
        ]
        mean_ssim = sum(channel_scores) / len(channel_scores)
        ssim_term = 1.0 - mean_ssim

        if self._mse_weight <= 0.0:
            return MSEFitness(ssim_term)

        rmse_term = float(np.sqrt(np.mean(np.square(target_arr - rendered_arr)))) / 255.0
        fitness = (1.0 - self._mse_weight) * ssim_term + self._mse_weight * rmse_term

        return MSEFitness(fitness)


class BlurredMSEEvaluator(FitnessEvaluator[TriangleIndividual, TriangleImageTarget, MSEFitness]):
    """MSE calculado sobre versiones desenfocadas (blur gaussiano) de ambas imágenes.

    Aplica un desenfoque gaussiano leve tanto al objetivo como al render antes de
    comparar. Esto evalúa la estructura general (formas grandes, distribución de
    color) en lugar de exigir coincidencia exacta píxel a píxel: pequeños
    corrimientos de bordes o detalles de alta frecuencia se difuminan de forma
    similar en ambas imágenes y dejan de penalizar tan fuerte, mientras que
    diferencias de forma o color a gran escala se siguen notando.
    """

    def __init__(self, sigma: float = 1.5) -> None:
        if sigma < 0:
            raise ValueError("sigma debe ser no negativo")
        self._sigma = sigma

        # El blur del objetivo no cambia durante la corrida: se cachea.
        self._cached_target: TriangleImageTarget | None = None
        self._blurred_target: np.ndarray | None = None

    def _blur(self, image_arr: np.ndarray) -> np.ndarray:
        # sigma=0 en el eje de canales: no mezclamos R, G y B entre sí.
        return gaussian_filter(image_arr, sigma=(self._sigma, self._sigma, 0))

    def _prepare(self, target: TriangleImageTarget) -> None:
        if self._cached_target is target:
            return
        self._blurred_target = self._blur(target.image.astype(np.float32))
        self._cached_target = target

    def evaluate(
        self,
        individual: TriangleIndividual,
        target: TriangleImageTarget,
        context: EvolutionContext,
    ) -> MSEFitness:
        self._prepare(target)
        assert self._blurred_target is not None

        rendered_image = render(individual, target.width, target.height)
        rendered_arr = np.array(rendered_image, dtype=np.float32)
        blurred_rendered = self._blur(rendered_arr)

        mse = np.mean(np.square(self._blurred_target - blurred_rendered))
        return MSEFitness(float(mse))


class MultiScaleMSEEvaluator(FitnessEvaluator[TriangleIndividual, TriangleImageTarget, MSEFitness]):
    """MSE combinado en varias escalas (resoluciones) de la imagen.

    Reescala el objetivo y el render a cada resolución en `scales` (1.0 = resolución
    de trabajo completa) y promedia (con `weights`) el MSE obtenido en cada una. Las
    escalas bajas capturan la estructura gruesa (formas y distribución de color a
    gran escala) mientras que las escalas altas siguen exigiendo precisión de
    detalle fino; combinarlas evita que el algoritmo sobre-ajuste a un solo nivel.
    """

    def __init__(
        self,
        scales: tuple[float, ...] = (1.0, 0.5, 0.25),
        weights: tuple[float, ...] | None = None,
    ) -> None:
        if not scales:
            raise ValueError("scales no puede estar vacío")
        if any(s <= 0 for s in scales):
            raise ValueError("cada escala debe ser positiva")
        if weights is not None and len(weights) != len(scales):
            raise ValueError("weights debe tener la misma longitud que scales")

        self._scales = scales
        raw_weights = weights if weights is not None else tuple(1.0 for _ in scales)
        total = sum(raw_weights)
        self._weights = tuple(w / total for w in raw_weights)

        self._cached_target: TriangleImageTarget | None = None
        self._target_levels: list[np.ndarray] = []

    @staticmethod
    def _resize(image: Image.Image, scale: float) -> np.ndarray:
        if scale >= 1.0:
            return np.array(image, dtype=np.float32)
        width = max(1, round(image.width * scale))
        height = max(1, round(image.height * scale))
        try:
            resample_filter = Image.Resampling.LANCZOS
        except AttributeError:
            resample_filter = Image.LANCZOS
        return np.array(image.resize((width, height), resample_filter), dtype=np.float32)

    def _prepare(self, target: TriangleImageTarget) -> None:
        if self._cached_target is target:
            return
        target_image = Image.fromarray(target.image.astype(np.uint8), mode="RGB")
        self._target_levels = [self._resize(target_image, scale) for scale in self._scales]
        self._cached_target = target

    def evaluate(
        self,
        individual: TriangleIndividual,
        target: TriangleImageTarget,
        context: EvolutionContext,
    ) -> MSEFitness:
        self._prepare(target)
        rendered_image = render(individual, target.width, target.height)

        weighted_sum = 0.0
        for scale, weight, target_level in zip(self._scales, self._weights, self._target_levels):
            rendered_level = self._resize(rendered_image, scale)
            mse = float(np.mean(np.square(target_level - rendered_level)))
            weighted_sum += weight * mse

        return MSEFitness(weighted_sum)


class ColorHistogramEvaluator(FitnessEvaluator[TriangleIndividual, TriangleImageTarget, MSEFitness]):
    """Fitness basado en la distancia entre histogramas de color globales.

    En vez de comparar píxel a píxel, compara la distribución global de colores
    (histograma normalizado por canal RGB) entre el objetivo y el render, sin
    importar dónde está ubicado cada color espacialmente.

    Advertencia: al ser ciego a la posición, un render que reparta los mismos
    colores en cualquier lugar del lienzo puede obtener buen puntaje sin parecerse
    estructuralmente al objetivo (análogo al colapso que se vio con SSIM puro, pero
    más severo porque ignora la posición por completo). Por eso conviene usarlo
    combinado con un evaluador espacial (`mse_weight`, o vía `CompositeEvaluator`)
    en vez de solo.
    """

    def __init__(self, bins: int = 32, mse_weight: float = 0.5) -> None:
        if bins < 2:
            raise ValueError("bins debe ser al menos 2")
        if not 0.0 <= mse_weight <= 1.0:
            raise ValueError("mse_weight debe estar entre 0.0 y 1.0")
        self._bins = bins
        self._mse_weight = mse_weight

        self._cached_target: TriangleImageTarget | None = None
        self._target_hist: np.ndarray | None = None  # (3, bins), normalizado por canal

    def _histogram(self, image_arr: np.ndarray) -> np.ndarray:
        hist = np.stack([
            np.histogram(image_arr[..., c], bins=self._bins, range=(0.0, 255.0))[0]
            for c in range(3)
        ]).astype(np.float64)
        totals = hist.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        return hist / totals

    def _prepare(self, target: TriangleImageTarget) -> None:
        if self._cached_target is target:
            return
        self._target_hist = self._histogram(target.image.astype(np.float64))
        self._cached_target = target

    def evaluate(
        self,
        individual: TriangleIndividual,
        target: TriangleImageTarget,
        context: EvolutionContext,
    ) -> MSEFitness:
        self._prepare(target)
        assert self._target_hist is not None

        rendered_image = render(individual, target.width, target.height)
        rendered_arr = np.array(rendered_image, dtype=np.float64)
        rendered_hist = self._histogram(rendered_arr)

        # Distancia L2 entre distribuciones normalizadas, promediada por canal.
        # 0 = distribuciones idénticas, máximo teórico sqrt(2) (masas disjuntas).
        per_channel = np.sqrt(np.sum(np.square(self._target_hist - rendered_hist), axis=1))
        hist_term = float(np.mean(per_channel))

        if self._mse_weight <= 0.0:
            return MSEFitness(hist_term)

        target_arr = target.image.astype(np.float64)
        rmse_term = float(np.sqrt(np.mean(np.square(target_arr - rendered_arr)))) / 255.0
        fitness = (1.0 - self._mse_weight) * hist_term + self._mse_weight * rmse_term

        return MSEFitness(fitness)


class NormalizedEvaluator(FitnessEvaluator[TriangleIndividual, TriangleImageTarget, MSEFitness]):
    """Envuelve otro evaluador y reescala su fitness a un rango ~[0, 1].

    Cada evaluador de este módulo devuelve su error en unidades distintas (MSE en
    píxeles al cuadrado hasta 65025, SSIM/histograma ya entre 0 y ~1.4, etc.). Para
    poder combinarlos en un fitness compuesto con pesos que signifiquen lo mismo
    entre sí, primero hay que llevarlos a una escala comparable dividiendo por el
    error máximo teórico de cada uno (`scale`). Ver `SCALES` para los valores usados
    por cada evaluador de este módulo.
    """

    def __init__(self, evaluator: FitnessEvaluator[TriangleIndividual, TriangleImageTarget, MSEFitness], scale: float) -> None:
        if scale <= 0:
            raise ValueError("scale debe ser positivo")
        self._evaluator = evaluator
        self._scale = scale

    def evaluate(
        self,
        individual: TriangleIndividual,
        target: TriangleImageTarget,
        context: EvolutionContext,
    ) -> MSEFitness:
        raw = self._evaluator.evaluate(individual, target, context)
        return MSEFitness(raw.value / self._scale)


# Error máximo teórico de cada evaluador de este módulo, para usar con `NormalizedEvaluator`.
# Los basados en MSE de píxeles están acotados por 255**2 (diferencia máxima por canal al
# cuadrado); SSIM y el histograma ya devuelven un término acotado (~1 y sqrt(2) respectivamente).
SCALES: dict[str, float] = {
    "mse": 255.0 ** 2,
    "regional": 255.0 ** 2,
    "blur": 255.0 ** 2,
    "multiscale": 255.0 ** 2,
    "ssim": 1.0,
    "histogram": 2.0 ** 0.5,
}


class CompositeEvaluator(FitnessEvaluator[TriangleIndividual, TriangleImageTarget, MSEFitness]):
    """Combina varios evaluadores (ya normalizados a rangos comparables) en un solo fitness.

    Evalúa cada componente y devuelve la suma ponderada de sus valores (los pesos se
    renormalizan para sumar 1). Permite mezclar criterios distintos, por ejemplo
    `0.4 * MSE + 0.3 * SSIM + 0.3 * histograma de color`, en vez de optimizar un
    solo objetivo a la vez.
    """

    def __init__(
        self,
        components: list[tuple[FitnessEvaluator[TriangleIndividual, TriangleImageTarget, MSEFitness], float]],
    ) -> None:
        if not components:
            raise ValueError("components no puede estar vacío")
        total_weight = sum(weight for _, weight in components)
        if total_weight <= 0:
            raise ValueError("la suma de los pesos debe ser positiva")
        self._components = [(evaluator, weight / total_weight) for evaluator, weight in components]

    def evaluate(
        self,
        individual: TriangleIndividual,
        target: TriangleImageTarget,
        context: EvolutionContext,
    ) -> MSEFitness:
        total = sum(
            weight * evaluator.evaluate(individual, target, context).value
            for evaluator, weight in self._components
        )
        return MSEFitness(total)
