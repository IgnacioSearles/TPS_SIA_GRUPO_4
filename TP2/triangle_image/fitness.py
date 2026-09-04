"""Imagen objetivo y evaluación de fitness por comparación de píxeles."""

from __future__ import annotations

import math
from typing import Any
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter, sobel, uniform_filter

from genetic_algorithm.domain.contracts import (EvolutionContext, Fitness,
                                                 FitnessComparator, FitnessEvaluator,
                                                 ImageTarget)
from triangle_image.gene import TriangleIndividual
from triangle_image.rendering import render

_TARGET_BACKGROUND = (255, 255, 255)
_TRANSPARENT_BACKGROUND_MSE_WEIGHT = 0.5


def _resample_filter() -> int:
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS


def _flatten_on_background(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Compone transparencias sobre el mismo fondo que usa el renderizador."""
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (*_TARGET_BACKGROUND, 255))
    background.alpha_composite(rgba)
    return background.convert("RGB"), rgba.getchannel("A")


def _mse_weights_from_alpha(alpha: np.ndarray) -> np.ndarray | None:
    """Da más peso al contenido visible que al fondo transparente del objetivo."""
    if np.all(alpha >= 1.0):
        return None
    return _TRANSPARENT_BACKGROUND_MSE_WEIGHT + (
        1.0 - _TRANSPARENT_BACKGROUND_MSE_WEIGHT
    ) * alpha


def _mean_squared_rgb_error(
    target_arr: np.ndarray,
    rendered_arr: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    diff_sq = np.square(target_arr - rendered_arr)
    if weights is None:
        return float(np.mean(diff_sq))
    return float(np.average(np.mean(diff_sq, axis=2), weights=weights))


class TriangleImageTarget(ImageTarget[np.ndarray]):
    """Imagen objetivo almacenada como array de numpy para cálculos rápidos."""

    def __init__(self, image: Image.Image, max_size: int | None = None) -> None:
        self._orig_width, self._orig_height = image.size
        self._scale_factor = 1.0
        flattened, alpha = _flatten_on_background(image)

        if max_size is not None and max(self._orig_width, self._orig_height) > max_size:
            self._scale_factor = max_size / float(max(self._orig_width, self._orig_height))
            new_width = max(1, round(self._orig_width * self._scale_factor))
            new_height = max(1, round(self._orig_height * self._scale_factor))
            size = (new_width, new_height)
            resample_filter = _resample_filter()
            flattened = flattened.resize(size, resample_filter)
            alpha = alpha.resize(size, resample_filter)

        self._width, self._height = flattened.size
        # Usamos int16 para evitar overflow al restar píxeles
        self._image_array = np.array(flattened, dtype=np.int16)
        alpha_array = np.asarray(alpha, dtype=np.float32) / 255.0
        self._mse_weights = _mse_weights_from_alpha(alpha_array)

    @property
    def image(self) -> np.ndarray:
        return self._image_array

    @property
    def mse_weights(self) -> np.ndarray | None:
        return self._mse_weights

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
    """Fitness normalizada en [0, 1], donde un valor mayor es mejor."""

    def __init__(self, error: float) -> None:
        self._error = error

    @property
    def value(self) -> float:
        return self._error


class MSEComparator(FitnessComparator[MSEFitness]):
    """Compara fitness ascendentes: un valor mayor representa una mejor solución."""

    def is_better(self, left: MSEFitness, right: MSEFitness) -> bool:
        return left.value > right.value


def _error_to_fitness(error: float, maximum_error: float) -> float:
    """Convierte un error acotado en una similitud normalizada ascendente."""
    if maximum_error <= 0:
        raise ValueError("maximum_error debe ser positivo")
    return float(np.clip(1.0 - error / maximum_error, 0.0, 1.0))


def _render_individual(
    individual: TriangleIndividual,
    target: TriangleImageTarget,
    context: EvolutionContext,
) -> Image.Image:
    """Renderiza usando el cache efímero del contexto, si está disponible."""
    cached_renderer = getattr(context, "render_individual", None)
    if callable(cached_renderer):
        return cached_renderer(individual, target.width, target.height)
    return render(individual, target.width, target.height)


def _render_array(
    individual: TriangleIndividual,
    target: TriangleImageTarget,
    context: EvolutionContext,
    dtype: Any = np.float32,
) -> np.ndarray:
    """Obtiene el array renderizado, reutilizando el cache del contexto si existe."""
    cached_renderer = getattr(context, "render_array", None)
    if callable(cached_renderer):
        return np.asarray(
            cached_renderer(individual, target.width, target.height), dtype=dtype
        )
    return np.asarray(_render_individual(individual, target, context), dtype=dtype)


def _global_mse(
    individual: TriangleIndividual,
    target: TriangleImageTarget,
    context: EvolutionContext,
) -> float:
    """MSE RGB global, compartido por los evaluadores que lo necesitan."""
    cached_mse = getattr(context, "global_mse", None)
    if callable(cached_mse):
        return float(
            cached_mse(
                individual,
                target.image,
                target.width,
                target.height,
                target.mse_weights,
            )
        )
    target_arr = target.image.astype(np.float32)
    rendered_arr = _render_array(individual, target, context, np.float32)
    return _mean_squared_rgb_error(target_arr, rendered_arr, target.mse_weights)


class MSEEvaluator(FitnessEvaluator[TriangleIndividual, TriangleImageTarget, MSEFitness]):
    """Evalúa renderizando los triángulos y calculando el MSE con numpy."""

    def __init__(self) -> None:
        self._cached_target: TriangleImageTarget | None = None
        self._target_arr: np.ndarray | None = None

    def _prepare(self, target: TriangleImageTarget) -> None:
        if self._cached_target is target:
            return
        self._target_arr = target.image.astype(np.float32)
        self._cached_target = target

    def evaluate(
        self,
        individual: TriangleIndividual,
        target: TriangleImageTarget,
        context: EvolutionContext,
    ) -> MSEFitness:
        self._prepare(target)
        assert self._target_arr is not None

        mse = _global_mse(individual, target, context)

        return MSEFitness(_error_to_fitness(float(mse), 255.0 ** 2))


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
        self._target_arr: np.ndarray | None = None
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

        self._target_arr = target.image.astype(np.float32)
        detail = np.zeros((self._grid_rows, self._grid_cols), dtype=np.float32)
        for i, (r0, r1) in enumerate(self._row_bounds):
            for j, (c0, c1) in enumerate(self._col_bounds):
                cell = self._target_arr[r0:r1, c0:c1]
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
        assert self._target_arr is not None

        rendered_image = _render_individual(individual, target, context)
        rendered_arr = np.array(rendered_image)
        diff_sq = np.square(self._target_arr - rendered_arr)
        alpha_weights = target.mse_weights

        weighted_sum = 0.0
        for i, (r0, r1) in enumerate(self._row_bounds):
            for j, (c0, c1) in enumerate(self._col_bounds):
                cell = diff_sq[r0:r1, c0:c1]
                if cell.size == 0:
                    continue
                if alpha_weights is None:
                    region_mse = float(np.mean(cell))
                else:
                    region_mse = float(
                        np.average(
                            np.mean(cell, axis=2),
                            weights=alpha_weights[r0:r1, c0:c1],
                        )
                    )
                weighted_sum += float(self._weights[i, j]) * region_mse

        return MSEFitness(_error_to_fitness(weighted_sum, 255.0 ** 2))


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
    1 = coincidencia perfecta, el término de error se convierte a una similitud
    normalizada ascendente.

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
        target_arr = target.image.astype(np.float32)
        rendered_arr = _render_array(individual, target, context, np.float32)

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
            return MSEFitness(_error_to_fitness(ssim_term, 2.0))

        rmse_term = float(np.sqrt(_global_mse(individual, target, context))) / 255.0
        error = (1.0 - self._mse_weight) * ssim_term + self._mse_weight * rmse_term
        maximum_error = (1.0 - self._mse_weight) * 2.0 + self._mse_weight

        return MSEFitness(_error_to_fitness(error, maximum_error))


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

        rendered_arr = _render_array(individual, target, context, np.float32)
        blurred_rendered = self._blur(rendered_arr)

        mse = _mean_squared_rgb_error(
            self._blurred_target, blurred_rendered, target.mse_weights
        )
        return MSEFitness(_error_to_fitness(float(mse), 255.0 ** 2))


class EdgeMSEEvaluator(FitnessEvaluator[TriangleIndividual, TriangleImageTarget, MSEFitness]):
    """MSE entre mapas de bordes Sobel en escala de grises.

    Prioriza que los límites de las formas estén en la misma posición que en la
    imagen objetivo. Opcionalmente suaviza antes de detectar bordes para reducir
    el efecto de píxeles aislados y ruido de alta frecuencia.
    """

    def __init__(self, blur_sigma: float = 1.0) -> None:
        if blur_sigma < 0:
            raise ValueError("blur_sigma debe ser no negativo")
        self._blur_sigma = blur_sigma
        self._cached_target: TriangleImageTarget | None = None
        self._target_edges: np.ndarray | None = None

    def _edge_map(self, image_arr: np.ndarray) -> np.ndarray:
        # Rec. 709: convierte RGB a luminancia sin mezclar las derivadas por canal.
        grayscale = np.dot(image_arr[..., :3], (0.2126, 0.7152, 0.0722))
        if self._blur_sigma > 0:
            grayscale = gaussian_filter(grayscale, sigma=self._blur_sigma)

        gradient_x = sobel(grayscale, axis=1, mode="reflect")
        gradient_y = sobel(grayscale, axis=0, mode="reflect")
        # Sobel tiene ganancia máxima de 4 por eje. Se normaliza y recorta para
        # mantener el mapa de bordes en [0, 255], igual que una imagen monocanal.
        return np.clip(np.hypot(gradient_x, gradient_y) / 4.0, 0.0, 255.0)

    def _prepare(self, target: TriangleImageTarget) -> None:
        if self._cached_target is target:
            return
        self._target_edges = self._edge_map(target.image.astype(np.float32))
        self._cached_target = target

    def evaluate(
        self,
        individual: TriangleIndividual,
        target: TriangleImageTarget,
        context: EvolutionContext,
    ) -> MSEFitness:
        self._prepare(target)
        assert self._target_edges is not None

        rendered_edges = self._edge_map(_render_array(individual, target, context, np.float32))
        error = float(np.mean(np.square(self._target_edges - rendered_edges)))
        return MSEFitness(_error_to_fitness(error, 255.0 ** 2))


def _luminance(image_arr: np.ndarray) -> np.ndarray:
    """Convierte RGB a luminancia Rec. 709."""
    return np.dot(image_arr[..., :3], (0.2126, 0.7152, 0.0722))


def _sobel_gradient(image_arr: np.ndarray, blur_sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve derivadas Sobel y su magnitud, normalizada a un máximo de 255."""
    grayscale = _luminance(image_arr)
    if blur_sigma > 0:
        grayscale = gaussian_filter(grayscale, sigma=blur_sigma)
    gradient_x = sobel(grayscale, axis=1, mode="reflect")
    gradient_y = sobel(grayscale, axis=0, mode="reflect")
    magnitude = np.clip(np.hypot(gradient_x, gradient_y) / 4.0, 0.0, 255.0)
    return gradient_x, gradient_y, magnitude


class GradientOrientationEvaluator(FitnessEvaluator[TriangleIndividual, TriangleImageTarget, MSEFitness]):
    """Compara intensidad y orientación de gradientes Sobel.

    El componente de orientación trata un borde claro-oscuro y uno oscuro-claro
    como el mismo contorno: compara la dirección de la recta, no el signo del
    gradiente. El resultado está normalizado en el rango aproximado [0, 1].
    """

    def __init__(self, blur_sigma: float = 1.0, orientation_weight: float = 0.5) -> None:
        if blur_sigma < 0:
            raise ValueError("blur_sigma debe ser no negativo")
        if not 0.0 <= orientation_weight <= 1.0:
            raise ValueError("orientation_weight debe estar entre 0.0 y 1.0")
        self._blur_sigma = blur_sigma
        self._orientation_weight = orientation_weight
        self._cached_target: TriangleImageTarget | None = None
        self._target_gradient: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def _prepare(self, target: TriangleImageTarget) -> None:
        if self._cached_target is target:
            return
        self._target_gradient = _sobel_gradient(target.image.astype(np.float32), self._blur_sigma)
        self._cached_target = target

    def evaluate(
        self,
        individual: TriangleIndividual,
        target: TriangleImageTarget,
        context: EvolutionContext,
    ) -> MSEFitness:
        self._prepare(target)
        assert self._target_gradient is not None
        target_x, target_y, target_magnitude = self._target_gradient
        rendered = _render_array(individual, target, context, np.float32)
        rendered_x, rendered_y, rendered_magnitude = _sobel_gradient(rendered, self._blur_sigma)

        magnitude_error = float(np.mean(np.square(target_magnitude - rendered_magnitude))) / (255.0 ** 2)
        denominator = np.hypot(target_x, target_y) * np.hypot(rendered_x, rendered_y)
        cosine = np.divide(
            target_x * rendered_x + target_y * rendered_y,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 1e-8,
        )
        # abs(cos) hace equivalente a theta y theta + pi: importa la orientación del borde.
        orientation_error = 1.0 - np.abs(cosine)
        edge_weight = np.maximum(target_magnitude, rendered_magnitude) / 255.0
        orientation_score = float(np.sum(orientation_error * edge_weight) / max(np.sum(edge_weight), 1e-8))

        score = ((1.0 - self._orientation_weight) * magnitude_error
                 + self._orientation_weight * orientation_score)
        return MSEFitness(_error_to_fitness(score, 1.0))


class ChamferEdgeEvaluator(FitnessEvaluator[TriangleIndividual, TriangleImageTarget, MSEFitness]):
    """Distancia Chamfer simétrica entre mapas de bordes Sobel binarizados.

    A diferencia del MSE de bordes, penaliza suavemente un contorno desplazado
    pocos píxeles; por eso es más adecuado durante etapas tempranas de evolución.
    El resultado se normaliza por la diagonal de la imagen y queda en [0, 1].
    """

    def __init__(self, blur_sigma: float = 1.0, threshold: float = 20.0) -> None:
        if blur_sigma < 0:
            raise ValueError("blur_sigma debe ser no negativo")
        if not 0.0 <= threshold <= 255.0:
            raise ValueError("threshold debe estar entre 0 y 255")
        self._blur_sigma = blur_sigma
        self._threshold = threshold
        self._cached_target: TriangleImageTarget | None = None
        self._target_edges: np.ndarray | None = None

    def _edge_mask(self, image_arr: np.ndarray) -> np.ndarray:
        _, _, magnitude = _sobel_gradient(image_arr, self._blur_sigma)
        return magnitude >= self._threshold

    def _prepare(self, target: TriangleImageTarget) -> None:
        if self._cached_target is target:
            return
        self._target_edges = self._edge_mask(target.image.astype(np.float32))
        self._cached_target = target

    @staticmethod
    def _directed_distance(source: np.ndarray, destination: np.ndarray, diagonal: float) -> float:
        if not np.any(source):
            return 0.0 if not np.any(destination) else 1.0
        if not np.any(destination):
            return 1.0
        distances = distance_transform_edt(~destination)
        return float(np.mean(distances[source]) / diagonal)

    def evaluate(
        self,
        individual: TriangleIndividual,
        target: TriangleImageTarget,
        context: EvolutionContext,
    ) -> MSEFitness:
        self._prepare(target)
        assert self._target_edges is not None
        rendered = _render_array(individual, target, context, np.float32)
        rendered_edges = self._edge_mask(rendered)
        diagonal = math.hypot(target.width, target.height)
        score = (self._directed_distance(rendered_edges, self._target_edges, diagonal)
                 + self._directed_distance(self._target_edges, rendered_edges, diagonal)) / 2.0
        return MSEFitness(_error_to_fitness(score, 1.0))


class SaliencyMSEEvaluator(FitnessEvaluator[TriangleIndividual, TriangleImageTarget, MSEFitness]):
    """MSE que da mayor importancia a zonas visualmente salientes del objetivo.

    La saliencia se estima con la magnitud de gradiente suavizada: bordes y zonas
    de alto contraste pesan más, pero todo píxel mantiene un peso base de uno.
    """

    def __init__(self, saliency_weight: float = 3.0, blur_sigma: float = 2.0) -> None:
        if saliency_weight < 0:
            raise ValueError("saliency_weight debe ser no negativo")
        if blur_sigma < 0:
            raise ValueError("blur_sigma debe ser no negativo")
        self._saliency_weight = saliency_weight
        self._blur_sigma = blur_sigma
        self._cached_target: TriangleImageTarget | None = None
        self._weights: np.ndarray | None = None

    def _prepare(self, target: TriangleImageTarget) -> None:
        if self._cached_target is target:
            return
        _, _, magnitude = _sobel_gradient(target.image.astype(np.float32), 0.0)
        saliency = gaussian_filter(magnitude, sigma=self._blur_sigma) if self._blur_sigma > 0 else magnitude
        maximum = float(saliency.max())
        normalized = saliency / maximum if maximum > 0 else np.zeros_like(saliency)
        self._weights = 1.0 + self._saliency_weight * normalized
        if target.mse_weights is not None:
            self._weights = self._weights * target.mse_weights
        self._cached_target = target

    def evaluate(
        self,
        individual: TriangleIndividual,
        target: TriangleImageTarget,
        context: EvolutionContext,
    ) -> MSEFitness:
        self._prepare(target)
        assert self._weights is not None
        rendered = _render_array(individual, target, context, np.float32)
        diff_squared = np.mean(np.square(target.image.astype(np.float32) - rendered), axis=2)
        error = float(np.average(diff_squared, weights=self._weights))
        return MSEFitness(_error_to_fitness(error, 255.0 ** 2))


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
        self._weight_levels: list[np.ndarray | None] = []

    @staticmethod
    def _resize(image: Image.Image, scale: float) -> np.ndarray:
        if scale >= 1.0:
            return np.array(image, dtype=np.float32)
        width = max(1, round(image.width * scale))
        height = max(1, round(image.height * scale))
        resample_filter = _resample_filter()
        return np.array(image.resize((width, height), resample_filter), dtype=np.float32)

    @staticmethod
    def _resize_weights(weights: np.ndarray | None, scale: float) -> np.ndarray | None:
        if weights is None or scale >= 1.0:
            return weights
        width = max(1, round(weights.shape[1] * scale))
        height = max(1, round(weights.shape[0] * scale))
        image = Image.fromarray(weights.astype(np.float32), mode="F")
        resized = np.array(image.resize((width, height), _resample_filter()), dtype=np.float32)
        return np.clip(resized, _TRANSPARENT_BACKGROUND_MSE_WEIGHT, 1.0)

    def _prepare(self, target: TriangleImageTarget) -> None:
        if self._cached_target is target:
            return
        target_image = Image.fromarray(target.image.astype(np.uint8), mode="RGB")
        self._target_levels = [self._resize(target_image, scale) for scale in self._scales]
        self._weight_levels = [
            self._resize_weights(target.mse_weights, scale) for scale in self._scales
        ]
        self._cached_target = target

    def evaluate(
        self,
        individual: TriangleIndividual,
        target: TriangleImageTarget,
        context: EvolutionContext,
    ) -> MSEFitness:
        self._prepare(target)
        rendered_image = _render_individual(individual, target, context)

        weighted_sum = 0.0
        for scale, weight, target_level, weight_level in zip(
            self._scales, self._weights, self._target_levels, self._weight_levels
        ):
            rendered_level = self._resize(rendered_image, scale)
            mse = _mean_squared_rgb_error(target_level, rendered_level, weight_level)
            weighted_sum += weight * mse

        return MSEFitness(_error_to_fitness(weighted_sum, 255.0 ** 2))


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

        rendered_arr = _render_array(individual, target, context, np.float64)
        rendered_hist = self._histogram(rendered_arr)

        # Distancia L2 entre distribuciones normalizadas, promediada por canal.
        # 0 = distribuciones idénticas, máximo teórico sqrt(2) (masas disjuntas).
        per_channel = np.sqrt(np.sum(np.square(self._target_hist - rendered_hist), axis=1))
        hist_term = float(np.mean(per_channel))

        if self._mse_weight <= 0.0:
            return MSEFitness(_error_to_fitness(hist_term, 2.0 ** 0.5))

        rmse_term = float(np.sqrt(_global_mse(individual, target, context))) / 255.0
        error = (1.0 - self._mse_weight) * hist_term + self._mse_weight * rmse_term
        maximum_error = (1.0 - self._mse_weight) * (2.0 ** 0.5) + self._mse_weight

        return MSEFitness(_error_to_fitness(error, maximum_error))


class NormalizedEvaluator(FitnessEvaluator[TriangleIndividual, TriangleImageTarget, MSEFitness]):
    """Adaptador legado para evaluadores que ya devuelven fitness normalizada.

    `scale` se conserva por compatibilidad, pero las métricas actuales ya
    normalizan internamente y no deben dividirse nuevamente.
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
        return MSEFitness(float(np.clip(raw.value, 0.0, 1.0)))


# Límites teóricos conservados para identificar métricas en la configuración y para
# compatibilidad con consumidores externos. La normalización se hace dentro de cada
# evaluador; no se aplican automáticamente al construir un combo.
SCALES: dict[str, float] = {
    "mse": 255.0 ** 2,
    "regional": 255.0 ** 2,
    "blur": 255.0 ** 2,
    "multiscale": 255.0 ** 2,
    "ssim": 1.0,
    "histogram": 2.0 ** 0.5,
    "edge": 255.0 ** 2,
    "gradient": 1.0,
    "chamfer": 1.0,
    "saliency": 255.0 ** 2,
}


class CompositeEvaluator(FitnessEvaluator[TriangleIndividual, TriangleImageTarget, MSEFitness]):
    """Combina fitness normalizadas mediante una suma ponderada ascendente."""

    def __init__(
        self,
        components: list[tuple[FitnessEvaluator[TriangleIndividual, TriangleImageTarget, MSEFitness], float]],
    ) -> None:
        if not components:
            raise ValueError("components no puede estar vacío")
        if any(not math.isfinite(weight) or weight < 0 for _, weight in components):
            raise ValueError("cada peso debe ser un número finito no negativo")
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
        begin_scope = getattr(context, "begin_render_scope", None)
        end_scope = getattr(context, "end_render_scope", None)
        if callable(begin_scope):
            begin_scope()
        try:
            total = sum(
                weight * evaluator.evaluate(individual, target, context).value
                for evaluator, weight in self._components
            )
            return MSEFitness(total)
        finally:
            if callable(end_scope):
                end_scope()
