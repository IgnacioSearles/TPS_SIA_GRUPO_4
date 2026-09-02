"""Imagen objetivo y evaluación de fitness por comparación de píxeles."""

from __future__ import annotations

from typing import Any
import numpy as np
from PIL import Image

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
