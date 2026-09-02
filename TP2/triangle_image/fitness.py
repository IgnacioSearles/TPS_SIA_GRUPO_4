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

    def __init__(self, image: Image.Image) -> None:
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
        rendered_array = np.array(rendered_image, dtype=np.int16)

        # Calculamos MSE
        # np.mean maneja automáticamente la suma total dividida por la cantidad de elementos
        mse = np.mean(np.square(target.image - rendered_array))
        
        return MSEFitness(float(mse))
