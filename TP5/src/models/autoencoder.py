"""Autoencoder determinista: codificador y decodificador densos."""

from __future__ import annotations

import numpy as np

from src.nn.activations import Activation
from src.nn.losses import ReconstructionLoss
from src.nn.network import MultilayerPerceptron

Array = np.ndarray
LossTerms = dict[str, float]


class Autoencoder:
    """Modelo = arquitectura + objetivo.

    El decodificador emite logits; `output_activation` los lleva al rango de
    los datos y la pérdida de reconstrucción se evalúa sobre esos logits.
    """

    def __init__(
        self,
        encoder: MultilayerPerceptron,
        decoder: MultilayerPerceptron,
        output_activation: Activation,
        reconstruction_loss: ReconstructionLoss,
    ) -> None:
        self._encoder = encoder
        self._decoder = decoder
        self._output_activation = output_activation
        self._reconstruction_loss = reconstruction_loss
        self.input_size = encoder.layer_sizes[0]
        self.latent_size = decoder.layer_sizes[0]

    @property
    def precision(self) -> np.dtype:
        """Tipo de punto flotante de los pesos; las entradas de entrenamiento deben usarlo."""
        return self.parameters()[0].dtype

    def encode(self, samples: Array) -> Array:
        return self._encoder.forward(samples)

    def decode(self, latent: Array) -> Array:
        return self._output_activation.apply(self._decoder.forward(latent))

    def reconstruct(self, samples: Array) -> Array:
        return self.decode(self.encode(samples))

    def loss_and_gradients(
        self, inputs: Array, targets: Array, epoch: int
    ) -> tuple[LossTerms, list[Array]]:
        """Pérdida media del lote y gradientes alineados con `parameters()`.

        `epoch` forma parte del contrato porque algunos objetivos (VAE) cambian
        con el avance del entrenamiento; este modelo no lo usa.
        """
        logits = self._decoder.forward(self._encoder.forward(inputs))
        loss = float(np.mean(self._reconstruction_loss.per_sample(logits, targets)))
        self._encoder.backward(self._backpropagate_reconstruction(logits, targets))
        return {"loss": loss}, self._gradients()

    def parameters(self) -> list[Array]:
        return [*self._encoder.parameters(), *self._decoder.parameters()]

    def _backpropagate_reconstruction(self, logits: Array, targets: Array) -> Array:
        """Retropropaga por el decodificador; devuelve el gradiente del latente."""
        logit_gradient = self._reconstruction_loss.logit_gradient(logits, targets) / len(targets)
        return self._decoder.backward(logit_gradient)

    def _gradients(self) -> list[Array]:
        return [*self._encoder.gradients(), *self._decoder.gradients()]
