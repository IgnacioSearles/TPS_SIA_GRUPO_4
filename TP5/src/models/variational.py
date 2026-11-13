"""Autoencoder variacional con posterior gaussiana diagonal y prior N(0, I)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.models.autoencoder import Array, Autoencoder, LossTerms
from src.nn.activations import Activation
from src.nn.losses import ReconstructionLoss
from src.nn.network import MultilayerPerceptron
from src.options import ConfigurationError


@dataclass(frozen=True)
class KlSchedule:
    """Peso (beta) del término KL, con rampa lineal opcional desde cero."""

    weight: float = 1.0
    warmup_epochs: int = 0

    def __post_init__(self) -> None:
        if self.weight < 0 or self.warmup_epochs < 0:
            raise ConfigurationError("kl.weight y kl.warmup_epochs no pueden ser negativos")

    def weight_at(self, epoch: int) -> float:
        if self.warmup_epochs == 0:
            return self.weight
        return self.weight * min(1.0, epoch / self.warmup_epochs)


class VariationalAutoencoder(Autoencoder):
    """El codificador emite [media, log-varianza]; `encode` devuelve la media."""

    def __init__(
        self,
        encoder: MultilayerPerceptron,
        decoder: MultilayerPerceptron,
        output_activation: Activation,
        reconstruction_loss: ReconstructionLoss,
        kl_schedule: KlSchedule,
        rng: np.random.Generator,
    ) -> None:
        super().__init__(encoder, decoder, output_activation, reconstruction_loss)
        if encoder.layer_sizes[-1] != 2 * self.latent_size:
            raise ValueError("El codificador de un VAE debe emitir 2 * latent_size valores")
        self._kl_schedule = kl_schedule
        self._rng = rng

    def encode(self, samples: Array) -> Array:
        mean, _ = self.encode_distribution(samples)
        return mean

    def encode_distribution(self, samples: Array) -> tuple[Array, Array]:
        """Media y log-varianza de q(z|x)."""
        statistics = self._encoder.forward(samples)
        return statistics[:, : self.latent_size], statistics[:, self.latent_size :]

    def loss_and_gradients(
        self, inputs: Array, targets: Array, epoch: int
    ) -> tuple[LossTerms, list[Array]]:
        noise = self._rng.standard_normal((len(inputs), self.latent_size), dtype=self.precision)
        return self.objective(inputs, targets, noise, self._kl_schedule.weight_at(epoch))

    def objective(
        self, inputs: Array, targets: Array, noise: Array, kl_weight: float
    ) -> tuple[LossTerms, list[Array]]:
        """ELBO negativa media: reconstrucción + kl_weight * KL, con `noise` explícito."""
        mean, log_variance = self.encode_distribution(inputs)
        deviation = np.exp(0.5 * log_variance)
        logits = self._decoder.forward(mean + deviation * noise)
        reconstruction = float(np.mean(self._reconstruction_loss.per_sample(logits, targets)))
        divergence = float(
            np.mean(-0.5 * np.sum(1.0 + log_variance - mean**2 - deviation**2, axis=1))
        )

        latent_gradient = self._backpropagate_reconstruction(logits, targets)
        scale = kl_weight / len(inputs)
        mean_gradient = latent_gradient + scale * mean
        log_variance_gradient = (
            latent_gradient * noise * 0.5 * deviation + scale * 0.5 * (deviation**2 - 1.0)
        )
        self._encoder.backward(np.hstack([mean_gradient, log_variance_gradient]))

        terms = {
            "loss": reconstruction + kl_weight * divergence,
            "reconstruction": reconstruction,
            "kl": divergence,
            "kl_weight": kl_weight,
        }
        return terms, self._gradients()

    def interpolation_loss_and_gradients(self, lower: Array, upper: Array, targets: Array):
        """Reconstruir el centro desde la media de los códigos de dos vecinos.

        Usa las medias posteriores (sin muestreo ni KL adicional). El gradiente
        vuelve a ambos extremos con factor 1/2; log-varianza recibe cero directo.
        La escala es la misma de reconstrucción: suma de características por
        muestra y promedio sobre el lote.
        """
        if (lower.ndim != 2 or lower.shape != upper.shape or lower.shape != targets.shape
                or len(lower) == 0 or lower.shape[1] != self.input_size):
            raise ValueError("Se requieren tres lotes no vacíos N x input_size de igual forma")
        count = len(lower)
        statistics = self._encoder.forward(np.concatenate([lower, upper]))
        means = statistics[:, :self.latent_size]
        logits = self._decoder.forward((means[:count] + means[count:]) * 0.5)
        loss = float(np.mean(self._reconstruction_loss.per_sample(logits, targets)))
        latent_gradient = self._backpropagate_reconstruction(logits, targets)
        encoder_gradient = np.zeros_like(statistics)
        encoder_gradient[:count, :self.latent_size] = latent_gradient * 0.5
        encoder_gradient[count:, :self.latent_size] = latent_gradient * 0.5
        self._encoder.backward(encoder_gradient)
        return loss, self._gradients()
