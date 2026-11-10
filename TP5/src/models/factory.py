"""Construcción y persistencia de modelos a partir de la sección `model`."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.models.autoencoder import Autoencoder
from src.models.variational import KlSchedule, VariationalAutoencoder
from src.nn.activations import build_activation
from src.nn.initializers import build_initializer
from src.nn.losses import build_loss
from src.nn.network import MultilayerPerceptron
from src.options import ConfigurationError, Options, choose

AUTOENCODER = "autoencoder"
VARIATIONAL = "variational"
# float32 entrena cerca del doble de rápido en redes grandes (espectrogramas).
PRECISIONS: dict[str, type[np.floating]] = {"float64": np.float64, "float32": np.float32}


def build_model(
    options: Options,
    input_size: int,
    initialization_rng: np.random.Generator,
    sampling_rng: np.random.Generator,
) -> Autoencoder:
    """Lee la sección `model`. Todas las claves son opcionales.

    `decoder_hidden_layers` omitido (o null) refleja al codificador.
    """
    model_type = options.take("type", AUTOENCODER)
    latent_size = options.take("latent_size", 2)
    encoder_hidden = _layer_list(options.take("hidden_layers", [16, 8]), "hidden_layers")
    decoder_hidden = options.take("decoder_hidden_layers", None)
    decoder_hidden = (
        encoder_hidden[::-1]
        if decoder_hidden is None
        else _layer_list(decoder_hidden, "decoder_hidden_layers")
    )
    hidden_activation = build_activation(options.take("activation", "tanh"))
    latent_activation = build_activation(options.take("latent_activation", "identity"))
    output_activation = build_activation(options.take("output_activation", "sigmoid"))
    initializer = build_initializer(options.take("initializer", "xavier"))
    loss = build_loss(options.take("loss", "bce"), output_activation)
    precision = choose(PRECISIONS, options.take("precision", "float64"), "Precisión")
    kl_options = options.optional_section("kl")
    options.ensure_consumed()

    if not isinstance(latent_size, int) or latent_size < 1:
        raise ConfigurationError(f"model.latent_size debe ser un entero positivo: {latent_size}")
    if model_type not in (AUTOENCODER, VARIATIONAL):
        raise ConfigurationError(f"model.type debe ser '{AUTOENCODER}' o '{VARIATIONAL}'")
    if kl_options is not None and model_type != VARIATIONAL:
        raise ConfigurationError("model.kl solo aplica a model.type 'variational'")

    is_variational = model_type == VARIATIONAL
    encoder_outputs = 2 * latent_size if is_variational else latent_size
    encoder = MultilayerPerceptron(
        [input_size, *encoder_hidden, encoder_outputs],
        hidden_activation,
        latent_activation,
        initializer,
        initialization_rng,
        precision,
    )
    decoder = MultilayerPerceptron(
        [latent_size, *decoder_hidden, input_size],
        hidden_activation,
        build_activation("identity"),
        initializer,
        initialization_rng,
        precision,
    )
    if not is_variational:
        return Autoencoder(encoder, decoder, output_activation, loss)
    return VariationalAutoencoder(
        encoder, decoder, output_activation, loss, _build_kl_schedule(kl_options), sampling_rng
    )


def save_parameters(model: Autoencoder, path: Path) -> None:
    arrays = {f"parameter_{index}": array for index, array in enumerate(model.parameters())}
    np.savez_compressed(path, **arrays)


def load_parameters(model: Autoencoder, path: Path) -> None:
    """Copia en `model` los pesos guardados; la arquitectura debe coincidir."""
    with np.load(path, allow_pickle=False) as saved:
        parameters = model.parameters()
        if len(saved.files) != len(parameters):
            raise ValueError(f"{path} no corresponde a la arquitectura del modelo")
        for index, parameter in enumerate(parameters):
            stored = saved[f"parameter_{index}"]
            if stored.shape != parameter.shape:
                raise ValueError(f"{path} no corresponde a la arquitectura del modelo")
            parameter[...] = stored


def _build_kl_schedule(options: Options | None) -> KlSchedule:
    if options is None:
        return KlSchedule()
    schedule = KlSchedule(options.take("weight", 1.0), options.take("warmup_epochs", 0))
    options.ensure_consumed()
    return schedule


def _layer_list(value: object, key: str) -> list[int]:
    if not isinstance(value, list) or any(not isinstance(size, int) or size < 1 for size in value):
        raise ConfigurationError(f"model.{key} debe ser una lista de enteros positivos: {value!r}")
    return value
