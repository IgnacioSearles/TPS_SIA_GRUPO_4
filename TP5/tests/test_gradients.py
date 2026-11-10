"""Los gradientes analíticos deben coincidir con diferencias finitas."""

import numpy as np
import pytest

from src.models.factory import build_model
from src.models.variational import KlSchedule
from src.nn.optimizers import build_optimizer
from src.options import Options

STEP = 1e-6
TOLERANCE = 1e-6
INPUT_SIZE = 6
CHECKS_PER_PARAMETER = 4


def _build(model_config):
    return build_model(
        Options(model_config, "model"),
        INPUT_SIZE,
        np.random.default_rng(0),
        np.random.default_rng(1),
    )


def _assert_gradients_match(model, evaluate):
    """`evaluate()` debe devolver (pérdida, gradientes) de forma determinista."""
    _, analytic = evaluate()
    analytic = [gradient.copy() for gradient in analytic]
    rng = np.random.default_rng(2)
    for parameter, gradient in zip(model.parameters(), analytic):
        for _ in range(CHECKS_PER_PARAMETER):
            index = tuple(rng.integers(size) for size in parameter.shape)
            original = parameter[index]
            parameter[index] = original + STEP
            loss_plus, _ = evaluate()
            parameter[index] = original - STEP
            loss_minus, _ = evaluate()
            parameter[index] = original
            numeric = (loss_plus - loss_minus) / (2 * STEP)
            assert gradient[index] == pytest.approx(numeric, abs=TOLERANCE)


@pytest.mark.parametrize(
    "model_config",
    [
        {"hidden_layers": [5, 4], "activation": "tanh", "loss": "bce"},
        {"hidden_layers": [5], "activation": "sigmoid", "loss": "mse"},
        {
            "hidden_layers": [5],
            "activation": "softplus",
            "loss": "mse",
            "output_activation": "identity",
            "decoder_hidden_layers": [3, 4],
        },
        {"hidden_layers": [], "activation": "leaky_relu", "latent_activation": "tanh"},
    ],
)
def test_autoencoder_gradients(model_config):
    model = _build(model_config)
    rng = np.random.default_rng(3)
    targets = rng.integers(0, 2, (3, INPUT_SIZE)).astype(float)
    inputs = rng.random((3, INPUT_SIZE))  # entrada distinta al objetivo, como en denoising

    def evaluate():
        terms, gradients = model.loss_and_gradients(inputs, targets, epoch=1)
        return terms["loss"], gradients

    _assert_gradients_match(model, evaluate)


@pytest.mark.parametrize("loss", ["bce", "mse"])
@pytest.mark.parametrize("kl_weight", [0.0, 0.7])
def test_variational_gradients(loss, kl_weight):
    model = _build({"type": "variational", "latent_size": 3, "hidden_layers": [5], "loss": loss})
    rng = np.random.default_rng(4)
    samples = rng.random((4, INPUT_SIZE))
    noise = rng.standard_normal((4, 3))

    def evaluate():
        terms, gradients = model.objective(samples, samples, noise, kl_weight)
        return terms["loss"], gradients

    _assert_gradients_match(model, evaluate)


def test_variational_kl_is_zero_for_standard_normal_posterior():
    model = _build({"type": "variational", "latent_size": 2, "hidden_layers": []})
    encoder_weights, encoder_biases = model.parameters()[:2]
    encoder_weights[...] = 0.0  # el codificador emite media 0 y log-varianza 0
    encoder_biases[...] = 0.0
    samples = np.random.default_rng(5).random((3, INPUT_SIZE))
    terms, _ = model.objective(samples, samples, np.zeros((3, 2)), kl_weight=1.0)
    assert terms["kl"] == pytest.approx(0.0)


@pytest.mark.parametrize("model_type", ["autoencoder", "variational"])
def test_float32_precision_is_preserved_through_training_step(model_type):
    model = _build({"type": model_type, "precision": "float32", "activation": "leaky_relu"})
    samples = np.random.default_rng(6).random((3, INPUT_SIZE)).astype(np.float32)
    _, gradients = model.loss_and_gradients(samples, samples, epoch=1)
    optimizer = build_optimizer(Options({"type": "adam"}, "optimizer"))
    optimizer.step(model.parameters(), gradients)
    assert all(gradient.dtype == np.float32 for gradient in gradients)
    assert all(parameter.dtype == np.float32 for parameter in model.parameters())
    assert model.reconstruct(samples).dtype == np.float32


def test_kl_warmup_reaches_full_weight():
    schedule = KlSchedule(weight=0.5, warmup_epochs=10)
    assert schedule.weight_at(1) == pytest.approx(0.05)
    assert schedule.weight_at(10) == pytest.approx(0.5)
    assert schedule.weight_at(500) == pytest.approx(0.5)
