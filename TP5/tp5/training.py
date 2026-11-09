"""Entrenamiento común: limpio -> limpio o ruidoso -> limpio."""

import numpy as np

from .data import flip_noise


def pixel_errors(targets, predictions):
    return np.count_nonzero(targets != (predictions >= 0.5), axis=1)


def train(model, clean, *, epochs=5000, learning_rate=0.01, batch_size=32,
          optimizer="adam", noise=0.0, seed=0, log_every=100):
    if epochs < 1 or batch_size < 1 or log_every < 1 or learning_rate <= 0:
        raise ValueError("Épocas, batch_size, log_every y learning_rate deben ser positivos")
    if optimizer not in ("sgd", "adam"):
        raise ValueError("optimizer debe ser sgd o adam")
    if clean.ndim != 2 or clean.shape[1] != 35 or not len(clean):
        raise ValueError("Se requiere una matriz no vacía de N x 35")
    if not np.isin(clean, [0, 1]).all():
        raise ValueError("Los patrones deben ser binarios")
    if not 0 <= noise <= 1:
        raise ValueError("noise debe estar entre 0 y 1")
    # Generadores independientes: el ruido no cambia el orden de las muestras.
    shuffle_rng = np.random.default_rng(seed)
    noise_rng = np.random.default_rng(seed + 1)
    params = model.parameters()
    moment = [np.zeros_like(p) for p in params]
    variance = [np.zeros_like(p) for p in params]
    history, step = [], 0
    for epoch in range(1, epochs + 1):
        order = shuffle_rng.permutation(len(clean))
        total_loss = 0.0
        for start in range(0, len(clean), batch_size):
            target = clean[order[start:start + batch_size]]
            inputs = flip_noise(target, noise, noise_rng) if noise else target
            loss, gradients = model.loss_and_gradients(inputs, target)
            if not np.isfinite(loss) or not all(np.isfinite(g).all() for g in gradients):
                raise FloatingPointError("Entrenamiento divergente; revisar learning_rate")
            total_loss += loss * len(target)
            step += 1
            for i, (parameter, gradient) in enumerate(zip(params, gradients)):
                if optimizer == "sgd":
                    parameter -= learning_rate * gradient
                else:
                    moment[i] = 0.9 * moment[i] + 0.1 * gradient
                    variance[i] = 0.999 * variance[i] + 0.001 * gradient ** 2
                    m = moment[i] / (1 - 0.9 ** step)
                    v = variance[i] / (1 - 0.999 ** step)
                    parameter -= learning_rate * m / (np.sqrt(v) + 1e-8)
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            errors = pixel_errors(clean, model.reconstruct(clean))
            history.append({"epoch": epoch, "train_bce": total_loss / len(clean),
                            "clean_max_pixel_errors": int(errors.max()),
                            "clean_mean_pixel_errors": float(errors.mean())})
            print(f"Época {epoch}: BCE={history[-1]['train_bce']:.5f}, "
                  f"máximo={errors.max()} píxeles", flush=True)
    return history
