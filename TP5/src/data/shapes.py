"""Figuras antialias en escala de grises; no se entregan parámetros al modelo."""

import hashlib
import numpy as np

from src.data.generative import assemble_generative, positive_integer, split_options
from src.options import ConfigurationError, Options


def render_shape(size, family, half_width, half_height, angle):
    """Rasterizar una figura centrada, rotada, con supersampling 4x."""
    supersampling = 4
    axis = (np.arange(size * supersampling) + 0.5) / supersampling - size / 2
    y, x = np.meshgrid(axis, axis, indexing="ij")
    u = np.cos(angle) * x + np.sin(angle) * y
    v = -np.sin(angle) * x + np.cos(angle) * y
    if family == "ellipse":
        inside = (u / half_width)**2 + (v / half_height)**2 <= 1
    elif family == "rectangle":
        inside = (abs(u) <= half_width) & (abs(v) <= half_height)
    else:
        raise ConfigurationError(f"Familia geométrica desconocida: {family}")
    return inside.reshape(size, supersampling, size, supersampling).mean(axis=(1, 3)).astype(np.float32)


def load_shapes(options: Options):
    train_count, test_count, seed = split_options(options)
    size = positive_integer(options.take("image_size", 32), "image_size")
    families = options.take("families", ["ellipse", "rectangle"])
    options.ensure_consumed()
    if not 16 <= size <= 64:
        raise ConfigurationError("image_size debe estar entre 16 y 64")
    if (not isinstance(families, list) or not families
            or any(f not in ("ellipse", "rectangle") for f in families)
            or len(set(families)) != len(families)):
        raise ConfigurationError("families debe contener ellipse y/o rectangle, sin repeticiones")
    rng = np.random.default_rng(seed)
    samples, records, seen = [], [], set()
    total = train_count + test_count
    attempts = 0
    while len(samples) < total:
        attempts += 1
        if attempts > total * 100:
            raise ConfigurationError("No se pudo generar suficiente variedad de figuras únicas")
        family = families[len(samples) % len(families)]
        a, b = rng.uniform(0.12 * size, 0.36 * size, size=2)
        angle = float(rng.uniform(-np.pi / 4, np.pi / 4))
        image = render_shape(size, family, a, b, angle)
        fingerprint = hashlib.sha256(image.tobytes()).digest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        records.append({"label": f"{family}_{len(samples):05d}", "family": family,
                        "half_width": float(a), "half_height": float(b), "angle_radians": angle})
        samples.append(image.ravel())
    return assemble_generative(samples, records, train_count, (size, size),
                                {"kind": "shapes", "data_seed": seed, "image_size": size,
                                 "families": families, "supersampling": 4})
