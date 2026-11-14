"""Arte geométrico RGB sintético: los parámetros nunca se entregan al VAE."""

import hashlib

import numpy as np

from src.data.generative import assemble_generative, positive_integer, split_options
from src.options import ConfigurationError, Options

FAMILIES = ("circle", "ellipse", "triangle", "rectangle")
BACKGROUND = (0.055, 0.075, 0.13)
PALETTE = np.array([[0.98, 0.35, 0.32], [1.0, 0.76, 0.22],
                    [0.16, 0.78, 0.72], [0.65, 0.48, 0.95]])


def render_art(size, objects, background=BACKGROUND):
    """Componer objetos opacos en orden de fondo a frente, antialias 4x."""
    scale = 4
    axis = (np.arange(size * scale) + 0.5) / scale
    y, x = np.meshgrid(axis, axis, indexing="ij")
    image = np.empty((size * scale, size * scale, 3), dtype=np.float32)
    image[:] = background
    for obj in objects:
        dx, dy = x - obj["center_x"], y - obj["center_y"]
        angle = obj["angle_radians"]
        u = (np.cos(angle) * dx + np.sin(angle) * dy) / obj["half_width"]
        v = (-np.sin(angle) * dx + np.cos(angle) * dy) / obj["half_height"]
        family = obj["family"]
        if family in ("circle", "ellipse"):
            mask = u*u + v*v <= 1
        elif family == "rectangle":
            mask = (abs(u) <= 1) & (abs(v) <= 1)
        elif family == "triangle":
            mask = (v >= -1) & (v <= 1) & (abs(u) <= (v + 1) / 2)
        else:
            raise ConfigurationError(f"Familia geométrica desconocida: {family}")
        image[mask] = obj["color_rgb"]
    return image.reshape(size, scale, size, scale, 3).mean(axis=(1, 3))


def load_art(options: Options):
    train, test, seed = split_options(options)
    size = positive_integer(options.take("image_size", 32), "image_size")
    object_count = options.take("objects", 1)
    families = options.take("families", list(FAMILIES))
    options.ensure_consumed()
    if not 16 <= size <= 64:
        raise ConfigurationError("image_size debe estar entre 16 y 64")
    if type(object_count) is not int or object_count not in (1, 2):
        raise ConfigurationError("objects debe ser 1 o 2")
    if (not isinstance(families, list) or not families
            or any(f not in FAMILIES for f in families) or len(set(families)) != len(families)):
        raise ConfigurationError("families debe contener familias conocidas sin repeticiones")
    rng = np.random.default_rng(seed)
    samples, records, seen = [], [], set()
    attempts = 0
    while len(samples) < train + test:
        attempts += 1
        if attempts > 100 * (train + test):
            raise ConfigurationError("No se pudo generar suficiente arte único")
        objects = []
        for layer in range(object_count):
            # La primera familia se balancea; los demás atributos son independientes.
            family = (families[len(samples) % len(families)] if layer == 0
                      else str(rng.choice(families)))
            low, high = ((.14, .30) if object_count == 1 else (.13, .23))
            a, b = rng.uniform(low, high, size=2) * size
            if family == "circle":
                b = a
            angle = float(rng.uniform(-np.pi / 4, np.pi / 4))
            shift = 0 if object_count == 1 else (-.12 if layer == 0 else .12)
            center = (np.array([.5 + shift, .5]) + rng.uniform(-.04, .04, 2)) * size
            colors = rng.choice(len(PALETTE), 2, replace=False)
            mix = float(rng.uniform(0, 1))
            color = PALETTE[colors[0]] * (1 - mix) + PALETTE[colors[1]] * mix
            objects.append({"family": family, "center_x": float(center[0]),
                            "center_y": float(center[1]), "half_width": float(a),
                            "half_height": float(b), "angle_radians": angle,
                            "color_rgb": color.tolist()})
        image = render_art(size, objects)
        fingerprint = hashlib.sha256(image.tobytes()).digest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        family = "+".join(o["family"] for o in objects)
        records.append({"label": f"art_{len(samples):05d}", "family": family,
                        "objects": objects})
        samples.append(image.ravel())
    return assemble_generative(samples, records, train, (size, size, 3),
        {"kind": "art", "data_seed": seed, "image_size": size, "channels": "RGB",
         "objects": object_count, "families": families, "supersampling": 4,
         "background_rgb": list(BACKGROUND), "palette_rgb": PALETTE.tolist(),
         "color_sampling": "uniform mixture of two distinct palette colors"})
