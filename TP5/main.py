"""Ejecutar desde TP5: python3 main.py --config configs/basic.json."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tp5.data import flip_noise, load_font
from tp5.model import Autoencoder
from tp5.plotting import latent, learning_curve, patterns
from tp5.training import pixel_errors, train


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # Todas las rutas del JSON son relativas al propio JSON.
    font_path = (config_path.parent / config["font"]).resolve()
    if not font_path.is_file():
        raise ValueError(f"Falta {font_path}. Copiar allí el font.h de la cátedra.")
    clean, labels = load_font(font_path)
    subset = config.get("subset")
    if subset is not None:
        if (not subset or any(type(i) is not int or not 0 <= i < len(clean) for i in subset)
                or len(set(subset)) != len(subset)):
            raise ValueError("subset debe tener índices únicos entre 0 y 31")
        clean, labels = clean[subset], [labels[i] for i in subset]
    evaluation = config.get("evaluation", {})
    levels = evaluation.get("noise_levels", [0, 0.05, 0.1, 0.2, 0.3])
    trials = evaluation.get("trials", 20)
    if (not levels or any(not 0 <= p <= 1 for p in levels)
            or type(trials) is not int or trials < 1):
        raise ValueError("La evaluación requiere niveles en [0,1] y trials positivo")
    seed = config.get("seed", 0)
    output = (config_path.parent / config["output"]).resolve()
    if output.exists():
        raise ValueError(f"La salida {output} ya existe; elegir otra para esta corrida")
    model = Autoencoder(config.get("hidden", [16, 8]), seed)
    history = train(model, clean, seed=seed, **config.get("training", {}))
    # Evitar sobrescribir resultados previos accidentalmente.
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    model.save(output / "model.npz")
    write_csv(output / "history.csv", history)
    prediction = model.reconstruct(clean)
    errors = pixel_errors(clean, prediction)
    write_csv(output / "reconstruction.csv", [
        {"pattern": label, "pixel_errors": int(error)} for label, error in zip(labels, errors)])
    points = model.encode(clean)
    write_csv(output / "latent.csv", [
        {"pattern": label, "z1": float(z[0]), "z2": float(z[1])}
        for label, z in zip(labels, points)])
    latent(points, labels, output / "latent.png")
    patterns([clean, prediction >= 0.5], ["Original", "Reconstrucción"],
             output / "reconstruction.png")
    learning_curve(history, output / "learning.png")
    # Evaluación con ruido nuevo, nunca reutilizado del entrenamiento.
    rng = np.random.default_rng(seed + 10000)
    noise_rows = []
    for level in levels:
        noisy = flip_noise(np.tile(clean, (trials, 1)), level, rng)
        reconstructed = model.reconstruct(noisy)
        target = np.tile(clean, (trials, 1))
        before = pixel_errors(target, noisy)
        after = pixel_errors(target, reconstructed)
        noise_rows.append({"noise": level, "trials": trials,
                           "input_mean_errors": float(before.mean()),
                           "output_mean_errors": float(after.mean()),
                           "output_max_errors": int(after.max()),
                           "fraction_at_most_one_error": float(np.mean(after <= 1))})
        patterns([clean, noisy[:len(clean)], reconstructed[:len(clean)] >= 0.5],
                 ["Original", "Ruido", "Salida"], output / f"noise_{level:g}.png")
    write_csv(output / "noise.csv", noise_rows)
    # Interpolar no garantiza novedad: registrar y verificar contra TODOS los caracteres.
    if len(points) >= 2:
        coordinates = np.linspace(points[0], points[1], 11)
        generated = model.decode(coordinates) >= 0.5
        full_dataset, _ = load_font(font_path)
        novel = [not np.any(np.all(full_dataset == sample, axis=1)) for sample in generated]
        np.savez_compressed(output / "interpolation.npz", z=coordinates,
                            patterns=generated, novel=np.asarray(novel))
        patterns([generated], ["Interpolación"], output / "interpolation.png")
    summary = {"patterns": len(clean), "max_pixel_errors": int(errors.max()),
               "all_patterns_at_most_one_error": bool(np.all(errors <= 1))}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Resultados: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejercicio 1: autoencoder de caracteres")
    parser.add_argument("--config", default="configs/basic.json")
    args = parser.parse_args()
    try:
        run(args.config)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")
