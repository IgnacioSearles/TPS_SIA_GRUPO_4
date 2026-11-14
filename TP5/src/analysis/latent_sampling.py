"""Control de muestreo del AE, sin reentrenar ni usar datos de prueba.

python -m src.analysis.latent_sampling
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src import plotting
from src.analysis.generative_suite import PROJECT_ROOT, compare_suite
from src.evaluation.context import write_csv
from src.evaluation.generative import nearest_training, pairwise_mse
from src.experiments.runner import load_run
from src.options import ConfigurationError


def fit_gaussian(codes):
    """Gaussiana empírica completa; raíz simétrica válida aun si es singular.

    Se usa covarianza muestral (N-1). Solo se recortan autovalores negativos
    producidos por redondeo, sin hiperparámetros ajustados sobre prueba.
    """
    codes = np.asarray(codes, dtype=np.float64)
    if codes.ndim != 2 or len(codes) < 2 or not np.isfinite(codes).all():
        raise ValueError("Se necesitan al menos dos códigos finitos en una matriz")
    mean = codes.mean(axis=0)
    centered = codes - mean
    covariance = centered.T @ centered / (len(codes) - 1)
    values, vectors = np.linalg.eigh(covariance)
    root = (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T
    return mean, covariance, root


def sample_fitted(model, training, epsilon):
    # Esta función no recibe referencias de prueba: el ajuste solo ve training.
    codes = model.encode(training.astype(model.precision, copy=False))
    mean, covariance, root = fit_gaussian(codes)
    coordinates = (epsilon @ root.T + mean).astype(model.precision)
    return model.decode(coordinates), coordinates, mean, covariance


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_control(root=PROJECT_ROOT):
    root = Path(root)
    # Valida presupuestos, particiones, semillas y finalización de las corridas.
    compare_suite(root, "shapes", write_outputs=False)
    output = root / "results/generative_latent_sampling_report"
    output.mkdir(parents=True, exist_ok=True)
    rows, provenance = [], []
    methods = ["AE normal estándar", "AE gaussiana ajustada", "VAE β=1 normal estándar"]
    for seed in (7, 17, 27):
        directories = [root / f"results/{name}/base__seed={seed}" for name in
                       ("ej2_new_2_shapes_ae", "ej2_new_1_shapes_vae")]
        models, datasets = zip(*(load_run(path) for path in directories))
        ae, vae = models
        dataset = datasets[0]
        manifests = [json.loads((p / "generative_manifest.json").read_text()) for p in directories]
        if manifests[0]["data"] != manifests[1]["data"]:
            raise ConfigurationError("Las particiones AE y VAE deben coincidir")
        count, generation_seed = manifests[0]["generation_count"], manifests[0]["generation_seed"]
        epsilon = np.random.default_rng(generation_seed).standard_normal((count, ae.latent_size))
        standard = epsilon.astype(ae.precision)
        fitted, coordinates, mean, covariance = sample_fitted(ae, dataset.samples, epsilon)
        samples = [ae.decode(standard), fitted, vae.decode(standard)]
        target = output / f"seed={seed}"
        target.mkdir(exist_ok=True)
        np.savez_compressed(target / "fitted_gaussian.npz", mean=mean, covariance=covariance,
                            epsilon=epsilon, coordinates=coordinates)
        for directory, regenerated in zip(directories, (samples[0], samples[2])):
            with np.load(directory / "generated.npz") as prior:
                if not np.allclose(regenerated, prior["samples"], atol=1e-6, rtol=1e-5):
                    raise ConfigurationError(f"La generación guardada no coincide con {directory}")
            provenance.append({"run": str(directory.relative_to(root)),
                               "model_sha256": _hash(directory / "model.npz"),
                               "config_sha256": _hash(directory / "config.json")})
        for index, (method, generated) in enumerate(zip(methods, samples)):
            nearest, indices = nearest_training(generated, dataset.samples)
            rows.append({"seed": seed, "method": method, "count": count,
                         "pairwise_mse": pairwise_mse(generated),
                         "nearest_train_mse_mean": float(nearest.mean())})
            write_csv(target / f"method_{index}_samples.csv", [
                {"sample": i, "nearest_train_mse": float(distance),
                 "nearest_train_label": dataset.labels[j]}
                for i, (distance, j) in enumerate(zip(nearest, indices))])
            np.savez_compressed(target / f"method_{index}.npz", samples=generated,
                                coordinates=coordinates if index == 1 else standard)
            side = int(np.ceil(np.sqrt(count)))
            plotting.save_image_mosaic(generated, (int(np.ceil(count / side)), side),
                                      dataset.image_shape, False, method, target / f"method_{index}.png")
            shown = np.linspace(0, count - 1, min(12, count)).round().astype(int)
            plotting.save_image_rows([generated[shown], dataset.samples[indices[shown]]],
                                     ["Generada", "Entrenamiento más próximo"],
                                     [str(i) for i in shown], dataset.image_shape, False,
                                     target / f"method_{index}_nearest.png")
        # Índices prefijados, iguales para todos los métodos y semillas.
        shown = np.arange(0, count, 8)
        plotting.save_image_rows([s[shown] for s in samples], methods, [str(i) for i in shown],
                                 dataset.image_shape, False, output / f"comparison_seed{seed}.png")
    write_csv(output / "runs.csv", rows)
    grouped = []
    for method in methods:
        selected = [r for r in rows if r["method"] == method]
        item = {"method": method, "seeds": len(selected)}
        for metric in ("pairwise_mse", "nearest_train_mse_mean"):
            values = [r[metric] for r in selected]
            item.update({metric + "_mean": float(np.mean(values)),
                         metric + "_std": float(np.std(values, ddof=1))})
        grouped.append(item)
    write_csv(output / "summary.csv", grouped)
    manifest = {"status": "completed", "training_performed": False,
                "fit_data": "training posterior codes only", "covariance": "full, sample N-1",
                "generation_seed": generation_seed, "count_per_method_seed": count,
                "data": manifests[0]["data"], "sources": provenance,
                "interpretation": "Descriptive metrics only: proximity and diversity do not certify "
                                  "plausibility or novelty. A Gaussian may bridge low-density regions."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(grouped, indent=2, ensure_ascii=False))
    print(f"Control sin reentrenamiento: {output}")
    return grouped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        run_control()
    except (ConfigurationError, FileNotFoundError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
