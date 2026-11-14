"""Evaluación común de modelos RGB: geometría y generación sin reentrenar."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import plotting
from src.analysis.latent_sampling import sample_fitted
from src.evaluation.art_geometry import reconstruction_scores, summarize_geometry
from src.evaluation.context import write_csv
from src.evaluation.generative import nearest_training, pairwise_mse
from src.models.factory import build_model, load_parameters
from src.options import Options

METRICS = ("macro_silhouette_iou", "macro_boundary_f1", "macro_contour_distance_px",
           "macro_mse", "macro_foreground_mse", "macro_interior_color_mse")


def load_source_model(root, source):
    directory = root / source["run"]
    config = json.loads((directory / "config.json").read_text())
    model = build_model(Options(config["model"], "model"), config["dataset"]["image_size"]**2 * 3,
                        np.random.default_rng(0), np.random.default_rng(0))
    load_parameters(model, directory / "model.npz")
    return model


def group_variants(rows, seeds):
    result = []
    for variant in dict.fromkeys(r["variant"] for r in rows):
        chosen = [r for r in rows if r["variant"] == variant]
        if len(chosen) != len(seeds) or {r["seed"] for r in chosen} != set(seeds):
            raise ValueError("La comparación requiere una corrida por semilla y variante")
        item = {"variant": variant, "label": chosen[0]["label"], "runs": len(chosen),
                "parameters": chosen[0]["parameters"]}
        for metric in METRICS:
            values = [r[metric] for r in chosen]
            item.update({metric+"_mean": float(np.mean(values)), metric+"_std": float(np.std(values, ddof=1))})
        result.append(item)
    return result


def evaluate_factorial(root, output, samples, records, sources, seeds, training=None):
    output.mkdir(parents=True, exist_ok=True)
    size = int(np.sqrt(samples.shape[1] / 3)); shape = (size, size, 3)
    shown = np.array([0, 1, 2, 3, len(samples)//2, len(samples)//2+1, len(samples)//2+2, len(samples)//2+3])
    epsilon = np.random.default_rng(1234).standard_normal((64, 8))
    rows, family_rows, reconstructions, generated_shown = [], [], {}, {}
    for source in sources:
        variant, seed, label = source["variant"], source["seed"], source["label"]
        print(f"{output.name}: {variant}, semilla {seed}", flush=True)
        model = load_source_model(root, source)
        reconstructed = model.reconstruct(samples.astype(model.precision))
        scores = [{"label": r["label"], "family": r["family"],
                   **reconstruction_scores(a.reshape(shape), b.reshape(shape), r)}
                  for a, b, r in zip(samples, reconstructed, records)]
        summary, families = summarize_geometry(scores)
        row = {"variant": variant, "label": label, "seed": seed,
               "parameters": sum(p.size for p in model.parameters()), **summary}
        rows.append(row)
        family_rows.extend({"variant": variant, "seed": seed, **r} for r in families)
        directory = output / variant / f"seed={seed}"
        directory.mkdir(parents=True, exist_ok=True)
        write_csv(directory / "reconstruction.csv", scores)
        reconstructions[variant, seed] = reconstructed[shown]
        if training is not None:
            train = training.samples.astype(model.precision)
            row["training_mse"] = float(np.mean((train - model.reconstruct(train))**2))
            coordinates = epsilon.astype(model.precision)
            standard = model.decode(coordinates)
            if source["model_type"] == "autoencoder":
                generated, coordinates, mean, covariance = sample_fitted(model, train, epsilon)
                np.savez_compressed(directory / "training_gaussian.npz", mean=mean, covariance=covariance)
                plotting.save_image_mosaic(standard, (8, 8), shape, False,
                    f"{label}: N(0,I), diagnóstico suplementario", directory / "standard_diagnostic.png")
                np.savez_compressed(directory / "standard_diagnostic.npz", samples=standard, coordinates=epsilon)
                method = "training_gaussian"
            else:
                generated, method = standard, "standard_prior"
            row["generation_method"] = method
            nearest, indices = nearest_training(generated, train)
            row["generated_pairwise_mse"] = pairwise_mse(generated)
            row["generated_nearest_train_mse"] = float(nearest.mean())
            generated_shown[variant, seed] = generated[::8]
            np.savez_compressed(directory / "generated.npz", samples=generated, coordinates=coordinates)
            write_csv(directory / "generated.csv", [{"sample": i, "method": method,
                "nearest_train_label": training.labels[j], "nearest_train_mse": float(d)}
                for i, (d, j) in enumerate(zip(nearest, indices))])
            plotting.save_image_mosaic(generated, (8, 8), shape, False,
                                      f"{label}, semilla {seed}: {method}", directory / "generated.png")
            selected = np.linspace(0, 63, 12).round().astype(int)
            plotting.save_image_rows([generated[selected], train[indices[selected]]],
                ["Generada", "Entrenamiento próximo"], [str(i) for i in selected], shape, False,
                directory / "generated_nearest.png")
    write_csv(output / "runs.csv", rows); write_csv(output / "families.csv", family_rows)
    grouped = group_variants(rows, seeds); write_csv(output / "summary.csv", grouped)
    variants = list(dict.fromkeys(s["variant"] for s in sources))
    labels = {s["variant"]: s["label"] for s in sources}
    for seed in seeds:
        plotting.save_image_rows([samples[shown]] + [reconstructions[v, seed] for v in variants],
            ["Original"] + [labels[v] for v in variants], [records[i]["family"] for i in shown],
            shape, False, output / f"reconstruction_seed{seed}.png")
        if generated_shown:
            plotting.save_image_rows([generated_shown[v, seed] for v in variants],
                [labels[v] + (" ajustado" if v.startswith("ae_") else " prior") for v in variants],
                [str(i) for i in range(0, 64, 8)], shape, False, output / f"generation_seed{seed}.png")
    _plot(grouped, family_rows, output)
    return grouped, rows


def _plot(grouped, families, output):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for axis, metric, title in zip(axes, METRICS[:3], ("IoU de silueta ↑", "F1 de borde ↑", "Contorno (px) ↓")):
        axis.bar(range(len(grouped)), [r[metric+"_mean"] for r in grouped],
                 yerr=[r[metric+"_std"] for r in grouped], capsize=3)
        axis.set_xticks(range(len(grouped)), [r["label"] for r in grouped], rotation=25, ha="right")
        axis.set_title(title)
    fig.tight_layout(); fig.savefig(output / "geometry.png", dpi=150); plt.close(fig)
    names = sorted({r["family"] for r in families})
    fig, axis = plt.subplots(figsize=(9, 4))
    width = .8 / len(grouped)
    for i, group in enumerate(grouped):
        values = [np.mean([r["silhouette_iou"] for r in families
                           if r["family"] == f and r["variant"] == group["variant"]]) for f in names]
        axis.bar(np.arange(len(names)) + i*width, values, width, label=group["label"])
    axis.set_xticks(np.arange(len(names)) + width*(len(grouped)-1)/2, names)
    axis.set(ylabel="IoU", ylim=(0, 1)); axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output / "families.png", dpi=150); plt.close(fig)
