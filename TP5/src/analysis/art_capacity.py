"""Capacidad RGB: preparar, comparar en validación y abrir una prueba final.

python -m src.analysis.art_capacity --prepare
python -m src.analysis.art_capacity --compare
python -m src.analysis.art_capacity --final
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import plotting
from src.data import art as art_data
from src.data.art_capacity import (CONFIGS, SEEDS, expected_configs, file_hash, load_probe,
                                   prepare_capacity, read_json)
from src.data.factory import build_dataset
from src.evaluation import art_geometry
from src.evaluation.art_geometry import reconstruction_scores, summarize_geometry
from src.evaluation.context import write_csv
from src.evaluation.generative import nearest_training, pairwise_mse
from src.experiments.grid import expand_runs
from src.models.factory import build_model, load_parameters
from src.options import ConfigurationError, Options

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS = ("macro_silhouette_iou", "macro_boundary_f1", "macro_contour_distance_px",
           "macro_mse", "macro_foreground_mse", "macro_interior_color_mse")


def _sources(root):
    configs = expected_configs(root)
    sources, identity = [], None
    for latent, config in configs.items():
        for run in expand_runs(config):
            directory = root / f"results/{CONFIGS[latent]}/{run.label}"
            if read_json(directory / "config.json") != run.config:
                raise ConfigurationError(f"Configuración guardada incompatible: {directory}")
            read_json(directory / "summary.json")  # marcador de corrida finalizada
            manifest = read_json(directory / "generative_manifest.json")
            if (manifest["model_seed"] != run.seed or manifest["parameters_given_to_model"] is not False
                    or manifest["train_count"] != config["dataset"]["train_count"]
                    or manifest["test_count"] != config["dataset"]["test_count"]):
                raise ConfigurationError(f"Partición o semilla incompatible: {directory}")
            if identity is not None and identity != manifest["data"]:
                raise ConfigurationError("Los modelos no usaron exactamente los mismos datos")
            identity = manifest["data"]
            with (directory / "history.csv").open() as stream:
                history = list(csv.DictReader(stream))
            if not history or int(history[-1]["epoch"]) != config["training"]["epochs"]:
                raise ConfigurationError(f"Historia incompleta: {directory}")
            sources.append({"latent": latent, "seed": run.seed, "run": str(directory.relative_to(root)),
                            "model_sha256": file_hash(directory / "model.npz"),
                            "config_sha256": file_hash(directory / "config.json")})
    return configs, sources, identity


def _model(root, source):
    directory = root / source["run"]
    config = read_json(directory / "config.json")
    model = build_model(Options(config["model"], "model"), config["dataset"]["image_size"]**2 * 3,
                        np.random.default_rng(0), np.random.default_rng(0))
    load_parameters(model, directory / "model.npz")
    return model


def select_latent(grouped, tolerance=.01):
    best = max(r["macro_silhouette_iou_mean"] for r in grouped)
    return min(r["latent"] for r in grouped if r["macro_silhouette_iou_mean"] >= best - tolerance)


def _group(rows):
    grouped = []
    for latent in sorted({r["latent"] for r in rows}):
        selected = [r for r in rows if r["latent"] == latent]
        if {r["seed"] for r in selected} != set(SEEDS):
            raise ConfigurationError("Faltan semillas para resumir la capacidad")
        result = {"latent": latent, "runs": len(selected)}
        for metric in METRICS:
            values = [r[metric] for r in selected]
            result.update({metric + "_mean": float(np.mean(values)),
                           metric + "_std": float(np.std(values, ddof=1))})
        grouped.append(result)
    return grouped


def _evaluate(root, output, samples, records, sources, training=None):
    shown = np.array([0, 1, 2, 3, len(samples)//2, len(samples)//2+1,
                      len(samples)//2+2, len(samples)//2+3])
    shape = (int(np.sqrt(samples.shape[1] / 3)),) * 2 + (3,)
    rows, family_rows, reconstructions, generations = [], [], {}, {}
    epsilon = np.random.default_rng(1234).standard_normal((64, 16))
    for source in sources:
        latent, seed = source["latent"], source["seed"]
        print(f"Evaluando {output.name}: latente {latent}, semilla {seed}", flush=True)
        model = _model(root, source)
        reconstructed = model.reconstruct(samples.astype(model.precision))
        scores = [{"label": r["label"], "family": r["family"],
                   **reconstruction_scores(a.reshape(shape), b.reshape(shape), r)}
                  for a, b, r in zip(samples, reconstructed, records)]
        summary, families = summarize_geometry(scores)
        rows.append({"latent": latent, "seed": seed, **summary})
        family_rows.extend({"latent": latent, "seed": seed, **f} for f in families)
        directory = output / f"latent{latent}/seed={seed}"
        directory.mkdir(parents=True, exist_ok=True)
        write_csv(directory / "reconstruction.csv", scores)
        reconstructions[latent, seed] = reconstructed[shown]
        if training is not None:
            # Prefijos de una matriz normal común; no implica ejes semánticos comunes.
            coordinates = epsilon[:, :latent].astype(model.precision)
            generated = model.decode(coordinates)
            generations[latent, seed] = generated[::8]
            nearest, indices = nearest_training(generated, training.samples)
            summary["generated_pairwise_mse"] = pairwise_mse(generated)
            summary["generated_nearest_train_mse"] = float(nearest.mean())
            np.savez_compressed(directory / "generated.npz", samples=generated, coordinates=coordinates)
            write_csv(directory / "generated.csv", [{"sample": i, "nearest_train_mse": float(d),
                "nearest_train_label": training.labels[j]} for i, (d, j) in enumerate(zip(nearest, indices))])
            plotting.save_image_mosaic(generated, (8, 8), shape, False,
                                      f"VAE RGB, latente {latent}, semilla {seed}", directory / "generated.png")
            chosen = np.linspace(0, 63, 12).round().astype(int)
            plotting.save_image_rows([generated[chosen], training.samples[indices[chosen]]],
                                     ["Generada", "Entrenamiento próximo"], [str(i) for i in chosen],
                                     shape, False, directory / "generated_nearest.png")
            rows[-1].update({k: v for k, v in summary.items() if k.startswith("generated_")})
    write_csv(output / "runs.csv", rows)
    write_csv(output / "families.csv", family_rows)
    grouped = _group(rows)
    write_csv(output / "summary.csv", grouped)
    latents = sorted({s["latent"] for s in sources})
    for seed in SEEDS:
        plotting.save_image_rows([samples[shown]] + [reconstructions[k, seed] for k in latents],
                                 ["Original"] + [f"Latente {k}" for k in latents],
                                 [records[i]["family"] for i in shown], shape, False,
                                 output / f"reconstruction_seed{seed}.png")
        if generations:
            plotting.save_image_rows([generations[k, seed] for k in latents],
                                     [f"Latente {k}" for k in latents], [str(i) for i in range(0, 64, 8)],
                                     shape, False, output / f"generation_seed{seed}.png")
    _plot(grouped, family_rows, output)
    return grouped


def _plot(grouped, families, output):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric, title in zip(axes, METRICS[:3],
        ("IoU de silueta ↑", "F1 de borde (1 px) ↑", "Distancia de contorno (px) ↓")):
        ax.bar([str(r["latent"]) for r in grouped], [r[metric+"_mean"] for r in grouped],
               yerr=[r[metric+"_std"] for r in grouped], capsize=3)
        ax.set(title=title, xlabel="Dimensiones latentes")
    fig.tight_layout(); fig.savefig(output / "geometry.png", dpi=150); plt.close(fig)
    names = sorted({r["family"] for r in families})
    fig, ax = plt.subplots(figsize=(8, 4))
    width = .8 / len(grouped)
    for i, group in enumerate(grouped):
        values = [np.mean([r["silhouette_iou"] for r in families
                           if r["family"] == family and r["latent"] == group["latent"]]) for family in names]
        ax.bar(np.arange(len(names)) + i*width, values, width, label=f"Latente {group['latent']}")
    ax.set_xticks(np.arange(len(names)) + width*(len(grouped)-1)/2, names)
    ax.set(ylabel="IoU de silueta", ylim=(0, 1)); ax.legend()
    fig.tight_layout(); fig.savefig(output / "families.png", dpi=150); plt.close(fig)


def _snapshot(root):
    configs, sources, identity = _sources(root)
    directory = root / "results/art_capacity_data"
    manifest = read_json(directory / "manifest.json")
    for split, partition in manifest["partitions"].items():
        if file_hash(directory / f"{split}.npz") != partition["file_sha256"]:
            raise ConfigurationError(f"Partición alterada: {split}")
    if manifest["training_sha256"] != identity["train_sha256"]:
        raise ConfigurationError("Los modelos no corresponden al entrenamiento fijado en el protocolo")
    for latent, name in CONFIGS.items():
        if file_hash(root / f"configs/{name}.json") != manifest["protocol"]["configuration_hashes"][str(latent)]:
            raise ConfigurationError("Las configuraciones cambiaron desde que se fijó el protocolo")
    calibration_inputs = {"metrics_sha256": file_hash(Path(art_geometry.__file__)),
                          "renderer_sha256": file_hash(Path(art_data.__file__)),
                          "validation_sha256": manifest["partitions"]["validation"]["file_sha256"]}
    calibration_path = directory / "segmentation_calibration.json"
    calibration = read_json(calibration_path) if calibration_path.exists() else {}
    if calibration.get("inputs") != calibration_inputs:
        samples, records = load_probe(directory, "validation")
        calibration = {"inputs": calibration_inputs, **art_geometry.calibrate_geometry(samples, records)}
        if calibration["nonperfect_references"]:
            raise ConfigurationError("La segmentación no reproduce exactamente la geometría de los originales")
        calibration_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    snapshot = {"sources": sources, "data_manifest_sha256": file_hash(directory / "manifest.json"),
                "metrics_code_sha256": file_hash(Path(art_geometry.__file__)),
                "renderer_code_sha256": file_hash(Path(art_data.__file__)),
                "calibration_sha256": file_hash(calibration_path),
                "metrics_version": art_geometry.METRICS_VERSION}
    return configs, sources, directory, manifest, snapshot


def compare_capacity(root=PROJECT_ROOT):
    root = Path(root)
    configs, sources, data, manifest, snapshot = _snapshot(root)
    output = root / "results/art_capacity_validation"
    if (output / "selection.json").exists():
        selection = read_json(output / "selection.json")
        if selection["inputs"] != snapshot:
            raise ConfigurationError("La selección ya está congelada y cambiaron sus fuentes o métricas")
        print(f"Se reutiliza selección: latente {selection['selected_latent']}")
        return selection
    samples, records = load_probe(data, "validation")
    training = build_dataset(Options(configs[4]["dataset"], "dataset"), root)
    if (training.generative_study.metadata["train_sha256"] != manifest["training_sha256"]
            or training.generative_study.metadata["test_sha256"] != manifest["previous_test_sha256"]):
        raise ConfigurationError("Los datos actuales no reproducen las particiones del estudio")
    output.mkdir(parents=True, exist_ok=True)
    grouped = _evaluate(root, output, samples, records, sources, training)
    selected = select_latent(grouped, manifest["protocol"]["iou_tolerance"])
    selection = {"selected_latent": selected, "inputs": snapshot, "validation_summary": grouped,
                 "criterion": manifest["protocol"]["selection"], "final_evaluated_at_selection": False,
                 "interpretation": manifest["protocol"]["selection_scope"]}
    (output / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"Selección congelada por validación: latente {selected}", flush=True)
    return selection


def evaluate_final(root=PROJECT_ROOT):
    root = Path(root)
    selection_path = root / "results/art_capacity_validation/selection.json"
    selection = read_json(selection_path)
    _, sources, data, _, snapshot = _snapshot(root)
    if selection["inputs"] != snapshot:
        raise ConfigurationError("Las fuentes cambiaron después de la selección")
    output = root / "results/art_capacity_final"
    status = {"selection_sha256": file_hash(selection_path), "inputs": snapshot,
              "selected_latent": selection["selected_latent"], "status": "completed",
              "interpretation": "Final confirmation of frozen choice against latent 4. Do not retune on this partition."}
    if (output / "manifest.json").exists():
        if read_json(output / "manifest.json") != status:
            raise ConfigurationError("Ya existe una evaluación final con otra selección o procedencia")
        print(f"Se reutiliza evaluación final: {output}")
        return output
    samples, records = load_probe(data, "reserved_final")
    chosen = [s for s in sources if s["latent"] in {4, selection["selected_latent"]}]
    output.mkdir(parents=True, exist_ok=True)
    _evaluate(root, output, samples, records, chosen)
    (output / "manifest.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"Evaluación final sin reentrenar: {output}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--compare", action="store_true")
    group.add_argument("--final", action="store_true")
    args = parser.parse_args()
    try:
        if args.prepare:
            prepare_capacity(PROJECT_ROOT)
        elif args.compare:
            compare_capacity()
        else:
            evaluate_final()
    except (ConfigurationError, ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
