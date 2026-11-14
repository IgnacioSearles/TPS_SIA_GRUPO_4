"""Preparar las dos aplicaciones del VAE para presentar, sin comparar sus MSE.

python -m src.analysis.art_audio --preview
python -m src.analysis.art_audio
python -m src.analysis.art_audio --compositions
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from src import plotting
from src.data.factory import build_dataset
from src.evaluation.context import EvaluationContext, write_csv
from src.evaluation.presentation import evaluate_presentation
from src.experiments.grid import expand_runs
from src.experiments.runner import load_run
from src.options import ConfigurationError, Options

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCES = {"art_single": "ej2_art_1_single", "timbres": "ej2_new_3_timbres_vae",
           "art_composition": "ej2_art_2_composition"}
METRICS = ("heldout_mean_squared_error", "generated_pairwise_mse", "generated_nearest_train_mse_mean")


def _read(path):
    if not path.is_file():
        raise ConfigurationError(f"Falta {path}. Completar esa corrida antes de preparar la presentación")
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preview(root=PROJECT_ROOT):
    root = Path(root)
    output = root / "results/art_audio_preview"
    output.mkdir(parents=True, exist_ok=True)
    for kind in ("art_single", "art_composition"):
        config = _read(root / f"configs/{SOURCES[kind]}.json")
        dataset = build_dataset(Options(config["dataset"], "dataset"), root)
        plotting.save_image_mosaic(dataset.samples[:64], (8, 8), dataset.image_shape,
                                   False, f"Dataset original: {kind} (no son generaciones del VAE)",
                                   output / f"{kind}.png")
        (output / f"{kind}_data.json").write_text(json.dumps(dataset.generative_study.metadata, indent=2))
    print(f"Vista previa de datos originales: {output}")
    return output


def prepare_report(root=PROJECT_ROOT, compositions=False):
    root = Path(root)
    kinds = ["art_single", "timbres"] + (["art_composition"] if compositions else [])
    output = root / ("results/art_audio_presentation_extended" if compositions else "results/art_audio_presentation")
    # Validar todas las fuentes antes de exportar; ninguna corrida es reentrenada.
    sources, signature = [], None
    for kind in kinds:
        name = SOURCES[kind]
        config = _read(root / f"configs/{name}.json")
        experiment = dict(config)
        destination = root / experiment.pop("output", f"results/{name}")
        experiment.pop("name", None)
        runs = expand_runs(experiment)
        if {r.seed for r in runs} != {7, 17, 27} or len(runs) != 3:
            raise ConfigurationError("La presentación requiere las semillas 7,17,27 sin repeticiones")
        current = {"model": config["model"], "training": config["training"],
                   "split": {k: config["dataset"][k] for k in ("train_count", "test_count", "data_seed")}}
        if signature is not None and signature != current:
            raise ConfigurationError("El protocolo principal cambió modelo, presupuesto o tamaños de partición")
        signature = current
        for run in runs:
            directory = destination / run.label
            summary = _read(directory / "summary.json")
            if _read(directory / "config.json") != run.config:
                raise ConfigurationError(f"La configuración guardada no coincide en {directory}")
            manifest = _read(directory / "generative_manifest.json")
            if (manifest["model_seed"] != run.seed or manifest["parameters_given_to_model"] is not False
                    or manifest["train_count"] != current["split"]["train_count"]
                    or manifest["test_count"] != current["split"]["test_count"]
                    or manifest["generation_seed"] != 1234
                    or manifest["generation_count"] != run.config["evaluation"]["generative"]["count"]
                    or manifest["data"]["kind"] != ("timbres" if kind == "timbres" else "art")):
                raise ConfigurationError(f"Partición o evaluación incompatible en {directory}")
            with (directory / "history.csv").open() as stream:
                history = list(csv.DictReader(stream))
            if not history or int(history[-1]["epoch"]) != config["training"]["epochs"]:
                raise ConfigurationError(f"Historia incompleta en {directory}")
            provenance = {"run": str(directory.relative_to(root)),
                          "model_sha256": _hash(directory / "model.npz"),
                          "config_sha256": _hash(directory / "config.json"),
                          "summary_sha256": _hash(directory / "summary.json"),
                          "generative_manifest_sha256": _hash(directory / "generative_manifest.json"),
                          "presentation_version": 1, "seed": run.seed}
            if not all(np.isfinite(float(summary[m])) for m in METRICS):
                raise ConfigurationError(f"Métricas no finitas en {directory}")
            sources.append((kind, directory, summary, manifest, provenance))
    output.mkdir(parents=True, exist_ok=True)
    rows, provenance_rows = [], []
    for kind, source, summary, manifest, provenance in sources:
        target = output / kind / f"seed={provenance['seed']}"
        stamp = target / "source.json"
        if not stamp.is_file() or _read(stamp) != provenance or not (target / "presentation/manifest.json").is_file():
            model, dataset = load_run(source)
            if dataset.generative_study.metadata != manifest["data"]:
                raise ConfigurationError(f"Los datos reconstruidos cambiaron en {source}")
            target.mkdir(parents=True, exist_ok=True)
            context = EvaluationContext(model, dataset, target, np.random.default_rng(1234), provenance["seed"])
            evaluate_presentation(context, Options({"pairs": 3, "steps": 11}, "presentation"))
            for filename in ("generated.png", "generated_nearest.png", "test_reconstruction.png"):
                shutil.copyfile(source / filename, target / filename)
            if (source / "audio/generative").is_dir():
                shutil.copytree(source / "audio/generative", target / "audio/generative", dirs_exist_ok=True)
            stamp.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        rows.append({"application": kind, "seed": provenance["seed"],
                     "generated_count": manifest["generation_count"],
                     **{metric: float(summary[metric]) for metric in METRICS}})
        provenance_rows.append(provenance)
    write_csv(output / "runs.csv", rows)
    grouped = []
    for kind in kinds:
        selected = [r for r in rows if r["application"] == kind]
        row = {"application": kind, "runs": len(selected)}
        for metric in METRICS:
            values = [r[metric] for r in selected]
            row.update({metric + "_mean": float(np.mean(values)), metric + "_std": float(np.std(values, ddof=1))})
        grouped.append(row)
    write_csv(output / "summary_by_application.csv", grouped)
    (output / "manifest.json").write_text(json.dumps({
        "status": "completed", "training_performed": False, "protocol": signature,
        "sources": provenance_rows, "interpolation": {"heldout_pairs": [[0, 1], [2, 3], [4, 5]], "steps": 11},
        "interpretation": "Applications share a learning procedure, not a common quality scale. "
                          "Pixel and mel MSE are not directly comparable. No targets exist for intermediate outputs."
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "README.md").write_text(
        "# Un VAE, dos aplicaciones\n\n"
        "Aprendemos de ejemplos para reconstruir, generar y recorrer combinaciones de características. "
        "Se entrenan modelos separados con la misma implementación y configuración; no comparten pesos "
        "ni un espacio latente conjunto. El encoder recibe RGB o mel; no recibe parámetros geométricos ni armónicos.\n\n"
        "Cada carpeta muestra todas las generaciones exportadas, vecinos de entrenamiento, reconstrucciones "
        "reservadas y tres recorridos prefijados de once puntos. Las figuras incluyen GIF; los sonidos "
        "incluyen WAV por paso, secuencia y referencias originales/pasadas por el códec.\n\n"
        "Las métricas se resumen por aplicación y semilla: no comparar MSE RGB con MSE mel ni combinar "
        "sus valores. La pérdida suma errores por característica y las dimensionalidades son distintas. "
        "La similitud del protocolo no iguala su dificultad. Afinación correcta, suavidad de un recorrido "
        "y distancia al entrenamiento no prueban por sí solas calidad o novedad.\n\n"
        "Consultar docs/art_audio_experiments.md para el guion y los límites.\n", encoding="utf-8")
    print(f"Presentación de {len(rows)} corridas, sin reentrenar: {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="solo renderizar datos originales, sin entrenar")
    parser.add_argument("--compositions", action="store_true", help="incluir la etapa exploratoria de dos figuras")
    args = parser.parse_args()
    try:
        preview() if args.preview else prepare_report(compositions=args.compositions)
    except (ConfigurationError, FileNotFoundError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
