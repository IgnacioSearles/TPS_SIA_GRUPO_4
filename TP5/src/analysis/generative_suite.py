"""Comparar las nuevas tandas sin declarar plausibilidad a partir de una sola métrica.

python -m src.analysis.generative_suite --shapes
python -m src.analysis.generative_suite --timbres
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.context import write_csv
from src.experiments.grid import expand_runs
from src.options import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = {"shapes": ["ej2_new_1_shapes_vae", "ej2_new_2_shapes_ae", "ej2_new_2_shapes_kl"],
           "timbres": ["ej2_new_3_timbres_vae"]}
METRICS = ["mean_squared_error", "heldout_mean_squared_error", "generated_pairwise_mse",
           "generated_nearest_train_mse_mean", "generated_fixed_pitch_hit_rate", "codec_fixed_pitch_hit_rate"]


def _read(path):
    if not path.is_file():
        raise ConfigurationError(f"Falta {path}. Completar las corridas de esta tanda antes de comparar")
    return json.loads(path.read_text(encoding="utf-8"))


def compare_suite(root, scope="all", *, write_outputs=True):
    root = Path(root)
    kinds = list(CONFIGS) if scope == "all" else [scope]
    rows, signatures, identities, warmups = [], {}, set(), {}
    for kind in kinds:
        for name in CONFIGS[kind]:
            config_path = root / f"configs/{name}.json"
            config = _read(config_path)
            output = root / config.pop("output", f"results/{config.pop('name', name)}")
            for run in expand_runs(config):
                directory = output / run.label
                summary = _read(directory / "summary.json")
                if _read(directory / "config.json") != run.config:
                    raise ConfigurationError(f"La configuración guardada cambió en {directory}")
                manifest = _read(directory / "generative_manifest.json")
                if (not (directory / "model.npz").is_file()
                        or manifest["data"]["kind"] != kind
                        or manifest["train_count"] != run.config["dataset"]["train_count"]
                        or manifest["test_count"] != run.config["dataset"]["test_count"]):
                    raise ConfigurationError(f"Modelo o partición incompatible en {directory}")
                settings = run.config["evaluation"]["generative"]
                if (manifest["model_seed"] != run.seed
                        or manifest["generation_seed"] != settings.get("seed", 1234)
                        or manifest["generation_count"] != settings.get("count", 64)
                        or manifest["parameters_given_to_model"] is not False):
                    raise ConfigurationError(f"Evaluación generativa incompatible en {directory}")
                with (directory / "history.csv").open(newline="", encoding="utf-8") as stream:
                    history = list(csv.DictReader(stream))
                if int(history[-1]["epoch"]) != run.config["training"]["epochs"]:
                    raise ConfigurationError(f"Entrenamiento incompleto en {directory}")
                model = dict(run.config["model"])
                model_type = model.pop("type")
                schedule = model.pop("kl", {})
                beta = schedule.get("weight")
                if model_type == "variational":
                    warmup = schedule.get("warmup_epochs", 0)
                    if kind in warmups and warmups[kind] != warmup:
                        raise ConfigurationError("El calentamiento KL cambió entre las configuraciones VAE")
                    warmups[kind] = warmup
                label = "AE" if model_type == "autoencoder" else f"VAE β={beta:g}"
                signature = json.dumps({"data": manifest["data"], "model": model,
                                        "training": run.config["training"],
                                        "evaluation": run.config["evaluation"]}, sort_keys=True)
                if kind in signatures and signatures[kind] != signature:
                    raise ConfigurationError(f"La comparación de {kind} cambió datos, arquitectura o presupuesto")
                signatures[kind] = signature
                identity = (kind, label, run.seed)
                if identity in identities:
                    raise ConfigurationError("Una configuración/semilla está repetida")
                identities.add(identity)
                row = {"dataset": kind, "model": label, "seed": run.seed,
                       "train_count": manifest["train_count"], "test_count": manifest["test_count"],
                       "run": str(directory.relative_to(root))}
                for metric in METRICS:
                    if metric in summary:
                        value = float(summary[metric])
                        if not np.isfinite(value):
                            raise ConfigurationError(f"Métrica no finita en {directory}: {metric}")
                        row[metric] = value
                    elif metric in METRICS[:4]:
                        raise ConfigurationError(f"Falta {metric} en {directory}")
                rows.append(row)
    grouped = []
    for kind, label in sorted({(r["dataset"], r["model"]) for r in rows}):
        selected = [r for r in rows if r["dataset"] == kind and r["model"] == label]
        if {r["seed"] for r in selected} != {7, 17, 27}:
            raise ConfigurationError("La comparación requiere las semillas 7,17,27")
        result = {"dataset": kind, "model": label, "runs": len(selected)}
        for metric in METRICS:
            values = [r[metric] for r in selected if metric in r]
            if values:
                result[metric + "_mean"] = float(np.mean(values))
                result[metric + "_std"] = float(np.std(values, ddof=1))
        grouped.append(result)
    if not write_outputs:
        return grouped
    output = root / f"results/generative_{scope}_report"
    output.mkdir(parents=True, exist_ok=True)
    # Unificar campos opcionales de audio, sin convertir faltantes a ceros.
    keys = list(dict.fromkeys(key for row in rows for key in row))
    write_csv(output / "runs.csv", [{k: row.get(k, "") for k in keys} for row in rows])
    keys = list(dict.fromkeys(key for row in grouped for key in row))
    write_csv(output / "summary.csv", [{k: row.get(k, "") for k in keys} for row in grouped])
    for kind in kinds:
        _plot([r for r in grouped if r["dataset"] == kind], output / f"{kind}.png")
    (output / "status.json").write_text(json.dumps({"status": "completed", "runs": len(rows),
        "scope": scope, "conclusion": "Inspect generated images/audio. Metrics are descriptive; "
        "diversity, distance to training and pitch alone do not certify valid novel samples."}, indent=2), encoding="utf-8")
    print(f"Comparación de {len(rows)} corridas en {output}")
    return grouped


def _plot(rows, path):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, metric, title in zip(axes, ["heldout_mean_squared_error", "generated_pairwise_mse"],
                                    ["Reconstrucción de prueba (menor es mejor)", "Diversidad generada (no mide calidad)"]):
        axis.bar(range(len(rows)), [r[metric + "_mean"] for r in rows],
                 yerr=[r[metric + "_std"] for r in rows], capsize=3)
        axis.set_xticks(range(len(rows)), [r["model"] for r in rows], rotation=25)
        axis.set_title(title, fontsize=10)
        axis.set_ylabel("MSE")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--shapes", action="store_true")
    group.add_argument("--timbres", action="store_true")
    args = parser.parse_args()
    try:
        compare_suite(PROJECT_ROOT, "shapes" if args.shapes else "timbres" if args.timbres else "all")
    except (ConfigurationError, KeyError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
