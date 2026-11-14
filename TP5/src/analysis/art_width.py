"""Diagnóstico AE/VAE x ancho: preparar, validar y confirmar contrastes fijados."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.data import art as renderer
from src.data.art_capacity import file_hash, read_json, load_probe
from src.data.art_width import VARIANTS, SEEDS, CONTRASTS, configurations, prepare_width
from src.data.factory import build_dataset
from src.evaluation import art_geometry, art_factorial
from src.evaluation.art_factorial import evaluate_factorial
from src.experiments.grid import expand_runs
from src.options import ConfigurationError, Options
from src.analysis import latent_sampling

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def snapshot(root):
    directory = root / "results/art_width_data"
    manifest = read_json(directory / "manifest.json")
    configs = configurations(root, manifest["protocol"]["wide_layers"])
    sources, data_identity = [], None
    for variant, config in configs.items():
        name = VARIANTS[variant][0]
        if file_hash(root / f"configs/{name}.json") != manifest["protocol"]["configuration_hashes"][variant]:
            raise ConfigurationError("La configuración cambió después de fijar el protocolo")
        for run in expand_runs(config):
            path = root / f"results/{name}/{run.label}"
            if read_json(path / "config.json") != run.config:
                raise ConfigurationError(f"Configuración guardada incompatible: {path}")
            read_json(path / "summary.json")
            data = read_json(path / "generative_manifest.json")
            if (data["model_seed"] != run.seed or data["parameters_given_to_model"] is not False
                    or data["train_count"] != config["dataset"]["train_count"]
                    or data["test_count"] != config["dataset"]["test_count"]):
                raise ConfigurationError(f"Partición incompatible: {path}")
            if data_identity is not None and data_identity != data["data"]:
                raise ConfigurationError("Las variantes no usaron los mismos datos de entrenamiento")
            data_identity = data["data"]
            with (path / "history.csv").open() as stream:
                history = list(csv.DictReader(stream))
            if not history or int(history[-1]["epoch"]) != config["training"]["epochs"]:
                raise ConfigurationError(f"Entrenamiento incompleto: {path}")
            kind = config["model"]["type"]
            label = ("AE" if kind == "autoencoder" else "VAE") + " " + str(config["model"]["hidden_layers"])
            sources.append({"variant": variant, "seed": run.seed, "model_type": kind, "label": label,
                            "run": str(path.relative_to(root)), "model_sha256": file_hash(path / "model.npz"),
                            "config_sha256": file_hash(path / "config.json")})
    if manifest["training_sha256"] != data_identity["train_sha256"]:
        raise ConfigurationError("El protocolo y los pesos no corresponden a los mismos datos")
    for split, item in manifest["partitions"].items():
        if file_hash(directory / f"{split}.npz") != item["file_sha256"]:
            raise ConfigurationError(f"Partición alterada: {split}")
    dependencies = (art_geometry, renderer, art_factorial, latent_sampling)
    inputs = {"sources": sources, "data_manifest_sha256": file_hash(directory / "manifest.json"),
              "evaluation_sha256": {module.__name__: file_hash(Path(module.__file__)) for module in dependencies}}
    return configs, manifest, sources, inputs


def choose_by_type(grouped, tolerance=.01):
    result = {}
    for kind in ("ae", "vae"):
        small, wide = (next(r for r in grouped if r["variant"] == f"{kind}_{size}") for size in ("small", "wide"))
        result[kind] = (small["variant"] if small["macro_silhouette_iou_mean"]
                       >= wide["macro_silhouette_iou_mean"] - tolerance else wide["variant"])
    return result


def contrasts(rows):
    mapping = {(r["variant"], r["seed"]): r for r in rows}
    result = []
    for name, (a, b) in CONTRASTS.items():
        values = [mapping[a, seed]["macro_silhouette_iou"] - mapping[b, seed]["macro_silhouette_iou"] for seed in SEEDS]
        result.append({"contrast": name, "a": a, "b": b, "iou_delta_mean": float(np.mean(values)),
                       "iou_delta_std": float(np.std(values, ddof=1)),
                       **{f"seed_{seed}": float(value) for seed, value in zip(SEEDS, values)}})
    return result


def run_stage(root=PROJECT_ROOT, final=False):
    root = Path(root)
    configs, manifest, sources, inputs = snapshot(root)
    directory = root / "results/art_width_data"
    validation = root / "results/art_width_validation"
    selection_path = validation / "selection.json"
    if final:
        selection = read_json(selection_path)
        if selection["inputs"] != inputs:
            raise ConfigurationError("Las fuentes cambiaron después de la selección")
        output = root / "results/art_width_final"
        stamp = output / "manifest.json"
        expected = {"inputs": inputs, "selection_sha256": file_hash(selection_path),
                    "status": "completed", "scope": "four predeclared conditions; no retuning"}
    else:
        output = validation; stamp = selection_path; expected = None
    if stamp.exists():
        saved = read_json(stamp)
        if saved.get("inputs") != inputs or (final and saved != expected):
            raise ConfigurationError("La evaluación congelada tiene otras fuentes")
        print(f"Se reutiliza {output}", flush=True)
        return output
    samples, records = load_probe(directory, "reserved_final" if final else "validation")
    training = None
    if not final:
        calibration = art_geometry.calibrate_geometry(samples, records)
        if calibration["nonperfect_references"]:
            raise ConfigurationError("Falló la calibración geométrica de los originales")
        training = build_dataset(Options(configs["vae_small"]["dataset"], "dataset"), root)
        if (training.generative_study.metadata["train_sha256"] != manifest["training_sha256"]
                or training.generative_study.metadata["test_sha256"] != manifest["previous_test_sha256"]):
            raise ConfigurationError("Los datos actuales cambiaron")
        output.mkdir(parents=True, exist_ok=True)
        (output / "calibration.json").write_text(json.dumps(calibration, indent=2))
    grouped, rows = evaluate_factorial(root, output, samples, records, sources, SEEDS, training)
    from src.evaluation.context import write_csv
    write_csv(output / "contrasts.csv", contrasts(rows))
    if not final:
        expected = {"inputs": inputs, "selected_by_type": choose_by_type(grouped, manifest["protocol"]["iou_tolerance"]),
                    "criterion": manifest["protocol"]["selection"], "final_evaluated_at_selection": False,
                    "scope": "geometry reconstruction, not certification of generative quality"}
    stamp.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    print(f"Etapa completada: {output}", flush=True)
    return output


def report(root=PROJECT_ROOT):
    root = Path(root)
    validation = root / "results/art_width_validation"
    final = root / "results/art_width_final"
    selection, completed = read_json(validation / "selection.json"), read_json(final / "manifest.json")
    _, _, _, inputs = snapshot(root)
    if (selection["inputs"] != inputs or completed["inputs"] != inputs
            or completed["selection_sha256"] != file_hash(validation / "selection.json")):
        raise ConfigurationError("El informe no corresponde a los modelos congelados")
    def table(path, contrast=False):
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        if contrast:
            lines = ["| Contraste de IoU (A − B) | Media ± desvío |", "|---|---:|"]
            lines += [f"| {r['a']} − {r['b']} | {float(r['iou_delta_mean']):+.4f} ± {float(r['iou_delta_std']):.4f} |" for r in rows]
        else:
            lines = ["| Modelo | IoU ↑ | Contorno (px) ↓ | MSE en figura ↓ |", "|---|---:|---:|---:|"]
            lines += [f"| {r['label']} | {float(r['macro_silhouette_iou_mean']):.4f} ± {float(r['macro_silhouette_iou_std']):.4f} | "
                      f"{float(r['macro_contour_distance_px_mean']):.4f} | {float(r['macro_foreground_mse_mean']):.5f} |" for r in rows]
        return "\n".join(lines)
    text = f'''# AE/VAE y ancho en arte RGB

Cuatro condiciones prefijadas, tres semillas por condición, latente 8 y los mismos
datos, MSE y presupuesto de 200 épocas. Se reutilizó el VAE [128,32] y se entrenaron
nueve modelos nuevos. Se evalúan 512 referencias nuevas de validación y otras
512 finales, independientes de los estudios anteriores.

## Validación

{table(validation / 'summary.csv')}

Elecciones geométricas por tipo, fijadas antes de la prueba final:
`{selection['selected_by_type']}`. Se prefiere la red menor si queda a 0,01 o menos
de la mejor IoU media del mismo tipo. Esto no certifica calidad generativa.

## Confirmación final de las cuatro condiciones

{table(final / 'summary.csv')}

{table(final / 'contrasts.csv', True)}

Los contrastes se emparejan por semilla y por imágenes. Son descriptivos: tres
semillas no justifican superioridad universal. Se abrió la prueba final después
de congelar el análisis, para confirmar los cuatro contrastes previamente fijados;
no se volvió a elegir ni se ajustaron hiperparámetros con ella.

## Interpretación y límites

La diferencia AE/VAE incluye KL, muestreo durante entrenamiento y cabeza de
varianza. No atribuirla exclusivamente a KL: eso requeriría otro control con VAE
beta=0. Ampliar las capas cambia capacidad y cantidad de parámetros; AE y VAE de
igual ancho tienen distinta cabeza de salida del encoder.

La comparación principal de generación usa AE con una gaussiana ajustada solo
a sus códigos de entrenamiento, y VAE con N(0,I). El AE con N(0,I) se conserva
como diagnóstico suplementario. Son procedimientos de generación distintos,
no una prueba de que un tipo de red siempre genere mejor. No se ajusta la
gaussiana con referencias reservadas ni se modifican pesos al evaluarlos.

Hay 64 generaciones por condición y semilla: 768 principales, sin filtrar fallos,
y 384 del diagnóstico AE estándar. Las comparaciones compactas muestran índices
0,8,...,56; las reconstrucciones muestran dos referencias por familia, prefijadas.
Las métricas de geometría se aplican a reconstrucciones con referencia, no
certifican validez de las muestras aleatorias. Inspeccionar esquinas, ruido y
diversidad junto a los números y vecinos de entrenamiento.

La silueta se segmenta con el evaluador calibrado del estudio anterior. Color y
forma no quedan perfectamente aislados si cambia mucho el brillo. Se conservan
fallos y predicciones vacías en todos los promedios.

El esquema variacional sigue siendo el mismo que en sonidos. Cambiar el ancho
para imágenes no obliga a igualar sus hiperparámetros con audio; no se comparten
pesos ni un espacio latente conjunto, ni se comparan MSE RGB con MSE mel.

Las tablas, gráficos por familia, reconstrucciones y galerías están en las
carpetas `art_width_validation` y `art_width_final`; los manifiestos guardan
hashes de las fuentes y del evaluador. No se modificaron resultados previos.
'''
    output = root / "results/art_width_report"
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.md").write_text(text, encoding="utf-8")
    print(f"Informe: {output / 'report.md'}", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "compare", "final", "report"):
        actions.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    try:
        if args.prepare: prepare_width(PROJECT_ROOT)
        elif args.compare: run_stage()
        elif args.final: run_stage(final=True)
        else: report()
    except (ConfigurationError, ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
