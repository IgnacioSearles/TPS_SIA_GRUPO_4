"""Comparación de un parámetro por vez, con controles y prueba final reservada."""

import copy
from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path

from src.evaluation.context import write_csv
from src.experiments.grid import expand_runs
from src.options import ConfigurationError

VALIDATION_TARGETS = [51, 57, 63]
FINAL_TARGETS = [55, 61, 67]
SEEDS = [7, 17, 27]
VARIANTS = ["v00", "v20", "v39"]


@dataclass(frozen=True)
class StudyComparison:
    name: str
    parameter_path: tuple[str, ...]
    values: tuple
    field: str
    value_prefix: str


LATENT = StudyComparison("latent", ("model", "latent_size"), (2, 4, 16), "latent_size", "z")
INTERPOLATION = StudyComparison("interploss", ("training", "interpolation", "weight"),
                                (0.0, 0.1, 1.0), "interpolation_weight", "lambda")


def parameter(config, path):
    for key in path:
        config = config[key]
    return config


def parameter_parent(config, path):
    return parameter(config, path[:-1])


def _config_paths(root, phase, study):
    return [root / f"configs/ej2_pitch_{study.name}_{phase}_{condition}.json"
            for condition in ("complete", "omitted")]


def _read(path):
    if not path.is_file():
        raise ConfigurationError(f"Falta {path}; completar esta etapa antes de comparar")
    return json.loads(path.read_text(encoding="utf-8"))


def compare_study(root, study, phase="validation"):
    """Sólo abre archivos de la fase indicada; nunca consulta resultados finales al elegir."""
    root = Path(root)
    groups, reference_signature, evidence = {}, None, {}
    for config_path in _config_paths(root, phase, study):
        config = _read(config_path)
        name = config.pop("name", config_path.stem)
        output = root / config.pop("output", f"results/{name}")
        runs = expand_runs(config)
        if sorted({r.seed for r in runs}) != SEEDS:
            raise ConfigurationError("La comparación requiere las semillas 7, 17 y 27")
        for run in runs:
            dimension = parameter(run.config, study.parameter_path)
            settings = run.config["dataset"]["pitch_study"]
            condition = "omitted" if settings["omit_targets"] else "complete"
            targets = VALIDATION_TARGETS if phase == "validation" else FINAL_TARGETS
            reserved = FINAL_TARGETS if phase == "validation" else VALIDATION_TARGETS
            if (settings["targets"] != targets or settings.get("reserved_targets") != reserved
                    or settings["evaluation_variants"] != VARIANTS
                    or settings["exclusion_radius"] != 0.75):
                raise ConfigurationError("La separación de validación/final no coincide con el protocolo")
            directory = output / run.label
            actual = _read(directory / "config.json")
            if actual != run.config:
                raise ConfigurationError(f"La configuración guardada no coincide en {directory}")
            training = _read(directory / "training.json")
            _read(directory / "summary.json")  # Sólo corridas terminadas.
            if not (directory / "model.npz").is_file():
                raise ConfigurationError(f"Faltan los pesos en {directory}")
            manifest = _read(directory / "pitch_manifest.json")
            metrics = _read(directory / "pitch_summary.json")
            if (manifest.get("reserved_targets") != reserved or manifest["targets"] != targets
                    or manifest["seed"] != run.seed):
                raise ConfigurationError(f"Manifiesto incompatible en {directory}")
            if not metrics["pitch_calibration_reliable"]:
                raise ConfigurationError(f"Calibración inconclusa en {directory}; no seleccionar un ganador")
            notes = manifest["notes"]
            if study.name == "interploss":
                _validate_triplet_manifest(directory, notes, training)
            for note in notes:
                if note["split"] == "training" and any(abs(note["pitch"] - p) <= 0.75 + 1e-9
                                                          for p in reserved + (targets if condition == "omitted" else [])):
                    raise ConfigurationError(f"Una altura reservada entró en entrenamiento en {directory}")
            normalized = copy.deepcopy(actual)
            parameter_parent(normalized, study.parameter_path).pop(study.parameter_path[-1])
            model = normalized["model"]
            dataset = copy.deepcopy(actual["dataset"])
            dataset["pitch_study"].pop("omit_targets")
            signature = json.dumps({"model": model, "dataset": dataset, "training": normalized["training"],
                                    "codec": manifest["codec"],
                                    "notes": [{k: v for k, v in n.items() if k != "split"} for n in notes]}, sort_keys=True)
            if reference_signature is not None and signature != reference_signature:
                raise ConfigurationError("Se cambió algo más que el parámetro comparado/condición")
            reference_signature = signature
            with (directory / "history.csv").open(newline="", encoding="utf-8") as stream:
                history = list(csv.DictReader(stream))
            if int(history[-1]["epoch"]) != actual["training"]["epochs"]:
                raise ConfigurationError(f"Entrenamiento incompleto en {directory}")
            table = directory / "pitch_interpolation.csv"
            if not table.is_file():
                raise ConfigurationError(f"Falta evaluación espectral en {directory}")
            with table.open(newline="", encoding="utf-8") as stream:
                rows = [r for r in csv.DictReader(stream) if float(r["fraction"]) == 0.5]
            methods = {"vae_interpolation", "spectrogram_mix", "lower_neighbor"}
            if condition == "complete":
                methods.add("target_reconstruction")
            keys = [(int(r["target_pitch"]), r["variant"], r["method"]) for r in rows]
            expected = {(p, v, m) for p in targets for v in VARIANTS for m in methods}
            if len(keys) != len(expected) or set(keys) != expected:
                raise ConfigurationError(f"Casos faltantes o repetidos en {directory}")
            group = groups.setdefault((condition, dimension), {"seeds": set(), "rows": [], "samples": []})
            if run.seed in group["seeds"]:
                raise ConfigurationError("Semilla repetida en una condición/dimensión")
            group["seeds"].add(run.seed)
            group["rows"].extend(rows)
            group["samples"].append(training["samples"])
            evidence[str(table.relative_to(root))] = hashlib.sha256(table.read_bytes()).hexdigest()
    dimensions = set(study.values) if phase == "validation" else {key[1] for key in groups}
    if phase == "final":
        locked = _read(root / f"results/pitch_{study.name}_validation_report/selection.json")
        if dimensions != {locked[f"selected_{study.field}"]}:
            raise ConfigurationError("La evaluación final debe usar sólo el valor elegido en validación")
    if set(groups) != {(condition, dim) for condition in ("complete", "omitted") for dim in dimensions}:
        raise ConfigurationError("Faltan condiciones o valores del parámetro")
    summary = []
    for (condition, dimension), group in sorted(groups.items()):
        if group["seeds"] != set(SEEDS):
            raise ConfigurationError("Faltan semillas")
        for method in sorted({r["method"] for r in group["rows"]}):
            rows = [r for r in group["rows"] if r["method"] == method]
            if len(rows) != 27 or any(not r.get("spectral_mse") for r in rows):
                raise ConfigurationError("Se requieren 27 casos por método y métricas espectrales")
            summary.append({"condition": condition, study.field: dimension, "method": method,
                            "n": len(rows), "pitch_hits": sum(r["hit"] == "True" for r in rows),
                            "spectral_hits": sum(r["spectral_hit"] == "True" for r in rows),
                            "joint_hits": sum(r["hit"] == "True" and r["spectral_hit"] == "True" for r in rows),
                            "spectral_mse": sum(float(r["spectral_mse"]) for r in rows) / len(rows)})
    candidates = [r for r in summary if r["condition"] == "omitted" and r["method"] == "vae_interpolation"]
    selected = min(candidates, key=lambda r: (-r["joint_hits"], r["spectral_mse"], r[study.field]))
    report = {"phase": phase, f"selected_{study.field}": selected[study.field],
              "selection_rule": "maximize joint pitch+spectral hits; then minimize spectral MSE; then smaller parameter",
              "selection_is_not_evidence_of_success": True, "summary": summary, "evidence_sha256": evidence}
    output = root / f"results/pitch_{study.name}_{phase}_report"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "comparison.csv", summary)
    (output / "selection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Etapa {phase}: {study.field} seleccionado {selected[study.field]}; informe en {output}")
    return report


def _validate_triplet_manifest(directory, notes, training):
    path = directory / "interpolation_triplets.csv"
    if not path.is_file():
        raise ConfigurationError(f"Falta el manifiesto de ternas en {directory}")
    with path.open(newline="", encoding="utf-8") as stream:
        triplets = list(csv.DictReader(stream))
    if not triplets or len(triplets) != training.get("interpolation_triplets"):
        raise ConfigurationError(f"Manifiesto de ternas vacío o incompatible en {directory}")
    by_label = {n["label"]: n for n in notes}
    for row in triplets:
        selected = [by_label.get(row[key]) for key in ("lower", "center", "upper")]
        if any(n is None or n["split"] != "training" for n in selected):
            raise ConfigurationError(f"Una terna usa referencias reservadas en {directory}")
        lower, center, upper = selected
        if (abs(center["pitch"] - lower["pitch"] - 1) > 1e-9
                or abs(upper["pitch"] - center["pitch"] - 1) > 1e-9
                or any(len({n.get(key, "base") for n in selected}) != 1
                       for key in ("variant", "instrument", "velocity"))):
            raise ConfigurationError(f"Una terna mezcla alturas/timbres/envolventes en {directory}")


def prepare_final_study(root, report, study):
    """Genera seis corridas nuevas; nunca reutiliza pesos que hayan visto objetivos finales."""
    if report["phase"] != "validation":
        raise ConfigurationError("La selección debe proceder de validación")
    root = Path(root)
    files = []
    for condition, path in zip(("complete", "omitted"), _config_paths(root, "validation", study)):
        config = _read(path)
        config.pop("grid", None)
        config.pop("output", None)
        parameter_parent(config, study.parameter_path)[study.parameter_path[-1]] = report[f"selected_{study.field}"]
        config["dataset"]["pitch_study"].update(targets=FINAL_TARGETS, reserved_targets=VALIDATION_TARGETS)
        config["name"] = (f"ej2_pitch_{study.name}_final_{study.value_prefix}"
                          f"{report[f'selected_{study.field}']}_{condition}")
        path = root / f"configs/ej2_pitch_{study.name}_final_{condition}.json"
        content = json.dumps(config, indent=2) + "\n"
        if path.exists() and _read(path) != config:
            raise ConfigurationError(f"{path} ya contiene otra selección; no se sobrescribe")
        files.append((path, content))
    for path, content in files:
        path.write_text(content, encoding="utf-8")
    print(f"Configuraciones finales listas: configs/ej2_pitch_{study.name}_final_*.json")
