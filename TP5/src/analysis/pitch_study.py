"""Agregar el protocolo completo sin seleccionar semillas ni puntos favorables.

python -m src.analysis.pitch_study results/ej2_pitch_* --output results/pitch_study_report
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
from src.options import ConfigurationError

EXPECTED_SEEDS = {7, 17, 27}
BASELINES = ("spectrogram_mix", "lower_neighbor")


def aggregate_study(paths, output):
    runs = {}
    for root in paths:
        for manifest_path in sorted(Path(root).rglob("pitch_manifest.json")):
            directory = manifest_path.parent.resolve()
            if directory in runs:
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            summary_path = directory / "pitch_summary.json"
            table_path = directory / "pitch_interpolation.csv"
            if not summary_path.is_file() or not table_path.is_file():
                raise ConfigurationError(f"Evaluación incompleta en {directory}; usar --evaluate-only")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            with table_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            runs[directory] = (manifest, summary, rows)
    if not runs:
        raise ConfigurationError("No se encontraron evaluaciones pitch_interpolation")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result, combined = {}, []
    for source in sorted({m["source"] for m, _, _ in runs.values()}):
        selected = [(directory, *value) for directory, value in runs.items() if value[0]["source"] == source]
        identities, signatures, reliable = set(), set(), True
        trials = []
        for directory, manifest, summary, rows in selected:
            identity = (manifest["omit_targets"], manifest["seed"])
            if identity in identities:
                raise ConfigurationError(f"Dos corridas para {source}, modo/semilla {identity}")
            identities.add(identity)
            # No mezclar selecciones, códecs, objetivos o versiones del protocolo.
            config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
            signature = {"protocol": manifest["protocol_version"], "codec": manifest["codec"],
                         "model": config.get("model", {}), "training": config.get("training", {}),
                         "targets": manifest["targets"],
                         "evaluation_variants": manifest.get("evaluation_variants", ["base"]),
                         "notes": [{k: v for k, v in n.items() if k != "split"}
                                   for n in manifest["notes"]]}
            signatures.add(json.dumps(signature, sort_keys=True))
            reliable &= summary["pitch_calibration_reliable"]
            midpoint_rows = [r for r in rows if float(r["fraction"]) == 0.5]
            expected_methods = {"vae_interpolation", *BASELINES}
            if not manifest["omit_targets"]:
                expected_methods.add("target_reconstruction")
            variants = manifest.get("evaluation_variants", ["base"])
            keys = [(int(r["target_pitch"]), r.get("variant", "base"), r["method"]) for r in midpoint_rows]
            expected = {(p, variant, method) for p in manifest["targets"]
                        for variant in variants for method in expected_methods}
            if len(keys) != len(expected) or set(keys) != expected:
                raise ConfigurationError(f"Predicciones centrales faltantes o repetidas en {directory}")
            for row in midpoint_rows:
                normalized = {"source": source, "seed": manifest["seed"],
                              "omit_targets": manifest["omit_targets"],
                              "target_pitch": int(row["target_pitch"]), "variant": row.get("variant", "base"),
                              "method": row["method"],
                              "hit": row["hit"] == "True", "valid": row["valid"] == "True",
                              "error_cents": row["error_cents"], "run": str(directory)}
                combined.append(normalized)
                if manifest["omit_targets"]:
                    trials.append(normalized)
        if len(signatures) != 1:
            raise ConfigurationError(f"Las corridas de {source} no tienen el mismo protocolo/dataset/códec")
        expected_runs = {(omit, seed) for omit in (True, False) for seed in EXPECTED_SEEDS}
        complete = identities == expected_runs and len(manifest["targets"]) == 3
        methods = {}
        for method in ("vae_interpolation", *BASELINES):
            values = [r["hit"] for r in trials if r["method"] == method]
            methods[method] = {"hits": sum(values), "trials": len(values),
                               "hit_rate": float(np.mean(values)) if values else None}
        main = methods["vae_interpolation"]
        per_variant = {}
        for variant in variants:
            values = [r["hit"] for r in trials if r["method"] == "vae_interpolation"
                      and r["variant"] == variant]
            per_variant[variant] = {"hits": sum(values), "trials": len(values)}
        favorable = (complete and reliable and main["trials"] == 9 * len(variants)
                     and all(v["trials"] == 9 and v["hits"] >= 8 for v in per_variant.values())
                     and all(main["hit_rate"] > methods[b]["hit_rate"] for b in BASELINES))
        status = ("incomplete" if not complete else "inconclusive_codec" if not reliable
                  else "favorable" if favorable else "not_favorable")
        result[source] = {"status": status, "complete": complete,
                          "calibration_reliable": bool(reliable), "favorable": bool(favorable),
                          "methods": methods, "per_variant": per_variant,
                          "missing_runs": [{"omit_targets": o, "seed": s}
                                           for o, s in sorted(expected_runs - identities)]}
    write_csv(output / "midpoints.csv", combined)
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _plot_comparison(combined, result, output / "comparison.png")
    print(json.dumps(result, indent=2))
    return result


def _plot_comparison(rows, summaries, path):
    fig, axes = plt.subplots(1, len(summaries), squeeze=False, figsize=(8 * len(summaries), 4))
    methods = ["vae_interpolation", *BASELINES, "target_reconstruction"]
    for axis, (source, summary) in zip(axes[0], summaries.items()):
        relevant = [r for r in rows if r["source"] == source]
        for index, method in enumerate(methods):
            values = [r for r in relevant if r["method"] == method
                      and (r["omit_targets"] if method != "target_reconstruction" else not r["omit_targets"])]
            if values:
                rate = np.mean([r["hit"] for r in values])
                axis.bar(index, rate, alpha=0.5)
                for seed in sorted({r["seed"] for r in values}):
                    subset = [r["hit"] for r in values if r["seed"] == seed]
                    axis.scatter(index + (seed - 17) / 100, np.mean(subset), color="black", s=20)
                axis.text(index, min(1.04, rate + 0.03), f"{sum(r['hit'] for r in values)}/{len(values)}",
                          ha="center", fontsize=9)
        axis.set_xticks(range(len(methods)), ["VAE omitidas", "Mezcla mel", "Vecino inferior", "Control conocido"],
                        rotation=15)
        axis.set(ylim=(0, 1.13), ylabel="Fracción con error ≤ 50 cents",
                 title=f"{source}: {summary['status']}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/pitch_study_report"))
    args = parser.parse_args()
    try:
        aggregate_study(args.experiments, args.output)
    except ConfigurationError as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
