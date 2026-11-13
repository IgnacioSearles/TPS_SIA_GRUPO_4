"""Protocolo fijo de notas omitidas: generación sin consultar el objetivo."""

from dataclasses import asdict
import hashlib
import json

import numpy as np

from src import plotting
from src.evaluation.context import write_csv
from src.evaluation.pitch import pitch_measurement
from src.options import ConfigurationError


def validate_pitch_interpolation(dataset, options):
    options.ensure_consumed()
    study = dataset.pitch_study
    if study is None or dataset.audio_codec is None:
        raise ConfigurationError("pitch_interpolation necesita dataset.pitch_study y audio")
    labels = set(dataset.labels)
    for variant in study.evaluation_variants:
        for target in study.targets:
            for neighbor in (target - 1, target + 1):
                if study.note(neighbor, variant).label not in labels:
                    raise ConfigurationError(f"Falta el vecino {neighbor} en entrenamiento")
            present = study.note(target, variant).label in labels
            if present == study.omit_targets:
                raise ConfigurationError("La separación de objetivos no coincide con omit_targets")


def latent_trajectory(model, neighbors):
    """La única entrada del encoder son dos vecinos; el objetivo no participa."""
    codes = model.encode(neighbors)
    fractions = np.linspace(0, 1, 11)
    return fractions, model.decode(codes[0] + fractions[:, None] * (codes[1] - codes[0]))


def evaluate_pitch_interpolation(context, options):
    validate_pitch_interpolation(context.dataset, options)
    study = context.dataset.pitch_study
    codec = context.dataset.audio_codec
    output = context.output_directory
    expanded = study.evaluation_variants != ("base",)
    microtonal = any(not float(n.pitch).is_integer() for n in study.notes)
    manifest = {
        "protocol_version": 4 if study.reserved_targets else 3 if microtonal else 2 if expanded else 1,
        "source": study.source, "seed": context.seed,
        "omit_targets": study.omit_targets, "targets": study.targets,
        "codec": asdict(codec),
        "notes": [{"label": n.label, "pitch": n.pitch, "instrument": n.instrument,
                   "velocity": n.velocity, "source": study.source,
                   "split": "training" if n.label in context.dataset.labels else "reference",
                   "is_target": n.pitch in study.targets,
                   "waveform_sha256": hashlib.sha256(n.waveform.astype("<f8").tobytes()).hexdigest(),
                   **({"variant": n.variant, "parameters": n.parameters} if expanded else {})}
                  for n in study.notes],
    }
    if expanded:
        manifest["evaluation_variants"] = study.evaluation_variants
        manifest["calibration_scope"] = "all_pitches_selected_variants"
    if study.reserved_targets:
        manifest["reserved_targets"] = study.reserved_targets
        manifest["calibration_scope"] = "integer_pitches_except_reserved_targets"
    manifest_path = output / "pitch_manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        checked_fields = ["protocol_version", "source", "seed", "omit_targets", "targets", "notes"]
        if expanded:
            checked_fields.append("evaluation_variants")
        if study.reserved_targets:
            checked_fields.append("reserved_targets")
        if any(previous[key] != json.loads(json.dumps(manifest[key]))
               for key in checked_fields):
            raise ConfigurationError("El dataset/protocolo cambió respecto del manifiesto guardado; "
                                     "usar un experimento nuevo para no mezclar resultados")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Todas las alturas de las variantes predefinidas; nunca elegir por resultados.
    # Los microtonos de entrenamiento no se calibran: la evaluación usa sólo semitonos.
    calibration, reference_spectrograms = [], {}
    for note in study.notes:
        if (note.variant not in study.evaluation_variants or not float(note.pitch).is_integer()
                or note.pitch in study.reserved_targets):
            continue
        name = str(note.pitch) if note.variant == "base" else f"{note.pitch}_{note.variant}"
        spec = codec.encode(note.waveform).ravel()
        reference_spectrograms[note.pitch, note.variant] = spec
        original = pitch_measurement(note.waveform, codec.sample_rate, note.pitch)
        recovered = context.write_audio(f"audio/pitch/references/{name}_codec.wav", spec)
        context.write_waveform(f"audio/pitch/references/{name}_original.wav", note.waveform)
        for method, waveform in [("original", note.waveform), ("codec", recovered)]:
            measurement = original if method == "original" else pitch_measurement(
                waveform, codec.sample_rate, note.pitch)
            calibration.append({"seed": context.seed, "pitch": note.pitch, "variant": note.variant, "method": method,
                                **measurement})
    write_csv(output / "pitch_calibration.csv", calibration)
    codec_hits = [r["hit"] for r in calibration if r["method"] == "codec"]
    calibration_rate = float(np.mean(codec_hits))
    calibration_by_variant = {
        variant: float(np.mean([r["hit"] for r in calibration
                                if r["method"] == "codec" and r["variant"] == variant]))
        for variant in study.evaluation_variants}
    reliable = all(rate >= 0.9 for rate in calibration_by_variant.values())

    rows, image_rows, image_names = [], [], []
    for variant in study.evaluation_variants:
        for target in study.targets:
            folder = str(target) if variant == "base" else f"{target}_{variant}"
            indices = [context.dataset.index_of(study.note(p, variant).label) for p in (target - 1, target + 1)]
            neighbors = context.dataset.samples[indices]
            fractions, decoded = latent_trajectory(context.model, neighbors)
            # El objetivo real sólo se usa DESPUÉS de generar, para medir (nunca como entrada).
            references = {p: reference_spectrograms[p, variant] for p in (target - 1, target, target + 1)}
            waveforms = []
            for index, (fraction, spec) in enumerate(zip(fractions, decoded)):
                waveform = context.write_audio(f"audio/pitch/{folder}/step_{index:02d}.wav", spec)
                waveforms.append(waveform)
                rows.append(_row(context, target, "vae_interpolation", float(fraction), waveform, variant,
                                 spec, references))
            context.write_waveform(f"audio/pitch/{folder}/midpoint.wav", waveforms[5])
            context.write_waveform(f"audio/pitch/{folder}/sequence.wav", np.concatenate(waveforms))
            comparison = [("spectrogram_mix", neighbors.mean(axis=0)),
                          ("lower_neighbor", neighbors[0])]
            if not study.omit_targets:
                # Control conocido: no interviene en la trayectoria latente.
                target_input = context.dataset.samples[[context.dataset.index_of(study.note(target, variant).label)]]
                comparison.append(("target_reconstruction", context.model.reconstruct(target_input)[0]))
            for method, spec in comparison:
                waveform = context.write_audio(f"audio/pitch/{folder}/{method}.wav", spec)
                rows.append(_row(context, target, method, 0.5, waveform, variant, spec, references))
            image_rows.append(np.vstack([reference_spectrograms[target, variant], neighbors[0], decoded[5],
                                         neighbors[1], neighbors.mean(axis=0)]))
            image_names.append(f"Objetivo MIDI {target} / {variant}")
    write_csv(output / "pitch_interpolation.csv", rows)
    plotting.save_pitch_trajectories(rows, output / "pitch_trajectories.png")
    plotting.save_image_rows(image_rows, image_names,
                             ["Objetivo reservado" if study.omit_targets else "Objetivo conocido",
                              "Vecino inferior", "VAE: punto medio", "Vecino superior", "Mezcla mel"],
                             codec.image_shape, False, output / "pitch_spectrograms.png")
    midpoints = [r for r in rows if r["method"] == "vae_interpolation" and r["fraction"] == 0.5]
    mixes = [r for r in rows if r["method"] == "spectrogram_mix"]
    summary = {"pitch_midpoint_hit_rate": float(np.mean([r["hit"] for r in midpoints])),
               "pitch_midpoint_spectral_hit_rate": float(np.mean([r["spectral_hit"] for r in midpoints])),
               "pitch_midpoint_spectral_mse": float(np.mean([r["spectral_mse"] for r in midpoints])),
               "pitch_mix_spectral_hit_rate": float(np.mean([r["spectral_hit"] for r in mixes])),
               "pitch_mix_spectral_mse": float(np.mean([r["spectral_mse"] for r in mixes])),
               "pitch_codec_hit_rate": calibration_rate, "pitch_calibration_reliable": reliable}
    detail = {**summary, "calibration_by_variant": calibration_by_variant,
              "midpoint_hits_by_variant": {
                  variant: {"pitch": int(sum(r["hit"] for r in midpoints if r["variant"] == variant)),
                            "spectral": int(sum(r["spectral_hit"] for r in midpoints
                                                if r["variant"] == variant)),
                            "total": sum(r["variant"] == variant for r in midpoints)}
                  for variant in study.evaluation_variants},
              "spectral_note": "spectral_mse: MSE against the true target spectrogram (same variant), "
                               "measured only after generation. spectral_hit: the target is the "
                               "nearest of {target-1, target, target+1} references. A mel mix of "
                               "both neighbours is a two-note chord and should not pass it.",
              "conclusion": "requires_aggregate" if reliable else "inconclusive_codec",
              "criterion": "at least 8/9 midpoints PER variant and strictly better than both baselines",
              "note": "Negative and unvoiced predictions are retained as failures; no octave correction."}
    (output / "pitch_summary.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")
    if not reliable:
        print(f"  Afinación inconclusa: alguna variante tiene menos de 90% de aciertos "
              f"tras el códec: {calibration_by_variant}", flush=True)
    return summary


def spectral_measurement(spec, references, target):
    """Distancia al espectrograma real del objetivo y a qué semitono se parece más."""
    distances = {p: float(np.mean((np.asarray(spec).ravel() - ref) ** 2)) for p, ref in references.items()}
    nearest = min(distances, key=distances.get)
    return {"spectral_mse": distances[target], "spectral_nearest_pitch": nearest,
            "spectral_hit": nearest == target}


def _row(context, target, method, fraction, waveform, variant="base", spec=None, references=None):
    row = {"seed": context.seed, "target_pitch": target, "variant": variant, "method": method, "fraction": fraction,
           **pitch_measurement(waveform, context.dataset.audio_codec.sample_rate, target)}
    if spec is not None:
        row.update(spectral_measurement(spec, references, target))
    return row
