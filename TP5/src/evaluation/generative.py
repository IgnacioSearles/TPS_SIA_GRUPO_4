"""Prueba independiente, diversidad y proximidad a entrenamiento: sin declarar éxito automático."""

import json
import numpy as np

from src import plotting
from src.evaluation.context import write_csv
from src.evaluation.pitch import pitch_measurement
from src.options import ConfigurationError


def validate_generative(dataset, options):
    if dataset.generative_study is None:
        raise ConfigurationError("evaluation.generative necesita un dataset generativo con prueba reservada")
    count = options.take("count", 64)
    examples = options.take("test_examples", 12)
    audio = options.take("audio_examples", 6)
    seed = options.take("seed", 1234)
    options.ensure_consumed()
    if type(count) is not int or count < 2:
        raise ConfigurationError("generative.count debe ser al menos 2")
    if any(type(v) is not int or v < 0 for v in (examples, audio, seed)):
        raise ConfigurationError("test_examples, audio_examples y seed deben ser enteros no negativos")
    return count, examples, audio, seed


def nearest_training(samples, training):
    """MSE y vecino más próximo, en bloques, sin tensor N x M x D."""
    samples = np.asarray(samples, dtype=float)
    distances = np.full(len(samples), np.inf)
    indices = np.zeros(len(samples), dtype=int)
    for start in range(0, len(training), 256):
        reference = np.asarray(training[start:start + 256], dtype=float)
        matrix = (np.sum(samples**2, axis=1)[:, None] + np.sum(reference**2, axis=1)[None, :]
                  - 2 * samples @ reference.T) / samples.shape[1]
        candidates = np.argmin(matrix, axis=1)
        values = np.maximum(0, matrix[np.arange(len(samples)), candidates])
        improved = values < distances
        distances[improved], indices[improved] = values[improved], start + candidates[improved]
    return distances, indices


def pairwise_mse(samples):
    """MSE media entre pares distintos: mide diversidad, no plausibilidad."""
    samples = np.asarray(samples, dtype=float)
    return float(2 * np.var(samples, axis=0).mean() * len(samples) / (len(samples) - 1))


def _shown(total, count):
    return sorted(set(np.linspace(0, total - 1, min(count, total)).round().astype(int))) if count else []


def evaluate_generative(context, options):
    count, examples, audio_count, seed = validate_generative(context.dataset, options)
    dataset, model, output = context.dataset, context.model, context.output_directory
    study = dataset.generative_study
    heldout = study.test_samples.astype(model.precision, copy=False)
    reconstructed = model.reconstruct(heldout)
    errors = np.mean((heldout - reconstructed)**2, axis=1)
    write_csv(output / "test_reconstruction.csv", [
        {"label": label, "mse": float(error)} for label, error in zip(study.test_labels, errors)])
    shown = _shown(len(heldout), examples)
    if shown:
        plotting.save_image_rows([heldout[shown], reconstructed[shown]], ["Prueba original", "Reconstrucción"],
                                 [study.test_labels[i] for i in shown], dataset.image_shape, False,
                                 output / "test_reconstruction.png")

    # El mismo conjunto de coordenadas para todas las configuraciones de igual latente.
    coordinates = np.random.default_rng(seed).standard_normal((count, model.latent_size)).astype(model.precision)
    generated = model.decode(coordinates)
    nearest, indices = nearest_training(generated, dataset.samples)
    rows = [{"sample": i, "nearest_training_label": dataset.labels[index],
             "nearest_training_mse": float(distance),
             **{f"z{j + 1}": float(value) for j, value in enumerate(coordinates[i])}}
            for i, (distance, index) in enumerate(zip(nearest, indices))]
    side = int(np.ceil(np.sqrt(count)))
    plotting.save_image_mosaic(generated, (int(np.ceil(count / side)), side), dataset.image_shape,
                               False, "Muestras desde N(0,I): inspeccionar plausibilidad y diversidad",
                               output / "generated.png")
    nearest_shown = _shown(count, min(12, count))
    plotting.save_image_rows([generated[nearest_shown], dataset.samples[indices[nearest_shown]]],
                             ["Generada", "Entrenamiento más próximo"],
                             [str(i) for i in nearest_shown], dataset.image_shape, False,
                             output / "generated_nearest.png")
    np.savez_compressed(output / "generated.npz", coordinates=coordinates, samples=generated)
    metrics = {"heldout_mean_squared_error": float(errors.mean()),
               "generated_nearest_train_mse_mean": float(nearest.mean()),
               "generated_pairwise_mse": pairwise_mse(generated)}
    if dataset.audio_codec is not None:
        metrics.update(_audio_evaluation(context, generated, reconstructed, audio_count, rows))
    write_csv(output / "generated.csv", rows)
    manifest = {"data": study.metadata, "model_seed": context.seed, "generation_seed": seed,
                "train_count": len(dataset), "test_count": len(heldout),
                "train": study.train_records, "test": study.test_records,
                "generation_count": count, "parameters_given_to_model": False,
                "interpretation": "Distance and diversity are descriptive, not tests of valid novelty. "
                                  "Fixed-pitch accuracy is not a timbre quality score."}
    (output / "generative_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return metrics


def _audio_evaluation(context, generated, reconstructed, audio_count, rows):
    dataset = context.dataset
    study, codec = dataset.generative_study, dataset.audio_codec
    target = 69 + 12 * np.log2(study.metadata["frequency_hz"] / 440)
    all_measurements = []
    for i, sample in enumerate(generated):
        waveform = codec.decode(sample.reshape(dataset.image_shape))
        measured = pitch_measurement(waveform, codec.sample_rate, target)
        rows[i].update(measured)
        all_measurements.append(measured)
        if i < audio_count:
            context.write_waveform(f"audio/generative/sample_{i:02d}.wav", waveform)
    calibration = []
    for index in _shown(len(study.test_samples), audio_count):
        waveform = codec.decode(study.test_samples[index].reshape(dataset.image_shape))
        calibration.append(pitch_measurement(waveform, codec.sample_rate, target))
        context.write_waveform(f"audio/test/{index:04d}__raw.wav", study.test_waveforms[index])
        context.write_waveform(f"audio/test/{index:04d}__codec.wav", waveform)
        context.write_audio(f"audio/test/{index:04d}__model.wav", reconstructed[index])
    if calibration:
        write_csv(context.output_directory / "codec_pitch_calibration.csv", calibration)
    result = {"generated_fixed_pitch_hit_rate": float(np.mean([m["hit"] for m in all_measurements]))}
    if calibration:
        result["codec_fixed_pitch_hit_rate"] = float(np.mean([m["hit"] for m in calibration]))
    return result
