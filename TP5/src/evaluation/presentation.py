"""Galería y recorridos del mismo protocolo generativo en imágenes y audio."""

import json

import numpy as np
from PIL import Image, ImageDraw

from src import plotting
from src.evaluation.context import write_csv
from src.options import ConfigurationError


def validate_presentation(dataset, options):
    if dataset.generative_study is None:
        raise ConfigurationError("presentation requiere un dataset generativo con prueba reservada")
    pairs = options.take("pairs", 3)
    steps = options.take("steps", 11)
    options.ensure_consumed()
    if type(pairs) is not int or pairs < 1 or 2*pairs > len(dataset.generative_study.test_samples):
        raise ConfigurationError("presentation.pairs necesita dos referencias de prueba por par")
    if type(steps) is not int or not 2 <= steps <= 61:
        raise ConfigurationError("presentation.steps debe estar entre 2 y 61")
    return pairs, steps


def save_rgb_animation(images, fractions, path):
    """GIF con fracción visible; vuelve por los mismos puntos, sin nuevos sorteos."""
    frames = []
    for image, fraction in zip(images, fractions):
        tile = Image.fromarray(np.round(np.clip(image, 0, 1) * 255).astype(np.uint8))
        frame = Image.new("RGB", (256, 286), (14, 19, 33))
        frame.paste(tile.resize((256, 256), Image.Resampling.NEAREST), (0, 0))
        ImageDraw.Draw(frame).text((10, 266), f"Interpolacion latente | t = {fraction:.2f}", fill="white")
        frames.append(frame)
    frames = frames + frames[-2:0:-1]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=200, loop=0)


def evaluate_presentation(context, options):
    pairs, steps = validate_presentation(context.dataset, options)
    dataset, model = context.dataset, context.model
    study = dataset.generative_study
    output = context.output_directory / "presentation"
    output.mkdir(parents=True, exist_ok=True)
    # Referencias prefijadas del conjunto reservado; ninguna selección por calidad.
    references = study.test_samples[:2*pairs].astype(model.precision, copy=False)
    codes = model.encode(references)  # media posterior en VAE
    fractions = np.linspace(0, 1, steps)
    rows, decoded_paths, coordinate_paths = [], [], []
    for pair in range(pairs):
        a, b = 2*pair, 2*pair+1
        path = codes[a] + fractions[:, None] * (codes[b] - codes[a])
        decoded = model.decode(path.astype(model.precision))
        decoded_paths.append(decoded)
        coordinate_paths.append(path)
        plotting.save_image_rows([references[[a]], model.reconstruct(references[[a]]),
                                  references[[b]], model.reconstruct(references[[b]])],
                                 ["A original", "A reconstruida", "B original", "B reconstruida"],
                                 [f"Par {pair}"], dataset.image_shape, False,
                                 output / f"pair_{pair}_references.png")
        if len(dataset.image_shape) == 3:
            save_rgb_animation(decoded.reshape((steps, *dataset.image_shape)), fractions,
                               output / f"pair_{pair}.gif")
        if dataset.audio_codec is not None:
            waves = [context.write_audio(f"presentation/pair_{pair}/step_{i:02d}.wav", sample)
                     for i, sample in enumerate(decoded)]
            silence = np.zeros(int(.15 * dataset.audio_codec.sample_rate))
            sequence = np.concatenate([piece for wave in waves for piece in (wave, silence)])
            context.write_waveform(f"presentation/pair_{pair}/sequence.wav", sequence)
            for name, index in (("a", a), ("b", b)):
                context.write_waveform(f"presentation/pair_{pair}/{name}_raw.wav", study.test_waveforms[index])
                context.write_audio(f"presentation/pair_{pair}/{name}_codec.wav", references[index])
        rows.extend({"pair": pair, "from": study.test_labels[a], "to": study.test_labels[b],
                     "fraction": float(fraction),
                     **{f"z{j+1}": float(value) for j, value in enumerate(z)}}
                    for fraction, z in zip(fractions, path))
    plotting.save_image_rows(decoded_paths, [f"Par {i}" for i in range(pairs)],
                             [f"{f:.2f}" for f in fractions], dataset.image_shape, False,
                             output / "trajectories.png")
    np.savez_compressed(output / "trajectories.npz", coordinates=np.asarray(coordinate_paths),
                        samples=np.asarray(decoded_paths), fractions=fractions)
    write_csv(output / "trajectories.csv", rows)
    (output / "manifest.json").write_text(json.dumps({
        "protocol": "heldout posterior-mean interpolation", "pairs": pairs, "steps": steps,
        "reference_indices": list(range(2*pairs)), "seed": context.seed,
        "data": study.metadata, "training_performed": False,
        "interpretation": "Endpoints are reconstructions. Intermediate outputs have no target. "
                          "Smooth transitions do not prove disentanglement, novelty or musical inference."
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return {}
