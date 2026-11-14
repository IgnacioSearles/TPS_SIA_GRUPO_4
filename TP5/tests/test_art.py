import base64
import copy
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest

from src.analysis.art_audio import prepare_report, SOURCES
from src.data.art import render_art
from src.data.dataset import Dataset
from src.data.factory import build_dataset
from src.evaluation.context import EvaluationContext
from src.evaluation.presentation import evaluate_presentation, validate_presentation
from src.experiments import runner
from src.explorer.session import ExplorerSession
from src.options import ConfigurationError, Options
from src.plotting import _draw_image


def art_config(objects=1):
    return {"type": "geometric_art", "image_size": 16, "objects": objects,
            "train_count": 12, "test_count": 6, "data_seed": 31}


@pytest.mark.parametrize("objects", [1, 2])
def test_rgb_dataset_is_reproducible_reserved_and_described(tmp_path, objects):
    a = build_dataset(Options(art_config(objects), "dataset"), tmp_path)
    b = build_dataset(Options(art_config(objects), "dataset"), tmp_path)
    assert a.image_shape == (16, 16, 3)
    assert a.samples.shape == (12, 768)
    np.testing.assert_array_equal(a.samples, b.samples)
    study = a.generative_study
    np.testing.assert_array_equal(study.test_samples, b.generative_study.test_samples)
    assert not set(a.labels).intersection(study.test_labels)
    assert not {r.tobytes() for r in a.samples}.intersection(r.tobytes() for r in study.test_samples)
    assert 0 <= a.samples.min() < a.samples.max() <= 1
    assert a.samples.dtype == np.float32
    assert study.metadata['objects'] == objects
    for i, record in enumerate(study.train_records):
        assert len(record['objects']) == objects
        assert record['objects'][0]['family'] == ['circle', 'ellipse', 'triangle', 'rectangle'][i % 4]
        np.testing.assert_array_equal(render_art(16, record['objects']).ravel(), a.samples[i])


def test_rendering_geometry_and_foreground_layer():
    rectangle = {'family': 'rectangle', 'half_width': 4, 'half_height': 4,
                 'center_x': 8, 'center_y': 8, 'angle_radians': 0, 'color_rgb': [1, 0, 0]}
    triangle = {**rectangle, 'family': 'triangle'}
    r, t = (render_art(16, [obj], (0, 0, 0)) for obj in (rectangle, triangle))
    assert 0 < t.sum() < r.sum()
    overlap = render_art(16, [rectangle, {**rectangle, 'color_rgb': [0, 1, 0]}])
    np.testing.assert_array_equal(overlap[8, 8], [0, 1, 0])


@pytest.mark.parametrize("extra", [{"objects": True}, {"objects": 3}, {"image_size": 8},
                                         {"families": ["unknown"]}, {"families": []}])
def test_invalid_art_options_fail(extra, tmp_path):
    with pytest.raises(ConfigurationError):
        build_dataset(Options({**art_config(), **extra}, "dataset"), tmp_path)


def test_plotting_keeps_rgb_channels_and_orientation():
    image = np.zeros((2, 3, 3))
    image[0, 0] = [1, 0, 0]
    fig, ax = plt.subplots()
    try:
        _draw_image(ax, image, False)
        np.testing.assert_array_equal(ax.images[0].get_array(), image)
        assert ax.images[0].origin == 'upper'
    finally:
        plt.close(fig)
    with pytest.raises(ValueError):
        Dataset(np.ones((2, 24)), ['a', 'b'], ['x', 'x'], (2, 3, 4), False)


def smoke_config(kind='art'):
    dataset = art_config()
    if kind == 'audio':
        dataset = {'type': 'fixed_pitch_timbres', 'train_count': 12, 'test_count': 6, 'data_seed': 31,
                   'spectrogram': {'duration_seconds': .5, 'fft_size': 256, 'hop_length': 128,
                                   'mel_bands': 8, 'griffin_lim_iterations': 1, 'mel_inversion_iterations': 2}}
    return {'seeds': [7, 17, 27], 'dataset': dataset,
            'model': {'type': 'variational', 'hidden_layers': [4], 'latent_size': 4,
                      'precision': 'float32', 'loss': 'mse', 'kl': {'weight': .5}},
            'training': {'epochs': 2, 'batch_size': 6},
            'evaluation': {'generative': {'count': 4, 'test_examples': 2, 'audio_examples': 0, 'seed': 1234},
                           'presentation': {'pairs': 1, 'steps': 3}}}


def test_rgb_end_to_end_preserves_weights_and_uses_heldout_mu(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'PROJECT_ROOT', tmp_path)
    config = smoke_config()
    config.pop('seeds')
    config['seed'] = 7
    path = tmp_path / 'rgb.json'
    path.write_text(json.dumps(config))
    directory = runner.run_experiment(path) / 'base__seed=7'
    before = hashlib.sha256((directory / 'model.npz').read_bytes()).hexdigest()
    model, dataset = runner.load_run(directory)
    with np.load(directory / 'presentation/trajectories.npz') as saved:
        codes = model.encode(dataset.generative_study.test_samples[:2])
        np.testing.assert_allclose(saved['coordinates'][0, 0], codes[0])
        np.testing.assert_allclose(saved['coordinates'][0, -1], codes[1], atol=1e-6)
        np.testing.assert_allclose(saved['coordinates'][0, 1], (codes[0] + codes[1])/2, atol=1e-6)
    with Image.open(directory / 'presentation/pair_0.gif') as gif:
        assert gif.size == (256, 286) and gif.n_frames >= 3
    session = ExplorerSession(model, dataset)
    assert session.describe()['image_shape'] == [16, 16, 3]
    decoded = session.decode({'random': True})
    assert len(base64.b64decode(decoded['image'])) == 768
    runner.run_experiment(path, evaluate_only=True)
    assert hashlib.sha256((directory / 'model.npz').read_bytes()).hexdigest() == before
    with pytest.raises(ConfigurationError):
        validate_presentation(dataset, Options({'pairs': 4}, 'presentation'))


def test_art_and_audio_configs_share_learning_protocol():
    root = Path(__file__).resolve().parents[1]
    audio = json.loads((root / 'configs/ej2_new_3_timbres_vae.json').read_text())
    for name in ('ej2_art_1_single', 'ej2_art_2_composition'):
        config = json.loads((root / f'configs/{name}.json').read_text())
        assert config['model'] == audio['model']
        assert config['training'] == audio['training']
        assert config['seeds'] == audio['seeds']
        for key in ('train_count', 'test_count', 'data_seed'):
            assert config['dataset'][key] == audio['dataset'][key]


def test_joint_report_small_audio_and_rgb_reuses_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'PROJECT_ROOT', tmp_path)
    (tmp_path / 'configs').mkdir()
    for kind, modality in [('art_single', 'art'), ('timbres', 'audio')]:
        path = tmp_path / f'configs/{SOURCES[kind]}.json'
        path.write_text(json.dumps(smoke_config(modality)))
        runner.run_experiment(path)
    models = list((tmp_path / 'results').glob('*/base__seed=*/model.npz'))
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in models}
    output = prepare_report(tmp_path)
    assert json.loads((output / 'manifest.json').read_text())['training_performed'] is False
    assert (output / 'art_single/seed=7/presentation/pair_0.gif').is_file()
    assert (output / 'timbres/seed=7/presentation/pair_0/sequence.wav').is_file()
    assert (output / 'summary_by_application.csv').is_file()
    # Una segunda exportación reutiliza los recorridos completos sin volver a decodificar.
    def unexpected(*args, **kwargs):
        pytest.fail('La presentación terminada debe reutilizarse')
    monkeypatch.setattr('src.analysis.art_audio.evaluate_presentation', unexpected)
    prepare_report(tmp_path)
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in models}
