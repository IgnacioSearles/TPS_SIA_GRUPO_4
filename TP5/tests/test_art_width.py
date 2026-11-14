import json

import numpy as np
import pytest

from src.analysis import art_width
from src.data.art_capacity import file_hash, load_probe, read_json
from src.data.art_width import BASE, VARIANTS, SEEDS, prepare_width, configurations
from src.data.factory import build_dataset
from src.evaluation.art_factorial import group_variants
from src.experiments.grid import expand_runs
from src.models.factory import build_model, save_parameters
from src.options import ConfigurationError, Options


def base_config():
    return {'seeds': list(SEEDS),
            'dataset': {'type': 'geometric_art', 'image_size': 16, 'objects': 1,
                        'train_count': 12, 'test_count': 8, 'data_seed': 111},
            'model': {'type': 'variational', 'latent_size': 8, 'hidden_layers': [4, 2],
                      'precision': 'float32', 'loss': 'mse', 'kl': {'weight': .5}},
            'training': {'epochs': 1},
            'evaluation': {'generative': {'count': 4, 'audio_examples': 0, 'test_examples': 0}}}


def setup(tmp_path):
    (tmp_path / 'configs').mkdir()
    (tmp_path / f'configs/{BASE}.json').write_text(json.dumps(base_config()))
    return prepare_width(tmp_path, count=8, wide_layers=(8, 4))


def test_factorial_changes_only_width_and_scheme_and_reserves_new_probes(tmp_path):
    directory = setup(tmp_path)
    configs = configurations(tmp_path, (8, 4))
    for name, config in configs.items():
        assert config['dataset'] == base_config()['dataset']
        assert config['training'] == base_config()['training']
        assert config['model']['latent_size'] == 8
        assert config['model']['hidden_layers'] == ([8, 4] if name.endswith('wide') else [4, 2])
        assert ('kl' in config['model']) == name.startswith('vae_')
    dataset = build_dataset(Options(base_config()['dataset'], 'dataset'), tmp_path)
    parts = [dataset.samples, dataset.generative_study.test_samples]
    parts += [load_probe(directory, split)[0] for split in ('validation', 'reserved_final')]
    seen = set()
    for part in parts:
        hashes = {r.tobytes() for r in part}
        assert not hashes.intersection(seen)
        seen.update(hashes)
    assert prepare_width(tmp_path, count=8, wide_layers=(8, 4)) == directory
    with pytest.raises(ConfigurationError, match='fijado'):
        prepare_width(tmp_path, count=12, wide_layers=(8, 4))


def test_choice_and_paired_contrasts():
    grouped = [{'variant': v, 'macro_silhouette_iou_mean': value} for v, value in
               [('ae_small', .9), ('ae_wide', .905), ('vae_small', .8), ('vae_wide', .9)]]
    assert art_width.choose_by_type(grouped) == {'ae': 'ae_small', 'vae': 'vae_wide'}
    rows = [{**r, 'seed': s, 'macro_silhouette_iou': r['macro_silhouette_iou_mean']}
            for r in grouped for s in SEEDS]
    scores = {r['contrast']: r for r in art_width.contrasts(rows)}
    assert scores['width_vae']['iou_delta_mean'] == pytest.approx(.1)
    assert scores['width_ae']['iou_delta_mean'] == pytest.approx(.005)
    assert scores['ae_minus_vae_small']['iou_delta_mean'] == pytest.approx(.1)


def test_complete_evaluation_is_inference_only_and_fitted_ae_uses_training(tmp_path, monkeypatch):
    setup(tmp_path)
    configs = configurations(tmp_path, (8, 4))
    dataset = build_dataset(Options(base_config()['dataset'], 'dataset'), tmp_path)
    # Pesos pequeños sintéticos para verificar el pipeline, no resultados de calidad.
    for variant, config in configs.items():
        for run in expand_runs(config):
            directory = tmp_path / f'results/{VARIANTS[variant][0]}/{run.label}'
            directory.mkdir(parents=True)
            model = build_model(Options(config['model'], 'model'), 768,
                                np.random.default_rng(run.seed), np.random.default_rng(run.seed))
            save_parameters(model, directory / 'model.npz')
            (directory / 'config.json').write_text(json.dumps(run.config))
            (directory / 'summary.json').write_text('{}')
            (directory / 'history.csv').write_text('epoch\n1\n')
            (directory / 'generative_manifest.json').write_text(json.dumps({
                'model_seed': run.seed, 'parameters_given_to_model': False,
                'train_count': 12, 'test_count': 8, 'data': dataset.generative_study.metadata}))
    models = list((tmp_path / 'results').glob('*/base__seed=*/model.npz'))
    before = {p: file_hash(p) for p in models}
    with pytest.raises(ConfigurationError, match='selection.json'):
        art_width.run_stage(tmp_path, final=True)
    calls = []
    from src.evaluation import art_factorial
    original = art_factorial.sample_fitted
    def checked(model, training, epsilon):
        np.testing.assert_array_equal(training, dataset.samples)
        calls.append(True)
        return original(model, training, epsilon)
    monkeypatch.setattr(art_factorial, 'sample_fitted', checked)
    output = art_width.run_stage(tmp_path)
    assert len(calls) == 6
    assert not (tmp_path / 'results/art_width_final').exists()
    assert (output / 'ae_wide/seed=7/training_gaussian.npz').is_file()
    assert (output / 'ae_wide/seed=7/standard_diagnostic.png').is_file()
    final = art_width.run_stage(tmp_path, final=True)
    assert (final / 'contrasts.csv').is_file()
    assert len(list(final.glob('*/seed=*/reconstruction.csv'))) == 12
    assert before == {p: file_hash(p) for p in models}
    def unexpected(*args, **kwargs):
        pytest.fail('Las evaluaciones completadas deben reutilizarse')
    monkeypatch.setattr(art_width, 'evaluate_factorial', unexpected)
    assert art_width.run_stage(tmp_path) == output
    assert art_width.run_stage(tmp_path, final=True) == final
    assert (art_width.report(tmp_path) / 'report.md').is_file()
