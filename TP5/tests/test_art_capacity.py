import copy
import json

import numpy as np
import pytest

from src.analysis import art_capacity
from src.data.art import BACKGROUND, render_art
from src.data.art_capacity import (BASE, CONFIGS, expected_configs, file_hash, load_probe,
                                   prepare_capacity, read_json)
from src.data.factory import build_dataset
from src.evaluation.art_geometry import reconstruction_scores, silhouette_scores, summarize_geometry
from src.experiments import runner
from src.options import ConfigurationError, Options


def record(family='rectangle'):
    return {'label': 'reference', 'family': family, 'objects': [
        {'family': family, 'center_x': 8., 'center_y': 8., 'half_width': 4.,
         'half_height': 4., 'angle_radians': 0., 'color_rgb': [.8, .3, .2]}]}


def test_geometry_identity_and_empty_predictions_are_counted():
    reference = record()
    image = render_art(16, reference['objects'])
    scores = reconstruction_scores(image, image, reference)
    assert scores['silhouette_iou'] == 1
    assert scores['boundary_f1'] == 1
    assert scores['contour_distance_px'] == 0
    assert scores['interior_color_mse'] == 0
    background = np.broadcast_to(BACKGROUND, image.shape)
    failed = reconstruction_scores(image, background, reference)
    assert failed['silhouette_iou'] == failed['boundary_f1'] == 0
    assert failed['empty_prediction']
    assert failed['contour_distance_px'] == pytest.approx(np.hypot(16, 16))
    summary, _ = summarize_geometry([{'family': 'rectangle', **scores}, {'family': 'rectangle', **failed}])
    assert summary['macro_silhouette_iou'] == .5 and summary['empty_predictions'] == 1


def test_color_change_can_preserve_geometry_and_rounding_is_penalized():
    reference = record()
    image = render_art(16, reference['objects'])
    changed = copy.deepcopy(reference['objects'])
    direction = np.array(changed[0]['color_rgb']) - BACKGROUND
    changed[0]['color_rgb'] = (direction[[1, 0, 2]] + BACKGROUND).tolist()
    scores = reconstruction_scores(image, render_art(16, changed), reference)
    assert scores['silhouette_iou'] == 1
    assert scores['interior_color_mse'] > 0
    changed[0].update(family='circle', color_rgb=reference['objects'][0]['color_rgb'])
    rounded = reconstruction_scores(image, render_art(16, changed), reference)
    assert rounded['silhouette_iou'] < 1 and rounded['contour_distance_px'] > 0


@pytest.mark.parametrize('family', ['circle', 'ellipse', 'triangle', 'rectangle'])
def test_antialias_roundoff_does_not_penalize_identical_reconstructions(family):
    reference = record(family)
    reference['objects'][0].update(half_width=5.3, half_height=6.7, angle_radians=.31,
                                    color_rgb=[.651271648, .42347386, .3234254])
    image = render_art(16, reference['objects'])
    assert reconstruction_scores(image, image, reference)['silhouette_iou'] == 1


def test_boundary_tolerance_and_invalid_references():
    a = np.zeros((16, 16), bool); a[4:12, 4:12] = True
    b = np.zeros_like(a); b[4:12, 5:13] = True
    scores = silhouette_scores(a, b)
    assert 0 < scores['silhouette_iou'] < 1
    assert scores['boundary_f1'] == 1 and scores['contour_distance_px'] > 0
    with pytest.raises(ValueError):
        silhouette_scores(np.zeros_like(a), b)


def test_selection_prefers_smallest_within_one_percentage_point():
    rows = [{'latent': k, 'macro_silhouette_iou_mean': v} for k, v in [(4, .7), (8, .8), (16, .805)]]
    assert art_capacity.select_latent(rows) == 8
    rows[-1]['macro_silhouette_iou_mean'] = .83
    assert art_capacity.select_latent(rows) == 16
    for row in rows:
        row['macro_silhouette_iou_mean'] = .8
    assert art_capacity.select_latent(rows) == 4


def small_base():
    return {'seeds': [7, 17, 27],
            'dataset': {'type': 'geometric_art', 'objects': 1, 'image_size': 16,
                        'train_count': 12, 'test_count': 8, 'data_seed': 23},
            'model': {'type': 'variational', 'latent_size': 4, 'hidden_layers': [8],
                      'precision': 'float32', 'loss': 'mse', 'kl': {'weight': .5}},
            'training': {'epochs': 2, 'batch_size': 6},
            'evaluation': {'generative': {'count': 4, 'test_examples': 0, 'audio_examples': 0, 'seed': 1234}}}


def setup_base(tmp_path):
    (tmp_path / 'configs').mkdir()
    base = small_base()
    (tmp_path / f'configs/{BASE}.json').write_text(json.dumps(base))
    return base


def test_probes_are_disjoint_fixed_and_never_added_to_training(tmp_path):
    base = setup_base(tmp_path)
    directory = prepare_capacity(tmp_path, count=8)
    dataset = build_dataset(Options(base['dataset'], 'dataset'), tmp_path)
    validation, records = load_probe(directory, 'validation')
    final, _ = load_probe(directory, 'reserved_final')
    partitions = [dataset.samples, dataset.generative_study.test_samples, validation, final]
    seen = set()
    for partition in partitions:
        fingerprints = {row.tobytes() for row in partition}
        assert not seen.intersection(fingerprints)
        seen.update(fingerprints)
    assert {r['family'] for r in records} == {'circle', 'ellipse', 'triangle', 'rectangle'}
    assert len(dataset) == 12
    assert prepare_capacity(tmp_path, count=8) == directory
    with pytest.raises(ConfigurationError, match='fijado'):
        prepare_capacity(tmp_path, count=12)
    config = read_json(tmp_path / f'configs/{CONFIGS[8]}.json')
    config['training']['epochs'] = 3
    (tmp_path / f'configs/{CONFIGS[8]}.json').write_text(json.dumps(config))
    with pytest.raises(ConfigurationError, match='factor adicional'):
        expected_configs(tmp_path)


def test_end_to_end_selection_final_and_reuse_without_weight_changes(tmp_path, monkeypatch):
    setup_base(tmp_path)
    monkeypatch.setattr(runner, 'PROJECT_ROOT', tmp_path)
    prepare_capacity(tmp_path, count=8)
    with pytest.raises(ConfigurationError, match='selection.json'):
        art_capacity.evaluate_final(tmp_path)
    for name in CONFIGS.values():
        runner.run_experiment(tmp_path / f'configs/{name}.json')
    models = list((tmp_path / 'results').glob('*/base__seed=*/model.npz'))
    assert len(models) == 9
    hashes = {p: file_hash(p) for p in models}
    selection = art_capacity.compare_capacity(tmp_path)
    assert selection['selected_latent'] in CONFIGS
    assert not (tmp_path / 'results/art_capacity_final').exists()
    output = art_capacity.evaluate_final(tmp_path)
    assert (output / 'reconstruction_seed7.png').is_file()
    assert read_json(output / 'manifest.json')['selected_latent'] == selection['selected_latent']
    assert hashes == {p: file_hash(p) for p in models}
    def unexpected(*args, **kwargs):
        pytest.fail('La evaluación terminada debe reutilizarse')
    monkeypatch.setattr(art_capacity, '_evaluate', unexpected)
    assert art_capacity.compare_capacity(tmp_path) == selection
    assert art_capacity.evaluate_final(tmp_path) == output
    changed = models[0]
    changed.write_bytes(changed.read_bytes() + b'changed')
    with pytest.raises(ConfigurationError, match='congelada'):
        art_capacity.compare_capacity(tmp_path)
