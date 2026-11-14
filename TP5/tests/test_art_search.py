import csv
import json

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from src.data.art import render_art
from src.evaluation.art_cleanliness import cleanliness,calibrate_cleanliness,clean_pass
from src.analysis.art_search import check_coverage
from src.analysis.art_search import native_references


def original(family):
    obj={'family':family,'half_width':7.,'half_height':8.,'angle_radians':.25,
         'center_x':16.,'center_y':16.,'color_rgb':[.8,.3,.2]}
    return render_art(32,[obj])


def test_cleanliness_accepts_originals_and_rejects_noise_blur_and_blank():
    images=[original(f) for f in ('circle','ellipse','triangle','rectangle')]
    limits=calibrate_cleanliness(images)
    assert all(clean_pass(cleanliness(image),limits) for image in images)
    assert not clean_pass(cleanliness(np.broadcast_to([.055,.075,.13],(32,32,3))),limits)
    assert not clean_pass(cleanliness(np.random.default_rng(1).random((32,32,3))),limits)
    blurred=gaussian_filter(images[-1],sigma=(2.5,2.5,0))
    assert not clean_pass(cleanliness(blurred),limits)


def test_manual_review_requires_distinct_clean_examples_of_all_families(tmp_path):
    coverage={}
    for seed in (7,17,27):
        directory=tmp_path/f'seed={seed}'
        directory.mkdir()
        with (directory/'cleanliness.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=['sample','clean']);writer.writeheader()
            writer.writerows({'sample':i,'clean':True} for i in range(64))
        coverage[str(seed)]={family:list(range(5*j,5*j+5)) for j,family in
                             enumerate(('circle','ellipse','triangle','rectangle'))}
    path=tmp_path/'coverage.json';path.write_text(json.dumps(coverage))
    assert check_coverage(path,tmp_path)==coverage
    coverage['7']['triangle']=coverage['7']['circle']
    path.write_text(json.dumps(coverage))
    with pytest.raises(ValueError):check_coverage(path,tmp_path)


def test_native_resolution_keeps_scene_parameters_and_original_data_untouched():
    record={'label':'x','family':'triangle','objects':[{'family':'triangle','half_width':7.,'half_height':8.,
            'angle_radians':.25,'center_x':16.,'center_y':16.,'color_rgb':[.8,.3,.2]}]}
    image=render_art(32,record['objects'])
    samples,records=native_references(image.reshape(1,-1),[record],64)
    assert samples.shape==(1,64*64*3)
    assert records[0]['objects'][0]['half_width']==14
    assert record['objects'][0]['half_width']==7
    assert records[0]['objects'][0]['color_rgb']==record['objects'][0]['color_rgb']
    limits=calibrate_cleanliness([image])
    assert clean_pass(cleanliness(samples[0].reshape(64,64,3)),limits)


def test_committed_loss_guard_works_without_local_results(tmp_path):
    import hashlib
    from src.data.art_search import assert_loss_unchanged
    from src.options import ConfigurationError
    source=tmp_path/'objective.py'
    source.write_text('original objective')
    guard={'objective.py':hashlib.sha256(source.read_bytes()).hexdigest()}
    committed=tmp_path/'reports/ej2_search/loss_guard.json'
    committed.parent.mkdir(parents=True)
    committed.write_text(json.dumps(guard))
    assert assert_loss_unchanged(tmp_path)==guard
    source.write_text('changed objective')
    with pytest.raises(ConfigurationError,match='Cambió una fórmula protegida'):
        assert_loss_unchanged(tmp_path)


def test_local_loss_guard_cannot_override_committed_baseline(tmp_path):
    from src.data.art_search import assert_loss_unchanged
    from src.options import ConfigurationError
    for relative,guard in [('results/art_search/loss_guard.json',{}),
                           ('reports/ej2_search/loss_guard.json',{'objective.py':'expected'})]:
        path=tmp_path/relative
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(guard))
    with pytest.raises(ConfigurationError,match='difiere de la registrada en Git'):
        assert_loss_unchanged(tmp_path)
