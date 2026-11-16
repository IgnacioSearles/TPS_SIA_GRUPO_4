import json
import numpy as np
import pytest
from src.analysis.nsynth_cnn_evaluation import (
    corruption,generation_codes,nearest_distances,mean_pairwise_mse,
    aggregate_note_rows,report,
)


def test_nearest_distances_match_bruteforce_with_partial_batch():
    rng=np.random.default_rng(18)
    reference=rng.random((7,11)).astype(np.float32)
    queries=np.vstack([reference[4],rng.random((4,11))])
    expected=((queries[:,None,:]-reference[None,:,:])**2).mean(axis=2)
    distances,indices=nearest_distances(queries,reference,batch_size=2)
    np.testing.assert_allclose(distances,expected.min(axis=1),atol=1e-14)
    np.testing.assert_array_equal(indices,expected.argmin(axis=1))
    assert indices[0]==4 and distances[0]<1e-14


def test_pairwise_diversity_counts_all_distinct_pairs():
    values=np.array([[0.,0.],[1.,1.],[0.,1.]])
    expected=np.mean([np.mean((values[i]-values[j])**2) for i in range(3) for j in range(i+1,3)])
    np.testing.assert_allclose(mean_pairwise_mse(values),expected,atol=1e-15)
    assert mean_pairwise_mse(np.zeros((4,3)))==0


def test_corruption_and_prior_pairing_independent_of_global_randomness():
    values=np.full((4,8),.5)
    before=values.copy()
    first=corruption(values,7,.1)
    np.random.default_rng(99).random(10000)
    np.testing.assert_array_equal(first,corruption(values,7,.1))
    np.testing.assert_array_equal(before,values)
    assert not np.array_equal(first,corruption(values,17,.1))
    np.testing.assert_array_equal(corruption(values,7,0),values)
    np.testing.assert_array_equal(generation_codes(7,16),generation_codes(7,16))
    assert not np.array_equal(generation_codes(7,16),generation_codes(17,16))


def test_aggregation_does_not_hide_invalid_case():
    rows=[{'condition':'dae_conv','task':'denoising','method':'model','noise_level':.1,'valid':True,'mse':.01},
          {'condition':'dae_conv','task':'denoising','method':'model','noise_level':.1,'valid':False,'mse':float('inf')}]
    result=aggregate_note_rows(rows)
    assert result[0]['count']==2 and result[0]['invalid']==1
    assert result[0]['mse']==float('inf')


def test_report_is_valid_before_any_model_finishes(tmp_path):
    value=report(tmp_path)
    assert value['completed_runs']==0 and value['expected_runs']==12
    assert value['status']=='incomplete'
    assert (tmp_path/'reports/nsynth_cnn_study/informe.md').exists()


@pytest.mark.parametrize('condition',['vae_mlp','dae_conv'])
def test_evaluate_one_run_formats_and_cache(tmp_path,monkeypatch,condition):
    from src.analysis import nsynth_cnn_evaluation as evaluation
    from src.data import nsynth_cnn_study
    from src.data.dataset import Dataset

    class Codec:
        sample_rate=16000
        image_shape=(128,64)
        calls=0
        def decode(self,spec):
            self.calls+=1
            return np.asarray(spec).ravel()[:64]

    class Model:
        precision=np.dtype('float32')
        latent_size=16
        calls=0
        def reconstruct(self,values):
            self.calls+=1
            return values*.9
        def decode(self,codes):
            self.calls+=1
            return np.repeat(np.clip(.5+codes[:,0:1]*.1,0,1),8192,axis=1)

    codec=Codec();model=Model()
    train=Dataset(np.full((2,8192),.4),['train0','train1'],['brass','guitar'],(128,64),False,codec)
    test=Dataset(np.full((2,8192),.6),['test0','test1'],['brass','guitar'],(128,64),False,codec)
    settings={'type':'nsynth','subset':train.labels}
    manifest={'eval_example_labels':test.labels}
    study={'train':train,'test':test,'manifest':manifest,'train_settings':settings}
    monkeypatch.setattr(nsynth_cnn_study,'build_study_data',lambda root:study)
    monkeypatch.setattr(evaluation,'load_run',lambda path:(model,train))
    run=tmp_path/f'results/ej2_nsynth_cnn_{condition}/base__seed=7';run.mkdir(parents=True)
    (run/'summary.json').write_text('{}');(run/'model.npz').write_bytes(b'fixed model fixture')
    model_type='variational' if condition.startswith('vae') else 'autoencoder'
    (run/'config.json').write_text(json.dumps({'dataset':settings,'model':{'type':model_type}}))
    data=tmp_path/'results/nsynth_cnn_study/data';data.mkdir(parents=True)
    np.savez(data/'example_originals.npz',labels=test.labels,waveforms=np.zeros((2,64)))
    result=evaluation.evaluate_run((str(tmp_path),condition,7))
    assert result['status']=='complete' and result['test_count']==2
    calls=(model.calls,codec.calls)
    second=evaluation.evaluate_run((str(tmp_path),condition,7))
    assert second==result
    assert (model.calls,codec.calls)==calls,'Cache should avoid model calls and audio inversions'
    folder=tmp_path/f'reports/nsynth_cnn_study/{condition}/seed=7'
    assert (folder/'per_note.csv').exists()
    assert (folder/'audio/references/test0_original.wav').exists()
    assert (folder/'audio/references/test0_codec.wav').exists()
    if condition.startswith('vae'):
        assert (folder/'generation.csv').exists() and (folder/'generated.npz').exists()
        assert result['generation_count']==64
    else:
        assert not (folder/'generated.npz').exists()
        assert set(result['corrupted_inputs_sha256'])=={'0.0','0.05','0.1','0.2'}
    partial=evaluation.report(tmp_path)
    assert partial['completed_runs']==1 and partial['status']=='incomplete'
    assert (tmp_path/'reports/nsynth_cnn_study/per_seed.csv').exists()
