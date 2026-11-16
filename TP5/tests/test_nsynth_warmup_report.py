import copy
import json
import numpy as np
import pytest

from src.analysis import nsynth_warmup_report as study


def config(seed=7,warmup=50):
    return {'seed':seed,'model':{'type':'variational','architecture':'mlp','latent_size':16,
                               'kl':{'weight':.5,'warmup_epochs':warmup}},
            'dataset':{'type':'nsynth','subset':['train0','train1']},
            'training':{'epochs':200,'optimizer':{'type':'adam','learning_rate':.001}}}


def test_matched_pair_requires_same_seed_data_and_budget():
    zero,fifty=config(warmup=0),config()
    study.validate_matched_pair(zero,fifty)
    for field,value in [('seed',17),('dataset',{'type':'nsynth','subset':['different']}),
                        ('training',{'epochs':100,'optimizer':{'type':'adam','learning_rate':.001}})]:
        changed=copy.deepcopy(zero);changed[field]=value
        with pytest.raises(ValueError):study.validate_matched_pair(changed,fifty)


def test_posterior_statistics_formula_and_strict_activity_boundary(monkeypatch):
    class Posterior:
        precision=np.float32
        def encode_distribution(self,samples):
            return samples,np.zeros_like(samples)
    samples=np.array([[-1.,0.,0.], [1.,0.,0.]],dtype=np.float32)
    values=study.posterior_statistics(Posterior(),samples,batch_size=1)
    assert values['active_dimensions']==1
    assert values['mean_posterior_kl']==.5
    np.testing.assert_array_equal(values['variance_mu_per_dimension'],[1.,0.,0.])
    np.testing.assert_array_equal(values['mean_kl_per_dimension'],[.5,0.,0.])
    monkeypatch.setattr(study,'ACTIVE_THRESHOLD',1.)
    assert study.posterior_statistics(Posterior(),samples)['active_dimensions']==0


def test_report_before_any_model_is_finished(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'assert_loss_unchanged',lambda root:None)
    value=study.report(tmp_path)
    assert value['expected_models']==24
    assert value['trained_models']==0 and value['evaluated_models']==0
    assert value['status']=='incomplete' and len(value['missing'])==24
    assert (tmp_path/'reports/nsynth_warmup/informe.md').exists()


def test_pairing_rejects_unmatched_codes_and_keeps_partial_pairs():
    base={'architecture':'mlp','warmup':50,'seed':7,'data_sha256':'data',
          'generation_codes_sha256':'codes','heldout_mse':.2,'final_kl':1.,'final_reconstruction':2.,
          'final_loss':2.5,'mean_posterior_kl':1.,'active_dimensions':8,
          'generation_train_mse':.1,'generation_test_mse':.2,'generation_pairwise_mse':.3}
    zero={**base,'warmup':0,'heldout_mse':.1}
    configs={('mlp',0,7):config(warmup=0),('mlp',50,7):config()}
    assert study.paired_differences([base],configs)==[]
    pair=study.paired_differences([base,zero],configs)
    np.testing.assert_allclose(pair[0]['heldout_mse'],-.1)
    with pytest.raises(ValueError,match='codes'):
        study.paired_differences([base,{**zero,'generation_codes_sha256':'different'}],configs)
    with pytest.raises(ValueError,match='split'):
        study.paired_differences([base,{**zero,'data_sha256':'different'}],configs)


def test_single_finished_model_without_audio_cache_is_incomplete(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'assert_loss_unchanged',lambda root:None)
    folder=tmp_path/'results/ej2_nsynth_cnn_vae_mlp/base__seed=7';folder.mkdir(parents=True)
    (folder/'summary.json').write_text('{}')
    (folder/'history.csv').write_text('epoch,loss,reconstruction,kl\n200,2.5,2,1\n')
    (folder/'config.json').write_text(json.dumps(config()))
    (folder/'training.json').write_text(json.dumps({'final_kl':1.,'final_reconstruction':2.,'final_loss':2.5}))
    (folder/'model.npz').write_bytes(b'frozen model')
    value=study.report(tmp_path,compute_latents=False)
    assert value['trained_models']==1 and value['evaluated_models']==0
    assert value['rows'][0]['heldout_mse'] is None
    assert value['status']=='incomplete' and value['paired_differences']==[]
    assert (tmp_path/'reports/nsynth_warmup/per_seed.csv').exists()


def test_report_rejects_stale_model_evaluation(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'assert_loss_unchanged',lambda root:None)
    folder=tmp_path/'results/ej2_nsynth_cnn_vae_mlp/base__seed=7';folder.mkdir(parents=True)
    (folder/'summary.json').write_text('{}')
    (folder/'history.csv').write_text('epoch,loss,reconstruction,kl\n200,2.5,2,1\n')
    (folder/'config.json').write_text(json.dumps(config()))
    (folder/'training.json').write_text('{}');(folder/'model.npz').write_bytes(b'new weights')
    evaluated=tmp_path/'reports/nsynth_cnn_study/vae_mlp/seed=7';evaluated.mkdir(parents=True)
    (evaluated/'summary.json').write_text(json.dumps({'model_sha256':'old weights'}))
    with pytest.raises(ValueError,match='Stale'):
        study.report(tmp_path,compute_latents=False)


def test_latent_cache_reuses_statistics_without_loading_decoder(tmp_path,monkeypatch):
    from types import SimpleNamespace
    folder=tmp_path/'model';folder.mkdir()
    (folder/'config.json').write_text(json.dumps(config()))
    (folder/'model.npz').write_bytes(b'fixed weights')
    data=tmp_path/'results/nsynth_cnn_study/data';data.mkdir(parents=True)
    (data/'manifest.json').write_text(json.dumps({'train_labels':['train0','train1'],'test_labels':['heldout']}))
    calls=[]
    class PosteriorOnly:
        precision=np.float32
        def encode_distribution(self,values):
            return values,np.zeros_like(values)
    def load(directory):
        calls.append(directory)
        return PosteriorOnly(),SimpleNamespace(labels=['train0','train1'],samples=np.array([[-1.],[1.]]))
    monkeypatch.setattr(study,'load_run',load)
    weights_sha=study.hash_bytes((folder/'model.npz').read_bytes())
    first=study.latent_statistics(tmp_path,folder,'vae_mlp',7,weights_sha)
    second=study.latent_statistics(tmp_path,folder,'vae_mlp',7,weights_sha)
    assert first==second and len(calls)==1
    assert first['mean_posterior_kl']==.5


def test_new_seed_is_paired_without_reclassifying_original_sample():
    base={'architecture':'mlp','warmup':50,'seed':37,'data_sha256':'data',
          'generation_codes_sha256':'codes','heldout_mse':.2,'final_kl':1.,'final_reconstruction':2.,
          'final_loss':2.5,'mean_posterior_kl':1.,'active_dimensions':8,
          'generation_train_mse':.1,'generation_test_mse':.2,'generation_pairwise_mse':.3}
    zero={**base,'warmup':0,'heldout_mse':.18}
    configs={('mlp',0,37):config(seed=37,warmup=0),('mlp',50,37):config(seed=37)}
    differences=study.paired_differences([base,zero],configs)
    assert len(differences)==1 and differences[0]['seed']==37
    summaries=study.paired_descriptions(differences)
    mse=[r for r in summaries if r['metric']=='heldout_mse']
    assert {r['scope'] for r in mse}=={'extension_three','all_six'}
    assert all(r['available_pairs']==1 and not r['complete_scope'] for r in mse)
    combined=next(r for r in mse if r['scope']=='all_six')
    assert combined['expected_pairs']==6 and combined['negative_differences']==1


def test_descriptive_extension_does_not_claim_six_pairs_when_only_originals_exist():
    differences=[{'architecture':'conv','seed':seed,'heldout_mse':-.001} for seed in study.ORIGINAL_SEEDS]
    summaries=study.paired_descriptions(differences)
    original=next(r for r in summaries if r['scope']=='original_three')
    combined=next(r for r in summaries if r['scope']=='all_six')
    assert original['available_pairs']==3 and original['complete_scope']
    assert combined['available_pairs']==3 and combined['expected_pairs']==6 and not combined['complete_scope']
    assert not any(r['scope']=='extension_three' for r in summaries)
