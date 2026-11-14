from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from src.analysis.nsynth_sampling import validate_pair
from src.options import ConfigurationError


def fixtures():
    ae={'seed':7,'dataset':{'type':'nsynth'},'model':{'type':'autoencoder','latent_size':16},
        'training':{'epochs':200},'evaluation':{}}
    vae=deepcopy(ae);vae['model'].update(type='variational',kl={'weight':.5})
    dataset=SimpleNamespace(samples=np.array([[1.,2.],[3.,4.]]),labels=['a','b'])
    return [ae,vae],[dataset,deepcopy(dataset)]


def test_matched_pair_accepts_only_variational_difference():
    configs,datasets=fixtures()
    validate_pair(configs,datasets)
    configs[1]['training']['epochs']=201
    with pytest.raises(ConfigurationError,match='arquitectura o entrenamiento'):
        validate_pair(configs,datasets)


def test_two_autoencoders_are_not_a_variational_control():
    configs,datasets=fixtures()
    configs[1]['model']['type']='autoencoder'
    with pytest.raises(ConfigurationError,match='AE y VAE'):
        validate_pair(configs,datasets)


@pytest.mark.parametrize('alter',['samples','labels'])
def test_different_training_notes_or_order_are_rejected(alter):
    configs,datasets=fixtures()
    if alter=='samples':datasets[1].samples[0,0]=9
    else:datasets[1].labels.reverse()
    with pytest.raises(ConfigurationError,match='mismas notas'):
        validate_pair(configs,datasets)
