import json
from pathlib import Path

import numpy as np
import pytest

from src.models.factory import build_model, save_parameters, load_parameters
from src.nn.activations import build_activation
from src.nn.initializers import build_initializer
from src.nn.spatial import Conv3, Upsample2
from src.options import ConfigurationError, Options


@pytest.mark.parametrize('stride', [1, 2])
def test_conv_forward_matches_direct_correlation_and_gradients(stride):
    rng = np.random.default_rng(4)
    layer = Conv3(2, 3, stride, build_activation('tanh'), build_initializer('he'), rng, np.float64)
    x = rng.normal(size=(2, 5, 6, 2))
    actual = layer.forward(x)
    padded = np.pad(x, ((0,0),(1,1),(1,1),(0,0)))
    expected = np.empty_like(actual)
    for n in range(2):
        for y in range(actual.shape[1]):
            for z in range(actual.shape[2]):
                patch = padded[n, y*stride:y*stride+3, z*stride:z*stride+3]
                expected[n,y,z] = np.tanh(np.sum(patch[:,:,:,None]*layer.weights, axis=(0,1,2))+layer.biases)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    upstream = rng.normal(size=actual.shape)
    dx = layer.backward(upstream).copy()
    dw = layer.weight_gradient.copy()
    for array, gradient in ((x, dx), (layer.weights, dw)):
        for _ in range(6):
            index = tuple(rng.integers(v) for v in array.shape)
            old = array[index]
            array[index] = old+1e-6; plus = np.sum(layer.forward(x)*upstream)
            array[index] = old-1e-6; minus = np.sum(layer.forward(x)*upstream)
            array[index] = old
            assert gradient[index] == pytest.approx((plus-minus)/2e-6, abs=1e-6)


def test_resize_gradient_sums_repeated_pixels():
    layer = Upsample2()
    x = np.arange(12.).reshape(1,2,3,2)
    result = layer.forward(x)
    assert result.shape == (1,4,6,2)
    np.testing.assert_array_equal(result[:,::2,::2], x)
    np.testing.assert_array_equal(layer.backward(np.ones_like(result)), np.full_like(x,4))


@pytest.mark.parametrize('model_type', ['autoencoder', 'variational'])
@pytest.mark.parametrize('refinement,refinement_init,depth', [(False,'random',None),(True,'random',1),(True,'identity',1),
                                                           (True,'random',2),(True,'identity',2)])
def test_spatial_model_objective_gradients_and_save_reload(model_type, refinement, refinement_init, depth, tmp_path):
    config = {'architecture':'conv','image_shape':[8,8,3],'conv_channels':[2,3],
              'type':model_type, 'hidden_layers':[4], 'latent_size':2, 'activation':'tanh',
              'initializer':'xavier', 'loss':'mse', 'conv_refinement':refinement,
              'conv_refinement_init':refinement_init}
    if depth is not None:config['conv_refinement_layers']=depth
    def model():
        return build_model(Options(config,'model'), 192, np.random.default_rng(1), np.random.default_rng(2))
    network = model()
    samples = np.random.default_rng(3).random((2,192))
    noise = np.random.default_rng(5).normal(size=(2,2))
    def evaluate():
        if model_type == 'variational':
            terms, gradients = network.objective(samples, samples, noise, .5)
        else:
            terms, gradients = network.loss_and_gradients(samples,samples,1)
        return terms['loss'], gradients
    loss, gradients = evaluate()
    gradients = [g.copy() for g in gradients]
    rng = np.random.default_rng(6)
    for parameter, gradient in zip(network.parameters(), gradients):
        for _ in range(3):
            index = tuple(rng.integers(v) for v in parameter.shape)
            old = parameter[index]
            parameter[index] = old+1e-6; plus,_ = evaluate()
            parameter[index] = old-1e-6; minus,_ = evaluate()
            parameter[index] = old
            assert gradient[index] == pytest.approx((plus-minus)/2e-6, abs=1e-6)
    before = network.reconstruct(samples)
    path = tmp_path/'model.npz'
    save_parameters(network,path)
    loaded = model(); load_parameters(loaded,path)
    np.testing.assert_array_equal(loaded.reconstruct(samples), before)


def test_spatial_float32_and_unchanged_objective_files():
    config = {'architecture':'conv','image_shape':[8,8,3],'conv_channels':[2,3],
              'type':'variational','hidden_layers':[4],'latent_size':2,'precision':'float32','loss':'mse'}
    model = build_model(Options(config,'model'),192,np.random.default_rng(1),np.random.default_rng(2))
    data = np.random.default_rng(3).random((2,192)).astype(np.float32)
    terms, gradients = model.loss_and_gradients(data,data,1)
    assert all(g.dtype == np.float32 for g in gradients)
    assert model.reconstruct(data).dtype == np.float32


def test_mlp_defaults_are_bitwise_unchanged():
    config = {'type':'variational','hidden_layers':[4],'latent_size':2}
    models = [build_model(Options(c,'model'),6,np.random.default_rng(1),np.random.default_rng(2))
              for c in (config, {**config, 'architecture':'mlp'})]
    for a,b in zip(models[0].parameters(),models[1].parameters()):
        np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('depth',[1,2])
def test_identity_refinement_preserves_initial_relu_function(depth):
    config = {'architecture':'conv','image_shape':[8,8,3],'conv_channels':[2,3],
              'type':'variational','hidden_layers':[4],'latent_size':2,'precision':'float32',
              'activation':'relu','loss':'mse'}
    plain = build_model(Options(config,'model'),192,np.random.default_rng(1),np.random.default_rng(2))
    refined = build_model(Options({**config,'conv_refinement':True,'conv_refinement_init':'identity','conv_refinement_layers':depth},'model'),
                          192,np.random.default_rng(1),np.random.default_rng(2))
    data = np.random.default_rng(3).random((2,192)).astype(np.float32)
    z = np.random.default_rng(4).normal(size=(2,2)).astype(np.float32)
    np.testing.assert_array_equal(plain.encode(data),refined.encode(data))
    np.testing.assert_array_equal(plain.decode(z),refined.decode(z))


@pytest.mark.parametrize('extra', [{'architecture':'bad'}, {'architecture':'conv','image_shape':[3,4,3]},
                                  {'architecture':'mlp','conv_channels':[2,3]},
                                  {'architecture':'mlp','conv_refinement':True},
                                  {'architecture':'conv','conv_refinement':1}])
def test_invalid_architecture_is_rejected(extra):
    with pytest.raises(ConfigurationError):
        build_model(Options(extra,'model'),48,np.random.default_rng(1),np.random.default_rng(2))
