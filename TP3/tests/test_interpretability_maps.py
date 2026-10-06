import numpy as np

from experiments.digits.interpretability_maps import class_scores, occlusion_map
from nn.network import build_model


def corner_detector():
    """A network whose digit-0 score only reads the top-left 4x4 pixels."""
    net = build_model({"layers": [784, 10], "activation": "identity", "output_activation": "sigmoid"},
                      np.random.default_rng(0))
    weights = np.zeros((28, 28))
    weights[:4, :4] = 1.0
    net.layers[0].W.value[...] = 0.0
    net.layers[0].W.value[:, 0] = weights.reshape(-1)
    net.layers[0].b.value[...] = 0.0
    return net


def test_class_scores_skip_the_final_sigmoid():
    net = corner_detector()
    images = np.ones((2, 784))

    np.testing.assert_allclose(class_scores(net, images)[:, 0], 16.0)


def test_occlusion_map_only_lights_up_where_the_class_score_looks():
    images = np.ones((3, 784))

    importance = occlusion_map(corner_detector(), images, digit=0, window=4, stride=2)

    assert importance[:4, :4].min() > 0
    assert importance[8:, :].max() == 0
    assert importance[:, 8:].max() == 0
