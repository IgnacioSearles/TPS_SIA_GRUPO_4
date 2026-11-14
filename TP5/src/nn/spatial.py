"""Capas espaciales NumPy con backprop manual y la interfaz de las redes densas.

Solo cambian encoder/decoder. La reparametrización y las pérdidas siguen en los
modelos existentes. El decoder usa resize + convolución, no un renderizador.
"""

import numpy as np

from src.nn.network import DenseLayer
from src.options import ConfigurationError


class Reshape:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def forward(self, inputs):
        self.input_shape = inputs.shape
        return inputs.reshape((len(inputs), *self.shape))

    def backward(self, gradient):
        return gradient.reshape(self.input_shape)


class Upsample2:
    def forward(self, inputs):
        self.input_shape = inputs.shape
        return np.repeat(np.repeat(inputs, 2, axis=1), 2, axis=2)

    def backward(self, gradient):
        n, h, w, c = self.input_shape
        return gradient.reshape(n, h, 2, w, 2, c).sum(axis=(2, 4))


class Conv3:
    """Correlación 3×3, padding de un píxel, NHWC y stride 1 o 2."""

    def __init__(self, inputs, outputs, stride, activation, initializer, rng, precision):
        if stride not in (1, 2):
            raise ValueError("El stride debe ser 1 o 2")
        self.stride, self.activation = stride, activation
        self.weights = initializer(rng, 9*inputs, outputs).reshape(3, 3, inputs, outputs).astype(precision)
        self.biases = np.zeros(outputs, dtype=precision)
        self.weight_gradient = np.zeros_like(self.weights)
        self.bias_gradient = np.zeros_like(self.biases)

    def forward(self, inputs):
        self.input_shape = inputs.shape
        padded = np.pad(inputs, ((0, 0), (1, 1), (1, 1), (0, 0)))
        windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(1, 2))
        patches = windows[:, ::self.stride, ::self.stride].transpose(0, 1, 2, 4, 5, 3)
        n, h, w = patches.shape[:3]
        self.columns = np.ascontiguousarray(patches).reshape(n*h*w, -1)
        self.preactivation = (self.columns @ self.weights.reshape(-1, self.weights.shape[-1])).reshape(n, h, w, -1) + self.biases
        self.outputs = self.activation.apply(self.preactivation)
        return self.outputs

    def backward(self, gradient):
        delta = gradient * self.activation.derivative(self.preactivation, self.outputs)
        n, h, w, outputs = delta.shape
        flat = delta.reshape(-1, outputs)
        self.weight_gradient = (self.columns.T @ flat).reshape(self.weights.shape)
        self.bias_gradient = flat.sum(axis=0)
        columns = (flat @ self.weights.reshape(-1, outputs).T).reshape(n, h, w, 3, 3, -1)
        _, ih, iw, channels = self.input_shape
        padded = np.zeros((n, ih+2, iw+2, channels), dtype=gradient.dtype)
        for y in range(3):
            for x in range(3):
                padded[:, y:y+self.stride*h:self.stride, x:x+self.stride*w:self.stride] += columns[:, :, :, y, x]
        return padded[:, 1:ih+1, 1:iw+1]


class SpatialNetwork:
    """Secuencia espacial/densa compatible con forward/backward y persistencia."""

    def __init__(self, layers, input_size, output_size):
        self.layers = layers
        self.layer_sizes = [input_size, output_size]

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, gradient):
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def parameters(self):
        return [p for layer in self.layers if hasattr(layer, "weights") for p in (layer.weights, layer.biases)]

    def gradients(self):
        return [g for layer in self.layers if hasattr(layer, "weights") for g in (layer.weight_gradient, layer.bias_gradient)]


def spatial_networks(shape, channels, hidden, decoder_hidden, latent, outputs,
                     activation, latent_activation, identity, initializer, rng, precision,
                     refinement=False, refinement_init='random', refinement_layers=1):
    if (not isinstance(shape, list) or len(shape) != 3
            or any(type(v) is not int or v < 1 for v in shape)
            or shape[0] % 4 or shape[1] % 4
            or not isinstance(channels, list) or len(channels) != 2
            or any(type(v) is not int or v < 1 for v in channels)):
        raise ConfigurationError("conv requiere image_shape=[alto,ancho,canales], lados múltiplos de 4 y dos conv_channels positivos")
    h, w, c = shape
    a, b = channels
    features = (h//4)*(w//4)*b
    encoder = [Reshape(shape), Conv3(c, a, 2, activation, initializer, rng, precision),
               Conv3(a, b, 2, activation, initializer, rng, precision), Reshape((-1,))]
    sizes = [features, *hidden, outputs]
    encoder += [DenseLayer(initializer(rng, i, o).astype(precision), latent_activation if j == len(sizes)-2 else activation)
                for j, (i, o) in enumerate(zip(sizes[:-1], sizes[1:]))]
    decoder = []
    sizes = [latent, *decoder_hidden, features]
    decoder += [DenseLayer(initializer(rng, i, o).astype(precision), activation) for i, o in zip(sizes[:-1], sizes[1:])]
    decoder += [Reshape((h//4, w//4, b)), Upsample2(), Conv3(b, a, 1, activation, initializer, rng, precision),
                Upsample2()]
    # En la variante identidad, sortear primero los logits preserva también su
    # inicialización original. La capa adicional puede aprender desde el inicio.
    final = Conv3(a, c, 1, identity, initializer, rng, precision) if refinement_init == 'identity' else None
    if refinement:
        # Aprende bordes/color a resolución completa antes de los logits RGB.
        for _ in range(refinement_layers):
            refine = Conv3(a, a, 1, activation, initializer, rng, precision)
            if refinement_init == 'identity':
                refine.weights.fill(0)
                refine.weights[1,1,np.arange(a),np.arange(a)] = 1
            decoder.append(refine)
    decoder += [final if final is not None else Conv3(a, c, 1, identity, initializer, rng, precision), Reshape((-1,))]
    return SpatialNetwork(encoder, h*w*c, outputs), SpatialNetwork(decoder, latent, h*w*c)
