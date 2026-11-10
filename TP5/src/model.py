"""MLP simétrico: tanh en capas ocultas, latente lineal y salida sigmoide."""

import numpy as np


class Autoencoder:
    def __init__(self, hidden=(16, 8), seed=0):
        if not hidden or any(type(size) is not int or size < 1 for size in hidden):
            raise ValueError("hidden debe contener tamaños enteros positivos")
        self.hidden = list(hidden)
        self.sizes = [35, *hidden, 2, *reversed(hidden), 35]
        self.latent_layer = len(hidden) + 1
        rng = np.random.default_rng(seed)
        self.weights = [rng.normal(0, np.sqrt(2 / (a + b)), (a, b))
                        for a, b in zip(self.sizes[:-1], self.sizes[1:])]
        self.biases = [np.zeros(b) for b in self.sizes[1:]]

    def _layer(self, x, index):
        z = x @ self.weights[index] + self.biases[index]
        if index == len(self.weights) - 1:
            # Sigmoide estable sin recortar el valor usado para el gradiente.
            return np.exp(-np.logaddexp(0, -z))
        if index + 1 == self.latent_layer:
            return z
        return np.tanh(z)

    def forward(self, x):
        activations = [np.asarray(x, dtype=float)]
        for i in range(len(self.weights)):
            activations.append(self._layer(activations[-1], i))
        return activations

    def reconstruct(self, x):
        return self.forward(x)[-1]

    def encode(self, x):
        for i in range(self.latent_layer):
            x = self._layer(x, i)
        return x

    def decode(self, z):
        for i in range(self.latent_layer, len(self.weights)):
            z = self._layer(z, i)
        return z

    def loss_and_gradients(self, inputs, targets):
        """BCE media sobre muestras y píxeles, calculada desde logits estables."""
        a = self.forward(inputs)
        logits = a[-2] @ self.weights[-1] + self.biases[-1]
        loss = np.mean(np.logaddexp(0, logits) - targets * logits)
        delta = (a[-1] - targets) / targets.size
        dw, db = [None] * len(self.weights), [None] * len(self.weights)
        for i in reversed(range(len(self.weights))):
            dw[i] = a[i].T @ delta
            db[i] = delta.sum(axis=0)
            if i:
                delta = delta @ self.weights[i].T
                if i != self.latent_layer:
                    delta *= 1 - a[i] ** 2
        return float(loss), [*dw, *db]

    def parameters(self):
        return [*self.weights, *self.biases]

    def save(self, path):
        np.savez_compressed(path, hidden=self.hidden,
                            **{f"p{i}": p for i, p in enumerate(self.parameters())})

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as saved:
            model = cls(saved["hidden"].tolist())
            for i, parameter in enumerate(model.parameters()):
                if saved[f"p{i}"].shape != parameter.shape:
                    raise ValueError("Pesos incompatibles con la arquitectura")
                parameter[:] = saved[f"p{i}"]
        return model
