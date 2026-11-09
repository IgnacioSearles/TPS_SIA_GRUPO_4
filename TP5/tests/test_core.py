import tempfile
import unittest
from pathlib import Path

import numpy as np

from tp5.data import flip_noise, load_font
from tp5.model import Autoencoder
from tp5.training import pixel_errors, train


class CoreTests(unittest.TestCase):
    def test_font_bit_orientation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "font.h"
            path.write_text("const int font[1][7] = { {16, 1, 0, 31, 0, 0, 0} };")
            x, _ = load_font(path, expected_patterns=1)
            np.testing.assert_array_equal(x[0, :5], [1, 0, 0, 0, 0])
            np.testing.assert_array_equal(x[0, 5:10], [0, 0, 0, 0, 1])

    def test_gradient_finite_differences(self):
        model = Autoencoder([3], seed=7)
        x = np.random.default_rng(1).integers(0, 2, (2, 35)).astype(float)
        inputs = 1 - x  # También verificar entrenamiento denoising.
        _, gradients = model.loss_and_gradients(inputs, x)
        for parameter, gradient in zip(model.parameters(), gradients):
            for index in [tuple(0 for _ in parameter.shape),
                          tuple(s - 1 for s in parameter.shape)]:
                original = parameter[index]
                parameter[index] = original + 1e-5
                plus = model.loss_and_gradients(inputs, x)[0]
                parameter[index] = original - 1e-5
                minus = model.loss_and_gradients(inputs, x)[0]
                parameter[index] = original
                self.assertAlmostEqual(gradient[index], (plus - minus) / 2e-5, places=7)

    def test_noise_and_metric(self):
        x = np.eye(35)[:2]
        np.testing.assert_array_equal(flip_noise(x, 0, np.random.default_rng(0)), x)
        np.testing.assert_array_equal(flip_noise(x, 1, np.random.default_rng(0)), 1 - x)
        np.testing.assert_array_equal(pixel_errors(x, x), [0, 0])

    def test_learning_and_persistence(self):
        x = np.zeros((2, 35))
        x[1] = 1
        for optimizer in ["sgd", "adam"]:
            model = Autoencoder([4], seed=2)
            initial = model.loss_and_gradients(x, x)[0]
            train(model, x, epochs=1000, learning_rate=0.5 if optimizer == "sgd" else 0.01,
                  optimizer=optimizer, log_every=1000)
            self.assertLess(model.loss_and_gradients(x, x)[0], initial * 0.8)
        np.testing.assert_allclose(model.decode(model.encode(x)), model.reconstruct(x))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.npz"
            model.save(path)
            np.testing.assert_array_equal(Autoencoder.load(path).reconstruct(x), model.reconstruct(x))


if __name__ == "__main__":
    unittest.main()
