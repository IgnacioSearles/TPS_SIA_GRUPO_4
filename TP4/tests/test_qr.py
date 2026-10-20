from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from src.tp4.qr import load_qr, pattern_from_image, pattern_to_image

QR_DIR = Path(__file__).resolve().parents[1] / 'data' / 'hopfield'
QR_FILES = sorted(QR_DIR.glob('*_qr.png'))


@unittest.skipUnless(len(QR_FILES) == 5, 'Faltan las imágenes locales de data/hopfield')
class RealQrTests(unittest.TestCase):
    def test_each_qr_is_1369_bipolar_modules(self):
        for path in QR_FILES:
            pattern = load_qr(path)
            self.assertEqual(pattern.shape, (1369,))
            self.assertTrue(np.isin(pattern, (-1, 1)).all())

    def test_round_trip_reproduces_the_png(self):
        for path in QR_FILES:
            pattern = load_qr(path)
            image = pattern_to_image(pattern)
            np.testing.assert_array_equal(np.asarray(image), np.asarray(Image.open(path).convert('L')))
            np.testing.assert_array_equal(pattern_from_image(image), pattern)

    def test_black_is_plus_one(self):
        # Patrón localizador de la esquina superior izquierda: borde negro de 7×7,
        # anillo blanco y centro negro de 3×3.
        for path in QR_FILES:
            grid = load_qr(path).reshape(37, 37)
            self.assertTrue((grid[0, :7] == 1).all())
            self.assertTrue((grid[1, 1:6] == -1).all())
            self.assertTrue((grid[2:5, 2:5] == 1).all())


class InvalidImageTests(unittest.TestCase):
    def test_rejects_wrong_size(self):
        with self.assertRaisesRegex(ValueError, '37×37'):
            pattern_from_image(Image.new('L', (36, 37), 0))

    def test_rejects_non_binary_image(self):
        pixels = np.zeros((37, 37), dtype=np.uint8)
        pixels[0, 0] = 128
        with self.assertRaisesRegex(ValueError, 'binaria'):
            pattern_from_image(Image.fromarray(pixels))


if __name__ == '__main__':
    unittest.main()
