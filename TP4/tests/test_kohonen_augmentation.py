"""Invariantes que hacen interpretable el experimento de augmentation."""
import unittest

import numpy as np

from analysis.kohonen_augmentation import AugmentedKohonen, local_neighbors, make_tasks, perturb
from src.tp4.config import KohonenConfig
from src.tp4.kohonen import Kohonen


class AugmentationTests(unittest.TestCase):
    def test_zero_strength_is_exactly_standard_kohonen(self):
        x = np.random.default_rng(3).normal(size=(12, 5))
        neighbors = local_neighbors(x, np.repeat([0, 1, 2], 4))
        cfg = KohonenConfig(rows=3, cols=3, seed=24, epochs=20, schedule_epochs=10)
        ordinary = Kohonen(cfg).fit(x)
        augmented = AugmentedKohonen(cfg, neighbors, 0.).fit(x)
        np.testing.assert_array_equal(ordinary.weights_, augmented.weights_)
        self.assertEqual(ordinary.history_, augmented.history_)
        self.assertEqual(augmented.updates_, 240)

    def test_mixing_keeps_compositions_bounds_and_main_topic(self):
        rng = np.random.default_rng(7)
        topics = np.repeat(np.arange(4), 6)
        raw_topics = rng.dirichlet(np.ones(4), 24)*25
        raw_topics[np.arange(24), topics] += 75
        content = rng.dirichlet(np.ones(4), 24)*100
        raw = np.column_stack((raw_topics, content, rng.uniform(100, 10000, 24)))
        mean, scale = raw.mean(0), raw.std(0, ddof=1)
        z = (raw-mean)/scale
        neighbors = local_neighbors(z, topics)
        for index, options in enumerate(neighbors):
            self.assertNotIn(index, options)
            self.assertTrue(np.all(topics[options] == topics[index]))
        for _ in range(25):
            generated = perturb(z, neighbors, .3, rng)*scale + mean
            np.testing.assert_allclose(generated[:, :4].sum(1), 100, atol=1e-10)
            np.testing.assert_allclose(generated[:, 4:8].sum(1), 100, atol=1e-10)
            np.testing.assert_array_equal(generated[:, :4].argmax(1), topics)
            self.assertTrue(np.all(generated >= raw.min(0)-1e-9))
            self.assertTrue(np.all(generated <= raw.max(0)+1e-9))

    def test_singleton_topic_is_left_intact(self):
        x = np.array([[0., 3.], [1., 2.], [2., 1.]])
        neighbors = local_neighbors(x, np.array(['dogs', 'art', 'art']))
        np.testing.assert_array_equal(neighbors[0], [0])
        np.testing.assert_array_equal(perturb(x, neighbors, .3, np.random.default_rng(0))[0], x[0])

    def test_strength_does_not_change_training_budget(self):
        x = np.random.default_rng(8).normal(size=(16, 4))
        neighbors = local_neighbors(x, np.repeat([0, 1], 8))
        cfg = KohonenConfig(rows=3, cols=3, seed=24, epochs=20)
        model = AugmentedKohonen(cfg, neighbors, .3).fit(x)
        self.assertEqual(model.updates_, 320)
        self.assertEqual(len(model.history_), 20)
        self.assertGreater(model.displacement_sum, 0.)
        metrics = model.evaluate(x)
        self.assertEqual(model.history_[-1].quantization_error, metrics.quantization_error)

    def test_three_seeds_share_each_split(self):
        tasks = make_tasks(20261023, [400, 401, 402])
        self.assertEqual(len(tasks), 60)
        for sample in range(20):
            group = [t for t in tasks if t['sample'] == sample]
            self.assertEqual(len(set(group[0]['train'])), 40)
            self.assertTrue(all(t['train'] == group[0]['train'] for t in group))


if __name__ == '__main__':
    unittest.main()
