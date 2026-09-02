"""Pruebas de los fitness visuales basados en bordes y saliencia."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from triangle_image import (
    ChamferEdgeEvaluator,
    CompositeEvaluator,
    GradientOrientationEvaluator,
    MSEEvaluator,
    MixedTriangleGeneMutator,
    NormalizedEvaluator,
    SaliencyMSEEvaluator,
    TriangleContext,
    TriangleColorMutator,
    TriangleGene,
    TriangleImageTarget,
    TriangleIndividual,
    TriangleOrientationMutator,
    TrianglePositionMutator,
    TriangleReplacementMutator,
    TriangleShapeMutator,
)
from triangle_image.mutation_schedule import (
    AdaptiveReheatMutationSchedule,
    ConstantMutationSchedule,
    MutationParameters,
)
from triangle_image.rendering import render


class StructuralFitnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = TriangleContext(seed=0)
        self.white_individual = TriangleIndividual(())

    def test_structural_fitnesses_are_zero_for_an_identical_uniform_image(self) -> None:
        target = TriangleImageTarget(Image.new("RGB", (24, 24), "white"))

        for evaluator in (
            GradientOrientationEvaluator(blur_sigma=0.0),
            ChamferEdgeEvaluator(blur_sigma=0.0),
            SaliencyMSEEvaluator(blur_sigma=0.0),
        ):
            with self.subTest(evaluator=type(evaluator).__name__):
                self.assertAlmostEqual(
                    evaluator.evaluate(self.white_individual, target, self.context).value,
                    0.0,
                )

    def test_structural_fitnesses_penalize_a_missing_target_contour(self) -> None:
        target_image = Image.new("RGB", (24, 24), "white")
        ImageDraw.Draw(target_image).rectangle((0, 0, 11, 23), fill="black")
        target = TriangleImageTarget(target_image)

        for evaluator in (
            GradientOrientationEvaluator(blur_sigma=0.0),
            ChamferEdgeEvaluator(blur_sigma=0.0),
            SaliencyMSEEvaluator(blur_sigma=0.0),
        ):
            with self.subTest(evaluator=type(evaluator).__name__):
                self.assertGreater(
                    evaluator.evaluate(self.white_individual, target, self.context).value,
                    0.0,
                )

    def test_composite_renders_once_and_evicts_the_image_after_evaluation(self) -> None:
        target = TriangleImageTarget(Image.new("RGB", (24, 24), "white"))
        evaluator = CompositeEvaluator([
            (NormalizedEvaluator(MSEEvaluator(), 255.0 ** 2), 0.5),
            (NormalizedEvaluator(SaliencyMSEEvaluator(), 255.0 ** 2), 0.3),
            (NormalizedEvaluator(ChamferEdgeEvaluator(), 1.0), 0.2),
        ])

        with patch("triangle_image.problem.render", wraps=render) as mocked_render:
            evaluator.evaluate(self.white_individual, target, self.context)

        self.assertEqual(mocked_render.call_count, 1)
        self.assertEqual(self.context._render_cache, {})


class TriangleMutationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = TriangleContext(seed=7)
        self.gene = TriangleGene(
            center_x=12.0, center_y=15.0, size=8.0,
            angle_a=50.0, angle_b=60.0, rotation=20.0,
            r=100, g=110, b=120, alpha=0.5,
        )

    def test_specialized_mutators_preserve_unrelated_gene_fields(self) -> None:
        color = TriangleColorMutator(0.5).mutate_gene(self.gene, self.context)
        orientation = TriangleOrientationMutator(0.5).mutate_gene(self.gene, self.context)
        shape = TriangleShapeMutator(40, 30, 0.5).mutate_gene(self.gene, self.context)
        position = TrianglePositionMutator(40, 30, 0.5).mutate_gene(self.gene, self.context)

        self.assertEqual(color.center_x, self.gene.center_x)
        self.assertEqual(color.size, self.gene.size)
        self.assertEqual(color.rotation, self.gene.rotation)
        self.assertEqual(orientation.center_x, self.gene.center_x)
        self.assertEqual(orientation.size, self.gene.size)
        self.assertEqual(orientation.r, self.gene.r)
        self.assertEqual(shape.center_x, self.gene.center_x)
        self.assertEqual(shape.rotation, self.gene.rotation)
        self.assertEqual(shape.r, self.gene.r)
        self.assertEqual(position.size, self.gene.size)
        self.assertEqual(position.rotation, self.gene.rotation)
        self.assertEqual(position.r, self.gene.r)

    def test_replacement_mutation_creates_a_new_valid_triangle(self) -> None:
        replacement = TriangleReplacementMutator(40, 30).mutate_gene(self.gene, self.context)

        self.assertNotEqual(replacement, self.gene)
        self.assertGreater(replacement.size, 0)
        self.assertGreater(replacement.angle_c, 0)
        self.assertGreaterEqual(replacement.center_x, 0)
        self.assertLessEqual(replacement.center_x, 40)
        self.assertGreaterEqual(replacement.center_y, 0)
        self.assertLessEqual(replacement.center_y, 30)

    def test_replacement_probability_can_decay_to_a_local_noop_mutation(self) -> None:
        mutator = MixedTriangleGeneMutator(
            40, 30,
            mutation_strength=0.0,
            replacement_probability=1.0,
            final_replacement_probability=0.0,
            decay_generations=10,
        )

        initial = mutator.mutate_gene(self.gene, self.context)
        self.context.set_generation(10)
        final = mutator.mutate_gene(self.gene, self.context)

        self.assertNotEqual(initial, self.gene)
        self.assertEqual(final, self.gene)

    def test_adaptive_reheat_ignores_improvements_below_delta(self) -> None:
        base = ConstantMutationSchedule(MutationParameters(0.2, 1.0, 0.1))
        schedule = AdaptiveReheatMutationSchedule(
            base, stagnation_generations=2, reheat_generations=3, improvement_delta=0.1
        )

        schedule.observe_best(0, 1.0)
        schedule.observe_best(1, 0.95)  # mejora real, pero menor que delta
        schedule.observe_best(2, 0.94)  # dispara el recalentamiento

        parameters = schedule.parameters_at(2)
        self.assertEqual(parameters.probability, 0.4)
        self.assertAlmostEqual(parameters.replacement_probability, 0.3)
