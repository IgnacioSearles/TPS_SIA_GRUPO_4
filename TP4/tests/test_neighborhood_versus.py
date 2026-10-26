"""Integridad de la comparación anidada y separación de la prueba externa."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from analysis.kohonen_neighborhood_versus import candidates,outer_partitions,choose,calibrate,configured
from src.tp4.preprocessing import Preprocessor


class NestedVersusTests(unittest.TestCase):
    def test_each_creator_is_tested_once_per_repetition_and_inner_sets_exclude_test(self):
        partitions=outer_partitions()
        self.assertEqual(len(partitions),15)
        for repetition in range(3):
            tested=[i for o in partitions if o['repetition']==repetition for i in o['test']]
            self.assertEqual(sorted(tested),list(range(50)))
        for outer in partitions:
            self.assertEqual(len(outer['train']),40)
            self.assertEqual(len(outer['test']),10)
            self.assertFalse(set(outer['train'])&set(outer['test']))
            for inner in outer['inner']:
                self.assertEqual(len(inner['train']),32)
                self.assertEqual(len(inner['validation']),8)
                self.assertFalse(set(inner['train'])&set(inner['validation']))
                self.assertEqual(set(inner['train'])|set(inner['validation']),set(outer['train']))

    def test_equal_candidate_budget_and_same_hyperparameters_between_families(self):
        options=candidates()
        self.assertEqual(len(options),20)
        for family in ['gaussian','uniform']:
            self.assertEqual(sum(c['family']==family for c in options),10)
        first=[c for c in options if c['family']=='gaussian']
        second=[c for c in options if c['family']=='uniform']
        for a,b in zip(first,second):
            ca,cb=a['config'].copy(),b['config'].copy()
            ca.pop('neighborhood');cb.pop('neighborhood')
            self.assertEqual(ca,cb)
            self.assertEqual(configured(a,123).seed,123)

    def test_validation_choice_never_prefers_invalid_low_qe_or_uses_test_scores(self):
        rows=[dict(family='gaussian',candidate_id='a',eligible=True,validation_qe=3.,validation_te=.2,
                   empty_fraction=.1,test_qe=100000.),
              dict(family='gaussian',candidate_id='b',eligible=False,validation_qe=1.,validation_te=.8,
                   empty_fraction=.1,test_qe=0.)]
        selected=choose(rows,'gaussian')
        self.assertEqual(selected['candidate_id'],'a')
        self.assertFalse(selected['fallback'])

    def test_no_eligible_candidate_is_explicit_fallback(self):
        rows=[dict(family='uniform',candidate_id='a',eligible=False,validation_qe=2.,validation_te=.4,empty_fraction=.1),
              dict(family='uniform',candidate_id='b',eligible=False,validation_qe=3.,validation_te=.3,empty_fraction=.1)]
        selected=choose(rows,'uniform')
        self.assertEqual(selected['candidate_id'],'b')
        self.assertTrue(selected['fallback'])
        self.assertEqual(selected['eligible_count'],0)

    def test_calibration_never_even_transforms_outer_test(self):
        outer=outer_partitions()[0]
        frame=pd.DataFrame({'value':np.arange(50,dtype=float)})
        seen=[]
        class Spy(Preprocessor):
            def fit(self,part):
                seen.append(('fit',set(part.index)))
                return super().fit(part)
            def transform(self,part):
                seen.append(('transform',set(part.index)))
                return super().transform(part)
        with patch('analysis.kohonen_neighborhood_versus.pd.read_csv',return_value=frame), \
             patch('analysis.kohonen_neighborhood_versus.Preprocessor',Spy), \
             patch('analysis.kohonen_neighborhood_versus.Kohonen') as model:
            model.return_value.fit.side_effect=RuntimeError('stop before fitting')
            with self.assertRaisesRegex(RuntimeError,'stop before fitting'):
                calibrate(dict(outer=outer,candidate=candidates()[0]),'unused')
            x=model.return_value.fit.call_args.args[0]
        self.assertEqual(x.shape,(32,1))
        np.testing.assert_allclose(x.mean(0),0.,atol=1e-12)
        for operation,indices in seen:
            self.assertFalse(indices&set(outer['test']))
        self.assertEqual(seen[0],('fit',set(outer['inner'][0]['train'])))


if __name__=='__main__':
    unittest.main()
