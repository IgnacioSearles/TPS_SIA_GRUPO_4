import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.tp4.config import KohonenConfig, OjaConfig, EarlyStopping
from src.tp4.geometry import Grid
from src.tp4.numerics import distances
from src.tp4.kohonen import Kohonen
from src.tp4.oja import Oja
from src.tp4.preprocessing import Preprocessor, PreprocessingConfig
from experiments.runner import expand, run
from experiments.results import ExecutionResult
from src.tp4.results import KohonenMetrics, RunMetrics

class NumericsTests(unittest.TestCase):
    def test_distances(self):
        rng=np.random.default_rng(8)
        x=rng.normal(size=(9,4)); w=np.vstack([x[0],rng.normal(size=(3,4))])
        np.testing.assert_allclose(distances(x,w),np.linalg.norm(x[:,None]-w[None,:],axis=2),atol=1e-7)

    def test_geometry(self):
        for topology, degree in [('rectangular',4),('hexagonal',6)]:
            grid=Grid(5,6,topology)
            self.assertEqual(grid.positions.shape,(30,2))
            np.testing.assert_array_equal(grid.distances,grid.distances.T)
            np.testing.assert_array_equal(grid.adjacency,grid.adjacency.T)
            self.assertEqual(grid.adjacency[2*6+2].sum(),degree)
            self.assertLess(grid.adjacency[0].sum(),degree)
            self.assertTrue((np.diag(grid.distances)==0).all())
            weights=np.arange(30,dtype=float)[:,None]
            expected=np.array([np.abs(weights[i]-weights[grid.adjacency[i]]).mean() for i in range(30)])
            np.testing.assert_allclose(grid.u_matrix(weights).ravel(),expected)
        hexgrid=Grid(3,3,'hexagonal')
        self.assertEqual(set(np.flatnonzero(hexgrid.adjacency[4])),{1,2,3,5,7,8})

class ModelTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(4)
        self.x=np.vstack([rng.normal(-2,.2,(15,2)),rng.normal(2,.2,(15,2))])

    def test_som_reproducible_and_improves(self):
        cfg=KohonenConfig(rows=2,cols=3,epochs=70,radius_end=.1,initialization='uniform')
        a=Kohonen(cfg).fit(self.x); b=Kohonen(cfg).fit(self.x)
        np.testing.assert_array_equal(a.weights_,b.weights_)
        self.assertLess(a.metrics(self.x)['quantization_error'],a.history_[0].quantization_error)
        self.assertEqual(a.occupancy(self.x).sum(),len(self.x))
        self.assertEqual(a.u_matrix().shape,(2,3))
        self.assertTrue(0<=a.metrics(self.x)['topographic_error']<=1)

    def test_vector_update(self):
        cfg=KohonenConfig(rows=1,cols=2,epochs=1,shuffle=False,seed=2,
                           learning_rate_start=.1,decay='constant',radius_start=1,radius_decay='constant')
        x=np.array([[1.,2.],[3.,1.]])
        rng=np.random.default_rng(2)
        expected=x[rng.choice(2,2,replace=False)].copy()
        grid=Grid(1,2,'rectangular')
        for sample in x:
            winner=np.linalg.norm(expected-sample,axis=1).argmin()
            for neuron in range(2):
                h=np.exp(-grid.distances[winner,neuron]**2/2)
                expected[neuron]+=.1*h*(sample-expected[neuron])
        np.testing.assert_allclose(Kohonen(cfg).fit(x).weights_,expected)

    def test_topographic_adjacency(self):
        model=Kohonen(KohonenConfig(rows=2,cols=2))
        model.weights_=np.array([[0.],[10.],[20.],[.1]])
        self.assertEqual(model.metrics(np.array([[.01]]))['topographic_error'],1)
        model.weights_=np.array([[0.],[.1],[20.],[10.]])
        self.assertEqual(model.metrics(np.array([[.01]]))['topographic_error'],0)

    def test_early_stopping(self):
        cfg=KohonenConfig(rows=2,cols=2,epochs=30,
                           early_stopping=EarlyStopping(True,1e6,2,5))
        model=Kohonen(cfg).fit(self.x)
        self.assertEqual(model.epochs_,5)
        self.assertEqual(model.stop_reason_,'early_stopping')
        self.assertEqual(model.selected_epoch_,model.best_epoch_)
        cfg=OjaConfig(epochs=30,early_stopping=EarlyStopping(True,2,2,5))
        self.assertEqual(Oja(cfg).fit(self.x).epochs_,5)

    def test_oja_pc1_and_reproducibility(self):
        rng=np.random.default_rng(3)
        x=rng.normal(size=(200,3))*[3,.3,.1]
        cfg=OjaConfig(epochs=100)
        a=Oja(cfg).fit(x); b=Oja(cfg).fit(x)
        np.testing.assert_array_equal(a.weights_,b.weights_)
        self.assertGreater(a.metrics(x)['pc1_alignment'],.999)
        before=a.metrics(x)['pc1_score_correlation']
        a.weights_ *= -1
        self.assertAlmostEqual(before,a.metrics(x)['pc1_score_correlation'])
        self.assertAlmostEqual(np.linalg.norm(a.weights_),1)

    def test_validation(self):
        with self.assertRaises(ValueError): KohonenConfig(rows=1,cols=1)
        with self.assertRaises(ValueError): OjaConfig(epochs=0)
        with self.assertRaises(ValueError): Kohonen().predict(self.x)
        with self.assertRaises(ValueError): Oja().transform(self.x)
        with self.assertRaises(ValueError): Kohonen().fit([[np.nan]])
        with self.assertRaises(FloatingPointError):
            Oja(OjaConfig(epochs=2,learning_rate_start=1e200,decay='constant')).fit(self.x)

class PreprocessingTests(unittest.TestCase):
    def test_round_trip(self):
        frame=pd.DataFrame({'a':[1.,2.,8.,5.],'b':[4.,1.,2.,8.]})
        for k in (None,1):
            prep=Preprocessor(PreprocessingConfig(exclude=[],log_columns=['a'],pca_components=k))
            z=prep.fit_transform(frame)
            restored=Preprocessor.from_state(json.loads(json.dumps(prep.state())))
            np.testing.assert_allclose(z,restored.transform(frame))
            np.testing.assert_allclose(z.mean(axis=0),0,atol=1e-14)
            np.testing.assert_allclose(prep.transform(frame.iloc[:1]),z[:1])
        with self.assertRaises(ValueError): prep.transform(frame.drop(columns='a'))
        with self.assertRaises(ValueError): Preprocessor().fit(pd.DataFrame({'a':[1,1]}))

class ExperimentTests(unittest.TestCase):
    def test_grid_and_failures(self):
        config={'algorithm':'kohonen','model':{'epochs':1,'rows':2,'cols':2},
                'seeds':[0,1], 'grid':{'model.topology':['rectangular','hexagonal']}}
        self.assertEqual(len(expand(config)),4)
        with tempfile.TemporaryDirectory() as temp:
            config['output']=temp
            self.assertEqual(len(run(config,True,True)),4)
            self.assertEqual(list(Path(temp).iterdir()),[])
            def execute(candidate,directory,control=None):
                if candidate['model']['seed']==1:
                    raise ValueError('fallo simulado')
                return ExecutionResult(candidate, RunMetrics(KohonenMetrics(2., 0.), 1, 30, 'max_epochs', 2))
            with patch('experiments.runner.execute',side_effect=execute):
                output=run(config,True)
            table=pd.read_csv(output/'runs.csv')
            self.assertEqual((table.status=='failed').sum(),2)
            summary=pd.read_csv(output/'summary.csv')
            self.assertTrue((summary.succeeded==1).all())
            self.assertTrue((summary.quantization_error_mean==2).all())

if __name__=='__main__':
    unittest.main()
