"""Ejemplos pequeños para estudiar el TP4; no genera archivos ni guarda redes.

Desde TP4: python3 presentation/ejemplos_estudio.py
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tp4.config import KohonenConfig
from src.tp4.hopfield import Hopfield
from src.tp4.kohonen import Kohonen
from src.tp4.oja import Oja
from src.tp4.pca import compute_pca
from src.tp4.unlearning import anti_hebbian_update


def pca_example():
    print('\n1. PCA: tres puntos sobre una diagonal')
    x = np.array([[-1., -1.], [0., 0.], [1., 1.]])
    result = compute_pca(x)
    np.testing.assert_allclose(result.explained_variance_ratio[0], 1.)
    print('Primera dirección:', result.components[0])
    print('Scores:', result.scores[:, 0])
    print('Varianza conservada:', result.explained_variance_ratio[0])


def som_example():
    print('\n2. Kohonen: ganadora y vecina se acercan con distinta intensidad')
    model = Kohonen(KohonenConfig(rows=2, cols=2))
    model.weights_ = np.array([[.6, -.5], [0., 0.], [-2., 2.], [-3., 3.]])
    model._update_sample(np.array([1., -1.]), learning_rate=.2, radius=1.)
    np.testing.assert_allclose(model.weights_[0], [.68, -.6])
    np.testing.assert_allclose(model.weights_[1], [.2*np.exp(-.5), -.2*np.exp(-.5)])
    print('Ganadora nueva:', model.weights_[0])
    print('Vecina nueva:', model.weights_[1])


def oja_example():
    print('\n3. Oja: una muestra y la normalización al terminar la época')
    weights = np.array([1., 0.])
    Oja._update_sample(np.array([2., 1.]), weights, .1)
    np.testing.assert_allclose(weights, [1., .2])
    print('Pesos después de la muestra:', weights)
    Oja._normalize(weights)
    print('Pesos normalizados:', weights)


def associative_example():
    print('\n4. Los mismos A, B y C en una red por regla')
    patterns = np.array([[1, 1, 1, 1, -1, -1],
                         [1, 1, 1, -1, 1, -1],
                         [1, 1, -1, 1, 1, -1]], dtype=float)
    query = patterns[0].copy()
    query[0] *= -1
    mixture = np.where(patterns.sum(axis=0) > 0, 1., -1.)
    print('A, B, C:\n', patterns)
    print('Consulta: A con la primera neurona invertida:', query)
    print('Gram:\n', patterns @ patterns.T)
    print('Pseudoinversa de Gram:\n', np.linalg.pinv(patterns @ patterns.T))
    for rule in ('hebb', 'pseudoinverse'):
        model = Hopfield(patterns, rule=rule)
        result = model.recall(query, return_states=True)
        print('\nRegla:', rule, '\nW:\n', model.weights_)
        print('Campo al presentar A perfecto:', model.weights_ @ patterns[0])
        for step, state in enumerate(result.states):
            print(f'  Estado {step}:', state)
        print('Parada:', result.stop_reason, '; pasos con cambio:', result.iterations)
        expected = patterns[0] if rule == 'pseudoinverse' else mixture
        np.testing.assert_array_equal(result.state, expected)
        np.testing.assert_array_equal(result.states[1], patterns[0])
    projector = patterns.T @ np.linalg.pinv(patterns @ patterns.T) @ patterns
    print('\nDiagonal antes de quitarla en pseudoinversa:', projector.diagonal())
    np.testing.assert_allclose(projector @ patterns[0], patterns[0], atol=1e-12)
    print('\n5. Un paso antihebbiano sobre la mezcla')
    model = Hopfield(patterns)
    before = model.energy(mixture)
    anti_hebbian_update(model, mixture, learning_rate=.1)
    increase = model.energy(mixture)-before
    np.testing.assert_allclose(increase, .1*(patterns.shape[1]-1)/2)
    print('Energía antes:', before, '; después:', model.energy(mixture))
    print('Aumento observado y previsto:', increase, '=', .1*(patterns.shape[1]-1)/2)
    print('Esto por sí solo no garantiza eliminar la mezcla ni recuperar un original.')


def zero_field_example():
    print('\n6. Campo cero por cancelación: se conserva el signo anterior')
    model = Hopfield(np.ones((1, 3)))
    # Matriz didáctica simétrica sin diagonal: las dos entradas de la primera fila cancelan.
    model.weights_ = np.array([[0., 1., -1.], [1., 0., 0.], [-1., 0., 0.]])
    for state in (np.ones(3), np.array([-1., 1., 1.])):
        fields = model.weights_ @ state
        following = model._synchronous(state)
        assert fields[0] == 0 and following[0] == state[0]
        print('Estado:', state, '; campos:', fields, '; siguiente:', following)


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    pca_example()
    som_example()
    oja_example()
    associative_example()
    zero_field_example()
    print('\nTodas las comprobaciones numéricas se cumplieron.')
