"""Regiones del SOM: agrupa neuronas vecinas con prototipos parecidos y mide su estabilidad entre semillas."""
from itertools import combinations

import numpy as np

from .kohonen_experiments import adjusted_rand_index


def contiguous_ward(weights, adjacency, region_count):
    """Ward aglomerativo sobre los prototipos, uniendo solo regiones vecinas en la grilla.

    La restricción de vecindad hace que cada región sea un área conexa del mapa, legible sobre la matriz U.
    Devuelve una etiqueta por neurona, numeradas en orden de su primera neurona.
    """
    weights, adjacency = np.asarray(weights, dtype=float), np.asarray(adjacency, dtype=bool)
    if adjacency.shape != (len(weights), len(weights)):
        raise ValueError('La adyacencia debe ser neuronas × neuronas')
    if not 1 <= region_count <= len(weights):
        raise ValueError(f'region_count debe estar entre 1 y {len(weights)}')
    regions = [[neuron] for neuron in range(len(weights))]
    while len(regions) > region_count:
        first, second = min(_adjacent_pairs(regions, adjacency), key=lambda pair: _ward_cost(weights, *pair))
        first.extend(second)
        regions.remove(second)
    labels = np.empty(len(weights), dtype=int)
    for label, members in enumerate(sorted(regions, key=min)):
        labels[members] = label
    return labels


def _adjacent_pairs(regions, adjacency):
    pairs = [(first, second) for first, second in combinations(regions, 2)
             if adjacency[np.ix_(first, second)].any()]
    if not pairs:
        raise ValueError('La grilla no es conexa: no se pueden unir más regiones')
    return pairs


def _ward_cost(weights, first, second):
    """Aumento de la suma de cuadrados intra-región al unir dos regiones."""
    size_first, size_second = len(first), len(second)
    gap = weights[first].mean(axis=0) - weights[second].mean(axis=0)
    return size_first * size_second / (size_first + size_second) * float(gap @ gap)


def mean_pairwise_ari(assignments):
    """Similitud media entre las particiones de distintas semillas; 1 = mismas regiones."""
    if len(assignments) < 2:
        raise ValueError('Se requieren al menos dos particiones')
    return float(np.mean([adjusted_rand_index(first, second) for first, second in combinations(assignments, 2)]))


def co_assignment(assignments):
    """Fracción de semillas en que cada par de creadores comparte región."""
    stacked = np.asarray(assignments)
    return (stacked[:, :, None] == stacked[:, None, :]).mean(axis=0)


def membership_firmness(reference, assignments):
    """Por creador: con qué frecuencia sigue junto a sus compañeros de región de la corrida de referencia."""
    together = co_assignment(assignments)
    reference = np.asarray(reference)
    firmness = np.empty(len(reference))
    for creator, region in enumerate(reference):
        mates = np.flatnonzero((reference == region) & (np.arange(len(reference)) != creator))
        firmness[creator] = together[creator, mates].mean() if len(mates) else 1.0
    return firmness
