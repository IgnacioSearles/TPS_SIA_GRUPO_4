"""Persistencia de resultados; delega la visualización al módulo plotting."""
import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from .plotting import plot_grid, plot_learning_curves
from .pca import compute_pca
from .storage import atomic_file


def write_json(path, content):
    with atomic_file(path) as stream:
        json.dump(content, stream, indent=2, allow_nan=False)
        stream.write('\n')


def export_model(output, model, preprocessing, frame, x, algorithm):
    write_json(output / 'preprocessing.json', preprocessing.state())
    pd.DataFrame([asdict(epoch) for epoch in model.history_]).to_csv(output / 'history.csv', index=False)
    identifiers = [column for column in ('creator_id', 'name') if column in frame]
    identity = frame[identifiers].reset_index(drop=True).copy()
    exporters = {'kohonen': export_kohonen, 'oja': export_oja}
    exporters[algorithm](output, model, identity, x)
    plot_learning_curves(output / 'learning_curves.png', model.history_)


def export_kohonen(output, model, identity, samples):
    np.savez_compressed(output / 'model.npz', weights=model.weights_)
    winners = model.predict(samples)
    identity['neuron'] = winners
    identity[['row', 'col']] = model.grid.indices[winners]
    identity.to_csv(output / 'assignments.csv', index=False)
    np.savez_compressed(
        output / 'grid.npz', indices=model.grid.indices, positions=model.grid.positions,
        adjacency=model.grid.adjacency, distances=model.grid.distances,
    )
    occupancy = model.occupancy(samples)
    u_matrix = model.u_matrix()
    np.savetxt(output / 'occupancy.csv', occupancy, delimiter=',', fmt='%d')
    np.savetxt(output / 'u_matrix.csv', u_matrix, delimiter=',')
    plot_grid(output / 'occupancy.png', model.grid, occupancy, 'Ocupación')
    plot_grid(output / 'u_matrix.png', model.grid, u_matrix, 'Matriz U')


def export_oja(output, model, identity, samples):
    np.savez_compressed(output / 'model.npz', weights=model.weights_, mean=model.mean_)
    pca = compute_pca(samples - model.mean_, center=False)
    reference = pca.components[0]
    sign = 1 if model.weights_ @ reference >= 0 else -1
    identity['oja_raw'] = model.transform(samples)
    identity['oja_aligned'] = identity.oja_raw * sign
    identity['pc1'] = (samples - model.mean_) @ reference
    identity.to_csv(output / 'projections.csv', index=False)
    write_json(output / 'comparison.json', {
        'presentation_sign': sign,
        'pc1_weights': reference.tolist(),
    })
