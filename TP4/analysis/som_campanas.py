"""Interpretación de Kohonen (perfiles, fronteras de la matriz U) y selección para campañas.

Uso desde TP4: python3 analysis/som_campanas.py
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from analysis.eda import TOPICS, cost_per_thousand_argentine_reach, main_topic
from src.tp4.config import KohonenConfig
from src.tp4.kohonen import Kohonen
from src.tp4.plotting import plot_grid
from src.tp4.preprocessing import PreprocessingConfig, Preprocessor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'ejercicio1'

# Mismos valores que configs/kohonen.json. Cambiar acá al fijar los hiperparámetros finales.
PREPROCESSING = PreprocessingConfig()
MODEL = KohonenConfig(rows=5, cols=5, epochs=500, seed=0)

CAMPAIGN_COLUMNS = ['name', 'neuron', 'followers', 'price_usd', 'cpm_arg', 'non_follower_reach_pct',
                    'content_tutorial_pct', 'saves_per_1000', 'argentina_audience_pct']
# Criterios de selección: (columna, mayor es mejor)
CRITERIA = {
    'followers': ('followers', True),
    'costo_alcance_ar': ('cpm_arg', False),
    'descubrimiento': ('non_follower_reach_pct', True),
    'profundidad': ('content_tutorial_pct', True),  # profundidad educativa: % de publicaciones que explican un procedimiento
}


def campaign_metrics(frame):
    frame = frame.copy()
    frame['cpm_arg'] = cost_per_thousand_argentine_reach(frame)
    frame['main_topic'] = main_topic(frame)
    return frame


def neuron_profiles(frame, columns, som, winners):
    """Por neurona: ocupación, temas, magnitudes y las variables más altas/bajas del prototipo."""
    rows = []
    for neuron, (row, col) in enumerate(som.grid.indices):
        members = frame[winners == neuron]
        prototype = pd.Series(som.weights_[neuron], index=columns).sort_values()
        rows.append({
            'neuron': f'{row},{col}',
            'creators': len(members),
            'topics': members.main_topic.value_counts().to_dict(),
            'followers_median': members.followers.median(),
            'price_usd_median': members.price_usd.median(),
            'high': ', '.join(f'{c} {v:+.1f}' for c, v in prototype.tail(3)[::-1].items()),
            'low': ', '.join(f'{c} {v:+.1f}' for c, v in prototype.head(3).items()),
            'names': ', '.join(members.name),
        })
    return pd.DataFrame(rows)


def frontiers(columns, som, top=5):
    """Pares de neuronas vecinas más distantes y las variables que más los separan."""
    i, j = np.where(np.triu(som.grid.adjacency, 1))
    gaps = np.linalg.norm(som.weights_[i] - som.weights_[j], axis=1)
    rows = []
    for k in np.argsort(-gaps)[:top]:
        diff = pd.Series(som.weights_[j[k]] - som.weights_[i[k]], index=columns)
        diff = diff.reindex(diff.abs().sort_values(ascending=False).index)
        rows.append({
            'from': '{},{}'.format(*som.grid.indices[i[k]]),
            'to': '{},{}'.format(*som.grid.indices[j[k]]),
            'distance': gaps[k],
            'main_differences': ', '.join(f'{c} {v:+.1f}' for c, v in diff.head(4).items()),
        })
    return pd.DataFrame(rows)


def campaigns(frame, top=3):
    rows = []
    for topic in TOPICS:
        candidates = frame[frame.main_topic == topic]  # candidato: creador cuyo tema principal es el de la campaña
        for criterion, (column, higher) in CRITERIA.items():
            chosen = candidates.sort_values(column, ascending=not higher).head(top)
            rows.append(chosen[CAMPAIGN_COLUMNS].assign(topic=topic, criterion=criterion))
    return pd.concat(rows)[['topic', 'criterion', *CAMPAIGN_COLUMNS]]


def run(output=OUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(ROOT / 'data' / 'creadores.csv')
    preprocessor = Preprocessor(PREPROCESSING)
    x = preprocessor.fit_transform(raw)  # antes de agregar las métricas derivadas, que no son variables del modelo
    frame = campaign_metrics(raw)
    som = Kohonen(MODEL).fit(x)
    winners = som.predict(x)
    frame['neuron'] = ['{},{}'.format(*som.grid.indices[w]) for w in winners]
    columns = preprocessor.columns_

    metrics = som.evaluate(x)
    print(f'Kohonen {MODEL.rows}x{MODEL.cols}: QE={metrics.quantization_error:.3f} TE={metrics.topographic_error:.2f}')

    profiles = neuron_profiles(frame, columns, som, winners)
    edges = frontiers(columns, som)
    chosen = campaigns(frame)
    profiles.to_csv(output / 'som_perfiles.csv', index=False)
    edges.to_csv(output / 'som_fronteras.csv', index=False)
    chosen.to_csv(output / 'campanas.csv', index=False)
    plot_grid(output / 'som_matriz_u.png', som.grid, som.u_matrix(), 'Matriz U')
    plot_grid(output / 'som_ocupacion.png', som.grid, som.occupancy(x), 'Creadores por neurona')

    with pd.option_context('display.width', 250, 'display.max_colwidth', 120):
        print('\nPerfiles por neurona:\n', profiles.drop(columns='names').to_string(index=False))
        print('\nFronteras más marcadas de la matriz U:\n', edges.round(2).to_string(index=False))
        summary = chosen.groupby(['topic', 'criterion'], sort=False)[CAMPAIGN_COLUMNS[2:]].mean()
        print('\nPromedio del top 3 por criterio:\n', summary.round(1).to_string())
    print('\nCorrelación de followers con:',
          {c: float(round(frame.followers.corr(frame[c]), 2))
           for c in ('cpm_arg', 'non_follower_reach_pct', 'total_engagement_per_1000', 'content_tutorial_pct', 'saves_per_1000')})


if __name__ == '__main__':
    run()
