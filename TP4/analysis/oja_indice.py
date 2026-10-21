"""Índice de Oja: convergencia, comparación con PC1 y uso para campañas.

Uso desde TP4: python3 -m analysis.oja_indice
"""
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.tp4.config import OjaConfig
from src.tp4.oja import Oja
from src.tp4.pca import compute_pca
from src.tp4.preprocessing import Preprocessor
from .eda import TOPICS, cost_per_thousand_argentine_reach, main_topic

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'ejercicio1' / 'oja'

SEEDS = range(5)
# Tasa inicial; la final es 100 veces menor, con el mismo decaimiento exponencial que configs/oja.json.
LEARNING_RATES = [.05, .01, .001, .0001]
REFERENCE = OjaConfig()
CAMPAIGN_COLUMNS = ['followers', 'price_usd', 'cpm_arg', 'argentina_audience_pct',
                    'non_follower_reach_pct', 'content_tutorial_pct', 'saves_per_1000']


@dataclass(frozen=True)
class OjaRun:
    learning_rate: float
    seed: int
    history: pd.DataFrame | None
    weights: np.ndarray | None
    error: str | None = None


def oriented(weights, reference):
    """El signo de Oja es arbitrario: se alinea con PC1 para poder comparar coeficientes y scores."""
    return weights if weights @ reference >= 0 else -weights


def train(x, learning_rate, seed):
    config = replace(REFERENCE, seed=seed, learning_rate_start=learning_rate,
                     learning_rate_end=learning_rate / 100)
    try:
        model = Oja(config).fit(x)
    except FloatingPointError as error:
        return OjaRun(learning_rate, seed, None, None, str(error))
    return OjaRun(learning_rate, seed, pd.DataFrame(model.history_), model.weights_.copy())


def runs_table(runs, pc1):
    rows = []
    for run in runs:
        row = {'learning_rate_start': run.learning_rate, 'seed': run.seed, 'error': run.error}
        if run.weights is not None:
            alignment = run.history.pc1_alignment.to_numpy()
            row.update(
                raw_sign=int(np.sign(run.weights @ pc1)),
                final_alignment=float(abs(run.weights @ pc1)),
                epochs_to_0_999=_first_epoch(alignment, .999),
                epochs_to_0_99999=_first_epoch(alignment, .99999),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _first_epoch(alignment, threshold):
    reached = np.flatnonzero(alignment >= threshold)
    return int(reached[0]) + 1 if len(reached) else None


def creator_index(frame, x, weights):
    scores = (x - x.mean(axis=0)) @ weights
    table = frame[['name']].assign(
        oja_index=scores, main_topic=main_topic(frame), cpm_arg=cost_per_thousand_argentine_reach(frame),
        **{column: frame[column] for column in CAMPAIGN_COLUMNS if column != 'cpm_arg'})
    table['rank'] = table.oja_index.rank(ascending=False).astype(int)
    return table.sort_values('rank')


def index_associations(index):
    """Spearman del índice con las variables de campaña: dice qué ordena y qué no."""
    return pd.Series({column: index.oja_index.corr(index[column], method='spearman')
                      for column in CAMPAIGN_COLUMNS}, name='spearman').to_frame()


def topic_leaders(index):
    """Por tema: quién gana con el índice y quién con el costo por alcance argentino."""
    rows = []
    for topic in TOPICS:
        candidates = index[index.main_topic == topic]
        if candidates.empty:
            continue
        by_index = candidates.nlargest(1, 'oja_index').iloc[0]
        by_cost = candidates.nsmallest(1, 'cpm_arg').iloc[0]
        rows.append({'topic': topic, 'index_leader': by_index['name'], 'index_leader_cpm_arg': by_index.cpm_arg,
                     'index_leader_argentina_pct': by_index.argentina_audience_pct,
                     'cost_leader': by_cost['name'], 'cost_leader_cpm_arg': by_cost.cpm_arg,
                     'cost_leader_argentina_pct': by_cost.argentina_audience_pct,
                     'cost_leader_index_rank': int(by_cost['rank'])})
    return pd.DataFrame(rows)


def coefficients(columns, weights, pca):
    return pd.DataFrame({
        'oja': weights, 'pc1': pca.components[0], 'pc1_correlation': pca.correlation_loadings[:, 0],
    }, index=columns).sort_values('pc1')


def comparison_summary(x, weights, pca):
    centered = x - x.mean(axis=0)
    oja_scores, pc1_scores = centered @ weights, pca.scores[:, 0]
    return {
        'alignment': float(abs(weights @ pca.components[0])),
        'max_coefficient_difference': float(np.abs(weights - pca.components[0]).max()),
        'max_score_difference': float(np.abs(oja_scores - pc1_scores).max()),
        'score_correlation': float(np.corrcoef(oja_scores, pc1_scores)[0, 1]),
        'rank_correlation': float(pd.Series(oja_scores).corr(pd.Series(pc1_scores), method='spearman')),
        'explained_variance_ratio': float(np.var(oja_scores, ddof=1) / np.var(centered, axis=0, ddof=1).sum()),
    }


def build(output=OUT, data=ROOT / 'data' / 'creadores.csv'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(data)
    preprocessor = Preprocessor()
    x = preprocessor.fit_transform(frame)
    pca = compute_pca(x)
    pc1 = pca.components[0]

    runs = [train(x, rate, seed) for rate in LEARNING_RATES for seed in SEEDS]
    table = runs_table(runs, pc1)
    reference = next(run for run in runs if run.learning_rate == REFERENCE.learning_rate_start and run.seed == 0)
    weights = oriented(reference.weights, pc1)
    index = creator_index(frame, x, weights)
    summary = comparison_summary(x, weights, pca)

    table.to_csv(output / 'corridas.csv', index=False)
    coefficients(preprocessor.columns_, weights, pca).to_csv(output / 'coeficientes.csv', float_format='%.6f')
    index.to_csv(output / 'indice.csv', index=False, float_format='%.4g')
    index_associations(index).to_csv(output / 'asociaciones.csv', float_format='%.3f')
    topic_leaders(index).to_csv(output / 'lideres_por_tema.csv', index=False, float_format='%.4g')
    pd.Series(summary).to_json(output / 'comparacion_pc1.json', indent=2)

    from .oja_plots import plot_coefficients, plot_convergence, plot_index, plot_index_campaigns
    plot_convergence(runs, output)
    plot_coefficients(coefficients(preprocessor.columns_, weights, pca), output)
    plot_index(index, output)
    plot_index_campaigns(index, output)
    _print(table, summary, index)
    return index


def _print(table, summary, index):
    with pd.option_context('display.width', 200):
        print(table.to_string(index=False))
        print(pd.Series(summary).to_string())
        print(index.head(5)[['name', 'main_topic', 'oja_index', 'followers', 'cpm_arg']].to_string(index=False))
        print(index.tail(5)[['name', 'main_topic', 'oja_index', 'followers', 'cpm_arg']].to_string(index=False))


if __name__ == '__main__':
    build()
