"""Kohonen 4×4 con entradas reducidas por PCA, comparado con el modelo de 27 variables.

Uso desde TP4: python3 -m analysis.som_pca_comparison
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.tp4.preprocessing import PreprocessingConfig, Preprocessor
from .kohonen_experiments import adjusted_rand_index
from .representation_summary import eta_squared, topic_purity
from .som_campanas import (
    REGION_COUNT, ROOT, catalog_map, creator_regions, prepare, strategy_selections, train_maps, with_regions,
)
from .som_regions import mean_pairwise_ari

OUT = ROOT / 'reports' / 'ejercicio1' / 'som_pca'
REFERENCE = '27 variables'
# Componentes: 2 = biplot, 3 = figura 3D, 6 = autovalor > 1, 10 = 90 % de la varianza.
REPRESENTATIONS = {REFERENCE: None, 'PCA 2': 2, 'PCA 3': 3, 'PCA 6': 6, 'PCA 10': 10}
SEPARATION_TARGETS = {
    'eta2_size': lambda frame: np.log(frame.followers), 'eta2_tutorials': lambda frame: frame.content_tutorial_pct,
    'eta2_argentina': lambda frame: frame.argentina_audience_pct, 'eta2_cost': lambda frame: np.log(frame.cpm_arg),
}


def model_inputs(frame, components):
    return Preprocessor(PreprocessingConfig(pca_components=components)).fit_transform(frame)


def aligned_to_reference(regions, reference):
    """Renombra cada región con la región de referencia con la que comparte más creadores."""
    table = pd.crosstab(regions, reference)
    return table.idxmax(axis=1).reindex(regions).to_numpy()


def fit_regions(frame, x):
    soms = train_maps(x)
    return soms, with_regions(frame, catalog_map(soms, x, frame.followers.to_numpy()))


def campaign_picks(regions, reference_regions):
    aligned = regions.assign(region=aligned_to_reference(regions.region.to_numpy(), reference_regions))
    return set(strategy_selections(aligned).query("strategy == 'som'").name)


def evaluate(x, soms, regions, reference):
    metrics = [som.evaluate(x) for som in soms]
    picks = campaign_picks(regions, reference.region.to_numpy())
    return {
        'components': x.shape[1],
        'quantization_error': float(np.mean([metric.quantization_error for metric in metrics])),
        'topographic_error': float(np.mean([metric.topographic_error for metric in metrics])),
        'ari_between_seeds': mean_pairwise_ari([creator_regions(som, x, REGION_COUNT) for som in soms]),
        'ari_with_reference': adjusted_rand_index(regions.region, reference.region),
        'topic_purity': topic_purity(regions.region, regions.main_topic),
        **{name: eta_squared(regions.region, target(regions)) for name, target in SEPARATION_TARGETS.items()},
        'shared_campaign_picks': len(picks & campaign_picks(reference, reference.region.to_numpy())),
        'campaign_picks': len(picks),
    }


def build(output=OUT, data=ROOT / 'data' / 'creadores.csv'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(data)
    _, _, frame = prepare(raw)
    # Las entradas salen del catálogo original: frame ya tiene columnas derivadas (costo, scores) que no son variables.
    inputs = {label: model_inputs(raw, components) for label, components in REPRESENTATIONS.items()}
    fitted = {label: fit_regions(frame, x) for label, x in inputs.items()}
    reference = fitted[REFERENCE][1]
    summary = pd.DataFrame([{'representation': label, **evaluate(inputs[label], soms, regions, reference)}
                            for label, (soms, regions) in fitted.items()])
    summary.to_csv(output / 'resumen.csv', index=False, float_format='%.3f')

    from .som_plots import plot_region_agreement
    plot_region_agreement(reference.region.to_numpy(),
                          {label: regions.region.to_numpy() for label, (_, regions) in fitted.items()
                           if label != REFERENCE}, output)
    with pd.option_context('display.width', 250):
        print(summary.round(3).to_string(index=False))
    return summary


if __name__ == '__main__':
    build()
