"""Serialización de tablas del EDA y resultados de PCA."""
from dataclasses import asdict
import json

import pandas as pd

from src.tp4.export import write_json
from .eda import ExploratoryTables, QualityReport, TopicTables, VariableBlocks
from .representation import PCAAnalysis


def export_exploration(output, quality: QualityReport, tables: ExploratoryTables):
    write_json(output / 'calidad.json', asdict(quality))
    tables.descriptive.to_csv(output / 'descriptivos.csv')
    tables.outliers.to_csv(output / 'atipicos.csv')
    tables.high_correlations.to_csv(output / 'correlaciones_altas.csv', index=False)


def export_use_case(output, topics: TopicTables, blocks: VariableBlocks, relations: pd.DataFrame):
    topics.medians.to_csv(output / 'temas_medianas.csv')
    topics.specialization.to_csv(output / 'temas_especializacion.csv', index=False)
    blocks.summary().to_csv(output / 'bloques_variables.csv', index=False)
    relations.to_csv(output / 'seguidores_relaciones.csv')


def loadings_table(analysis: PCAAnalysis):
    return pd.DataFrame(
        analysis.pca.correlation_loadings,
        index=analysis.columns, columns=analysis.component_names,
    )


def export_representation(output, frame, analysis: PCAAnalysis):
    result = analysis.pca
    names = analysis.component_names
    pd.DataFrame({
        'componente': names, 'autovalor': result.eigenvalues,
        'varianza': result.explained_variance_ratio, 'acumulada': result.cumulative_variance,
    }).to_csv(output / 'varianza.csv', index=False)
    pd.DataFrame(result.components.T, index=analysis.columns, columns=names).to_csv(output / 'coeficientes.csv')
    loadings_table(analysis).to_csv(output / 'cargas_correlacion.csv')
    scores = pd.DataFrame(result.scores, columns=names)
    pd.concat([frame[['creator_id', 'name']], scores], axis=1).to_csv(output / 'proyecciones.csv', index=False)
    analysis.standardized.to_csv(output / 'variables_estandarizadas.csv', index=False)
    pd.DataFrame({'media': analysis.means, 'desvio_muestral': analysis.scales}).to_csv(output / 'escalado.csv')


def print_summary(quality: QualityReport, comparisons, selected: PCAAnalysis):
    print(json.dumps(asdict(quality), indent=2))
    print(comparisons.to_string(index=False))
    loadings = loadings_table(selected)
    for component in ('PC1', 'PC2', 'PC3', 'PC4'):
        print(component, loadings[component].sort_values().to_string())
