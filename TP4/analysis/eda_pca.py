"""Entrada del EDA: python3 analysis/eda_pca.py o python3 -m analysis.eda_pca."""
from dataclasses import dataclass
from pathlib import Path
import sys

# Permite la ejecución directa desde la raíz del proyecto.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from analysis.eda import (
    FOLLOWER_RELATION_TARGETS, TOPIC_PROFILE_COLUMNS, ExploratoryTables, QualityReport, TopicTables,
    VariableBlocks, exploratory_tables, follower_relations, quality_report, topic_tables, validate_catalog,
    variable_blocks,
)
from analysis.plots import (
    plot_biplot, plot_biplot_pc3, plot_correlations, plot_pca_3d, plot_distributions, plot_follower_relations, plot_scale_comparison,
    plot_topic_composition, plot_topic_profiles, plot_variable_blocks, plot_variance,
)
from analysis.reporting import export_exploration, export_representation, export_use_case, print_summary
from analysis.representation import PCAAnalysis, analyze_pca, analyze_variants
from src.tp4.preprocessing import Preprocessor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'ejercicio1'


@dataclass(frozen=True)
class CatalogAnalysis:
    quality: QualityReport
    exploration: ExploratoryTables
    variants: list[PCAAnalysis]
    topics: TopicTables
    blocks: VariableBlocks
    relations: pd.DataFrame

    @property
    def selected(self) -> PCAAnalysis:
        return next(variant for variant in self.variants if variant.name == 'seleccion_z')

    def comparisons(self) -> pd.DataFrame:
        return pd.DataFrame([variant.summary() for variant in self.variants])


def pca(frame):
    """Adaptador de la función original; delega en la implementación compartida."""
    analysis = analyze_pca(frame, 'custom')
    result = analysis.pca
    return analysis.standardized, result.scores, result.components.T, result.eigenvalues


def analyze_catalog(frame: pd.DataFrame) -> CatalogAnalysis:
    numeric = validate_catalog(frame)
    return CatalogAnalysis(
        quality=quality_report(frame, numeric),
        exploration=exploratory_tables(numeric),
        variants=analyze_variants(frame, numeric),
        topics=topic_tables(frame),
        blocks=variable_blocks(numeric[_selected_columns(frame)]),
        relations=follower_relations(frame, FOLLOWER_RELATION_TARGETS),
    )


def _selected_columns(frame):
    return list(Preprocessor().fit(frame).columns_)


def export_analysis(output, frame, analysis: CatalogAnalysis):
    export_exploration(output, analysis.quality, analysis.exploration)
    export_representation(output, frame, analysis.selected)
    analysis.comparisons().to_csv(output / 'comparacion_pca.csv', index=False)
    plot_variance(analysis.selected, output)
    plot_biplot(frame, analysis.selected, output)
    plot_biplot_pc3(frame, analysis.selected, output)
    plot_pca_3d(frame, analysis.selected, output)
    plot_correlations(analysis.exploration.correlations, output)
    plot_distributions(frame.select_dtypes(include='number'), output)
    export_use_case(output, analysis.topics, analysis.blocks, analysis.relations)
    selected = analysis.selected
    plot_scale_comparison(frame[selected.columns], selected.standardized, output)
    plot_variable_blocks(analysis.blocks, output)
    plot_topic_composition(frame, analysis.topics, output)
    plot_topic_profiles(frame, TOPIC_PROFILE_COLUMNS, output)
    plot_follower_relations(frame, analysis.relations, output)


def run(output=OUT) -> CatalogAnalysis:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ROOT / 'data' / 'creadores.csv')
    analysis = analyze_catalog(frame)
    export_analysis(output, frame, analysis)
    print_summary(analysis.quality, analysis.comparisons(), analysis.selected)
    return analysis


if __name__ == '__main__':
    run()
