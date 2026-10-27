"""Versiones simplificadas de los gráficos para la presentación: menos rótulos y letra grande.

No calculan resultados nuevos: rehacen con los mismos datos y modelos lo que ya muestran los informes.
Requiere las salidas de `analysis.som_campanas` (usa `som/cantidad_regiones.csv`) y de
`analysis/hopfield_qr_controles.py`, `analysis/hopfield_qr_variante.py` y `analysis/hopfield_qr_dos_redes.py`
(usa sus resúmenes en CSV), más la evidencia publicada del desaprendizaje en reports/ejercicio2/desaprendizaje/.

Uso desde TP4: python3 analysis/presentation_figures.py
Salidas en reports/presentacion/.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from analysis import presentation_catalog_figures as catalog_figures
from analysis import presentation_qr_figures as qr_figures
from analysis.eda import TOPICS
from analysis.hopfield_qr import load_patterns
from analysis.som_campanas import (
    MODEL, catalog_map, criterion_selections, prepare, random_purity_by_count, strategy_selections, train_maps,
    with_regions,
)
from src.tp4.pca import compute_pca
from src.tp4.shared_modules import SharedModules

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'presentacion'
QR_REPORTS = ROOT / 'reports' / 'ejercicio2'
# Vecindades: una ganadora interior y el radio al empezar, en la época 250 y desde la 500 (decae de 2 a 0,5).
NEIGHBORHOOD_WINNER = 5
NEIGHBORHOOD_RADII = {2.0: 'al empezar', 1.0: 'época 250', .5: 'desde la época 500'}
# Letra grande y sin cajas: ni bordes superior y derecho ni marco en las leyendas.
SLIDE_STYLE = {'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 16,
               'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False}


def catalog(out):
    raw = pd.read_csv(ROOT / 'data' / 'creadores.csv')
    x, preprocessor, frame = prepare(raw)
    columns = preprocessor.columns_
    topics = frame.main_topic.to_numpy()
    pca = compute_pca(x)
    soms = train_maps(x)
    som_catalog = catalog_map(soms, x, frame.followers.to_numpy())
    frame = with_regions(frame, som_catalog)
    planes = {column: values for column, values in zip(columns, (som_catalog.som.weights_ * preprocessor.scale_
                                                                 + preprocessor.mean_).T)}
    selections, criteria = strategy_selections(frame), criterion_selections(frame)

    catalog_figures.plot_variance_share(raw, columns, out / 'e1_escala.png')
    catalog_figures.plot_topic_composition(raw, topics, out / 'e1_temas.png')
    catalog_figures.plot_topic_profiles(raw, topics, out / 'e1_perfiles.png')
    catalog_figures.plot_blocks(raw, columns, out / 'e1_bloques.png')
    catalog_figures.plot_biplot(frame, topics, pca, columns, out / 'e1_biplot.png')
    catalog_figures.plot_followers(raw, out / 'e1_seguidores.png')
    catalog_figures.plot_variance(pca, out / 'e1_varianza.png')
    catalog_figures.plot_occupancy(x, out / 'e1_ocupacion.png')
    catalog_figures.plot_neighborhoods(MODEL, NEIGHBORHOOD_WINNER, NEIGHBORHOOD_RADII, out / 'e1_vecindades.png')
    catalog_figures.plot_region_merges(som_catalog, out / 'e1_regiones_union.png')
    catalog_figures.plot_region_count(pd.read_csv(ROOT / 'reports' / 'ejercicio1' / 'som' / 'cantidad_regiones.csv'),
                                      random_purity_by_count(soms, x, topics), out / 'e1_regiones_cantidad.png')
    catalog_figures.plot_planes(som_catalog, planes, catalog_figures.VISIBILITY_PLANES, out / 'e1_planos_eje.png')
    catalog_figures.plot_planes(som_catalog, planes, catalog_figures.TOPIC_PLANES, out / 'e1_planos_temas.png')
    catalog_figures.plot_reach_on_pc1(frame, out / 'e1_masivo_descubrimiento.png')
    catalog_figures.plot_football(frame, selections, criteria, out / 'e1_futbol.png')
    for topic in TOPICS:
        catalog_figures.plot_client_map(som_catalog, topic, out / f'e1_mapa_{topic}.png')
    catalog_figures.plot_decision(frame, selections, som_catalog.region_names, out / 'e1_decision.png')
    catalog_figures.plot_cost(selections, out / 'e1_costo.png')


def qr(out):
    patterns = load_patterns()
    qr_figures.plot_queries_and_outputs(patterns, out / 'e2_consultas_y_salidas.png')
    qr_figures.plot_energy(patterns, out / 'e2_energia.png')
    qr_figures.plot_random_vs_qr(pd.read_csv(QR_REPORTS / 'controles' / 'control1_resumen.csv', index_col='noise'),
                                 out / 'e2_azar_vs_qr.png')
    qr_figures.plot_shared(patterns, out / 'e2_compartidos.png')
    qr_figures.plot_unstable(patterns, out / 'e2_crosstalk.png')
    qr_figures.plot_shared_fraction(
        pd.read_csv(QR_REPORTS / 'controles' / 'control2_resumen.csv', index_col='shared_fraction'),
        1 - SharedModules(patterns).count / patterns.shape[1], len(patterns), out / 'e2_fraccion_compartida.png')
    qr_figures.plot_stable_subsets(pd.read_csv(QR_REPORTS / 'variante' / 'subconjuntos.csv'),
                                   out / 'e2_subconjuntos.png')
    qr_figures.plot_broken_corner(patterns, out / 'e2_esquina.png')
    qr_figures.plot_two_networks(pd.read_csv(QR_REPORTS / 'dos_redes' / 'eleccion_resumen.csv', index_col='level'),
                                 out / 'e2_dos_redes.png')
    qr_figures.plot_overlap_before_after(pd.read_csv(QR_REPORTS / 'variante' / 'solapamientos.csv'),
                                         out / 'e2_solapamiento_antes_despues.png')
    qr_figures.plot_similar_pair(pd.read_csv(QR_REPORTS / 'variante' / 'resultados.csv'),
                                 out / 'e2_par_parecido.png')
    # Las tres reglas sobre las mismas consultas de confirmación; el desaprendizaje sale de la confirmación.
    evidence = QR_REPORTS / 'desaprendizaje' / 'evidencia'
    pseudoinverse = pd.read_csv(evidence / 'comparacion_pseudoinversa_20261022' / 'resumen.csv')
    qr_figures.plot_rules(pd.concat([
        pd.read_csv(evidence / 'confirmacion_sin_compartidos_20261022' / 'comparacion_por_ruido.csv'),
        pseudoinverse[pseudoinverse.method != 'unlearning']]), out / 'e2_reglas.png')
    qr_figures.plot_iterations(
        pd.read_csv(evidence / 'confirmacion_sin_compartidos_20261022' / 'resultados.csv.gz'),
        out / 'e2_iteraciones_desaprendizaje.png')
    qr_published(out)


def qr_published(out):
    """Gráficos del ejercicio 2 que sólo usan los QR y la evidencia publicada en el repositorio, sin los CSV
    regenerables de controles/, variante/ ni dos_redes/."""
    patterns = load_patterns()
    evidence = QR_REPORTS / 'desaprendizaje' / 'evidencia'
    confirmation = evidence / 'confirmacion_sin_compartidos_20261022'
    by_noise = pd.read_csv(confirmation / 'comparacion_por_ruido.csv')
    qr_figures.plot_energy_levels(patterns, out / 'e2_energia_niveles.png')
    qr_figures.plot_shared_annotated(patterns, out / 'e2_compartidos_anotado.png')
    qr_figures.plot_unlearning_training(pd.read_csv(confirmation / 'estabilidad.csv'),
                                        pd.read_csv(confirmation / 'entrenamiento.csv.gz'),
                                        qr_figures.hebb_sink_rejections(patterns),
                                        out / 'e2_desaprendizaje_entrenamiento.png')
    qr_figures.plot_unlearning_result(by_noise, pd.read_csv(confirmation / 'resultados.csv.gz'),
                                      out / 'e2_desaprendizaje_resultado.png')
    qr_figures.plot_rules_by_noise(by_noise, pd.read_csv(evidence / 'comparacion_pseudoinversa_20261022' / 'resumen.csv'),
                                   out / 'e2_reglas_por_ruido.png')
    damage = pd.concat([pd.read_csv(QR_REPORTS / run / 'resumen.csv') for run in
                        ('dano_realista_20261026', 'dano_realista_extremo_20261026', 'dano_realista_40_20261027')])
    qr_figures.plot_realistic_damage(damage, out / 'e2_dano_realista.png')
    qr_figures.plot_realistic_damage(damage, out / 'e2_dano_desgaste_y_mancha.png', kinds=('desgaste', 'desgaste_y_mancha'))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(SLIDE_STYLE):
        catalog(OUT)
        qr(OUT)


if __name__ == '__main__':
    main()
