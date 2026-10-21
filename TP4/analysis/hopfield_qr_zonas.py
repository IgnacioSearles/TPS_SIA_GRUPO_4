"""Ej. 2, una red de Hopfield por zona del QR.

Cada red guarda los cinco QR recortados a un cuadrado de la imagen. Se mide, para varios tamaños de
zona, en cuántas zonas los cinco QR quedan estables, y con el mismo barrido de ruido que el informe
principal, si la unión de las zonas recupera el QR y si el lector la decodifica.

Uso desde TP4: python3 analysis/hopfield_qr_zonas.py
Salidas (los CSV fuera de git) en reports/ejercicio2/zonas/.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from analysis.hopfield_qr import MAX_ITERATIONS, NOISE_LEVELS, QRS, SEEDS, _save, load_patterns
from analysis.hopfield_qr_reader import Reader
from src.tp4.queries import make_queries
from src.tp4.zoned_hopfield import ZonedHopfield

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'ejercicio2' / 'zonas'
SIDE = 37
ZONE_SIZES = (37, 19, 13, 10, 7, 5)  # 37 = una sola red, el informe principal
SWEEP_ZONE_SIZES = (13, 10, 5)
MAP_ZONE_SIZES = (10, 5)
VARIANTS = {'completa': False, 'sin_compartidos': True}

# Mismos colores que variante/: zona completa en el slot 1, sólo módulos que difieren en el 2.
VARIANT_COLORS = {'completa': '#2a78d6', 'sin_compartidos': '#eb6834'}
VARIANT_LABELS = {'completa': 'zona completa', 'sin_compartidos': 'sólo módulos que difieren en la zona'}
SIZE_COLORS = {13: '#86b6ef', 10: '#3987e5', 5: '#0d366b'}  # secuencial: más oscuro = zonas más chicas
INK, MUTED = '#1f1f1f', '#888888'


def stability_table(patterns):
    """Por tamaño de zona y variante: zonas, zonas iguales en los cinco QR y zonas con los cinco estables."""
    rows = []
    for size in ZONE_SIZES:
        for variant, only_differing in VARIANTS.items():
            zoned = ZonedHopfield(patterns, SIDE, size, only_differing)
            learned = zoned.learned_zones
            stable = [zoned.stable_patterns(zone) for zone in learned]
            rows.append({'zone': size, 'variant': variant, 'zones': len(zoned.zones_),
                         'identical_zones': len(zoned.zones_) - len(learned), 'learned_zones': len(learned),
                         'all_stable_zones': sum(count == len(patterns) for count in stable),
                         'mean_stable_qrs': float(np.mean(stable))})
    return pd.DataFrame(rows)


def plot_stability(table, path):
    sizes = list(ZONE_SIZES)
    x = np.arange(len(sizes))
    width = .38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for offset, variant in zip((-width / 2, width / 2), VARIANTS):
        rows = table[table.variant == variant].set_index('zone').loc[sizes]
        bars = ax.bar(x + offset, rows.all_stable_zones / rows.learned_zones, width - .04,
                      color=VARIANT_COLORS[variant], label=VARIANT_LABELS[variant])
        ax.bar_label(bars, [f'{s}/{n}' for s, n in zip(rows.all_stable_zones, rows.learned_zones)],
                     padding=3, fontsize=8, color=INK)
    identical = table[table.variant == 'completa'].set_index('zone').loc[sizes, 'identical_zones']
    ax.set_xticks(x, [f'{size}×{size}\niguales: {count}' for size, count in zip(sizes, identical)])
    ax.set(xlabel='Lado de la zona (zonas iguales en los cinco QR, sin red)',
           ylabel='Zonas con los cinco QR estables', ylim=(0, 1.1))
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper right', frameon=False, fontsize=9)
    _save(fig, path)


def plot_zone_maps(patterns, path):
    """Sobre el QR del menú principal: en cada zona con red, cuántos de los cinco QR quedan estables."""
    fig, axes = plt.subplots(1, len(MAP_ZONE_SIZES), figsize=(6 * len(MAP_ZONE_SIZES), 6))
    for ax, size in zip(axes, MAP_ZONE_SIZES):
        zoned = ZonedHopfield(patterns, SIDE, size, only_differing=True)
        ax.imshow(patterns[0].reshape(SIDE, SIDE), cmap='Greys', vmin=-3, vmax=3)
        for zone in zoned.zones_:
            height = min(size, SIDE - zone.row)
            width = min(size, SIDE - zone.column)
            corner = (zone.column - .5, zone.row - .5)
            if zone.network is None:
                ax.add_patch(Rectangle(corner, width, height, facecolor='white', alpha=.7, edgecolor=MUTED))
                continue
            stable = zoned.stable_patterns(zone)
            all_stable = stable == len(patterns)
            ax.add_patch(Rectangle(corner, width, height, fill=False, lw=3 if all_stable else 1,
                                   edgecolor=VARIANT_COLORS['sin_compartidos'] if all_stable else INK))
            ax.text(zone.column + width / 2 - .5, zone.row + height / 2 - .5, f'{stable}/5', ha='center',
                    va='center', fontsize=10 if size > 5 else 7, color=INK,
                    bbox={'facecolor': 'white', 'alpha': .85, 'edgecolor': 'none', 'pad': 1})
        ax.set(title=f'Zonas de {size}×{size}', xticks=[], yticks=[])
    fig.suptitle('QR estables por zona (sólo módulos que difieren); en blanco, zonas iguales en los cinco')
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def sweep(patterns, queries, reader):
    """Cada consulta del barrido principal en cada tamaño de zona, con la variante sin compartidos."""
    originals = [reader.read(pattern).text for pattern in patterns]
    rows = []
    for size in SWEEP_ZONE_SIZES:
        zoned = ZonedHopfield(patterns, SIDE, size, only_differing=True)
        for query in queries:
            result = zoned.recall(query.pattern, max_iterations=MAX_ITERATIONS)
            rows.append({'zone': size, 'qr': list(QRS)[query.origin], 'noise': round(float(query.noise), 2),
                         'seed': query.seed, 'correct': result.match == query.origin,
                         'consistent': result.consistent,
                         'zones_on_stored': result.zones_on_stored / len(result.zone_matches),
                         'hamming': int(np.sum(result.state != patterns[query.origin])),
                         'output_decoded': reader.read(result.state).text == originals[query.origin],
                         'query_decoded': reader.read(query.pattern).text == originals[query.origin]})
    return pd.DataFrame(rows)


def plot_sweep(summary, path):
    panels = {'correct': 'QR recuperado exacto', 'zones_on_stored': 'Zonas que terminan en un QR guardado',
              'output_decoded': 'El lector decodifica la salida'}
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 4.5), sharey=True)
    for ax, (column, title) in zip(axes, panels.items()):
        for size in SWEEP_ZONE_SIZES:
            values = summary.loc[size, column]
            ax.plot(values.index, values, marker='o', lw=2, color=SIZE_COLORS[size], label=f'zonas de {size}×{size}')
        ax.set(title=title, xlabel='Nivel de ruido p', ylim=(-.03, 1.03))
        ax.spines[['top', 'right']].set_visible(False)
    reader_alone = summary.loc[SWEEP_ZONE_SIZES[0], 'query_decoded']
    axes[2].plot(reader_alone.index, reader_alone, ls='--', color=MUTED, label='lector solo, sin red')
    axes[0].set_ylabel('Fracción de consultas')
    axes[1].legend(frameon=False)
    axes[2].legend(frameon=False)
    _save(fig, path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    patterns = load_patterns()

    stability = stability_table(patterns)
    stability.to_csv(OUT / 'estabilidad.csv', index=False)
    plot_stability(stability, OUT / '1_estabilidad.png')
    plot_zone_maps(patterns, OUT / '2_mapa_zonas.png')
    print(stability.round(2).to_string(index=False))

    results = sweep(patterns, make_queries(patterns, NOISE_LEVELS, SEEDS), Reader())
    results.to_csv(OUT / 'resultados.csv', index=False)
    summary = results.groupby(['zone', 'noise'])[['correct', 'consistent', 'zones_on_stored', 'hamming',
                                                  'output_decoded', 'query_decoded']].mean()
    summary.to_csv(OUT / 'resumen.csv')
    plot_sweep(summary, OUT / '3_barrido.png')
    print(summary.round(3).to_string())


if __name__ == '__main__':
    main()
