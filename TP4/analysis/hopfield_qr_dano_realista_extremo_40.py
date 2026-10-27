"""Ej. 2: daño realista hasta 40–50 % de módulos alterados en manchas y rayones.

En las dos corridas anteriores el desgaste, el desgaste con mancha y la foto llegaron a
47–50 % de módulos alterados, pero las manchas negras se quedaron en 29 %, las blancas en
31 % y los rayones en 24 %. Esta corrida completa esos tres tipos con niveles más altos,
con los mismos métodos congelados y semillas 200–219. Se agregó para que todos los tipos
lleguen al mismo rango del gráfico: es una prueba de estrés descriptiva, no una nueva
selección de parámetros. El desgaste en un borde no se repite: con probabilidad 1 ya
borra todo su borde y no puede pasar de 26 %.

Uso desde TP4: python3 analysis/hopfield_qr_dano_realista_extremo_40.py --output reports/ejercicio2/dano_realista_40_20261027
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import hopfield_qr_dano_realista as realistic

UP_TO_40 = {
    'mancha_negra': (.7, .75, .8, .85, .9, .95),
    'mancha_blanca': (.7, .75, .8, .85, .9, .95),
    'rayones': (30, 36, 42, 50, 60, 70),
}


def one_row_curves(plot_curves):
    """Con tres tipos, la grilla de 2 × 4 de la corrida completa deja paneles vacíos: una sola fila de
    cuatro (tres tipos y la leyenda)."""
    def plot(summary, path):
        subplots = realistic.plt.subplots
        realistic.plt.subplots = lambda *args, **kwargs: subplots(1, 4, figsize=(16, 4.4), sharey=True, squeeze=False)
        try:
            plot_curves(summary, path)
        finally:
            realistic.plt.subplots = subplots
    return plot


def configure():
    realistic.DAMAGES = {kind: (realistic.DAMAGES[kind][0], levels) for kind, levels in UP_TO_40.items()}
    realistic.SOURCES = realistic.SOURCES + ('analysis/hopfield_qr_dano_realista_extremo_40.py',)
    realistic.plot_curves = one_row_curves(realistic.plot_curves)


def main():
    configure()
    realistic.main()


if __name__ == '__main__':
    main()
