"""Ej. 2: daño realista con niveles extremos, para ubicar dónde se rompe cada método.

Con los niveles de `hopfield_qr_dano_realista.py` el desaprendizaje y la pseudoinversa
recuperaron casi todo, así que no se veía el punto de quiebre. Esta corrida usa los
mismos daños, métodos congelados y semillas 200–219 con niveles más altos. Se agregó
después de ver la primera corrida: es una prueba de estrés descriptiva, no una nueva
selección de parámetros. El desgaste en un borde ya llegaba a probabilidad 1 y no se
repite.

Uso desde TP4: python3 analysis/hopfield_qr_dano_realista_extremo.py --output reports/ejercicio2/dano_realista_extremo_20261026
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import hopfield_qr_dano_realista as realistic

EXTREME = {
    'desgaste': (.6, .7, .8, .85, .9, .95),
    'mancha_negra': (.35, .4, .45, .5, .55, .6),
    'mancha_blanca': (.35, .4, .45, .5, .55, .6),
    'rayones': (10, 12, 14, 16, 20, 24),
    'desgaste_y_mancha': (.5, .6, .7, .8, .85, .9),
    'foto': (.65, .7, .75, .8, .85, .9),
}


def main():
    realistic.DAMAGES = {kind: (realistic.DAMAGES[kind][0], levels) for kind, levels in EXTREME.items()}
    realistic.SOURCES = realistic.SOURCES + ('analysis/hopfield_qr_dano_realista_extremo.py',)
    realistic.main()


if __name__ == '__main__':
    main()
