"""Módulos compartidos: los que valen lo mismo en todos los patrones almacenados.

Sirve para la variante que almacena en la red sólo los módulos que difieren. Los
compartidos no aportan nada para distinguir un patrón de otro, así que se sacan antes
de almacenar o recuperar, y después se reponen con su valor conocido.
"""
import numpy as np


class SharedModules:
    def __init__(self, patterns):
        patterns = np.asarray(patterns)
        self.template_ = patterns[0]
        self.differing = ~(patterns == patterns[0]).all(axis=0)

    @property
    def count(self):
        """Cantidad de módulos que difieren en algún patrón (neuronas de la variante)."""
        return int(self.differing.sum())

    def strip(self, patterns):
        """Deja sólo los módulos que difieren (último eje)."""
        return np.asarray(patterns)[..., self.differing]

    def restore(self, values):
        """Patrón completo: los módulos compartidos con su valor conocido y los que
        difieren con `values`."""
        pattern = self.template_.copy()
        pattern[self.differing] = values
        return pattern
