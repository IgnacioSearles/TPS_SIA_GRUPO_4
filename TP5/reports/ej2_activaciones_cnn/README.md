# Mapas de activación del VAE CNN

Visualización posterior al cierre del estudio; no se entrenó ni cambió el modelo.
Referencias fijas 0–3 de validación, y triángulo índice 2 para los gráficos.
El gráfico principal conserva los primeros ocho canales de cada capa; los
gráficos completos incluyen los 16 y 32 canales, incluso mapas oscuros.

Se usa una escala común por capa sobre las cuatro referencias. Un tono claro
significa activación mayor dentro de esa escala; no se comparan intensidades
entre capas. Son salidas posteriores a ReLU, no filtros, atención ni máscaras
de figuras. El refinamiento está en el decoder y se documenta por separado.

Las matrices completas, selección, escalas y hashes quedan en NPZ/manifest.
Los pesos originales conservan su hash. No se usan referencias finales.
