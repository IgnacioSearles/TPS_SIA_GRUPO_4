# Evidencia liviana de los experimentos de desaprendizaje

Este directorio permite auditar los resultados sin entrenar ni descargar una red.
**No contiene pesos, checkpoints, matrices de la red ni imágenes originales.**
Los archivos de métricas ocupan aproximadamente 554 KiB; el manifiesto y este
índice agregan unos pocos KiB.

## Contenido

- [Inventario](inventario.csv): siete ejecuciones, fase, filas almacenadas y nuevas
  evaluaciones. La comparación reutiliza predicciones del candidato; no se cuentan
  otra vez como muestras nuevas.
- [Manifiesto y hashes](manifest.json): tamaño y SHA-256 de cada archivo publicado,
  SHA-256 de su fuente local y formato de almacenamiento.
- `config.json`: parámetros efectivos, versiones y hashes de los patrones. Las
  rutas del equipo de ejecución se normalizaron a rutas relativas al proyecto;
  por eso el hash del JSON publicado puede diferir del de su fuente.
- `resultados.csv.gz`: registros individuales comprimidos, sin estados ni pesos.
- `entrenamiento.csv.gz`: intentos, actualizaciones y motivos de no actualización.
- `estabilidad.csv`, `resumen.csv`, `candidatos.csv` y comparaciones: agregados
  legibles, según la etapa. `resumen_por_qr.csv.gz` conserva el detalle por QR.
- `seleccion.json`: candidato congelado o decisión explícita de no confirmar.
- `COMPLETED`: finalización de la ejecución de origen.

Gzip conserva exactamente los bytes del CSV de origen. Se comprueba al publicar.
La evidencia incluye las dos pruebas cortas como antecedente; no son datos de
confirmación ni se mezclan con ella para elegir parámetros.

## Leer y reproducir sin los datos originales

Desde TP4, con NumPy, pandas y matplotlib disponibles:

```bash
OPENBLAS_NUM_THREADS=1 python3 reports/ejercicio2/desaprendizaje/reproducir_analisis.py --published
```

Regenera el gráfico y calcula los intervalos leyendo estos registros comprimidos.
No carga imágenes, pesos ni una red entrenada. Para inspeccionar filas:

```python
import pandas as pd

table = pd.read_csv(
    'reports/ejercicio2/desaprendizaje/evidencia/'
    'confirmacion_sin_compartidos_20261022/resultados.csv.gz'
)
```

Para volver a entrenar sí se requieren los cinco QR locales de `data/hopfield/`,
que siguen ignorados por Git. Ver [protocolo y comandos](../../../../docs/experimentos_desaprendizaje.md).
El hash de los patrones permite detectar un cambio de datos.

## Actualizar esta copia

Con las ejecuciones locales originales completas:

```bash
OPENBLAS_NUM_THREADS=1 python3 analysis/publish_hopfield_unlearning.py
```

El publicador usa una lista explícita de archivos permitidos, comprime los
registros grandes y rechaza archivos de más de 250000 bytes o un paquete de más
de 1.5 MB. No recorre ni copia indiscriminadamente `results/`.

La interpretación y sus límites están en el [informe completo](../PRUEBAS_COMPLETAS.md)
y la [comparación con pseudoinversa](../PSEUDOINVERSA_VS_DESAPRENDIZAJE.md).
