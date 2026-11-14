"""Informe legible de capacidad, después de congelar y evaluar la selección."""

import csv
from pathlib import Path

import numpy as np

from src.analysis.art_capacity import PROJECT_ROOT, _snapshot
from src.data.art_capacity import CONFIGS, SEEDS, read_json
from src.options import ConfigurationError


def _csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _table(rows):
    lines = ["| Latente | IoU ↑ | F1 de borde ↑ | Contorno (px) ↓ | MSE en figura ↓ |",
             "|---:|---:|---:|---:|---:|"]
    metrics = ("macro_silhouette_iou", "macro_boundary_f1", "macro_contour_distance_px", "macro_foreground_mse")
    for row in rows:
        values = [f"{float(row[m+'_mean']):.4f} ± {float(row[m+'_std']):.4f}" for m in metrics]
        lines.append(f"| {row['latent']} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def report_capacity(root=PROJECT_ROOT):
    root = Path(root)
    validation, final = (root / f"results/art_capacity_{kind}" for kind in ("validation", "final"))
    selection = read_json(validation / "selection.json")
    final_manifest = read_json(final / "manifest.json")
    _, _, _, _, snapshot = _snapshot(root)
    if selection["inputs"] != snapshot or final_manifest["inputs"] != snapshot:
        raise ConfigurationError("El informe no corresponde a las fuentes congeladas")
    selected = selection["selected_latent"]
    if selected != final_manifest["selected_latent"]:
        raise ConfigurationError("La selección y la evaluación final no coinciden")
    rows = _csv(final / "runs.csv")
    by_run = {(int(r['latent']), int(r['seed'])): r for r in rows}
    differences = [float(by_run[selected, seed]['macro_silhouette_iou'])
                   - float(by_run[4, seed]['macro_silhouette_iou']) for seed in SEEDS]
    output = root / "results/art_capacity_report"
    output.mkdir(parents=True, exist_ok=True)
    text = f'''# Capacidad del VAE RGB

Se entrenaron seis modelos nuevos, con latentes 8 y 16 y semillas 7,17,27.
Los tres modelos de latente 4 se reutilizaron. Se mantuvieron datos de
entrenamiento, capas [128,32], MSE, beta 0,5, Adam y 200 épocas.

## Validación y elección

Se fijaron 512 referencias nuevas, independientes del entrenamiento y de la
prueba anterior. La elección fue **latente {selected}**: el menor tamaño situado
a no más de 0,01 de la mejor IoU macro media de validación.

{_table(_csv(validation / 'summary.csv'))}

La IoU se promedia dentro de cada familia y entre familias; los desvíos son
entre tres semillas. La tolerancia de selección no es un intervalo de confianza.
La elección optimiza reconstrucción geométrica, no calidad generativa global.

## Confirmación final reservada

Después de congelar la elección se evaluaron otras 512 imágenes nuevas. Se
comparó exclusivamente la elección con la base, sin reentrenar ni volver a elegir.

{_table(_csv(final / 'summary.csv'))}

Cambio de IoU frente a la base, emparejado por semilla: **{np.mean(differences):+.4f}
± {np.std(differences, ddof=1):.4f}**. Cambios individuales:
{', '.join(f'{seed}: {d:+.4f}' for seed, d in zip(SEEDS, differences))}.
Con tres semillas estos números son descriptivos y no prueban superioridad universal.

Validación y prueba final contienen imágenes distintas; no comparar sus valores
como si fueran exactamente los mismos casos. Este estudio mantiene las mismas
familias y distribución, no mide generación de clases geométricas ausentes.

## Qué revisar visualmente

Se guardan 64 generaciones por variante y semilla: 576 en total, sin filtrar
artefactos. Las galerías completas y sus vecinos de entrenamiento están dentro
de `../art_capacity_validation/latentN/seed=S/`.

- `reconstruction_seed7.png`, `17` y `27`: las mismas referencias para cada capacidad.
- `generation_seed7.png`, `17` y `27`: índices prefijados 0,8,...,56 de las generaciones.
- `families.png`: resultados separados de círculos, elipses, triángulos y rectángulos.
- `geometry.png`: siluetas, bordes y distancia de contorno.
- CSV por muestra, familia y semilla: incluyen salidas vacías y fallos.

Una mejora de IoU no garantiza esquinas perfectas ni generaciones válidas desde
el prior. Distancia al entrenamiento y diversidad tampoco certifican novedad o
calidad. Las imágenes deben acompañar cualquier conclusión numérica.

La segmentación se calibra con referencias idénticas: todas deben dar IoU/F1=1
y contorno=0. Se usa un umbral de contraste fijo, con tolerancia numérica 1e-6
para antialias float32. Un objeto muy tenue puede perder silueta; no se separan
perfectamente los errores de color y de forma.

## Relación con audio y presentación

Se conserva la implementación matemática del VAE. Adaptar el tamaño del latente
a imágenes no cambia la pregunta compartida con sonidos: reconstruir, generar
y recorrer variaciones de una familia aprendida. Son modelos independientes,
sin pesos ni latente conjunto. No se compara MSE RGB con MSE mel.

La corrida elegida puede explorarse con:

```bash
cd ~/sia/TPS_SIA_GRUPO_4/TP5 && source .venv/bin/activate &&
OPENBLAS_NUM_THREADS=1 python -m src.explorer results/{CONFIGS[selected]}/base__seed=7 --port 8000 --no-browser
```

Sus animaciones están en esa corrida, dentro de `presentation/`. Esas animaciones
usan referencias de la prueba histórica; las métricas de confirmación anteriores
usan la nueva prueba final reservada. No se eligieron pares por su apariencia.

Los hashes de modelos, configuraciones, particiones y evaluador están en
`selection.json` y en el manifiesto final. Una búsqueda posterior de beta, ancho
de red o pérdida debe conservar esta prueba como evaluación y definir otra
validación y una nueva confirmación final, sin retocar esta elección.
'''
    path = output / "report.md"
    path.write_text(text, encoding="utf-8")
    print(f"Informe de capacidad: {path}")
    return path


def main():
    try:
        report_capacity()
    except (ConfigurationError, FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
