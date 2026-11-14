"""Resumen reproducible de candidatos terminados; no abre la prueba reservada."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.data.art_capacity import read_json, file_hash
from src.data.art_search import assert_loss_unchanged

ROOT = Path(__file__).resolve().parents[2]


def phase_epochs(root, config, seed, visited=()):
    count=config['training']['epochs']
    specification=config.get('initial_parameters')
    if specification is None:return [count]
    path=specification['path'] if isinstance(specification,dict) else specification
    source=(root/path.replace('{seed}',str(seed))).parent/'config.json'
    if not source.exists():return [None,count]
    if source in visited:raise ValueError('Ciclo en la procedencia de los pesos')
    return phase_epochs(root,read_json(source),seed,(*visited,source))+[count]


def refinement_ablation(root, output):
    names=('cnn4k_wide100','cnn4k_wide_tune50','cnn4k_wide_identity_tune50')
    labels=('Inicial: 100 épocas\nβ=0,5','Afinación: 100+50 épocas\nβ=0,1',
            'Afinación + capa\n100+50 épocas; β=0,1')
    sources=[root/f'results/art_search/validation/{name}/seeds_7' for name in names]
    if not all((p/'status.json').exists() for p in sources):return
    pick=np.arange(0,64,8)
    fig,axes=plt.subplots(3,8,figsize=(13,5.8))
    coordinates=None
    for row,(directory,label) in enumerate(zip(sources,labels)):
        status=read_json(directory/'status.json')
        with np.load(directory/'seed=7/generated.npz') as data:
            if coordinates is not None:np.testing.assert_array_equal(coordinates,data['coordinates'])
            coordinates=data['coordinates'].copy()
            shown=data['samples'][pick].reshape(-1,32,32,3)
        for column,image in enumerate(shown):
            axis=axes[row,column]
            axis.imshow(image,interpolation='nearest')
            axis.set_xticks([]);axis.set_yticks([])
            for spine in axis.spines.values():spine.set_visible(False)
            if row==0:axis.set_title(f'Índice {pick[column]}',fontsize=9)
        count=round(status['clean_rate_mean']*64)
        axes[row,0].text(-.25,.5,f'{label}\nLimpias: {count}/64',
                         transform=axes[row,0].transAxes,ha='right',va='center',fontsize=10)
    fig.suptitle('Mismas coordenadas de N(0,I), semilla 7\n'
                 'Índices fijos 0,8,…,56; tasas sobre las 64 salidas sin filtrar',fontsize=12)
    fig.subplots_adjust(left=.24,right=.99,bottom=.03,top=.85,hspace=.18,wspace=.06)
    fig.savefig(output/'refinement_ablation.png',dpi=170);plt.close(fig)


def report(root=ROOT):
    root = Path(root)
    guard = assert_loss_unchanged(root)
    base = root/'results/art_search'
    limits = read_json(base/'data/manifest.json')['cleanliness_limits']
    entries = []
    for path in sorted((base/'validation').glob('*/seeds_*/status.json')):
        status = read_json(path)
        config = read_json(root/f"configs/ej2_search_art_{status['candidate']}.json")
        # An earlier single-seed pilot stays in the report after promotion.
        rows = []
        failures = {key: 0 for key in ('edge_band_ratio', 'background_rmse',
                                      'chromatic_rmse', 'interior_rmse')}
        for seed in status['seeds']:
            with (path.parent/f'seed={seed}/cleanliness.csv').open() as stream:
                rows.extend(csv.DictReader(stream))
        for row in rows:
            for key in failures:
                failures[key] += float(row[key]) > limits[key]
        entries.append({
            'candidate': status['candidate'], 'seeds': status['seeds'],
            'resolution': status.get('resolution', config['dataset']['image_size']),
            'train_count': config['dataset']['train_count'],
            'epochs': config['training']['epochs'],
            'phase_epochs':phase_epochs(root,config,status['seeds'][0]),
            'beta': config['model']['kl']['weight'],
            'status_sha256': file_hash(path),
            'macro_iou': status['macro_iou_mean'],
            'triangle_iou': status['polygon_iou_means']['triangle'],
            'rectangle_iou': status['polygon_iou_means']['rectangle'],
            'clean_rate': status['clean_rate_mean'],
            'automatic_pass': status['automatic_pass'],
            'failed_counts_nonexclusive': failures,
            'generated_count': len(rows),
            'directory': str(path.parent.relative_to(root)),
        })
    output = root/'reports/ej2_search'
    output.mkdir(exist_ok=True, parents=True)
    result = {'loss_guard': guard, 'reserved_final_opened': any((base/'reserved_final').glob('*/seeds_*/status.json')),
              'satisfactory': (base/'satisfactory.json').exists(), 'candidates': entries}
    (output/'progress.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    lines = [
        '# Búsqueda de calidad del VAE geométrico', '',
        'Estado de los candidatos con evaluación de validación terminada. '
        'Una sola semilla es un piloto, no una confirmación. Las corridas en curso '
        'no se incluyen hasta que tienen pesos y evaluación completos.', '',
        'Se conserva la fórmula MSE + beta·KL. El calentamiento inicial es de 50 épocas; '
        'las fases de afinación posteriores no lo repiten. '
        'La fase inicial usa beta=0,5; la ampliación prueba los hiperparámetros '
        'beta=0,1 y 0,02, ambos positivos. '
        'No se agregan pérdidas geométricas, etiquetas al modelo ni filtros de salida. '
        'La prueba reservada solo se abre después de congelar la selección.', '',
        '| Candidato | Semillas | Lado | Datos | Épocas por fase | Beta final | IoU macro | Triángulo | Rectángulo | Limpias | Pasa automático |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in entries:
        budget='+'.join(str(n) if n is not None else '?' for n in row['phase_epochs'])
        lines.append(f"| {row['candidate']} | {','.join(map(str,row['seeds']))} | "
                     f"{row['resolution']} | {row['train_count']} | {budget} | {row['beta']} | "
                     f"{row['macro_iou']:.4f} | {row['triangle_iou']:.4f} | "
                     f"{row['rectangle_iou']:.4f} | {row['clean_rate']:.1%} | "
                     f"{'Sí' if row['automatic_pass'] else 'No'} |")
    if any(len(r['phase_epochs'])>1 for r in entries):
        lines += ['', 'Las sumas de épocas indican fases sucesivas, con pesos heredados y Adam '
                  'reiniciado. Cada configuración y `training.json` conservan los hiperparámetros '
                  'y el origen exacto; no se interpretan como una reanudación exacta.']
    lines += ['', '![Comparación de reconstrucción y limpieza](progress.png)', '',
              '## Control de refinamiento', '',
              'Se muestran índices fijos, no los mejores casos. Los dos modelos afinados '
              'comparten datos, pesos iniciales, beta, tasa, cantidad de pasos y ruido base. '
              'La capa nueva empieza como identidad y luego aprende por el mismo error.', '',
              '![Ablación con coordenadas comunes](refinement_ablation.png)', '']
    lines += ['', '## Causas de fallo de limpieza', '',
              'Los recuentos se solapan: una imagen puede fallar por varios motivos. '
              'Un 0 % de limpieza estricta no equivale a 0 figuras reconocibles.', '',
              '| Candidato / semillas | Muestras | Borde | Fondo | Color | Interior |',
              '|---|---:|---:|---:|---:|---:|']
    for row in entries:
        f = row['failed_counts_nonexclusive']
        lines.append(f"| {row['candidate']} / {row['seeds']} | {row['generated_count']} | "
                     f"{f['edge_band_ratio']} | {f['background_rmse']} | "
                     f"{f['chromatic_rmse']} | {f['interior_rmse']} |")
    lines += ['', '## Evidencia completa', '',
              'Cada directorio conserva `status.json`, CSV por figura y generación, '
              'muestras crudas NPZ, galerías con las 64 salidas y vecinos de entrenamiento. '
              'Las distancias al vecino más próximo no prueban por sí solas novedad.', '']
    pruned = base/'pruned_random_refinement.json'
    if pruned.exists():
        record=read_json(pruned)
        lines += [f"Se interrumpieron dos pilotos de refinamiento aleatorio tras registrar "
                  f"{record['last_observed_epoch']} épocas por estancamiento. No se cuentan como "
                  'evaluaciones generativas terminadas; se conserva el diagnóstico en '
                  '`results/art_search/pruned_random_refinement.json`.', '']
    lines.extend(f"- `{r['directory']}`" for r in entries)
    lines += ['', '## Alcance', '',
              'La IoU usa 512 imágenes nuevas, equilibradas entre las cuatro familias. '
              'La limpieza se calibró sobre originales antes de mirar generaciones. '
              'Además se requiere revisión visual de diversidad y confirmación con '
              'tres semillas y referencias reservadas; el criterio completo está en `plan.md`.', '',
              'Las imágenes a 64×64 se renderizan a su resolución nativa desde las mismas '
              'escenas continuas. Al sumar más errores de píxel con beta fijo, la presión '
              'relativa de KL cambia. No se compara la MSE absoluta entre resoluciones, '
              'ni entre RGB y espectrogramas de audio.', '',
              'Imágenes y sonidos comparten el experimento conceptual: aprender una '
              'distribución y generar ejemplos nuevos de esa familia. Usan pesos y '
              'representaciones diferentes; no comparten espacio latente ni escalas de error.', '']
    (output/'progress.md').write_text('\n'.join(lines), encoding='utf-8')
    if entries:
        fig, axes = plt.subplots(1, 2, figsize=(13, max(3.5, .5*len(entries))))
        labels = [f"{r['candidate']} ({len(r['seeds'])} sem.)" for r in entries]
        y = np.arange(len(entries))
        for ax, key, title, target in zip(axes, ('macro_iou', 'clean_rate'),
                                         ('IoU de reconstrucción', 'Generaciones limpias'), (.92, .90)):
            ax.barh(y, [r[key] for r in entries])
            ax.set_yticks(y, labels if key == 'macro_iou' else [])
            ax.axvline(target, color='firebrick', linestyle='--', label='Criterio medio')
            ax.set_xlim(0, 1); ax.set_title(title); ax.invert_yaxis(); ax.legend()
        fig.tight_layout(); fig.savefig(output/'progress.png', dpi=160); plt.close(fig)
    refinement_ablation(root,output)
    print(json.dumps({'completed_evaluations': len(entries), 'report': str(output)}, indent=2))


if __name__ == '__main__':
    report()
