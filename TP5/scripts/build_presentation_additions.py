"""Build an additive Reveal proposal and two offline, independently usable copies.

Run from TP5 with PYTHONPATH=. .venv/bin/python -m scripts.build_presentation_additions.
Existing presentation files are treated as immutable; preservation is checked.
"""
from pathlib import Path
import hashlib
import html
import json
import re
import shutil
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / 'presentation'
OUT = ROOT / 'results/slides_review_2026-11-15'
PREFIX = 'extra_cierre_'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def section(title, body, notes, cls=''):
    return f'\n<section class="extra-cierre {cls}">\n<h2>{title}</h2>\n{body}\n<aside class="notes">{notes}</aside>\n</section>\n'

def picture(name, alt, cls=''):
    return f'<img class="{cls}" src="assets/{PREFIX}{name}" alt="{html.escape(alt)}">'

def run():
    original = (P / 'index.html').read_text()
    baseline = {str(f.relative_to(P)): digest(f) for f in P.rglob('*') if f.is_file()
                and not f.name.startswith(PREFIX)
                and f.name not in ('con_agregados.html', 'extra_cierre.css', 'AGREGADOS_CIERRE.md')}
    sections = re.findall(r'<section\b.*?</section>', original, re.S)
    assert len(sections) == 55
    OUT.mkdir(parents=True, exist_ok=True)
    # Snapshot first, before any new presentation asset is created.
    shutil.copytree(P, OUT/'actual', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('con_agregados.html', 'extra_cierre.css',
                                                 'AGREGADOS_CIERRE.md', PREFIX+'*'))
    source = ROOT/'reports/ej2_cierre_generativo_2026-11-15'
    files = ['rgb_reconstruccion.png', 'rgb_generaciones.png', 'limpia_pero_deforme.png',
             'comparacion_final.png']
    for name in files:
        shutil.copyfile(source/'figuras'/name, P/'assets'/(PREFIX+name))
    for name in ['mapas_encoder.png', 'encoder_2_completo.png']:
        shutil.copyfile(ROOT/'reports/ej2_activaciones_cnn'/name, P/'assets'/(PREFIX+name))
    methods = [('ae_standard', 'AE · normal estándar', '0,0321'),
               ('ae_fitted', 'AE · gaussiana ajustada', '0,0176'),
               ('vae_standard', 'VAE · normal estándar', '0,0171')]
    rows = []
    for method, label, distance in methods:
        data = np.load(ROOT/'results/nsynth_sampling_control'/method/'generated.npz')
        values = data['generated'] if 'generated' in data else data['samples']
        cells = []
        for i in range(3):
            name = f'{method}_{i}'
            # Fixed first indices, fixed color scale in the codec representation.
            plt.imsave(P/'assets'/(PREFIX+name+'.png'), values[i].reshape(128,64),
                       cmap='magma', vmin=0, vmax=1, origin='lower')
            shutil.copyfile(source/'audio'/(name+'.wav'), P/'assets'/(PREFIX+name+'.wav'))
            cells.append('<div>'+picture(name+'.png', label+f', muestra {i}')+
                         f'<button class="play" data-sound="assets/{PREFIX}{name}.wav">Escuchar {i+1}</button></div>')
        rows.append(f'<div class="extra-audio-row"><p>{label}<strong>{distance}</strong></p>'+''.join(cells)+'</div>')
    a = section('El muestreo también cambia el resultado',
        '<p class="extra-context">Control complementario de NSynth · semilla 7 · 256 generaciones por método</p>'+
        '<div class="extra-audio-head">MSE al vecino de entrenamiento más próximo</div>'+''.join(rows)+
        '<p class="take">Ajustar la gaussiana acercó el AE al entrenamiento. No mide calidad auditiva ni novedad.</p>',
        'Reproducciones nuevas con la configuración de NSynth; no son los pesos certificados de la comparación anterior. '
        'Un único modelo por tipo. Se muestran los primeros tres índices, no una selección por calidad. '
        'El mismo AE usa códigos deterministas y pesos idénticos: después de entrenar ajustamos una media y covarianza globales '
        'y generamos z=m+L·epsilon. Su distancia media bajó 45,3 %. El VAE aprende mu(x) y sigma(x) por entrada, '
        'reparametriza durante el entrenamiento y recibe KL hacia el prior. Agregar una gaussiana al AE no lo convierte en VAE. '
        'La misma realización de ruido no alinea semánticamente los espacios de modelos distintos. '
        'Son distancias a entrenamiento, no a una prueba independiente; escuchar las muestras es complementario, sin evaluación humana controlada.',
        'extra-audio')
    b = section('El mismo mecanismo variacional con figuras RGB',
        '<p class="extra-context">Píxeles 32×32 · latente de 8 dimensiones · cuatro familias · sin etiquetas como entrada</p>'+
        '<div class="extra-rgb"><div><p>Originales de validación y reconstrucciones</p>'+
        picture('rgb_reconstruccion.png','Cuatro referencias y sus reconstrucciones')+'</div><div><p>Generaciones desde N(0,I)</p>'+
        picture('rgb_generaciones.png','Ocho generaciones de índices prefijados')+'</div></div>'+
        '<p class="take">Modelos independientes: comparten el mecanismo variacional, no pesos ni escalas de MSE.</p>',
        'Extensión visual del ejercicio 2: aprender una representación para reconstruir y generar ejemplos plausibles. '
        'Son 3.072 valores RGB y 8 coordenadas latentes: reducción dimensional de 384 a 1, no tasa de compresión de archivos. '
        'Aquí usamos CNN y beta final 0,02; en audio, latente 16 y beta 0,5. No aislamos el efecto causal de la modalidad. '
        'Referencias 0–3 de validación e índices generados 0,8,…,56 de la semilla 7. '
        'Las figuras ayudan a distinguir limpieza de geometría, así como la altura correcta no garantiza un espectro correcto. '
        'Generar muestras plausibles es distinto de inferir una nota eliminada del entrenamiento.')
    c = section('Qué activan los filtros de la CNN',
        '<p class="extra-context">Encoder · misma entrada · primeros ocho canales de cada capa</p>'+
        picture('mapas_encoder.png','Activaciones reales de dos capas ante un triángulo','extra-map')+
        '<p class="take">Respuestas reales después de ReLU. El brillo solo es comparable dentro de cada capa.</p>',
        'Inferencia sin entrenamiento ni cambios de pesos. Referencia 2 de validación, seleccionada antes de inspeccionar mapas; '
        'mostramos canales 0–7 en orden, incluyendo respuestas débiles. La primera convolución produce 16 canales de 16×16; '
        'la segunda, 32 de 8×8. Se reduce resolución con stride 2; no hay max-pooling. '
        'Cada filtro aprende combinaciones locales de color y posición. Estos mapas no son atención ni máscaras semánticas '
        'y no prueban que un canal identifique triángulos. Una escala común por capa se fija con las cuatro referencias. '
        'El apéndice incluye los 32 canales de la segunda capa. El decoder usa upsampling y convoluciones, sin convolución transpuesta.')
    d = section('Limpieza y geometría son propiedades diferentes',
        '<div class="extra-deform"><div>'+picture('limpia_pero_deforme.png','Dos generaciones limpias con contornos deformados')+
        '<p class="extra-context">Casos ilustrativos: semilla 27, índices 13 y 61.</p></div><div>'+
        '<table><tr><th>Decoder</th><th>Limpias</th></tr><tr><td>Un bloque</td><td>91,1 %</td></tr><tr><td>Dos bloques</td><td>94,8 %</td></tr></table>'+
        '<p>192 generaciones por variante, tres semillas.</p><p>IoU media de reconstrucción:<br>0,9336 → 0,9337</p></div></div>'+
        '<p class="take">La convolución aprendida mejoró limpieza, pero no aseguró figuras inequívocas.</p>',
        'Control pareado: mismos padres, datos y pasos de continuación de 50 épocas. La convolución extra 3×3 de 16 a 16 '
        'canales agrega 2.320 parámetros, inicia como identidad y luego aprende todos sus coeficientes. '
        'No cambiamos MSE, KL ni reparametrización. Limpieza pasó de 175/192 a 182/192; auditoría: 694/768 a 716/768. '
        'Los dos fallos ilustran un límite, no estiman una tasa geométrica. MSE compara píxeles, no exige tres lados ni ángulos rectos. '
        'Resolución baja, promedios bajo incertidumbre e intermedios latentes son explicaciones posibles, no causas aisladas demostradas. '
        'El criterio visual completo no se cumplió: 3/5/4 círculos claros por semilla frente al mínimo de cinco. '
        'La revisión fue conservadora, no ciega y de un evaluador; no abrimos la prueba final reservada.')
    e = section('Dónde elegimos los códigos',
        '<div class="extra-gmm"><div>'+picture('comparacion_final.png','Muestras RGB del prior y de una mezcla ajustada')+
        '<p class="extra-context">Ocho índices fijos · semilla 7 · las filas no son objetos pareados.</p></div>'+
        '<table><tr><th>Muestreo</th><th>Limpias</th></tr><tr><td>N(0,I)</td><td>93,5 %</td></tr>'+
        '<tr><td>GMM · 1</td><td>96,1 %</td></tr><tr><td>GMM · 4</td><td>98,4 %</td></tr><tr><td>GMM · 8</td><td>98,8 %</td></tr></table></div>'+
        '<p class="take">Mismo decoder, otra distribución de generación. Mejora limpieza; no demuestra novedad.</p>',
        '768 muestras por método: tres modelos y 256 generaciones por modelo. GMM de covarianza completa ajustada '
        'después del entrenamiento a 4.096 códigos, una muestra posterior por imagen, sin etiquetas de familia. '
        'La GMM no modifica la red ni convierte sus componentes en clases. Es un procedimiento de dos etapas: VAE más densidad ajustada. '
        'Prior:718/768; GMM1:738; GMM4:756; GMM8:759. Cuatro frente a ocho difieren en solo tres casos. '
        'GMM8 queda 26,7 % más cerca del entrenamiento en MSE; no prueba ausencia de memorización. '
        'La semilla 17 de GMM8 llegó al límite de 100 iteraciones; no proclamamos óptimo global. '
        'Las figuras híbridas persisten. La generación estándar desde N(0,I) no cambió.')
    f = section('Controles y alcance de los resultados',
        '<div class="extra-controls"><div><h3>Subir beta no alcanzó en el piloto</h3>'+
        '<table><tr><th>Beta</th><th>Limpias</th><th>KL</th></tr><tr><td>0,02</td><td>235/256</td><td>28,92</td></tr>'+
        '<tr><td>0,10</td><td>235/256</td><td>22,32</td></tr></table><p>20 épocas pareadas · una semilla.<br>No se promovió esta variante.</p></div>'+
        '<div><h3>Qué podemos sostener</h3><p>Tres semillas en RGB; una en el control de NSynth.</p>'+
        '<p>Cercanía al dataset y limpieza no certifican novedad ni calidad perceptual.</p>'+
        '<p>La prueba final reservada quedó sin abrir.</p></div></div>'+
        '<p class="take">Reportamos mejoras concretas y fallos visibles. El criterio visual completo no se alcanzó.</p>',
        'Ambas ramas parten del mismo modelo y usan 1.280 pasos. En la grilla principal también empatan en 62/64. '
        'La mayor penalización redujo KL pero no mejoró estas generaciones; no es una ley universal sobre beta. '
        'No se comparan valores de MSE entre audio y RGB ni se infiere equivalencia por proyecciones 2D o momentos gaussianos. '
        'Barras entre muestras de un modelo no representan variación entre entrenamientos. '
        'Las notas omitidas requerían una relación de altura no observada: esta extensión no revierte ese resultado negativo. '
        'Estudio cerrado el 15/11/2026. No se realizaron nuevos entrenamientos para preparar esta presentación.')
    g = section('La segunda capa: los 32 mapas completos',
        picture('encoder_2_completo.png','Todos los canales de la segunda capa, incluidos los oscuros','extra-fullmap')+
        '<p class="take">Misma referencia y misma escala para todos los canales. No son etiquetas geométricas.</p>',
        'Referencia 2, triángulo de validación, semilla 7. Canales en su orden original, sin selección por apariencia. '
        'Las activaciones después de ReLU son no negativas: un canal oscuro responde poco a esta entrada, '
        'no necesariamente a todas las entradas. Reporte reproducible: reports/ej2_activaciones_cnn. '
        'También guarda los 16 mapas de la primera capa, dos capas de refinamiento del decoder '
        'y arrays completos para cuatro referencias, junto con hashes de los pesos inalterados.')
    additions = {35:[a],43:[b,c,d],54:[e,f,g]}
    variant = original
    for index in sorted(additions, reverse=True):
        old = sections[index]
        assert variant.count(old) == 1
        variant = variant.replace(old, old+'\n'+''.join(additions[index]), 1)
    css_link = '<link rel="stylesheet" href="extra_cierre.css">\n'
    variant = variant.replace('</head>', css_link+'</head>', 1)
    added_sections = [re.search(r'<section\b.*?</section>', s, re.S).group() for group in additions.values() for s in group]
    all_sections = re.findall(r'<section\b.*?</section>', variant, re.S)
    assert len(all_sections) == 62
    assert [s for s in all_sections if s not in added_sections] == sections
    restored = variant.replace(css_link,'')
    for index, group in additions.items():
        restored = restored.replace(sections[index]+'\n'+''.join(group), sections[index], 1)
    assert restored == original, 'Only section insertions and one stylesheet link are allowed'
    (P/'con_agregados.html').write_text(variant)
    (P/'extra_cierre.css').write_text('''/* Additive rules. Original style.css is unchanged. */
.reveal .extra-cierre .extra-context {font-size:21px;color:var(--muted);margin-bottom:14px;line-height:1.25;}
.reveal .extra-cierre img {display:block;margin:0;object-fit:contain;max-width:100%;}
.reveal .extra-cierre table {font-size:27px;border-collapse:collapse;width:100%;margin:0;}
.reveal .extra-cierre th,.reveal .extra-cierre td {padding:15px 9px;border-bottom:1px solid var(--rule);text-align:left;}
.reveal .extra-cierre th {color:var(--accent);font-size:23px;}
.reveal .extra-cierre h3 {font-size:29px;margin:0 0 20px;}
.reveal .extra-cierre .take {font-size:26px;}
.reveal .extra-audio-head {font-size:20px;text-align:right;color:var(--muted);margin:0 0 9px;}
.reveal .extra-audio-row {display:grid;grid-template-columns:300px repeat(3,1fr);gap:20px;align-items:center;margin-bottom:14px;}
.reveal .extra-audio-row p {font-size:24px;}
.reveal .extra-audio-row strong {display:block;color:var(--accent);font-size:34px;margin-top:6px;}
.reveal .extra-audio-row img {width:100%;height:92px;object-fit:fill;}
.reveal .extra-audio-row .play {font-size:17px;padding:3px 13px;margin-top:5px;}
.reveal .extra-rgb {display:grid;grid-template-columns:1fr 1fr;gap:32px;align-items:center;height:405px;}
.reveal .extra-rgb p {font-size:24px;margin-bottom:12px;}
.reveal .extra-rgb img {width:100%;height:330px;}
.reveal .extra-map {width:1160px;height:405px;}
.reveal .extra-fullmap {width:1160px;height:475px;}
.reveal .extra-deform {display:grid;grid-template-columns:650px 1fr;gap:32px;align-items:start;margin-top:30px;}
.reveal .extra-deform img {height:350px;width:650px;}
.reveal .extra-deform p:not(.extra-context) {font-size:23px;margin-top:20px;}
.reveal .extra-gmm {display:grid;grid-template-columns:790px 1fr;gap:28px;align-items:center;margin-top:35px;}
.reveal .extra-gmm img {width:790px;height:340px;}
.reveal .extra-gmm table {font-size:25px;}
.reveal .extra-controls {display:grid;grid-template-columns:1fr 1fr;gap:55px;margin-top:45px;}
.reveal .extra-controls p {font-size:25px;margin:22px 0;line-height:1.3;}
''')
    new_pages = [dict(page=i+1,title=re.sub('<.*?>','',re.search(r'<h2>(.*?)</h2>',s,re.S).group(1)))
                 for i,s in enumerate(all_sections) if s in added_sections]
    manifest = dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    current_slides=55,proposed_slides=62,original_sections_unchanged=True,
                    original_files_sha256=baseline,added_pages=new_pages,training_performed=False)
    for rel, checksum in baseline.items():
        assert digest(P/rel) == checksum, rel
        assert digest(OUT/'actual'/rel) == checksum, rel
    shutil.copytree(OUT/'actual', OUT/'con_agregados', dirs_exist_ok=True)
    (OUT/'con_agregados'/'index.html').write_text(variant)
    shutil.copyfile(P/'extra_cierre.css', OUT/'con_agregados'/'extra_cierre.css')
    for path in (P/'assets').glob(PREFIX+'*'):
        shutil.copyfile(path, OUT/'con_agregados'/'assets'/path.name)
    for folder in ['actual','con_agregados']:
        page = (OUT/folder/'index.html').read_text()
        for ref in re.findall(r'(?:src|href|data-sound)="([^"]+)"',page):
            if not re.match(r'^(https?:|#|data:)',ref):
                assert (OUT/folder/ref.split('?')[0]).exists(), (folder,ref)
    (OUT/'verificacion.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    page_list = '\n'.join(f"- {p['page']}: {p['title']}" for p in new_pages)
    (P/'AGREGADOS_CIERRE.md').write_text(f'''# Propuesta adicional, 15/11/2026

Abrir `con_agregados.html`; `index.html` conserva exactamente la presentación vigente.
55 páginas originales sin cambios, en el mismo orden relativo; siete agregados (62 total).

## Páginas nuevas en la propuesta

{page_list}

Las cuatro primeras se integran a la explicación; las tres últimas son respaldo.
Todos los agregados incluyen notas del orador. No se reemplazan las notas anteriores.
Los botones de audio conservan el controlador existente.

## Fuentes e interpretación

- `reports/ej2_cierre_generativo_2026-11-15/guion_solo_agregados.md`: motivación y guion detallado.
- `reports/ej2_activaciones_cnn/`: inferencia real sin entrenar; mapas completos y pesos verificados.
- `results/nsynth_sampling_control/`: control complementario nuevo, una semilla, tres distribuciones.
- `results/art_latent_mixture/`: GMM sobre códigos de entrenamiento, sin etiquetas; tres semillas.
- `reports/ej2_cierre_generativo_2026-11-15/`: tablas, figuras y alcance del cierre.

El AE con densidad ajustada conserva su entrenamiento determinista. No es un VAE.
Las GMM cambian el muestreo posterior al entrenamiento; no mejoran retroactivamente el prior N(0,I).
Limpieza no equivale a geometría correcta; cercanía al entrenamiento no demuestra novedad.
RGB y audio comparten mecanismo variacional, sin compartir pesos ni comparar sus MSE.
La prueba final quedó sin abrir. No se hicieron nuevos entrenamientos.

## Reproducir y verificar

Desde TP5: `PYTHONPATH=. .venv/bin/python -m scripts.export_cnn_activations`
y `PYTHONPATH=. .venv/bin/python -m scripts.build_presentation_additions`.
El segundo comando escribe dos paquetes en `results/slides_review_2026-11-15/`
y `verificacion.json` con hashes de todos los archivos originales.
Verifica 55 secciones literales preservadas y 62 finales, referencias locales existentes,
y que quitar las siete inserciones y el vínculo CSS reconstruye el HTML original exactamente.
''')
    print(json.dumps(dict(output=str(OUT),added_pages=new_pages,verified_files=len(baseline)),ensure_ascii=False,indent=2))

if __name__ == '__main__':
    run()
