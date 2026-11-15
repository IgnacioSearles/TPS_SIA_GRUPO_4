"""Rewrite only the seven proposed additions; the original 55 sections are immutable."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'presentation'
ASSETS=P/'assets'
PREFIX='extra_claro_'
OUT=ROOT/'results/slides_claras_2026-11-15'

def image(name,alt,cls=''):
    return f'<img class="{cls}" src="assets/{PREFIX}{name}" alt="{alt}">'

def save_image(name,values):
    Image.fromarray(np.uint8(np.clip(values,0,1)*255)).save(ASSETS/(PREFIX+name))

def cloud_svg(kind):
    rng=np.random.default_rng(20261115)
    if kind in ('standard','fitted'):
        centers=[(350,105)]
        data=[(x,y) for x,y in rng.normal(size=(36,2))*[35,23]+centers[0]]
        bubbles=[(135,145,83,66)] if kind=='standard' else [(350,105,78,54)]
    else:
        centers=[(110,76),(355,76),(145,210),(365,205)]
        data=[(x,y) for c in centers for x,y in rng.normal(size=(15,2))*[24,15]+c]
        bubbles=[(245,140,205,123)] if kind=='single' else [(x,y,61,42) for x,y in centers]
    svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 275">']
    for x,y,rx,ry in bubbles:
        svg.append(f'<ellipse cx="{x}" cy="{y}" rx="{rx}" ry="{ry}" fill="#176a98" fill-opacity=".07" stroke="#176a98" stroke-width="2"/>')
        svg.append(f'<ellipse cx="{x}" cy="{y}" rx="{rx*.65}" ry="{ry*.65}" fill="#176a98" fill-opacity=".12"/>')
        svg.append(f'<ellipse cx="{x}" cy="{y}" rx="{rx*.3}" ry="{ry*.3}" fill="#176a98" fill-opacity=".21"/>')
    for x,y in data:
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="#384953"/>')
    if kind=='standard':
        svg += ['<text x="135" y="239" text-anchor="middle" fill="#176a98" font-family="Arial" font-size="20">Nube alrededor de 0</text>',
                '<text x="350" y="51" text-anchor="middle" fill="#384953" font-family="Arial" font-size="20">Códigos del AE</text>']
    if kind=='fitted':
        svg += ['<text x="250" y="239" text-anchor="middle" fill="#176a98" font-family="Arial" font-size="20">Centro y anchura ajustados a los códigos</text>']
    svg.append('</svg>')
    return ''.join(svg)

def section(title,body,take,notes):
    return '<section class="claro"><h2>'+title+'</h2>'+body+'<p class="take">'+take+'</p><aside class="notes">'+notes+'</aside></section>'

def row_stat(left,right,note):
    return f'<div class="statrow"><strong>{left}</strong><span class="arrow">→</span><strong class="orange">{right}</strong></div><p class="small">{note}</p>'

def run():
    original=(P/'index.html').read_text()
    baseline={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest()
              for f in P.rglob('*') if f.is_file() and not f.name.startswith(PREFIX)
              and f.name not in ('con_agregados_v2.html','extra_claro.css','AGREGADOS_CLAROS.md')}
    sections=re.findall(r'<section\b.*?</section>',original,re.S)
    assert len(sections)==55
    arrays=np.load(ROOT/'reports/ej2_activaciones_cnn/activaciones.npz')
    save_image('original.png',arrays['references'][2].reshape(32,32,3))
    save_image('recon.png',arrays['reconstructed'][2].reshape(32,32,3))
    prior=np.load(ROOT/'results/art_sampling_diagnostic/seed=7/prior/generated.npz')['samples']
    for i in [0,16,32,48]:save_image(f'gen_{i}.png',prior[i].reshape(32,32,3))
    bad=np.load(ROOT/'results/art_search/validation/cnn4k_lowbeta_conv2_50/seeds_7_17_27/seed=27/generated.npz')['samples']
    for i in [13,61]:save_image(f'bad_{i}.png',bad[i].reshape(32,32,3))
    # Actual activations, same per-layer scale as the complete map report.
    for layer,channel in [(1,1),(2,5)]:
        values=arrays[f'encoder_{layer}']
        plt.imsave(ASSETS/(PREFIX+f'map_{layer}.png'),values[2,:,:,channel],
                   cmap='cividis',vmin=0,vmax=float(values.max()))
    for kind in ['standard','fitted','single','mixture']:
        (ASSETS/(PREFIX+kind+'.svg')).write_text(cloud_svg(kind))
    for method in ['ae_standard','ae_fitted']:
        shutil.copyfile(ROOT/f'reports/ej2_cierre_generativo_2026-11-15/audio/{method}_0.wav',ASSETS/(PREFIX+method+'.wav'))

    a=section('Gaussiana: una nube para sortear códigos',
        '<p class="intro">El <b>código latente</b> son 16 números que el decoder convierte en sonido.</p>'+
        '<p class="definition"><b>Una normal o gaussiana</b> da más probabilidad cerca de su centro y menos lejos. Su anchura marca cuánto se dispersan los sorteos.</p>'+
        '<div class="pair clouds"><div><h3>AE: sorteo alrededor de 0</h3>'+image('standard.svg','Esquema: una gaussiana estándar separada de los códigos del AE')+
        '<button class="play" data-sound="assets/'+PREFIX+'ae_standard.wav">Escuchar un ejemplo</button></div>'+
        '<div><h3>Mismo AE: sorteo adaptado</h3>'+image('fitted.svg','Esquema: gaussiana ajustada a los códigos del mismo AE')+
        '<button class="play" data-sound="assets/'+PREFIX+'ae_fitted.wav">Escuchar un ejemplo</button></div></div>'+
        '<p class="small">Puntos oscuros: códigos aprendidos. Nube azul: dónde sorteamos. Esquema en 2D; el modelo usa 16 dimensiones.</p>',
        'La red no cambió: ajustamos la nube después de entrenar el AE.',
        'Primero definir código: una lista de 16 números, no un archivo de sonido. Para generar sin entrada debemos elegir esos números. '
        'Una normal o gaussiana es una distribución de probabilidad; aquí se dibuja como nube en dos dimensiones solo para explicar centro y dispersión. '
        'El esquema NO es una proyección medida del modelo. En 16 dimensiones la covarianza también describe relaciones entre coordenadas. '
        'El AE no impone que sus códigos caigan cerca de cero. Ajustamos una media y una covarianza globales a códigos de entrenamiento y mantenemos los pesos. '
        'Los audios son el primer índice de cada método del control nuevo de NSynth, semilla 7, sin selección por calidad. '
        'Sobre 256 generaciones la MSE al vecino de entrenamiento pasó de 0,03213 a 0,01757; VAE estándar:0,01709. '
        'Eso mide cercanía al entrenamiento, no calidad perceptual ni novedad. Fuente: reports/ej2_search/nsynth_sampling_control. '
        'Una gaussiana ajustada después del entrenamiento no convierte el AE en VAE; se explica en la última página adicional.')

    b=section('Reconstruir y generar son dos pruebas distintas',
        '<p class="intro">Mismo mecanismo variacional que en audio; otro modelo, entrenado con píxeles RGB.</p>'+
        '<div class="visualsplit"><div><h3>Con una imagen de entrada</h3><div class="reconpair">'+image('original.png','Triángulo original de validación')+
        '<span class="arrow">→</span>'+image('recon.png','Reconstrucción de ese triángulo por el VAE')+'</div>'+
        '<div class="labels"><span>Original</span><span>Reconstrucción</span></div><p>Mirar si conserva <b>contorno y color</b>.</p></div>'+
        '<div><h3>Sin imagen de entrada</h3><div class="generation">'+''.join(image(f'gen_{i}.png',f'Generación real, semilla 7, índice {i}') for i in [0,16,32,48])+
        '</div><p>Se sortean códigos y el decoder dibuja.<br>Mirar si aparece una <b>figura reconocible</b>.</p></div></div>'+
        '<p class="small">Entrenamiento: círculos, elipses, triángulos y rectángulos de distintos colores, tamaños y orientaciones. Imágenes de 32×32.</p>',
        'Reconstruir una figura conocida no garantiza dibujar bien desde un código sorteado.',
        'Referencias reales: triángulo índice 2 de validación y su reconstrucción del modelo cnn4k_lowbeta_conv2_50, semilla 7. '
        'Generaciones reales desde el prior: índices 0,16,32,48 del diagnóstico de muestreo, elegidos como secuencia equiespaciada, no como mejores casos. '
        'Cada salida sin entrada no tiene una clase objetivo asignada: es un VAE no condicional. '
        'Hay 3.072 valores RGB y z tiene 8 coordenadas: reducción dimensional 384 a 1, no tasa de compresión de archivos. '
        'En audio z tiene 16 coordenadas. Son modelos independientes, sin pesos ni escalas de MSE compartidas. '
        'La pregunta común es si la representación aprendida permite reconstruir y generar ejemplos con estructura coherente.')

    c=section('Un mapa muestra dónde responde un filtro',
        '<p class="intro"><b>Claro = respuesta mayor.</b> Oscuro = respuesta menor ante esta entrada.</p>'+
        '<div class="maps"><div><h3>Imagen de entrada</h3>'+image('original.png','Entrada fija: triángulo rosa')+'<p>32×32 píxeles RGB</p></div>'+
        '<span class="arrow">→</span><div><h3>Primera capa</h3>'+image('map_1.png','Respuesta del canal 1 de la primera convolución')+'<p>16×16 · canal 1</p></div>'+
        '<span class="arrow">→</span><div><h3>Segunda capa</h3>'+image('map_2.png','Respuesta del canal 5 de la segunda convolución')+'<p>8×8 · canal 5</p></div></div>'+
        '<p class="look"><b>Qué mirar:</b> las zonas claras cambian entre mapas y la cuadrícula se vuelve más gruesa.</p>'+
        '<p class="small">Activaciones reales después de ReLU. Escalas distintas por capa: comparar ubicación, no brillo entre capas.</p>',
        'Son respuestas aprendidas; no podemos llamar a estos canales “detectores de triángulos”.',
        'Mapas reales guardados en reports/ej2_activaciones_cnn/activaciones.npz. Entrada 2 de validación, semilla 7. '
        'Canales 1 y 5 elegidos como ejemplos ilustrativos legibles, no como prueba cuantitativa ni por resultados musicales. '
        'Se usa exactamente la escala común por capa del informe completo: máximos sobre las cuatro referencias. '
        'Un filtro calcula combinaciones locales de las entradas; en capas siguientes recibe mapas de la capa anterior. '
        'La resolución disminuye por stride 2, no por max-pooling. Los contornos de color no se pasan como etiquetas. '
        'Una respuesta localizada no prueba reconocimiento semántico. Se conservan todos los mapas, incluidos oscuros, en el reporte. '
        'El encoder tiene convoluciones; el decoder también. Estos son ejemplos del encoder, no atención ni máscaras geométricas.')

    d=section('Una imagen limpia puede estar mal dibujada',
        '<div class="definitions"><p><b class="blue">Limpieza:</b> fondo sin manchas, color parejo y una sola región principal.</p>'+
        '<p><b class="orange">Geometría:</b> un contorno que permita reconocer alguna de las cuatro figuras.</p></div>'+
        '<div class="badpair"><div>'+image('bad_13.png','Salida real con curva irregular: semilla 27, índice 13')+
        '<p><b class="orange">Mirar la curva:</b><br>es asimétrica e irregular.</p></div><div>'+image('bad_61.png','Salida real con lados deformados: semilla 27, índice 61')+
        '<p><b class="orange">Mirar los lados:</b><br>no forman un rectángulo claro.</p></div></div>'+
        '<p class="small">Ambos casos pasan el control automático de limpieza. Son ejemplos de fallos, no una estimación de su frecuencia.</p>',
        'El control de limpieza no cuenta lados ni verifica círculos o ángulos rectos.',
        'Casos reales del VAE cnn4k_lowbeta_conv2_50, semilla 27, índices 13 y 61 del conjunto de 64 generaciones de validación. '
        'Los ejemplos se muestran precisamente por su deformación y no se usan para estimar una tasa de fallos geométricos. '
        'La métrica de limpieza revisa también contraste, tamaño de la región principal, conectividad, suavidad de borde y uniformidad cromática. '
        'Umbrales calibrados sobre figuras del generador; no es una calificación humana de belleza. '
        'Por geometría entendemos pertenencia visual reconocible a círculo, elipse, triángulo o rectángulo, tolerando la cuadrícula 32×32. '
        'No les asignamos una figura objetivo inexistente: la generación no es condicional. La revisión visual fue conservadora, no ciega y de un evaluador. '
        'El criterio visual completo no se alcanzó y la prueba final reservada no se abrió. Fuente: src/evaluation/art_cleanliness.py y reports/ej2_search/presentation_evidence.')

    e=section('GMM: combinar varias nubes de probabilidad',
        '<p class="definition"><b>GMM</b> significa <b>mezcla de gaussianas</b> (Gaussian Mixture Model). Para generar se elige una nube y se sortea un código dentro de ella.</p>'+
        '<div class="pair clouds gmm"><div><h3>Una nube ajustada a todos los códigos</h3>'+image('single.svg','Esquema de una gaussiana global sobre varios grupos de códigos')+
        '<p><b>96,1 %</b> limpias · 738/768</p></div><div><h3>Cuatro nubes ajustadas a los códigos</h3>'+image('mixture.svg','Esquema de cuatro gaussianas sobre los mismos puntos')+
        '<p><b class="orange">98,4 %</b> limpias · 756/768</p></div></div>'+
        '<p class="small">Puntos oscuros: códigos de entrenamiento. Dibujo explicativo en 2D; las nubes no representan clases de figuras.</p>',
        'El decoder es el mismo. Cambia dónde sorteamos el código que le entregamos.',
        'Una gaussiana global puede cubrir espacios entre agrupaciones. Una mezcla puede describir varias regiones de densidad, '
        'sin exigir que una componente coincida con una familia geométrica. El dibujo es un esquema y NO una proyección de los códigos medidos. '
        'Los porcentajes sí son resultados reales del experimento RGB: tres VAE ya entrenados, 256 generaciones por modelo y método, total 768. '
        'GMM1:738/768; GMM4:756/768; GMM8:759/768. Ajustes sobre una muestra posterior por imagen de entrenamiento, sin etiquetas. '
        'No se reentrena el decoder y no se modifica la pérdida. Es una variante de dos etapas, no el muestreo estándar del VAE desde N(0,I). '
        'Estos son porcentajes de limpieza, definidos en la página anterior; no porcentajes de geometría correcta ni prueba de novedad. '
        'No se usa una componente por imagen ni se abren referencias finales. Fuentes: results/art_latent_mixture y reports/ej2_cierre_generativo_2026-11-15/resultados_gmm.csv.')

    f=section('Mejoró la limpieza; la forma sigue siendo el límite',
        '<div class="resultblocks"><div><h3>1. Agregar una convolución al decoder</h3><p>Control pareado · 192 imágenes por variante</p>'+
        row_stat('91,1 %','94,8 %','175 → 182 imágenes limpias. Mismos padres, datos y pasos.')+
        '</div><div><h3>2. Cambiar el sorteo, manteniendo el decoder</h3><p>Normal estándar → mezcla de ocho gaussianas</p>'+
        row_stat('93,5 %','98,8 %','718 → 759 imágenes limpias de 768. Es otro experimento.')+'</div></div>'+
        '<p class="look"><b>Lo que falta:</b> que las salidas limpias también tengan contornos inequívocos.</p>',
        'Como en audio: acertar una altura tampoco garantiza conservar sus armónicos.',
        'Son dos comparaciones diferentes y no se suman sus mejoras. Primera: una versus dos convoluciones de refinamiento, '
        'misma fuente de pesos y continuación pareada de 50 épocas, tres semillas, 64 generaciones por modelo. '
        '175/192 frente a182/192. La convolución extra 3×3 de16 a16 canales suma2320 parámetros; inicia como identidad y luego aprende. '
        'MSE, KL y reparametrización se conservan. La IoU media de reconstrucción varió apenas de0,9336 a0,9337. '
        'Segunda: prior estándar versus GMM8 sobre los mismos tres modelos, 256 generaciones por modelo:718/768 frente a759/768. '
        'La limpieza no certifica geometría, calidad perceptual o novedad. Las notas omitidas requerían generalizar una relación musical: '
        'esta mejora visual no revierte ese resultado negativo. La prueba final de figuras sigue sin abrir. '
        'El piloto beta0,02/0,1 tampoco mejoró limpieza:235/256 en ambas ramas; queda como detalle de respaldo en el informe, no como nueva conclusión universal.')

    g=section('Un AE con gaussiana sigue siendo un AE',
        '<div class="aevae"><div><h3>AE: un punto por entrada</h3><div class="mini-flow"><span>Entrada</span><span>→ encoder →</span><i class="point"></i></div>'+
        '<p>Aprende a reconstruir desde un código fijo.</p><div class="after">DESPUÉS de entrenar</div><p>Ajustamos <b>una nube global</b> a todos sus códigos para poder sortear.</p></div>'+
        '<div><h3>VAE: una nube por entrada</h3><div class="mini-flow"><span>Entrada</span><span>→ encoder →</span><i class="cloud-icon"><i></i></i></div>'+
        '<p>Aprende el centro y la anchura de cada nube; de allí sale el código.</p><div class="after">DURANTE el entrenamiento</div><p>La penalización <b>KL</b> acerca esas nubes a una normal estándar.</p></div></div>',
        'La diferencia está en cómo se entrena el modelo, no en ajustar una nube al final.',
        'El AE tiene encoder determinista y un único código por entrada. El ajuste global de una gaussiana ocurre después del entrenamiento '
        'y no introduce distribución posterior por entrada, reparametrización ni KL. Por eso sigue siendo AE. '
        'El VAE aprende una normal diagonal q(z|x) por entrada: mu(x) y logvar(x), muestrea con reparametrización y entrena con reconstrucción más beta KL. '
        'La KL penaliza desajuste por entrada con una normal estándar; no garantiza que la distribución agregada de códigos coincida exactamente con el prior. '
        'Al reconstruir en nuestra implementación encode devuelve la media posterior. Al generar desde el prior no se utiliza el encoder. '
        'Las convoluciones pueden estar en cualquiera de los dos modelos: no definen la diferencia AE/VAE. '
        'La GMM ajustada a códigos del VAE es otra distribución de generación después del entrenamiento, no un cambio retroactivo de su objetivo.')

    new=[a,b,c,d,e,f,g]
    insertions={35:[a],43:[b,c,d],54:[e,f,g]}
    full=original
    for i in sorted(insertions,reverse=True):full=full.replace(sections[i],sections[i]+'\n'+''.join(insertions[i]),1)
    link='<link rel="stylesheet" href="extra_claro.css">\n'
    full=full.replace('</head>',link+'</head>',1)
    restored=full.replace(link,'')
    for i,group in insertions.items():restored=restored.replace(sections[i]+'\n'+''.join(group),sections[i],1)
    assert restored==original
    (P/'con_agregados_v2.html').write_text(full)
    css='''/* Only the seven revised additions. */
.reveal .claro .intro{font-size:27px;margin-bottom:16px;line-height:1.3}
.reveal .claro .definition{font-size:27px;line-height:1.3;margin-bottom:18px;max-width:1135px}
.reveal .claro h3{font-size:27px;margin:0 0 10px;line-height:1.2}
.reveal .claro b{font-weight:700}.reveal .claro .blue{color:#176a98}.reveal .claro .orange{color:#d4602a}
.reveal .claro .small{font-size:20px;color:#5d7180;line-height:1.25;margin-top:14px}
.reveal .claro .pair{display:grid;grid-template-columns:1fr 1fr;gap:44px}
.reveal .claro img{display:block;max-width:100%;object-fit:contain;margin:0}
.reveal .claro .clouds img{width:100%;height:225px}.reveal .claro .clouds p{font-size:25px}
.reveal .claro .play{font-size:21px;padding:5px 14px;margin-top:0}
.reveal .claro .take{font-size:27px;bottom:32px}
.reveal .claro .visualsplit{display:grid;grid-template-columns:1fr 1fr;gap:50px;margin-top:34px}
.reveal .claro .visualsplit p{font-size:25px;line-height:1.35;margin-top:20px}
.reveal .claro .reconpair{display:grid;grid-template-columns:1fr 50px 1fr;gap:12px;align-items:center}
.reveal .claro .reconpair img{width:215px;height:215px;image-rendering:pixelated}
.reveal .claro .arrow{color:#176a98;font-size:46px;text-align:center}
.reveal .claro .labels{display:flex;justify-content:space-between;font-size:23px;margin-top:9px}
.reveal .claro .generation{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;width:260px}
.reveal .claro .generation img{width:124px;height:124px;image-rendering:pixelated}
.reveal .claro .maps{display:grid;grid-template-columns:1fr 50px 1fr 50px 1fr;gap:24px;align-items:center;margin-top:30px}
.reveal .claro .maps img{width:260px;height:260px;image-rendering:pixelated}.reveal .claro .maps p{font-size:23px;margin-top:12px}
.reveal .claro .look{font-size:28px;line-height:1.3;margin-top:26px}.reveal .claro .look b{color:#176a98}
.reveal .claro .definitions{display:grid;grid-template-columns:1fr 1fr;gap:45px;margin-bottom:22px}.reveal .claro .definitions p{font-size:27px}
.reveal .claro .badpair{display:grid;grid-template-columns:1fr 1fr;gap:46px}.reveal .claro .badpair>div{display:grid;grid-template-columns:245px 1fr;gap:18px;align-items:center}
.reveal .claro .badpair img{width:245px;height:245px;image-rendering:pixelated}.reveal .claro .badpair p{font-size:24px;line-height:1.35}
.reveal .claro .gmm img{height:247px}.reveal .claro .gmm p b{font-size:38px;color:#176a98}
.reveal .claro .resultblocks{margin-top:25px;display:grid;gap:26px}.reveal .claro .resultblocks>div{border-bottom:1px solid #d5e0e5;padding-bottom:16px}
.reveal .claro .resultblocks p{font-size:22px;color:#5d7180}.reveal .claro .statrow{display:flex;align-items:center;gap:36px;margin-top:3px}
.reveal .claro .statrow strong{font-size:63px;letter-spacing:-2px;line-height:1.05;color:#176a98}.reveal .claro .statrow .orange{color:#d4602a}
.reveal .claro .resultblocks .small{margin-top:7px;font-size:21px}.reveal .claro .resultblocks+.look{margin-top:20px}
.reveal .claro .aevae{display:grid;grid-template-columns:1fr 1fr;gap:65px;margin-top:35px}.reveal .claro .aevae p{font-size:29px;line-height:1.35;margin:15px 0}
.reveal .claro .mini-flow{display:flex;align-items:center;gap:20px;height:120px;font-size:25px}.reveal .claro .point{display:block;width:38px;height:38px;border-radius:50%;background:#176a98}
.reveal .claro .cloud-icon{display:flex;align-items:center;justify-content:center;width:155px;height:100px;border:2px solid #176a98;border-radius:50%;background:#176a9820}.reveal .claro .cloud-icon i{display:block;width:60px;height:42px;border-radius:50%;background:#176a9840}
.reveal .claro .after{font-size:22px;font-weight:bold;color:#d4602a;margin-top:25px}
'''
    (P/'extra_claro.css').write_text(css)
    OUT.mkdir(parents=True,exist_ok=True)
    shutil.copytree(P,OUT/'deck',dirs_exist_ok=True)
    (OUT/'deck/index.html').write_text(full)
    only=original[:original.index('<section')]+''.join(new)+original[original.rindex('</section>')+10:]
    only=only.replace('</head>',link+'</head>',1)
    (OUT/'deck/solo_agregados.html').write_text(only)
    meta={'original_sections':55,'total_sections':62,'rewritten_additions':7,'pages':[37,46,47,48,60,61,62],
          'original_sections_byte_preserved':True,'old_files_sha256':baseline,
          'titles':[re.search(r'<h2>(.*?)</h2>',s).group(1) for s in new],
          'source_selection':{'reconstruction_reference':2,'prior_indices':[0,16,32,48],
                              'failure_seed':27,'failure_indices':[13,61],'activation_channels':[1,5]},
          'schematic_clouds_not_model_projections':True}
    for rel,value in baseline.items():assert hashlib.sha256((P/rel).read_bytes()).hexdigest()==value,rel
    assert len(re.findall(r'<section\b',full))==62
    assert len(re.findall(r'<section\b',only))==7
    (OUT/'verification.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False))
    print(json.dumps({k:v for k,v in meta.items() if k!='old_files_sha256'},indent=2,ensure_ascii=False))

if __name__=='__main__':run()
