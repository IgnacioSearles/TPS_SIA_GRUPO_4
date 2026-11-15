"""Replace only the previously added figure sections; preserve the remote 58 and audio additions."""
from pathlib import Path
import csv
import hashlib
import json
import re
import shutil

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data.art_capacity import load_probe

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'presentation'
PREFIX='extra_figuras_hilo_'
OUT=ROOT/'results/figure_story'
REPORT=ROOT/'reports/figure_story'
BLUE='#176a98'; ORANGE='#d4602a'; GRAY='#5d7180'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.spines.top':False,
                     'axes.spines.right':False,'text.color':'#182b3a','axes.labelcolor':'#182b3a'})

def pic(name,alt,cls=''):
    return f'<img class="{cls}" src="assets/{PREFIX}{name}" alt="{alt}">'

def rgb(name,array):
    Image.fromarray(np.uint8(np.clip(array.reshape(32,32,3),0,1)*255)).save(P/'assets'/(PREFIX+name))

def slide(title,body,take,notes):
    return f'<section class="figura-hilo"><h2>{title}</h2>{body}<p class="take">{take}</p><aside class="notes">{notes}</aside></section>'

def paired_chart():
    with (ROOT/'reports/ej2_search/matched_control/seeds.csv').open() as stream:rows=list(csv.DictReader(stream))
    assert [int(r['seed']) for r in rows]==[7,17,27]
    one=np.array([int(r['one_block_clean']) for r in rows]);two=np.array([int(r['two_blocks_clean']) for r in rows])
    assert one.sum()==175 and two.sum()==182
    fig,ax=plt.subplots(figsize=(10,3.75),constrained_layout=True)
    x=np.arange(3);width=.32
    for offset,counts,color,label in [(-width/2,one,BLUE,'Una capa de refinamiento'),(width/2,two,ORANGE,'Dos capas de refinamiento')]:
        bars=ax.bar(x+offset,100*counts/64,width,color=color,label=label)
        for bar,count in zip(bars,counts):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1.8,f'{count}/64',ha='center',fontsize=14)
    ax.set_ylim(0,112);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Imágenes limpias (%)')
    ax.set_xticks(x,['Semilla 7','Semilla 17','Semilla 27'])
    ax.legend(loc='lower left',fontsize=12)
    fig.savefig(P/'assets'/(PREFIX+'control_pareado.png'),dpi=150);plt.close(fig)
    shutil.copyfile(ROOT/'reports/ej2_search/matched_control/seeds.csv',REPORT/'control_pareado.csv')
    return {'one_clean':int(one.sum()),'two_clean':int(two.sum()),'total':192,
            'one_iou_mean':float(np.mean([float(r['one_block_iou']) for r in rows])),
            'two_iou_mean':float(np.mean([float(r['two_blocks_iou']) for r in rows]))}

def visual_results(arrays):
    generated=np.load(ROOT/'results/art_search/validation/cnn4k_lowbeta_conv2_50/seeds_7_17_27/seed=27/generated.npz')['samples']
    images=[arrays['references'][3],arrays['reconstructed'][3],generated[13],generated[61]]
    titles=['Original reservado','Reconstrucción','Generación: caso 13','Generación: caso 61']
    fig,axes=plt.subplots(1,4,figsize=(13.3,3.3),constrained_layout=True)
    for ax,values,title in zip(axes,images,titles):
        ax.imshow(values.reshape(32,32,3),interpolation='nearest');ax.set_title(title,fontsize=16)
        ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_visible(False)
    axes[0].set_xlabel('Rectángulo conocido',fontsize=14,color=BLUE)
    axes[1].set_xlabel('Esquinas redondeadas',fontsize=14,color=ORANGE)
    axes[2].set_xlabel('Curva asimétrica',fontsize=14,color=ORANGE)
    axes[3].set_xlabel('Lado hundido',fontsize=14,color=ORANGE)
    axes[2].annotate('',xy=(12,11),xytext=(2,3),arrowprops={'arrowstyle':'->','color':'white','lw':2})
    axes[3].annotate('',xy=(16,10),xytext=(27,3),arrowprops={'arrowstyle':'->','color':'white','lw':2})
    fig.savefig(P/'assets'/(PREFIX+'lectura_visual.png'),dpi=150);plt.close(fig)

def run():
    REPORT.mkdir(parents=True,exist_ok=True)
    baseline={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in P.rglob('*') if f.is_file()
              and not f.name.startswith(PREFIX) and f.name not in ('con_agregados_figuras.html','extra_figuras_hilo.css','FIGURAS_EXPLICADAS.md')}
    source=(P/'con_agregados_actualizados.html').read_text()
    matches=list(re.finditer(r'<section\b.*?</section>',source,re.S))
    old=[m.group() for m in matches]
    assert len(old)==64
    assert 'Reconstruir una figura' in old[49] and 'Una imagen limpia' in old[50] and 'GMM' in old[63]
    arrays=np.load(ROOT/'reports/ej2_activaciones_cnn/activaciones.npz')
    examples,records=load_probe(ROOT/'results/art_search/data','validation')
    ellipse_indices=[i for i,r in enumerate(records) if r['family']=='ellipse']
    aspect=lambda i:max(records[i]['objects'][0]['half_width']/records[i]['objects'][0]['half_height'],
                        records[i]['objects'][0]['half_height']/records[i]['objects'][0]['half_width'])
    ellipse=max(ellipse_indices,key=aspect)
    example_indices=[0,ellipse,2,3]
    for family,index in enumerate(example_indices):rgb(f'familia_{family}.png',examples[index])
    spec=np.load(ROOT/'results/nsynth_sampling_control/vae_standard/generated.npz')['samples'][0].reshape(128,64)
    plt.imsave(P/'assets'/(PREFIX+'audio.png'),spec,origin='lower',cmap='magma',vmin=0,vmax=1)
    stats=paired_chart();visual_results(arrays)
    for name in ['gmm_limpieza.png','gmm_0.png','gmm_64.png','gmm_128.png','gmm_192.png']:
        shutil.copyfile(P/'assets'/('extra_actualizado_'+name),P/'assets'/(PREFIX+name))
    families=['Círculo','Elipse','Triángulo','Rectángulo']
    intro=slide('La misma pregunta del audio, ahora visible',
        '<p class="lead">En audio, una altura correcta podía venir con un espectro equivocado.</p>'+
        '<div class="bridge"><div><h3>Audio: estructura espectral</h3>'+pic('audio.png','Espectrograma real generado por el VAE de audio')+
        '<p>Armónicos y evolución del sonido.</p></div><div><h3>Figuras: estructura del contorno</h3><div class="families">'+
        ''.join('<div>'+pic(f'familia_{i}.png',name)+f'<p>{name}</p></div>' for i,name in enumerate(families))+'</div>'+
        '<p>Variamos tamaño, posición, giro y color.</p></div></div>'+
        '<p class="question"><b>Objetivo:</b> ver si el VAE genera formas reconocibles al sortear códigos, sin darle una figura de entrada.</p>',
        'Otro modelo: 4.096 imágenes RGB de 32×32, latente 8 y tres semillas. Misma idea variacional.',
        'Esta extensión no busca resolver notas omitidas usando imágenes. Hace visible una pregunta común: qué estructura se conserva al generar desde un código sin entrada. '
        f'Dataset sintético: círculo, elipse, triángulo, rectángulo, variando tamaño, orientación, posición y color. Las cuatro imágenes son referencias de validación{example_indices}, no entradas de entrenamiento. '
        'Para hacer legible la introducción se elige la elipse más alargada de validación; no es selección de resultados ni se usa para calcular tasas. Algunas elipses del generador son casi circulares por construcción. '
        'El VAE recibe píxeles, sin etiquetas ni parámetros geométricos. Modelo independiente del audio, z8 frente a16 y beta final0,02 frente a0,5. No se comparan sus escalas de MSE ni se comparten pesos. '
        'El espectrograma es una muestra real del control nuevo de audio, solo como puente conceptual; no corresponde a ninguna figura. '
        'La reconstrucción usa una entrada; la generación libre se prueba sorteando z desde N(0,I). El criterio completo incluye revisión de formas, no solo una métrica de píxeles.')
    control=slide('Probamos más capacidad local en el decoder',
        '<p class="lead"><b>Hipótesis:</b> otra convolución antes de producir RGB podría mejorar la definición.</p>'+
        '<p class="setup">Mismos pesos padres, datos y 50 épocas adicionales. Solo cambia una capa 3×3 de refinamiento.</p>'+
        '<figure>'+pic('control_pareado.png','Comparación pareada por semilla: una frente a dos capas de refinamiento')+'</figure>'+
        '<p class="result"><b>175/192 → 182/192</b> imágenes limpias: 91,1 % → 94,8 %.</p>',
        'Limpieza: fondo y color uniformes, una región principal. MSE + KL sin cambios.',
        'Este es el control pareado, no una comparación entre los mejores modelos de búsquedas diferentes. Un bloque de refinamiento significa una convolución3×3 de16 a16 canales dentro del decoder antes de la salida RGB; no significa que el decoder entero tenga solo una convolución. '
        'La capa extra suma2320 parámetros e inicia como identidad, preservando la función inicial. Las dos ramas parten de padres idénticos por semilla y continúan50 épocas con iguales datos, beta0,02, tasa, minibatches y coordenadas de generación. '
        'Semillas7:59/64 a62/64;17:61 a62;27:55 a58. Total175/192 a182/192:7 casos más,3,65 puntos porcentuales. En auditoría de256 por semilla:694/768 a716/768. '
        'Son tres entrenamientos pareados, no192 modelos independientes. No se afirma significación universal. Fuente: reports/ej2_search/matched_control/seeds.csv. '
        'Limpieza se define en la siguiente página. La hipótesis de capacidad local queda apoyada para esa métrica en este control, no para geometría general.')
    visual=slide('Más limpieza todavía deja formas incorrectas',
        '<div class="definitions"><p><b>Limpieza:</b> fondo uniforme, color parejo y una región principal.</p>'+
        '<p><b>Forma:</b> un contorno reconocible como alguna de las cuatro familias.</p></div>'+
        '<figure>'+pic('lectura_visual.png','Rectángulo original y reconstrucción; dos generaciones limpias con contornos deformados')+'</figure>'+
        '<p class="small">Los dos casos generados pasan limpieza. Son fallos ilustrativos; no estiman una tasa geométrica.</p>',
        'El indicador de limpieza no cuenta lados ni verifica círculos o ángulos rectos.',
        'La imagen original es la referencia3 de validación y su reconstrucción del VAE con dos capas de refinamiento, semilla7. La pérdida de esquinas se observa al comparar la misma entrada con su salida. '
        'Los casos generados son13 y61 de la semilla27, elegidos expresamente para ilustrar deformaciones a pesar de pasar limpieza. No son una muestra aleatoria para estimar frecuencia y no tienen clase objetivo asignada: la generación es no condicional. '
        'Las flechas blancas son anotaciones de lectura, no atención ni salidas adicionales de la red. '
        'La limpieza también controla contraste, tamaño, conectividad y transición de bordes con umbrales calibrados previamente. '
        f'La coincidencia media de contornos al reconstruir fue casi igual en el control: IoU{stats["one_iou_mean"]:.5f} a{stats["two_iou_mean"]:.5f}. '
        'La MSE no impone una regla explícita de lados o simetría. Eso no significa que un VAE nunca pueda aprender geometría; aquí el criterio visual completo no se alcanzó.')
    sampling=slide('También probamos dónde sortear los códigos',
        '<p class="lead"><b>Hipótesis:</b> cambiar el sorteo podría evitar regiones poco aprendidas.</p>'+
        '<p class="definition"><b>GMM = mezcla de gaussianas:</b> varias nubes para sortear los códigos.</p>'+
        '<div class="sampling"><div>'+pic('gmm_limpieza.png','Limpieza al cambiar solo la distribución de generación')+'</div><div><h3>Ejemplos con ocho nubes</h3><div class="samples">'+
        ''.join(pic(f'gmm_{i}.png',f'Generación de GMM8 índice{i}') for i in [0,64,128,192])+'</div>'+
        '<p>El decoder se conserva;<br>cambia cómo elegimos z.</p></div></div>',
        '93,5 % → 98,8 % limpias. Ese porcentaje no mide si las figuras están bien dibujadas.',
        'Se mantiene el mismo decoder de cada uno de los tres VAE. Se ajusta una densidad a una muestra posterior por cada una de4096 imágenes de entrenamiento, sin etiquetas de familia. '
        'Se elige una componente según su peso y se sortea dentro de su gaussiana. Una componente no equivale a una clase de figura. '
        'Se comparan256 generaciones por modelo,768 por método: prior718;GMM1:738;GMM4:756;GMM8:759. Eje completo0–100. '
        'Los ejemplos usan índices0,64,128,192, una subsecuencia equiespaciada del recorrido prefijado, no los mejores casos. '
        'Este es otro experimento, no el antes/después de agregar una convolución. No se suman ambos incrementos. La GMM cambia la distribución de generación tras entrenar; no mejora retroactivamente el prior estándar ni reentrena el VAE. '
        'GMM4 y8 difieren en solo3/768 casos limpios. La tasa no evalúa geometría ni demuestra novedad. Fuentes: results/art_latent_mixture y results/art_sampling_diagnostic.')
    conclusion=slide('La mejora fue parcial; la lección vuelve al audio',
        '<div class="conclusion"><div><h3>Qué sí mejoró</h3><p>Otra convolución:<br><strong>91,1 % → 94,8 %</strong> limpias.</p>'+
        '<p>Otro sorteo, mismo decoder:<br><strong>93,5 % → 98,8 %</strong> limpias.</p>'+
        '<p class="small">Dos controles distintos: no se suman sus mejoras.</p></div><div><h3>Qué no alcanzamos</h3>'+
        '<p>La revisión visual de formas <b>no superó el criterio fijado</b>.</p><p>Mejorar fondo y color no aseguró contornos inequívocos.</p>'+
        '<p class="small">La prueba final reservada quedó sin abrir.</p></div></div>'+
        '<div class="return"><p><b>Audio:</b> afinación correcta ≠ espectro correcto.</p><p><b>Figuras:</b> imagen limpia ≠ forma correcta.</p></div>',
        'Una métrica parcial puede mejorar sin resolver el objetivo completo de generación.',
        'Respuesta al objetivo de la introducción: hubo mejoras parciales de limpieza, con reconstrucción casi igual en el control, pero no obtuvimos un generador de figuras inequívocas que cumpla el criterio visual completo. '
        'La revisión conservadora identificó3,5 y4 círculos claros por semilla frente al mínimo de5 en cada una. Es el juicio no ciego de un evaluador, no una tasa geométrica con etiquetas verdaderas para generaciones no condicionales. '
        'No se abrió la prueba final. Las tasas91,1/94,8 corresponden al control pareado de192 casos;93,5/98,8 al diagnóstico de muestreo con768. No son una única serie causal. '
        'La conexión con sonidos es evaluar la estructura que interesa preservar. No se demuestra una regla musical mediante figuras, ni se equiparan métricas o pesos. '
        'El resultado negativo de notas omitidas sigue vigente; los nuevos experimentos CNN de audio son una comparación independiente y no se presentan como resueltos en este bloque.')
    new=[intro,control,visual,sampling,conclusion]
    pieces=[];last=0
    for i,match in enumerate(matches):
        pieces.append(source[last:match.start()])
        if i==49:pieces.append('\n'.join(new))
        elif i not in (50,63):pieces.append(match.group())
        last=match.end()
    pieces.append(source[last:])
    full=''.join(pieces)
    link='<link rel="stylesheet" href="extra_figuras_hilo.css">\n'
    full=full.replace('</head>',link+'</head>',1)
    final=re.findall(r'<section\b.*?</section>',full,re.S)
    assert len(final)==66
    assert [s for s in final if s not in new]==[s for i,s in enumerate(old) if i not in (49,50,63)]
    originals=re.findall(r'<section\b.*?</section>',(P/'index.html').read_text(),re.S)
    assert len(originals)==58 and [s for s in final if s in originals]==originals
    (P/'con_agregados_figuras.html').write_text(full)
    css='''/* Scoped to the replacement figures block, all other sections unchanged. */
.reveal .figura-hilo .lead{font-size:28px;line-height:1.3;margin-bottom:16px}.reveal .figura-hilo h3{font-size:27px;margin:0 0 14px}
.reveal .figura-hilo img{display:block;object-fit:contain;max-width:100%;margin:0}.reveal .figura-hilo .take{font-size:26px}
.reveal .figura-hilo .bridge{display:grid;grid-template-columns:320px 1fr;gap:55px;align-items:start;margin-top:24px}
.reveal .figura-hilo .bridge>div>img{width:260px;height:190px;object-fit:fill}.reveal .figura-hilo .bridge p{font-size:24px;margin-top:12px}
.reveal .figura-hilo .families{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}.reveal .figura-hilo .families img{width:160px;height:160px;image-rendering:pixelated}
.reveal .figura-hilo .families p{font-size:22px;margin-top:7px}.reveal .figura-hilo .question{font-size:29px;line-height:1.3;margin-top:28px;color:var(--accent)}
.reveal .figura-hilo .setup{font-size:23px;line-height:1.3;color:var(--muted);margin-bottom:10px}
.reveal .figura-hilo figure{margin:0;height:280px}.reveal .figura-hilo figure img{width:1160px;height:100%}
.reveal .figura-hilo .result{font-size:30px;margin-top:12px}.reveal .figura-hilo .result b{color:var(--accent)}
.reveal .figura-hilo .definitions{display:grid;grid-template-columns:1fr 1fr;gap:45px;margin-bottom:18px}.reveal .figura-hilo .definitions p{font-size:26px;line-height:1.3}
.reveal .figura-hilo .groupnames{display:grid;grid-template-columns:1fr 1fr;text-align:center;font-size:23px;color:var(--accent);margin-bottom:14px}
.reveal .figura-hilo .small{font-size:21px;line-height:1.3;color:var(--muted);margin-top:16px}
.reveal .figura-hilo .definition{font-size:26px;line-height:1.3;margin-bottom:18px}
.reveal .figura-hilo .sampling{display:grid;grid-template-columns:800px 1fr;gap:25px;align-items:center}.reveal .figura-hilo .sampling>div>img{width:800px;height:320px}
.reveal .figura-hilo .samples{display:grid;grid-template-columns:repeat(2,100px);gap:10px}.reveal .figura-hilo .samples img{width:100px;height:100px;image-rendering:pixelated}
.reveal .figura-hilo .sampling h3{font-size:24px}.reveal .figura-hilo .sampling p{font-size:23px;margin-top:14px}
.reveal .figura-hilo .conclusion{display:grid;grid-template-columns:1fr 1fr;gap:65px;margin-top:35px}.reveal .figura-hilo .conclusion p{font-size:27px;line-height:1.35;margin:18px 0}
.reveal .figura-hilo .conclusion strong{color:var(--accent);font-size:40px}.reveal .figura-hilo .conclusion .small{font-size:21px}
.reveal .figura-hilo .return{margin-top:20px;border-top:1px solid var(--rule);padding-top:18px}.reveal .figura-hilo .return p{font-size:28px;line-height:1.4}
'''
    (P/'extra_figuras_hilo.css').write_text(css)
    OUT.mkdir(parents=True,exist_ok=True);shutil.copytree(P,OUT/'deck',dirs_exist_ok=True)
    (OUT/'deck/index.html').write_text(full)
    only=full[:full.index('<section')]+''.join(new)+full[full.rindex('</section>')+10:]
    (OUT/'deck/solo_figuras.html').write_text(only)
    for relative,digest in baseline.items():assert hashlib.sha256((P/relative).read_bytes()).hexdigest()==digest,relative
    meta={'original_slides_preserved':58,'nonfigure_sections_of_previous_version_preserved':61,
          'replaced_previous_pages':[50,51,64],'figure_pages':[i for i,s in enumerate(final,1) if s in new],
          'total_slides':66,'paired_control':stats,'protected_files_sha256':baseline,
          'plot_source':'reports/ej2_search/matched_control/seeds.csv','intro_example_indices':example_indices,
          'ellipse_selection':'largest axis ratio in validation, illustrative only','new_training':False}
    (REPORT/'verification.json').write_text(json.dumps(meta,indent=2));(OUT/'verification.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps({k:v for k,v in meta.items() if k!='protected_files_sha256'},indent=2))

if __name__=='__main__':run()
