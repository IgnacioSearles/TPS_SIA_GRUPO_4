"""Rebuild only the added figure block around observations -> hypotheses -> controls."""
from pathlib import Path
import base64
import hashlib
import json
import re
import shutil

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data.art_capacity import load_probe
from src.experiments.runner import load_run

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'presentation'
PREFIX='extra_hilo_causal_'
OUT=ROOT/'results/figure_thread'
REPORT=ROOT/'reports/figure_thread'
BLUE='#176a98';ORANGE='#d4602a'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'text.color':'#182b3a'})

def pic(name,alt):return f'<img src="assets/{PREFIX}{name}" alt="{alt}">'
def section(title,body,take,notes):return f'<section class="figura-causal"><h2>{title}</h2>{body}<p class="take">{take}</p><aside class="notes">{notes}</aside></section>'

def make_baseline():
    run=ROOT/'results/ej2_width_art_vae_wide/base__seed=7'
    digest=hashlib.sha256((run/'model.npz').read_bytes()).hexdigest()
    model,_=load_run(run)
    probes,records=load_probe(ROOT/'results/art_width_data','validation')
    ellipses=[i for i,r in enumerate(records) if r['family']=='ellipse']
    ratio=lambda i:max(records[i]['objects'][0]['half_width']/records[i]['objects'][0]['half_height'],records[i]['objects'][0]['half_height']/records[i]['objects'][0]['half_width'])
    ids=[0,max(ellipses,key=ratio),2,3]
    refs=probes[ids].astype(model.precision)
    reconstructed=model.reconstruct(refs)
    gen=np.load(ROOT/'results/art_width_validation/vae_wide/seed=7/generated.npz')['samples'][:4]
    fig,axes=plt.subplots(3,4,figsize=(9,6.4))
    for row,values in enumerate([refs,reconstructed,gen]):
        for ax,value in zip(axes[row],values):
            ax.imshow(value.reshape(32,32,3),interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_visible(False)
        axes[row,0].text(-.12,.5,['Originales','Reconstrucciones','Generaciones\nindependientes'][row],ha='right',va='center',transform=axes[row,0].transAxes,fontsize=13,color=BLUE)
    fig.subplots_adjust(left=.22,right=.99,top=.98,bottom=.025,hspace=.17,wspace=.10)
    fig.savefig(P/'assets'/(PREFIX+'denso_original.png'),dpi=170);plt.close(fig)
    assert hashlib.sha256((run/'model.npz').read_bytes()).hexdigest()==digest
    return {'model':str(run.relative_to(ROOT)),'sha256':digest,'reference_indices':ids,'generation_indices':[0,1,2,3],
            'config':json.loads((run/'config.json').read_text()),'role':'historical dense result, before CNN; not a matched comparison'}

def make_early_cnn():
    stages=[('CNN básica','cnn4k_100'),('CNN más ancha','cnn4k_wide100')]
    fig,axes=plt.subplots(2,4,figsize=(12,4.6))
    sources=[]
    for row,(label,name) in enumerate(stages):
        path=ROOT/f'results/art_search/validation/{name}/seeds_7/seed=7/generated.npz'
        saved=np.load(path)
        for ax,value in zip(axes[row],saved['samples'][:4]):
            ax.imshow(value.reshape(32,32,3),interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_visible(False)
        axes[row,0].text(-.12,.5,label,ha='right',va='center',transform=axes[row,0].transAxes,fontsize=17,color=BLUE)
        sources.append({'candidate':name,'generated':str(path.relative_to(ROOT)),'indices':[0,1,2,3]})
    fig.subplots_adjust(left=.18,right=.99,top=.98,bottom=.025,hspace=.15,wspace=.12)
    fig.savefig(P/'assets'/(PREFIX+'primeras_cnn.png'),dpi=150);plt.close(fig)
    return sources

def convolution_diagram():
    image=P/'assets/extra_figuras_hilo_familia_3.png'
    encoded=base64.b64encode(image.read_bytes()).decode()
    svg=f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1160 300">
<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="{BLUE}"/></marker></defs>
<image href="data:image/png;base64,{encoded}" x="15" y="18" width="240" height="240" style="image-rendering:pixelated"/>
<rect x="82.5" y="78" width="22.5" height="22.5" fill="none" stroke="#ffcc55" stroke-width="4"/>
<line x1="108" y1="90" x2="345" y2="90" stroke="{BLUE}" stroke-width="3" marker-end="url(#arrow)"/>
<text x="242" y="70" text-anchor="middle" font-family="Arial" font-size="23" fill="{BLUE}">Una ventana local</text>
<rect x="365" y="35" width="144" height="144" fill="#eef4f7" stroke="{BLUE}" stroke-width="3"/>
<path d="M413,35v144 M461,35v144 M365,83h144 M365,131h144" stroke="{BLUE}" stroke-width="2"/>
<text x="437" y="217" text-anchor="middle" font-family="Arial" font-size="26" fill="#182b3a">Filtro 3×3 aprendido</text>
<line x1="540" y1="105" x2="675" y2="105" stroke="{BLUE}" stroke-width="3" marker-end="url(#arrow)"/>
<text x="760" y="55" font-family="Arial" font-size="27" fill="#182b3a">Los mismos pesos</text>
<text x="760" y="95" font-family="Arial" font-size="27" fill="#182b3a">se aplican en distintas zonas.</text>
<text x="760" y="155" font-family="Arial" font-size="25" fill="{ORANGE}">No le damos reglas</text>
<text x="760" y="190" font-family="Arial" font-size="25" fill="{ORANGE}">para dibujar figuras.</text>
<text x="135" y="290" text-anchor="middle" font-family="Arial" font-size="23" fill="#5d7180">Imagen RGB</text>
</svg>'''
    (P/'assets'/(PREFIX+'convolucion.svg')).write_text(svg)

def run():
    REPORT.mkdir(parents=True,exist_ok=True)
    protected={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in P.rglob('*') if f.is_file()
        and not f.name.startswith(PREFIX) and f.name not in ('con_agregados_hilo_completo.html','extra_hilo_causal.css')}
    base=(P/'con_agregados_figuras.html').read_text();matches=list(re.finditer(r'<section\b.*?</section>',base,re.S));old=[m.group() for m in matches]
    assert len(old)==66 and 'misma pregunta' in old[49] and 'lección vuelve' in old[53]
    baseline=make_baseline();early=make_early_cnn();convolution_diagram()
    intro=old[49].replace('En audio, una altura correcta podía venir con un espectro equivocado.',
        'Partimos de un VAE denso como el del audio, pero entrenado con figuras.')
    intro=intro.replace('Otro modelo: 4.096 imágenes RGB de 32×32, latente 8 y tres semillas. Misma idea variacional.',
        'VAE independiente: RGB de 32×32 y latente 8. Ampliamos de 1.024 a 4.096 imágenes.')
    intro=intro.replace('Esta extensión no busca resolver notas omitidas usando imágenes.',
        'Hilo del bloque: resultado denso observado, hipótesis de localidad, primeras CNN, afinación y control pareado de una capa. Esta extensión no busca resolver notas omitidas usando imágenes.')
    dense=section('El VAE denso generaba contornos difusos',
        '<p class="context">Resultado histórico anterior a las CNN · VAE denso [256,128] · latente 8 · 1.024 imágenes · 200 épocas</p>'+
        '<div class="baseline"><div>'+pic('denso_original.png','Originales y reconstrucciones del VAE denso anterior; sus primeras cuatro generaciones')+'</div><div>'+
        '<h3>Qué observamos</h3><p>Al reconstruir ya se pierden esquinas.</p><p>Al generar aparecen <b>fondos con ruido y bordes difusos</b>.</p>'+
        '<p class="small">La tercera fila no corresponde a las entradas de arriba: son códigos sorteados.</p></div></div>',
        'Eso motivó probar una arquitectura que aproveche explícitamente los píxeles vecinos.',
        'Resultado real anterior a las CNN: results/ej2_width_art_vae_wide/base__seed=7, 1024 imágenes,200 épocas,latente8,ocultas[256,128],beta0,5. '
        'Se muestran referencias de validación del protocolo art_width y sus reconstrucciones actuales desde la media posterior, con pesos congelados verificados por hash. '
        'La elipse se elige por relación de ejes para que el ejemplo sea legible, no para medir calidad. La tercera fila usa las primeras cuatro generaciones guardadas0–3 del prior. '
        'No son el mismo objeto que las entradas ni un antes/después de CNN. Es una observación visual que motiva una hipótesis, no una prueba de que una red densa nunca pueda aprender geometría. '
        'El piloto posterior CNN cambia datos, arquitectura y presupuesto; no se presenta una diferencia causal aislada entre estas etapas. El control pareado posterior sí aísla una capa adicional.')
    conv=section('Convolución: aprender con píxeles vecinos',
        '<p class="lead">Un filtro pequeño aprende combinaciones locales y se reutiliza por toda la imagen.</p>'+
        '<figure>'+pic('convolucion.svg','Esquema de un filtro local 3 por 3 cuyos pesos se comparten entre posiciones')+'</figure>'+
        '<div class="paths"><p><b>Encoder:</b> imagen → mapas locales → media y desvío → z.</p>'+
        '<p><b>Decoder:</b> z → mapas → ampliar + convolucionar → imagen.</p></div>',
        'Hay convoluciones en ambos lados. Se conservan reparametrización, MSE y KL.',
        'Introducir la arquitectura antes de hablar de otra convolución. Una densa recibe la imagen aplanada y aprende conexiones específicas por posición; también puede aprender relaciones espaciales, pero no comparte por construcción un filtro local. '
        'La CNN reutiliza pesos de una ventana en posiciones diferentes. El esquema muestra el tamaño3×3 y no son valores aprendidos, atención ni un filtro manual de bordes. '
        'Varios filtros producen varios mapas o canales. El encoder usa stride2, sin max-pooling, y el decoder duplica resolución y aplica convoluciones. No usa un renderizador ni reglas geométricas. '
        'La arquitectura convolucional no define la diferencia AE/VAE: ambos pueden usar CNN. Aquí seguimos entrenando un VAE; cambian las redes que lo codifican y decodifican, no su fórmula de error.')
    conv=conv.replace('class="figura-causal"','class="figura-causal conv-bridge"',1)
    cnn=section('Las primeras CNN también dejaban bordes difusos',
        '<p class="context">Pilotos: 4.096 imágenes · latente 8 · 100 épocas · mismo peso de KL · semilla 7</p>'+
        '<figure>'+pic('primeras_cnn.png','Primeras cuatro generaciones de una CNN básica y una más ancha; persiste el borde difuso')+'</figure>'+
        '<p class="bridge">Después exploramos más capacidad, entrenamiento, peso de KL y refinamiento. Para medir el aporte de <b>una sola capa</b>, hicimos el control siguiente.</p>',
        'Pasar a CNN no resolvió todo. Estos pilotos exploratorios no son una comparación causal con el denso.',
        'Resultados guardados de cnn4k_100 y cnn4k_wide100, semilla7. Se muestran índices consecutivos0–3, no los mejores. Son generaciones independientes, no la transformación de los cuatro objetos del denso. '
        'Ambos pilotos usan4096 imágenes100 épocas beta0,5; cambian anchura/capacidad. Respecto del modelo denso histórico también cambian datos y presupuesto: no se atribuye una diferencia numérica exclusivamente a usar CNN. '
        'Se observan contornos con transiciones amplias incluso cuando algunas formas se reconocen. El control automático de limpieza no es clasificación de figuras. '
        'Después la búsqueda exploró anchura, entrenamiento adicional, menor peso de KL y una primera capa de refinamiento. Varias cosas cambiaron y no se ordenan sus resultados como una ablación causal. '
        'La comparación siguiente parte de una CNN ya afinada común a cada par; por eso puede aislar el efecto de una segunda capa de refinamiento. Fuente: reports/ej2_search/plan.md,progress.md y configuraciones guardadas.')
    control=old[50].replace('Probamos más capacidad local en el decoder','Una segunda capa mejora un poco la limpieza')
    control=control.replace('<b>Hipótesis:</b> otra convolución antes de producir RGB podría mejorar la definición.',
        'Partimos de una CNN ya afinada. <b>Refinamiento</b> es una convolución al final del decoder.')
    control=control.replace('Este es el control pareado, no una comparación entre los mejores modelos de búsquedas diferentes.',
        'Separar esta prueba de pasar de denso a CNN. Aquí ya hay una CNN afinada con una capa de refinamiento; se mide únicamente añadir la segunda. Este es el control pareado, no una comparación entre los mejores modelos de búsquedas diferentes.')
    final=old[53].replace('Otra convolución:<br>','Segunda capa, en el control pareado:<br>')
    final=final.replace('Respuesta al objetivo de la introducción:',
        'La secuencia distingue el resultado denso que motivó localidad, las primeras CNN, la búsqueda exploratoria y la ablación final. No atribuir toda la mejora histórica a convoluciones. Respuesta al objetivo de la introducción:')
    new=[intro,dense,conv,cnn,control,old[51],old[52],final]
    pieces=[];last=0
    for i,m in enumerate(matches):
        pieces.append(base[last:m.start()])
        if i==49:pieces.append('\n'.join(new))
        elif i not in range(50,54):pieces.append(m.group())
        last=m.end()
    pieces.append(base[last:]);full=''.join(pieces)
    full=full.replace('</head>','<link rel="stylesheet" href="extra_hilo_causal.css">\n</head>',1)
    current=re.findall(r'<section\b.*?</section>',full,re.S)
    assert len(current)==69
    assert [s for s in current if s not in new]==[s for i,s in enumerate(old) if i not in range(49,54)]
    originals=re.findall(r'<section\b.*?</section>',(P/'index.html').read_text(),re.S)
    assert len(originals)==58 and [s for s in current if s in originals]==originals
    (P/'con_agregados_hilo_completo.html').write_text(full)
    css='''/* New causal bridge only; all original sections and styles remain untouched. */
.reveal .figura-causal .context{font-size:21px;color:var(--muted);line-height:1.3;margin-bottom:15px}
.reveal .figura-causal .lead{font-size:28px;line-height:1.3;margin-bottom:20px}
.reveal .figura-causal h3{font-size:28px;margin:0 0 18px}.reveal .figura-causal img{display:block;margin:0;object-fit:contain;max-width:100%}
.reveal .figura-causal .take{font-size:26px}.reveal .figura-causal .baseline{display:grid;grid-template-columns:790px 1fr;gap:28px;align-items:center}
.reveal .figura-causal .baseline img{width:790px;height:420px}.reveal .figura-causal .baseline p{font-size:27px;line-height:1.35;margin:22px 0}
.reveal .figura-causal .baseline .small{font-size:21px;color:var(--muted)}
.reveal .figura-causal figure{margin:0;height:325px}.reveal .figura-causal figure img{width:1160px;height:100%}
.reveal .figura-causal.conv-bridge figure{height:260px}
.reveal .figura-causal .paths p{font-size:28px;line-height:1.45;margin:8px 0}
.reveal .figura-causal .bridge{font-size:27px;line-height:1.3;margin-top:18px}
'''
    (P/'extra_hilo_causal.css').write_text(css)
    OUT.mkdir(parents=True,exist_ok=True);shutil.copytree(P,OUT/'deck',dirs_exist_ok=True);(OUT/'deck/index.html').write_text(full)
    for name,digest in protected.items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest,name
    meta={'original_slides_preserved':58,'other_previous_slides_preserved':61,'replaced_previous_pages':[50,51,52,53,54],
          'figure_pages':list(range(50,58)),'total_slides':69,'baseline':baseline,'early_cnn':early,
          'protected_files_sha256':protected,'no_new_training':True,'historical_pilots_are_not_matched_comparisons':True}
    (REPORT/'verification.json').write_text(json.dumps(meta,indent=2));(OUT/'verification.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps({k:v for k,v in meta.items() if k not in ('protected_files_sha256','baseline')},indent=2))

if __name__=='__main__':run()
