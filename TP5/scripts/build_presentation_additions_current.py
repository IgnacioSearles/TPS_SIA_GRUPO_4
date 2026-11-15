"""Add six evidence-based sections to the remote 58-slide deck; touch no old file."""
from pathlib import Path
import csv
import hashlib
import json
import re
import shutil
import subprocess

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data.factory import build_dataset
from src.options import Options

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'presentation'
PREFIX='extra_actualizado_'
OUT=ROOT/'results/presentation_additions_ea4ccad'
REPORT=ROOT/'reports/presentation_additions_ea4ccad'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.labelcolor':'#182b3a','text.color':'#182b3a'})
BLUE='#176a98'; ORANGE='#d4602a'; GRAY='#5d7180'

def read_csv(path):
    with path.open(newline='') as stream:return list(csv.DictReader(stream))

def save_figure(fig,name):
    fig.savefig(P/'assets'/(PREFIX+name),dpi=150,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def image(name,alt,cls=''):
    return f'<img class="{cls}" src="assets/{PREFIX}{name}" alt="{alt}">'

def section(title,body,take,notes,cls=''):
    return f'<section class="actualizado {cls}"><h2>{title}</h2>{body}<p class="take">{take}</p><aside class="notes">{notes}</aside></section>'

def rgb(name,value):
    Image.fromarray(np.uint8(np.clip(value.reshape(32,32,3),0,1)*255)).save(P/'assets'/(PREFIX+name))

def sampling_chart():
    folder=ROOT/'reports/ej2_search/nsynth_sampling_control'
    methods=['ae_standard','ae_fitted','vae_standard']
    labels=['AE · normal estándar','Mismo AE · normal ajustada','VAE · normal estándar']
    fig,ax=plt.subplots(figsize=(8.4,4.7))
    means=[];export=[]
    for i,(method,label,color) in enumerate(zip(methods,labels,['#c85a51',ORANGE,BLUE])):
        rows=read_csv(folder/method/'samples.csv')
        assert len(rows)==256
        values=np.array([float(r['nearest_train_mse']) for r in rows])*1000
        means.append(float(values.mean()/1000))
        jitter=np.random.default_rng(20+i).uniform(-.17,.17,len(values))
        ax.scatter(values,2-i+jitter,s=13,color=color,alpha=.42,linewidths=0)
        ax.plot([values.mean()]*2,[2-i-.27,2-i+.27],color='#182b3a',lw=2.8)
        ax.text(values.mean(),2-i+.32,f'media {values.mean():.1f}'.replace('.',','),ha='center',fontsize=14)
        export.extend([{'method':method,**r} for r in rows])
    ax.set_yticks([2,1,0],labels,fontsize=13)
    ax.set_xlabel('Distancia a la nota de entrenamiento más cercana (MSE × 1000)')
    ax.set_xlim(left=0);ax.set_ylim(-.5,2.65)
    fig.tight_layout();save_figure(fig,'sorteo_256.png')
    with (REPORT/'sorteo_256.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=export[0]);writer.writeheader();writer.writerows(export)
    return means

def novelty_chart():
    directory=ROOT/'reports/ej2_search/nsynth_sampling_control/vae_standard'
    row=read_csv(directory/'samples.csv')[0]
    config=json.loads((ROOT/'results/ej2_nsynth_sampling_control_vae/base__seed=7/config.json').read_text())
    dataset=build_dataset(Options(config['dataset'],'dataset'),ROOT)
    generated=np.load(ROOT/'results/nsynth_sampling_control/vae_standard/generated.npz')['samples'][0].reshape(128,64)
    reference=dataset.samples[dataset.labels.index(row['nearest_train_label'])].reshape(128,64)
    assert np.isclose(np.mean((generated-reference)**2),float(row['nearest_train_mse']),rtol=1e-5)
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,values,title in zip(axes,[generated,reference,abs(generated-reference)],
            ['Generada por el VAE','Referencia más cercana','Diferencia absoluta']):
        im=ax.imshow(values,origin='lower',aspect='auto',extent=(0,1,0,128),
                     cmap='magma' if ax is not axes[2] else 'cividis',vmin=0,vmax=1)
        ax.set_title(title,fontsize=17);ax.set_xlabel('Tiempo (s)');ax.set_yticks([0,64,128])
    axes[0].set_ylabel('Bandas mel (grave → agudo)')
    fig.colorbar(im,ax=axes[2],fraction=.04,pad=.03,label='|Δ espectrograma| (0–1)')
    save_figure(fig,'distinto_no_novedad.png')
    (REPORT/'ejemplo_distancia.json').write_text(json.dumps({'sample':0,'nearest_label':row['nearest_train_label'],
        'mse':float(row['nearest_train_mse']),'source':'new NSynth control, seed 7; not original deck weights'},indent=2))

def failure_figure():
    values=np.load(ROOT/'results/art_search/validation/cnn4k_lowbeta_conv2_50/seeds_7_17_27/seed=27/generated.npz')['samples']
    fig,axes=plt.subplots(1,2,figsize=(12,3.8),constrained_layout=True)
    for ax,index in zip(axes,[13,61]):
        ax.imshow(values[index].reshape(32,32,3),interpolation='nearest')
        ax.set_xticks([]);ax.set_yticks([])
        ax.set_title('Fondo y color uniformes',color=BLUE,fontsize=17)
        for spine in ax.spines.values():spine.set_visible(False)
    axes[0].annotate('Curva\nasimétrica',xy=(12,11),xytext=(-10,13),color=ORANGE,fontsize=18,
                     ha='right',va='center',arrowprops={'arrowstyle':'->','color':ORANGE,'lw':2.3})
    axes[1].annotate('Lado superior\nhundido',xy=(16,10),xytext=(35,10),color=ORANGE,fontsize=18,
                     va='center',arrowprops={'arrowstyle':'->','color':ORANGE,'lw':2.3})
    save_figure(fig,'limpia_deforme.png')

def protocol_figure():
    fig,ax=plt.subplots(figsize=(12.8,4.5))
    pitches=np.arange(-1,1.001,.25)
    for y,radius,name in [(1,.5,'Omisión normal'),(0,.75,'Omisión estricta')]:
        ax.fill_betweenx([y-.20,y+.20],-radius,radius,color=ORANGE,alpha=.12)
        ax.hlines(y,-1,1,color='#a3b1ba',lw=2)
        for pitch in pitches:
            excluded=abs(pitch)<=radius+1e-9
            endpoint=abs(pitch)==1
            ax.scatter(pitch,y,s=145 if endpoint else 85,marker='x' if excluded else 'o',
                       color=ORANGE if excluded else BLUE if endpoint else GRAY,
                       linewidths=2.3,zorder=3)
        ax.text(-1.18,y,name,ha='right',va='center',fontsize=17)
        ax.text(1.13,y,f'Más cercana entrenada:\n{.75 if radius==.5 else 1:g} semitono'.replace('.',','),
                fontsize=16,va='center',color=GRAY)
    ax.annotate('Extremo que codificamos',xy=(-1,1.04),xytext=(-1,1.55),ha='center',color=BLUE,
                fontsize=15,arrowprops={'arrowstyle':'->','color':BLUE})
    ax.annotate('Extremo que codificamos',xy=(1,1.04),xytext=(1,1.55),ha='center',color=BLUE,
                fontsize=15,arrowprops={'arrowstyle':'->','color':BLUE})
    ax.text(0,-.45,'Objetivo retirado',ha='center',color=ORANGE,fontsize=16)
    ax.set_xlim(-1.6,1.72);ax.set_ylim(-.62,1.78)
    ax.set_xticks(pitches,['−1','−0,75','−0,5','−0,25','0','+0,25','+0,5','+0,75','+1'])
    ax.tick_params(axis='x',labelsize=12)
    ax.set_xlabel('Altura respecto del objetivo (semitonos)',labelpad=12)
    ax.set_yticks([])
    for side in ('left','right','top'):ax.spines[side].set_visible(False)
    fig.tight_layout();save_figure(fig,'protocolo_huecos.png')

def gmm_chart():
    gmm=read_csv(ROOT/'results/art_latent_mixture/methods.csv')
    prior=[r for r in read_csv(ROOT/'results/art_sampling_diagnostic/methods.csv') if r['method']=='prior']
    records=[('Normal estándar',sum(int(r['clean_count']) for r in prior),768)]
    for k in (1,4,8):
        rows=[r for r in gmm if int(r['components'])==k]
        assert len(rows)==3 and sum(int(r['count']) for r in rows)==768
        records.append((f'Mezcla de {k} nube'+('s' if k!=1 else ''),sum(int(r['clean_count']) for r in rows),768))
    fig,ax=plt.subplots(figsize=(8,4.4))
    for i,(name,hits,total) in enumerate(records):
        y=3-i;rate=100*hits/total
        ax.barh(y,100,color='#e5edf1',height=.56)
        ax.barh(y,rate,color=BLUE if i==0 else ORANGE,height=.56)
        ax.text(102,y,f'{rate:.1f} %\n{hits}/{total}'.replace('.',','),va='center',fontsize=15)
    ax.set_yticks([3,2,1,0],[r[0] for r in records],fontsize=14)
    ax.set_xlim(0,116);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Imágenes que pasan el control de limpieza (%)')
    fig.tight_layout();save_figure(fig,'gmm_limpieza.png')
    with (REPORT/'gmm_limpieza.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['method','clean','total']);writer.writerows(records)
    samples=np.load(ROOT/'results/art_latent_mixture/seed=7/components=8/generated.npz')['samples']
    for i in [0,64,128,192]:rgb(f'gmm_{i}.png',samples[i])

def progress_snapshot():
    states=[];fig,ax=plt.subplots(figsize=(10,4.8))
    for seed in (7,17,27):
        path=ROOT/f'results/ej2_audio_conv_conv_complete/base__seed={seed}/checkpoint.npz'
        if not path.exists():continue
        with np.load(path,allow_pickle=False) as data:meta=json.loads(str(data['metadata']))
        history=meta['history']
        selected=[r for r in history if 'mean_squared_error' in r]
        if selected:ax.plot([r['epoch'] for r in selected],[r['mean_squared_error'] for r in selected],marker='.',label=f'Semilla {seed}')
        states.append({'seed':seed,'saved_epoch':meta['epoch'],'elapsed_seconds':meta['elapsed']})
    ax.set_yscale('log');ax.set_xlabel('Época guardada');ax.set_ylabel('MSE sobre entrenamiento (escala log)')
    ax.set_title('CNN de audio: avance provisional del control completo');ax.legend()
    ax.text(.01,.01,'No mide notas omitidas ni compara todavía contra el VAE denso.',transform=ax.transAxes,fontsize=11,color=GRAY)
    fig.tight_layout();fig.savefig(REPORT/'avance_audio.png',dpi=150);plt.close(fig)
    (REPORT/'estado_audio.json').write_text(json.dumps({'expected_runs':12,
        'finished':[str(p.parent.relative_to(ROOT)) for p in (ROOT/'results').glob('ej2_audio_conv_*/*/summary.json') if 'smoke' not in str(p)],
        'checkpoints':states,'interpretation':'Training only; no architecture superiority conclusion yet'},indent=2))

def run():
    REPORT.mkdir(parents=True,exist_ok=True)
    baseline={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in P.rglob('*') if f.is_file()
              and not f.name.startswith(PREFIX) and f.name not in ('con_agregados_actualizados.html','extra_actualizado.css','AGREGADOS_ACTUALIZADOS.md')}
    original=(P/'index.html').read_text();old=re.findall(r'<section\b.*?</section>',original,re.S)
    assert len(old)==58,'Use only the reviewed ea4ccad deck'
    means=sampling_chart();novelty_chart();failure_figure();protocol_figure();gmm_chart();progress_snapshot()
    arrays=np.load(ROOT/'reports/ej2_activaciones_cnn/activaciones.npz')
    rgb('rect_original.png',arrays['references'][3]);rgb('rect_recon.png',arrays['reconstructed'][3])
    samples=np.load(ROOT/'results/art_sampling_diagnostic/seed=7/prior/generated.npz')['samples']
    for i in [0,16,32,48]:rgb(f'gen_{i}.png',samples[i])
    a=section('Otro sorteo acerca el AE a notas conocidas',
        '<p class="context">Control complementario: 256 muestras por método · nuevas reproducciones · semilla 7</p>'+
        '<div class="sampling"><div>'+image('sorteo_256.png','Las 256 distancias por método, con la media marcada')+'</div><div>'+
        '<p><b>Gaussiana:</b> una nube de probabilidad. Su centro y anchura indican dónde sorteamos los 16 números del código.</p>'+
        '<strong class="big">−45 %</strong><p>de distancia media al entrenamiento al adaptar el sorteo del mismo AE.</p></div></div>',
        'Los pesos del AE no cambian. Ajustar el sorteo después de entrenar no lo convierte en VAE.',
        'Compara tres métodos del control nuevo de NSynth, no los pesos originales del deck. Todas las 256 muestras aparecen, sin recortar valores extremos. '
        'La línea marca la media; no mezclar con las medianas de500 puntos de la página anterior. AE estándar0,0321286; mismo AE ajustado0,0175658; VAE estándar0,0170866. '
        'Mejora relativa45,3%. Se ajustan media y covarianza globales a códigos de entrenamiento, con los mismos pesos del AE. '
        'No es VAE: no aprende incertidumbre por entrada ni entrena con KL. Una sola semilla de modelos, no prueba universal ni de calidad auditiva. '
        'Fuente: reports/ej2_search/nsynth_sampling_control/{summary.csv,manifest.json,*/samples.csv}. Los pesos de las slides originales no están disponibles localmente.')
    b=section('Una salida diferente no demuestra novedad',
        '<figure>'+image('distinto_no_novedad.png','Espectrograma generado, vecino más cercano y mapa de diferencias reales')+'</figure>'+
        '<div class="questions"><p><b>Lo observado:</b> la salida no es idéntica a su referencia más cercana.</p>'+
        '<p><b>Lo que falta probar:</b> que no sea una variación de ejemplos aprendidos.</p></div>',
        'La distancia mide parecido; no certifica calidad auditiva, novedad ni ausencia de memorización.',
        'Muestra0 del VAE del control complementario de NSynth, semilla7. La referencia es la más cercana entre todas las notas de entrenamiento; se comprueba la MSE contra el CSV. '
        'Ambos espectrogramas comparten escala0–1; el tercero muestra valor absoluto de su diferencia, con barra propia. '
        'MSE positiva al vecino más próximo descarta identidad exacta en esa representación frente al conjunto comparado; no descarta memorización con cambios. '
        'Llegar a la distancia típica entre dos notas reales tampoco implica necesariamente copia. Una distancia pequeña no demuestra mala generación ni una grande demuestra novedad. '
        '500 salidas de un modelo no son500 entrenamientos independientes. No se realizó un estudio auditivo humano controlado. '
        'Esta página aclara el alcance de la evidencia sin modificar la afirmación previa ni sus notas.')
    c=section('Reconstruir una figura no garantiza generarla bien',
        '<p class="context">Extensión visual: mismo mecanismo variacional que en audio, con otro modelo y otros datos.</p>'+
        '<div class="rgb"><div><h3>Con una imagen de entrada</h3><div class="recon">'+image('rect_original.png','Rectángulo de validación')+'<span>→</span>'+image('rect_recon.png','Reconstrucción del rectángulo')+'</div>'+
        '<div class="labels"><span>Original</span><span>Reconstrucción</span></div><p>Mirar las esquinas y el contorno.</p></div>'+
        '<div><h3>Sin imagen de entrada</h3><div class="grid">'+''.join(image(f'gen_{i}.png',f'Generación del VAE, índice {i}') for i in [0,16,32,48])+
        '</div><p>Códigos sorteados → figuras nuevas.<br>¿Se reconoce alguna de las cuatro familias?</p></div></div>',
        'Círculos, elipses, triángulos y rectángulos de 32×32. No comparamos su MSE con la del audio.',
        'Modelo RGB cnn4k_lowbeta_conv2_50, semilla7; referencia reservada de validación índice3 y su reconstrucción. '
        'Generaciones desde prior:0,16,32,48 del diagnóstico de muestreo. Secuencia de índices equiespaciados, no elegidos como mejores casos. '
        'La reconstrucción sí tiene referencia; la generación libre no tiene una clase objetivo asignada. '
        'RGB usa latent8 y beta final0,02; audio16 y beta0,5. Redes independientes, sin compartir pesos. '
        'La comparación es conceptual: reconstrucción, muestreo y conservación de estructura; no un estudio causal de modalidad ni una comparación numérica de pérdidas.')
    d=section('Una imagen limpia puede estar mal dibujada',
        '<div class="definitions"><p><b>Limpieza:</b> fondo uniforme, color parejo y una región principal.</p>'+
        '<p><b>Forma correcta:</b> un contorno reconocible como círculo, elipse, triángulo o rectángulo.</p></div>'+
        '<figure>'+image('limpia_deforme.png','Dos fallos reales señalados: curva asimétrica y lado hundido')+'</figure>'+
        '<p class="context">Casos de fallos que pasan limpieza; no estiman una tasa de formas incorrectas.</p>',
        'Como en audio: acertar la altura no garantiza conservar los armónicos.',
        'Casos de validación de cnn4k_lowbeta_conv2_50, semilla27, índices13 y61. Se muestran por su deformación, no como muestra aleatoria para estimar una tasa. '
        'Las flechas son anotaciones de lectura, no salidas del modelo ni una prueba geométrica automática. '
        'Limpieza mide fondo, color, contraste, conectividad, tamaño y transición de bordes; no exige tres lados ni ángulos rectos. '
        'La generación no es condicional: no se les asigna una figura objetivo inexistente. Revisión visual conservadora, no ciega y de un evaluador. '
        'Se mejoró limpieza, pero no se cumplió el criterio visual completo. La prueba final reservada de figuras sigue sin abrirse. '
        'La analogía con audio distingue propiedades evaluadas; no equipara métricas entre dominios.')
    e=section('El hueco cambia; los extremos son los mismos',
        '<figure>'+image('protocolo_huecos.png','Exclusión normal y estricta con los mismos extremos de interpolación')+'</figure>'+
        '<p class="context">Cruces naranjas: alturas excluidas. Puntos grises: entrenamiento. Puntos azules: extremos que codificamos.</p>',
        'Cambia el hueco del entrenamiento; la interpolación sigue usando objetivo −1 y objetivo +1.',
        'Aclaración de las etiquetas de la página47 original. La exclusión normal de radio0,5 retira microtonos a distancia≤0,5, dejando como más cercanos±0,75. '
        'La estricta de radio0,75 deja como más cercanos±1. En ambas, evaluate_pitch_interpolation codifica study.note(target−1,variant) y target+1. '
        'Por tanto las etiquetas vecinas a3/4 y a1 describen disponibilidad de entrenamiento, no cambio en extremos. '
        'El punto medio se fijó antes de evaluar. Esto no cambia5/27,2/27 ni la conclusión negativa. '
        'Fuente: src/evaluation/pitch_interpolation.py y configs/ej2_pitch_microtonal_{omitted,strict_omitted}.json.')
    f=section('La GMM mejora limpieza, no asegura la forma',
        '<p class="definition"><b>GMM = mezcla de gaussianas:</b> elegimos una de varias nubes y sorteamos un código dentro.</p>'+
        '<div class="mixture"><div>'+image('gmm_limpieza.png','Proporción de imágenes limpias para cada distribución de muestreo')+'</div><div><h3>Generaciones con ocho nubes</h3><div class="grid">'+
        ''.join(image(f'gmm_{i}.png',f'Generación GMM8, semilla7, índice{i}') for i in [0,64,128,192])+
        '</div><p>Mirar esquinas y curvas:<br>la tasa no evalúa esa geometría.</p></div></div>',
        'Mismo decoder; cambia el sorteo. 98,8 % limpias no significa 98,8 % de formas correctas.',
        'Barras completas0–100%, sin eje truncado.768 muestras por método: tres VAE y256 generaciones por VAE. '
        'Prior718/768;GMM1:738;GMM4:756;GMM8:759. Son tasas de limpieza, no evaluación geométrica. '
        'Los mismos decoders se conservan; las GMM se ajustan a una muestra posterior por cada imagen de entrenamiento, sin etiquetas. '
        'No es el muestreo estándar N(0,I), no reentrena la red ni modifica MSE o KL. '
        'Las imágenes son las posiciones0,64,128,192: una de cada dos posiciones de la secuencia prefijada, cubriendo el rango256. No se seleccionan las mejores. '
        'No son un antes/después de los casos de otra página. GMM4 y8 difieren en solo3/768 salidas limpias. '
        'Ni limpieza ni cercanía al entrenamiento prueban novedad. La prueba final permanece cerrada. '
        'Fuentes: results/art_sampling_diagnostic/methods.csv y results/art_latent_mixture/methods.csv.')
    added=[a,b,c,d,e,f]
    insertions={38:[a],42:[b],46:[c,d],56:[e],57:[f]}
    full=original
    for index in sorted(insertions,reverse=True):full=full.replace(old[index],old[index]+'\n'+''.join(insertions[index]),1)
    link='<link rel="stylesheet" href="extra_actualizado.css">\n'
    full=full.replace('</head>',link+'</head>',1)
    restored=full.replace(link,'')
    for index,group in insertions.items():restored=restored.replace(old[index]+'\n'+''.join(group),old[index],1)
    assert restored==original
    (P/'con_agregados_actualizados.html').write_text(full)
    css='''/* Scoped only to new sections; original style.css is immutable. */
.reveal .actualizado .context{font-size:21px;color:var(--muted);line-height:1.3;margin-bottom:14px}
.reveal .actualizado .definition{font-size:27px;line-height:1.3;margin-bottom:20px}
.reveal .actualizado .take{font-size:26px}.reveal .actualizado h3{font-size:27px;margin:0 0 14px}
.reveal .actualizado img{display:block;margin:0;max-width:100%;object-fit:contain}
.reveal .actualizado figure{margin:0;height:410px}.reveal .actualizado figure img{width:1160px;height:100%}
.reveal .actualizado .sampling{display:grid;grid-template-columns:800px 1fr;gap:24px;align-items:center}
.reveal .actualizado .sampling img{width:800px;height:415px}.reveal .actualizado .sampling p{font-size:25px;line-height:1.3}
.reveal .actualizado .big{display:block;font-size:76px;color:var(--warm);margin-top:24px;line-height:1.05}
.reveal .actualizado .questions{display:grid;grid-template-columns:1fr 1fr;gap:40px;margin-top:16px}.reveal .actualizado .questions p{font-size:25px}
.reveal .actualizado .rgb{display:grid;grid-template-columns:1fr 1fr;gap:60px;margin-top:30px}.reveal .actualizado .rgb p{font-size:25px;margin-top:20px;line-height:1.35}
.reveal .actualizado .recon{display:grid;grid-template-columns:1fr 40px 1fr;align-items:center;gap:12px}
.reveal .actualizado .recon img{width:215px;height:215px;image-rendering:pixelated}.reveal .actualizado .recon span{font-size:44px;color:var(--accent)}
.reveal .actualizado .labels{display:flex;justify-content:space-between;font-size:23px;margin-top:8px}
.reveal .actualizado .grid{display:grid;grid-template-columns:repeat(2,130px);gap:12px}
.reveal .actualizado .grid img{width:130px;height:130px;image-rendering:pixelated}
.reveal .actualizado .definitions{display:grid;grid-template-columns:1fr 1fr;gap:40px;margin-bottom:14px}.reveal .actualizado .definitions p{font-size:26px}
.reveal .actualizado .definitions+figure{height:265px}.reveal .actualizado .definitions+figure+.context{margin-top:12px;font-size:21px}
.reveal .actualizado .mixture{display:grid;grid-template-columns:800px 1fr;gap:30px;align-items:center}
.reveal .actualizado .mixture>div>img{width:800px;height:390px}.reveal .actualizado .mixture p{font-size:24px;margin-top:18px}
.reveal .actualizado .mixture h3{font-size:24px}
.reveal .actualizado .mixture .grid{grid-template-columns:repeat(2,110px)}.reveal .actualizado .mixture .grid img{width:110px;height:110px}
'''
    (P/'extra_actualizado.css').write_text(css)
    OUT.mkdir(parents=True,exist_ok=True)
    shutil.copytree(P,OUT/'deck',dirs_exist_ok=True)
    (OUT/'deck/index.html').write_text(full)
    only=original[:original.index('<section')]+''.join(added)+original[original.rindex('</section>')+10:]
    (OUT/'deck/solo_agregados.html').write_text(only.replace('</head>',link+'</head>',1))
    all_sections=re.findall(r'<section\b.*?</section>',full,re.S)
    new_pages=[i for i,s in enumerate(all_sections,1) if s in added]
    assert len(all_sections)==64 and len(new_pages)==6
    for rel,value in baseline.items():assert hashlib.sha256((P/rel).read_bytes()).hexdigest()==value,rel
    for ref in re.findall(r'(?:src|href|data-sound)="([^"]+)"',full):
        if not ref.startswith(('http:','https:','#','data:')):assert (P/ref).is_file(),ref
    meta={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'original_slides':58,'added_slides':6,'total_slides':64,'added_pages':new_pages,
          'original_sections_byte_preserved':True,'old_files_sha256':baseline,
          'nsynth_control_means':means,'nsynth_control_count':256,
          'selection':{'nsynth_generated_index':0,'rgb_reference':3,'rgb_prior_indices':[0,16,32,48],
                       'failure_seed':27,'failure_indices':[13,61],'gmm_preview_indices':[0,64,128,192]}}
    (REPORT/'verification.json').write_text(json.dumps(meta,indent=2))
    (OUT/'verification.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps({k:v for k,v in meta.items() if k!='old_files_sha256'},indent=2))

if __name__=='__main__':run()
