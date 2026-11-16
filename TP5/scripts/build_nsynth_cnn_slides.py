"""Two slides only: CNN protocol and the complete denoising comparison."""
from pathlib import Path
import csv,hashlib,json,re,shutil,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'presentation';R=ROOT/'reports/nsynth_cnn_slides_compact'
PREFIX='extra_nsynth_denoise_'
BLUE='#176a98';ORANGE='#d4602a';INK='#182b3a'

def asset(name):return f'assets/{PREFIX}{name}'
def img(name,alt):return f'<img src="{asset(name)}" alt="{alt}">'
def btn(name,label='Escuchar'):return f'<button class="play" data-sound="{asset(name)}">{label}</button>'
def sec(title,body,take,notes):
    return f'<section class="nsynth-denoise-add"><h2>{title}</h2>{body}'+(f'<p class="take">{take}</p>' if take else '')+f'<aside class="notes">{notes}</aside></section>'

def spectrum(values,name):
    fig,ax=plt.subplots(figsize=(2.3,1.6))
    ax.imshow(values.reshape(128,64),origin='lower',aspect='auto',vmin=0,vmax=1,cmap='magma')
    ax.axis('off');fig.subplots_adjust(0,0,1,1)
    fig.savefig(P/asset(name),dpi=140);plt.close(fig)

def chart():
    with (ROOT/'reports/nsynth_denoising_noise_sweep/per_seed.csv').open() as stream:rows=list(csv.DictReader(stream))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'text.color':INK,'axes.labelcolor':INK})
    fig,ax=plt.subplots(figsize=(7.6,3.15),constrained_layout=True)
    levels=(0.,.05,.1,.2,.3,.4,.5)
    definitions=[('dae_mlp','input','Entrada sin limpiar',INK,'--'),('dae_mlp','model','Denso',BLUE,'-'),('dae_conv','model','CNN',ORANGE,'-')]
    output=[]
    for condition,method,label,color,style in definitions:
        means=[]
        for level in levels:
            selected=[r for r in rows if r['condition']==condition and r['method']==method and float(r['noise_level'])==level]
            assert sorted(int(r['seed']) for r in selected)==[7,17,27]
            values=[float(r['mse']) for r in selected];means.append(np.mean(values))
            ax.scatter([level]*3,values,color=color,s=18,alpha=.35,zorder=3)
            output.extend(selected)
        ax.plot(levels,means,style,color=color,marker='o',lw=2.2,ms=5,label=label)
    ax.set_xticks(levels,['0','0,05','0,10','0,20','0,30','0,40','0,50']);ax.set_ylim(0,.135)
    ax.set_xlabel('Desvío del ruido mel (σ)');ax.set_ylabel('MSE frente a la nota limpia')
    ax.legend(frameon=False,loc='upper left',fontsize=12)
    ax.grid(axis='y',alpha=.18);ax.spines[['top','right']].set_visible(False)
    fig.savefig(P/asset('all_noise_levels.png'),dpi=170,bbox_inches='tight',facecolor='white');plt.close(fig)
    with (R/'plotted_values.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)

def build():
    R.mkdir(parents=True,exist_ok=True)
    source=(P/'index.html').read_text()
    matches=list(re.finditer(r'<section\b.*?</section>',source,re.S))
    original=[m.group() for m in matches if not any(x in m.group() for x in ('class="nsynth-cnn-add"','class="nsynth-denoise-add"'))]
    remote=subprocess.check_output(['git','show','9744710:TP5/presentation/index.html'],cwd=ROOT,text=True)
    assert original==re.findall(r'<section\b.*?</section>',remote,re.S),'Unrelated slides changed; review before rebuilding'
    base=source
    for m in reversed(matches):
        if any(x in m.group() for x in ('class="nsynth-cnn-add"','class="nsynth-denoise-add"')):
            base=base[:m.start()]+base[m.end():]
    base=base.replace('<link rel="stylesheet" href="extra_nsynth_cnn.css">\n','')
    protected={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in P.rglob('*')
               if p.is_file() and p.name!='index.html' and not p.name.startswith(PREFIX) and p.name!='extra_nsynth_denoise.css'}
    chart()
    folder=ROOT/'reports/nsynth_cnn_study/dae_conv/seed=7'
    with np.load(folder/'examples.npz',allow_pickle=False) as data:
        assert str(data['labels'][0])=='bass_electronic_027-048-025'
        spectrum(data['noisy_examples'][0],'input.png')
        spectrum(data['denoised_examples'][0],'output.png')
    label='bass_electronic_027-048-025'
    for level in (.1,.5):
        for kind in ('clean','noisy','dense','cnn'):
            filename=ROOT/f'reports/nsynth_denoising_listening/audio/seed=7/sigma={level}/{kind}/{label}.wav'
            assert filename.is_file(),filename
            shutil.copy2(filename,P/asset(f'{kind}_{level}.wav'))
    architecture=sec('Probamos CNN para quitar ruido',
        '<p class="compact-intro">Entrenamos para recuperar la nota limpia a partir de una entrada con ruido.</p>'
        '<div class="cnn-flow">'
        '<figure>'+img('input.png','Espectrograma mel deteriorado del bajo prefijado')+'<figcaption>Entrada con ruido</figcaption></figure>'
        '<div class="operation"><span>Encoder</span><b>Conv 3 × 3</b><small>8 y 16 canales</small></div>'
        '<div class="bottleneck"><b>16</b><span>valores</span></div>'
        '<div class="operation"><span>Decoder</span><b>Upsample + Conv</b><small>Reconstruye el mel</small></div>'
        '<figure>'+img('output.png','Salida del AE convolucional para el mismo ejemplo')+'<figcaption>Salida de la CNN</figcaption></figure>'
        '</div>'
        ''
        '<p class="protocol">884 notas de entrenamiento y 224 de prueba.</p>',
        'Denso y CNN reciben exactamente el mismo ruido.',
        'Propósito: evaluar si localidad y pesos compartidos mejoran tareas del audio NSynth ya presentado. '
        'Los filtros se aprenden por backpropagation. Conv3 es correlación cruzada3×3,padding1,stride2enencoder y1endecoder. '
        'Encoder:128×64×1→Conv8→64×32×8→Conv16→32×16×16→Flatten8192→Dense512→128. '
        'EnAE salida lineal16; enVAE salida32para media ylogvar de16dimensiones y reparametrización. '
        'Decoder:16→Dense128→512→8192→reshape32×16×16→Upsample2→Conv16a8→Upsample2→Conv8a1→sigmoid. '
        'ReLU interna, float32, Adam0.001,batch64,200épocas. VAE:MSE+0.5KL,warmup50;AE:MSE. '
        'Mismo conjunto1108notas de10familias,split porinstrumento:41train y12test; no hay instrumento común. '
        'No se modificó ninguna fórmula de pérdida. El diagrama ilustra el primer bajo prefijado de test, σ0.1,semilla7; no es selección del mejor caso. '
        'En esta versión se omiten las diapositivas de resultados de reconstrucción y generación porque las diferencias fueron modestas; '
        'siguen documentadas en reports/nsynth_cnn_study. No se implica que denoising resuelva las notas omitidas.')
    audio='<div class="audio-caption"><span>El mismo bajo · MIDI 48 · semilla 7</span>'+btn('clean_0.1.wav','Referencia limpia')+'</div>'
    audio+='<table class="audio-matrix"><tr><th>Ruido mel</th><th>Entrada con ruido</th><th>Denso</th><th>CNN</th></tr>'
    for level in (.1,.5):
        audio+='<tr><th>σ = '+f'{level:.2f}'.replace('.',',')+'</th>'+''.join('<td>'+btn(f'{kind}_{level}.wav')+'</td>' for kind in ('noisy','dense','cnn'))+'</tr>'
    audio+='</table>'
    result=sec('La CNN reduce el error con ruido fuerte',
        '<p class="compact-intro">Evaluación de σ = 0 a 0,50.</p>'
        '<div class="denoise-evidence"><figure>'+img('all_noise_levels.png','Error de entrada, AE denso y CNN para siete niveles de ruido y todas las semillas')+'</figure>'
        '<div class="findings"><p><b>σ = 0,10</b><span>Ambos empeoran<br>la entrada.</span></p>'
        '<p><b>σ = 0,50</b><span class="big">−76,1 %</span><span>error frente a la entrada</span></p></div></div>'
        +audio,
        '',
        'Fuentes: reports/nsynth_denoising_noise_sweep/per_seed.csv yper_note.csv. Nuevos niveles0.3,0.4,0.5 sin reentrenar. Aσ0.5: entrada0.118569,denso0.062864,CNN0.028396, reducción76.1%vsentrada. Se usan dosAE, noVAE, en denoising.  Gráfico completo con niveles0,0.05,0.1,0.2,0.3,0.4,0.5; '
        'puntos transparentes:tressemillas ylíneas:medias sobre224notas porsemilla. No son672entrenamientos independientes. '
        'Ruido gaussiano con desvíoσ sobrevalores mel normalizados,clip0..1. Los dos modelos reciben exactamente la misma entrada porsemilla y nivel. '
        'Baseline: devolver la entrada sin procesarla. MSE aσ0.1:entrada0.007413,denso0.016585,CNN0.015362; '
        'aσ0.2:entrada0.027086,denso0.020476,CNN0.016126. Las reducciones24.4%y40.5%se calculan respecto de la entrada, no entre redes. '
        'La CNN mejora frente al denso pero con ruido bajo ambos pierden detalles útiles y empeoran frente a la identidad. '
        'Los audios son el primer bajo prefijado,bass_electronic_027-048-025,semilla7; no se eligió por ventaja deCNN. '
        'Referencia limpia,entrada ruidosa y ambas salidas se exportan conPGHI+60GriffinLim e inversiónmel200. '
        'La página reports/nsynth_denoising_listening/index.html permite escuchar las seis notas fijas,los cuatro niveles ylas tressemillas. '
        'No hay efecto de fase desigual entre arquitecturas ni afirmación de calidad auditiva a partir de MSE sola.')
    anchor=next(s for s in original if 'No logramos generar una nota que no vio' in s)
    updated=base.replace(anchor,anchor+'\n\n'+architecture+'\n\n'+result,1)
    if 'extra_nsynth_denoise.css' not in updated:updated=updated.replace('</head>','<link rel="stylesheet" href="extra_nsynth_denoise.css">\n</head>')
    css='''/* Only the two compact CNN/denoising slides. */
.reveal .nsynth-denoise-add .compact-intro{font-size:24px;color:var(--muted);margin:0 0 20px;line-height:1.3}
.reveal .nsynth-denoise-add .cnn-flow{display:grid;grid-template-columns:180px 235px 100px 250px 180px;gap:30px;align-items:center;margin:50px 0 44px}
.reveal .nsynth-denoise-add .cnn-flow figure{margin:0;text-align:center}
.reveal .nsynth-denoise-add .cnn-flow img{width:180px;height:150px;object-fit:contain;max-height:none;margin:0}
.reveal .nsynth-denoise-add figcaption{font-size:23px;margin-top:14px;color:var(--muted)}
.reveal .nsynth-denoise-add .operation{display:flex;flex-direction:column;gap:12px;position:relative;font-size:27px;line-height:1.2;text-align:center}
.reveal .nsynth-denoise-add .operation::before{content:'→';position:absolute;left:-30px;top:36px;color:var(--faint)}
.reveal .nsynth-denoise-add .operation::after{content:'→';position:absolute;right:-30px;top:36px;color:var(--faint)}
.reveal .nsynth-denoise-add .operation span{color:var(--muted);font-size:24px}
.reveal .nsynth-denoise-add .operation small{font-size:21px;color:var(--muted)}
.reveal .nsynth-denoise-add .bottleneck{text-align:center;display:flex;flex-direction:column;color:var(--accent)}
.reveal .nsynth-denoise-add .bottleneck b{font-size:70px;line-height:1}.reveal .nsynth-denoise-add .bottleneck span{font-size:23px}
.reveal .nsynth-denoise-add .tasks{display:flex;justify-content:space-between;border-top:1px solid var(--rule);padding-top:24px;font-size:29px}
.reveal .nsynth-denoise-add .tasks b{color:var(--accent)}
.reveal .nsynth-denoise-add .protocol{font-size:22px;color:var(--muted);margin-top:30px}
.reveal .nsynth-denoise-add .denoise-evidence{display:grid;grid-template-columns:850px 270px;gap:30px;align-items:center;height:290px}
.reveal .nsynth-denoise-add .denoise-evidence figure{margin:0;height:290px}
.reveal .nsynth-denoise-add .denoise-evidence img{width:850px;height:290px;object-fit:contain;max-height:none;margin:0}
.reveal .nsynth-denoise-add .findings p{font-size:23px;line-height:1.25;margin:0 0 20px}
.reveal .nsynth-denoise-add .findings b,.reveal .nsynth-denoise-add .findings span{display:block}
.reveal .nsynth-denoise-add .findings b{color:var(--muted);font-size:23px;margin-bottom:6px}
.reveal .nsynth-denoise-add .findings .big{font-size:42px;color:var(--accent);line-height:1.1}
.reveal .nsynth-denoise-add .audio-caption{display:flex;gap:24px;align-items:center;font-size:22px;margin:13px 0 8px}
.reveal .nsynth-denoise-add .audio-caption button{margin:0}
.reveal .nsynth-denoise-add .audio-matrix{width:100%;font-size:21px;border-collapse:collapse;margin:0;text-align:center}
.reveal .nsynth-denoise-add .audio-matrix th{font-weight:400;color:var(--muted);padding:3px 10px;text-align:center}
.reveal .nsynth-denoise-add .audio-matrix td{padding:2px 10px}
.reveal .nsynth-denoise-add .audio-matrix button{margin:0}
.reveal .nsynth-denoise-add .take{font-size:25px;line-height:1.25}
'''
    (P/'index.html').write_text(updated);(P/'extra_nsynth_denoise.css').write_text(css)
    sections=re.findall(r'<section\b.*?</section>',updated,re.S)
    assert len(sections)==71 and [s for s in sections if 'class="nsynth-denoise-add"' not in s]==original
    for name,digest in protected.items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest,name
    record={'base_sections':69,'slides':71,'replacement_sections':2,'added_pages':[50,51],
            'old_seven_slide_block_removed':True,'all_other_sections_notes_and_order_exact':True,
            'new_order':'audio through omitted notes -> CNN protocol -> complete denoising -> figures -> conclusions -> appendix',
            'noise_levels':[0,.05,.1,.2,.3,.4,.5],'same_audio_phase':'pghi','fixed_audio_example':label,'seed':7,
            'all_previous_assets_preserved':True}
    (R/'verification.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))

if __name__=='__main__':build()
