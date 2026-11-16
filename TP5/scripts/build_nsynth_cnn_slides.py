"""Append evidence-backed NSynth CNN slides to the current remote-style deck."""
from pathlib import Path
import csv,hashlib,json,re,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'presentation';R=ROOT/'reports/nsynth_cnn_slides'
PREFIX='extra_nsynth_cnn_';BLUE='#176a98';ORANGE='#d4602a';INK='#182b3a'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'text.color':INK,'axes.labelcolor':INK})

def rows(path):
    with path.open() as stream:return list(csv.DictReader(stream))
def asset(name):return f'assets/{PREFIX}{name}'
def image(name,alt,cls=''):return f'<img class="{cls}" src="{asset(name)}" alt="{alt}">'
def button(name,label='Escuchar'):return f'<button class="play" data-sound="{asset(name)}">{label}</button>'
def sec(title,body,take,notes):
    return f'<section class="nsynth-cnn-add"><h2>{title}</h2>{body}<p class="take">{take}</p><aside class="notes">{notes}</aside></section>'

def save(fig,name):
    fig.savefig(P/asset(name),dpi=165,bbox_inches='tight',facecolor='white');plt.close(fig)

def make_charts():
    per=rows(ROOT/'reports/nsynth_cnn_study/per_seed.csv')
    rec=[r for r in per if r['task']=='reconstruction']
    fig,ax=plt.subplots(figsize=(8.2,3.7),constrained_layout=True)
    for marker,seed in zip(('o','s','^'),(7,17,27)):
        values=[float(next(r['mse'] for r in rec if r['condition']==c and int(r['seed'])==seed)) for c in ('vae_mlp','vae_conv')]
        ax.plot([0,1],values,marker=marker,color='#91a5b3',lw=1.8,ms=8,label=f'Semilla {seed}')
    means=[np.mean([float(r['mse']) for r in rec if r['condition']==c]) for c in ('vae_mlp','vae_conv')]
    ax.scatter([0,1],means,color=[BLUE,ORANGE],s=160,zorder=5)
    for x,value in enumerate(means):ax.text(x-.08,.00075,f'Media {value:.5f}'.replace('.',','),ha='center',color=[BLUE,ORANGE][x],fontsize=14)
    ax.set_xticks([0,1],['VAE denso','VAE convolucional']);ax.set_xlim(-.35,1.52);ax.set_ylim(0,.018)
    ax.set_ylabel('MSE en instrumentos no vistos');ax.grid(axis='y',alpha=.18)
    ax.legend(loc='center right',frameon=False,fontsize=12)
    ax.spines[['top','right']].set_visible(False)
    save(fig,'reconstruction.png')
    # Same relative-energy distance used by the newly pulled sampling slides.
    generation=[]
    for architecture in ('mlp','conv'):
        for seed in (7,17,27):
            folder=ROOT/f'reports/nsynth_cnn_study/vae_{architecture}/seed={seed}'
            with np.load(folder/'generated.npz',allow_pickle=False) as data:
                samples=data['samples'].astype(float)
            source=rows(folder/'generation.csv')
            for i,row in enumerate(source):
                energy=float(np.mean(samples[i]**2))
                generation.append({'architecture':architecture,'seed':seed,'sample':i,
                                   'relative_difference':float(row['train_mse'])/energy,
                                   'train_mse':float(row['train_mse']),'sample_energy':energy})
    fig,ax=plt.subplots(figsize=(9.1,3.1),constrained_layout=True);jitter=np.random.default_rng(16)
    medians={}
    for position,(a,label,color) in enumerate((('mlp','Denso',BLUE),('conv','CNN',ORANGE))):
        values=np.array([r['relative_difference'] for r in generation if r['architecture']==a])
        ax.scatter(values,position+jitter.uniform(-.17,.17,len(values)),s=17,color=color,alpha=.4)
        medians[a]=float(np.median(values));ax.plot([medians[a]]*2,[position-.25,position+.25],color=INK,lw=3)
        ax.text(medians[a],position+.31,f'Mediana {medians[a]:.3f}'.replace('.',','),ha='center',fontsize=13)
    ax.set_yticks([0,1],['Denso','CNN']);ax.set_ylim(-.4,1.6);ax.invert_yaxis()
    ax.set_xlim(left=0);ax.set_xlabel('Diferencia con la nota de entrenamiento más parecida / energía de la muestra')
    ax.spines[['top','right','left']].set_visible(False);ax.grid(axis='x',alpha=.17)
    save(fig,'generation_relative.png')
    fig,ax=plt.subplots(figsize=(9.2,3.7),constrained_layout=True)
    levels=(0.,.05,.1,.2)
    for condition,method,label,color,style in [('dae_mlp','input','Entrada sin limpiar',INK,'--'),('dae_mlp','model','AE denso',BLUE,'-'),('dae_conv','model','AE convolucional',ORANGE,'-')]:
        means=[]
        for level in levels:
            values=[float(r['mse']) for r in per if r['condition']==condition and r['method']==method and float(r['noise_level'])==level]
            means.append(np.mean(values));ax.scatter([level]*len(values),values,color=color,s=20,alpha=.4)
        ax.plot(levels,means,style,color=color,lw=2.5,marker='o',label=label)
    ax.axvline(.1,color='#a3b1ba',ls=':',lw=1);ax.text(.103,.028,'Ruido de entrenamiento',fontsize=11,color='#5d7180')
    ax.set_xticks(levels,['0','0,05','0,10','0,20']);ax.set_ylim(0,.031)
    ax.set_xlabel('Desvío del ruido gaussiano sobre el espectrograma mel');ax.set_ylabel('MSE respecto de la nota limpia')
    ax.legend(loc='upper left',frameon=False,fontsize=12);ax.grid(axis='y',alpha=.17);ax.spines[['top','right']].set_visible(False)
    save(fig,'denoising.png')
    with (R/'generation_relative.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(generation[0]));writer.writeheader();writer.writerows(generation)
    return means,medians

def spec_image(values,name):
    fig,ax=plt.subplots(figsize=(3.3,1.65),constrained_layout=True)
    ax.imshow(values.reshape(128,64),origin='lower',aspect='auto',extent=[0,1,0,128],cmap='magma',vmin=0,vmax=1)
    ax.set_xticks([0,.5,1],['0','0,5','1 s']);ax.set_yticks([0,64,128]);ax.tick_params(labelsize=9)
    save(fig,name)

def make_examples():
    data=ROOT/'results/nsynth_cnn_study/data'
    manifest=json.loads((data/'manifest.json').read_text())
    labels=manifest['eval_example_labels'];selected=[0,2]
    spectra={}
    for a in ('mlp','conv'):
        with np.load(ROOT/f'reports/nsynth_cnn_study/vae_{a}/seed=7/examples.npz',allow_pickle=False) as stored:
            assert list(stored['labels'])==labels
            spectra[a]=stored['reconstructed_examples'].copy()
            if a=='mlp':spectra['codec']=stored['clean_examples'].copy()
    for i in selected:
        for a in ('codec','mlp','conv'):spec_image(spectra[a][i],f'{a}_{i}.png')
    # Audio copies are produced by the common listening export, all with PGHI.
    kinds={'original':'original','codec':'codec','mlp':'dense','conv':'cnn'}
    for i in selected:
        for key,kind in kinds.items():
            source=ROOT/f'reports/nsynth_cnn_listening/audio/pghi/seed=7/{kind}/{labels[i]}.wav'
            if not source.exists():raise FileNotFoundError(f'Waiting for shared listening export: {source}')
            shutil.copy2(source,P/asset(f'{key}_{i}.wav'))
    return labels,selected

def build():
    R.mkdir(parents=True,exist_ok=True)
    base=(P/'index.html').read_text()
    existing=re.findall(r'<section\b.*?</section>',base,re.S)
    own=[s for s in existing if 'class="nsynth-cnn-add"' in s]
    for s in own:base=base.replace(s,'')
    old=re.findall(r'<section\b.*?</section>',base,re.S)
    assert len(old)==69,'Review the new base deck before rebuilding additions'
    (R/'base_sections.json').write_text(json.dumps(old,ensure_ascii=False,indent=2))
    protected={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in P.rglob('*') if p.is_file()
               and p.name!='index.html' and not p.name.startswith(PREFIX) and p.name!='extra_nsynth_cnn.css'}
    _,medians=make_charts();labels,selected=make_examples()
    sources='Fuentes: reports/nsynth_cnn_study/per_note.csv, per_seed.csv y generation_summary.csv. '
    intro=sec('Volvemos a NSynth: convoluciones sobre el sonido',
        '<p class="intro">La pregunta de las figuras, sobre los mismos espectrogramas mel del audio.</p>'
        '<div class="two"><div><h3>Qué cambia</h3><p>Filtros aprendidos de 3 × 3 sobre tiempo y frecuencia.</p><p>Antes de aplanar, dos convoluciones conservan la vecindad local.</p></div>'
        '<div><h3>Qué mantenemos</h3><p>Latente 16, ReLU, Adam 0,001 y 200 épocas. Misma pérdida en cada par denso/CNN.</p><p>VAE para reconstruir y generar. AE con entrada deteriorada para denoising.</p></div></div>'
        '<div class="split"><b>884 notas para entrenar</b><span>224 para probar, de otros instrumentos</span></div>',
        'Mismo conjunto y partición para las dos arquitecturas. Tres semillas por condición.',
        sources+'Se usan las1108notas NSynth de10familias del ejemplo. Split nuevo por instrumento exacto:41instrumentos de entrenamiento y12 de prueba, sin compartir instrumentos entre partes. No se reutilizan los pesos viejos entrenados sobre las1108notas. El objetivo aquí es reconstrucción, generación libre y restauración del espectrograma, no encontrar notas omitidas. Las capas se implementan con NumPy y backprop manual, sin cambiar MSE ni KL. La localidad no impone relaciones armónicas. Conv3 implementa correlación cruzada sin invertir el filtro, compartiendo pesos por posición. Kernel3, padding1, stride2 en encoder y1 en decoder.')
    architecture=sec('La CNN mantiene la compresión a 16 valores',
        '<p class="intro">Dimensiones por nota: alto × ancho × canales. ReLU en las capas internas.</p>'
        '<table class="architecture"><tr><th>Encoder CNN</th><th>Decoder CNN</th></tr>'
        '<tr><td>Mel 128 × 64 × 1<br>Conv 3×3, stride 2, 8 canales<br>64 × 32 × 8<br>Conv 3×3, stride 2, 16 canales<br>32 × 16 × 16<br>Flatten 8192, Dense 512, 128</td>'
        '<td>Código 16, Dense 128, 512, 8192<br>Reshape 32 × 16 × 16<br>Upsample ×2, Conv 3×3 de 16 a 8<br>64 × 32 × 8<br>Upsample ×2, Conv 3×3 de 8 a 1<br>Sigmoid: 128 × 64 × 1</td></tr></table>'
        '<p class="detail">En el VAE: salida lineal de 32 valores, μ y log σ²; en el AE: código lineal de 16.</p>'
        '<p class="detail">Control denso: 8192, 512, 128 y la misma salida latente. Decoder espejo.</p>',
        'Upsample repite cada píxel. Las convoluciones aprenden los filtros que siguen a esa ampliación.',
        'Fuentes: src/nn/spatial.py y configs/nsynth_cnn. En el VAE, reparametrización z=mu+exp(0.5logvar)*epsilon, epsilon normalestándar16. Decoder Dense usaReLU, la últimaConv produce logits lineales y el modelo aplica sigmoid. No hay maxpool,ConvTranspose ni capas RGB de refinamiento. EnAE salidaencoder16; enVAE32parámetros de posterior. CNNVAE8537809 parámetros; densoVAE8535328, diferencia0.029%. MSE+0.5KLconwarmup50enVAE, MSEenAE. La comparación por tipo conserva datos, batch64 y pasos, pero no equipara tiempo de cómputo. Fórmula de Conv: Yij,o=bo+sum(u,v=0..2,c)Wuvco Xpad(si+u,sj+v,c).')
    reconstruction=sec('La CNN reduce un 5,5 % el error de reconstrucción',
        '<p class="intro">224 notas de instrumentos no vistos por cada modelo. Cada línea une una semilla.</p>'
        '<figure class="chart">'+image('reconstruction.png','MSE de reconstrucción pareada por semilla, con eje desde cero')+'</figure>',
        'Mejora media de 0,01570 a 0,01483. La CNN mejora en dos de las tres semillas.',
        sources+'MSE media sobre8192valores mel normalizados de cada nota, después promedio de224notas y3semillas. No es la medida de diferencia relativa a energía de las slides de generación. No son672entrenamientos independientes: hay tres por condición. Semilla17: CNN0.0148008 frente denso0.0147501, empeora ligeramente. No se afirma significación estadística ni calidad auditiva a partir de MSE. Ambas arquitecturas comparten mismo split y VAEbeta0.5warmup50.')
    listening='<p class="intro">Misma nota, semilla 7 y exportación PGHI para ambos modelos.</p><div class="listen-grid"><div></div><h3>Referencia con códec</h3><h3>VAE denso</h3><h3>VAE CNN</h3>'
    names={0:'Bajo electrónico · MIDI 48',2:'Flauta · MIDI 67'}
    for i in selected:
        listening+=f'<div class="note-label"><b>{names[i]}</b><p>Velocidad 25</p>'+button(f'original_{i}.wav','Original NSynth')+'</div>'
        for a in ('codec','mlp','conv'):
            listening+='<figure>'+image(f'{a}_{i}.png',f'{labels[i]}: {a}, mismo rango mel de0a1')+button(f'{a}_{i}.wav')+'</figure>'
    listening+='</div>'
    listening_slide=sec('La misma nota, reconstruida por los dos VAE',listening,
        'La referencia con códec permite separar la degradación de audio del error que agrega la red.',
        'Dos ejemplos ilustrativos de los seis prefijados antes de evaluar: posiciones0y2delmanifest, bass_electronic_027-048-025 yflute_acoustic_002-067-025. Todos los ejemplos y tres semillas están en la página de escucha reports/nsynth_cnn_listening/index.html. Selección por etiquetas, no por ventajaCNN. Todos los audios de comparación se exportaron desde sus espectrogramas finales con el mismo códec PGHI+60iteracionesGriffinLim e inversiónmel200. OriginalNSynth no pasa por red ni códec. La MSE de las tablas no cambia al sustituir fase aleatoria por PGHI: se mide en mel antes de la inversión. PGHI es mejora del códec, no evidencia de aprendizajeCNN. Espectrogramas mismaescala0..1, bandas128 ytiempo1segundo.')
    generation=sec('Generación: diferencias pequeñas entre los VAE',
        '<p class="intro">Los mismos 64 códigos N(0, I) por semilla. Se muestran las 192 muestras de cada arquitectura.</p>'
        '<figure class="chart">'+image('generation_relative.png','Diferencia relativa a energía para todas las generaciones densas y CNN')+'</figure>'
        '<p class="detail">Dividimos el error con la nota más cercana por la energía de la muestra generada.</p>',
        'La distancia describe cercanía. Escuchar y revisar diversidad sigue siendo necesario para valorar la generación.',
        'Fuentes: reports/nsynth_cnn_study/vae_*/seed=*/generated.npz y generation.csv. Misma normalización de scripts/presentacion/far_sampling.py: min MSE(generada,nota de entrenamiento)/media(generada²). Se divide por energía en mel de la muestra generada, no por potencia de audioWAV. Referencias884train constantes entre modelos. Se exporta cada una de192muestras en reports/nsynth_cnn_slides/generation_relative.csv. Mediana sobre192salidas, no192modelos: tresmodelos porarquitectura. No se compara este protocolo con el conjunto completo de1108notas de las diapositivas previas. Menor distancia no prueba novedad ni ausencia de memorización. La media MSE entre pares es0.03703denso y0.03532CNN: no permite concluir superioridad tímbrica. No usarla como objetivo de búsqueda.')
    denoising=sec('El denoising ayudó con ruido fuerte',
        '<p class="intro">AE entrenados con ruido gaussiano σ = 0,10 en mel. Mismas entradas deterioradas para ambos.</p>'
        '<figure class="chart">'+image('denoising.png','Curvas de MSE de entrada, AE denso y CNN según ruido mel')+'</figure>',
        'Con σ = 0,10 ambos empeoran la entrada. Con σ = 0,20, la CNN reduce su error un 40,5 %.',
        sources+'No se usa VAE para este experimento: ambos sonAEdeMSE,latente16,200épocas. Ruido gaussiano independiente añadido a valores mel yclip0..1, no ruido acústico de micrófono. Aσ0.10entradaMSE0.007413, AE denso0.016585 yCNN0.015362. Aσ0.20entrada0.027086, denso0.020476 yCNN0.016126: reducciones24.4%y40.5%vsidentidad. Sinruido yconruidopequeño la reconstrucción destruye detalles útiles. Los límites se muestran junto con el resultado favorable; no afirmar limpieza universal. Los puntos transparentes son las tres semillas y las líneas sus medias en el mismo test224notas.')
    conclusions=sec('Qué aportaron las CNN en audio',
        '<div class="conclusions"><p><b>Reconstrucción:</b> mejora media del 5,5 % en instrumentos no vistos, con una semilla ligeramente peor.</p>'
        '<p><b>Generación:</b> distancias próximas y muestras para escuchar. La métrica sola no alcanza para valorar el sonido.</p>'
        '<p><b>Denoising:</b> mejora con deterioro fuerte, pero empeora las entradas con menos ruido.</p>'
        '<p><b>Notas omitidas:</b> en la prueba sintética aparte, la CNN no logró altura y espectro correctos: 0/27; el denso, 3/27.</p></div>',
        'La arquitectura espacial ayuda en algunas tareas. La interpolación musical ausente sigue abierta.',
        sources+'La conclusión conecta tareas y modalidades, sin sumar mejoras de figuras con sonido ni confundir reconstrucción con generación o interpolación. Los resultados de notas omitidas corresponden a otro dataset sintético, no a las224notasNSynth. No hay cambio en fórmula de error delAE/VAE. CalentamientoKL se mantiene fijoenelcontrasteprincipal; una ablación0vs50con200épocas y3semillas no mostró ventaja consistente delwarmup, y se amplió a6semillasparacondición con réplicas independientes, todavía pendientesalprepararestebloque. Los resultados nuevos de esa réplica deben actualizarse después, sin anticiparlos.')
    additions=[intro,architecture,reconstruction,listening_slide,generation,denoising,conclusions]
    anchor=next(s for s in old if 'La mejora fue parcial; la lección vuelve al audio' in s)
    updated=base.replace(anchor,anchor+'\n\n'+'\n\n'.join(additions),1)
    if 'extra_nsynth_cnn.css' not in updated:updated=updated.replace('</head>','<link rel="stylesheet" href="extra_nsynth_cnn.css">\n</head>')
    (P/'index.html').write_text(updated)
    css='''/* Only NSynth CNN additions. Existing layouts remain unchanged. */
.reveal .nsynth-cnn-add .intro{font-size:24px;color:var(--muted);line-height:1.3;margin:0 0 22px}
.reveal .nsynth-cnn-add .two{display:grid;grid-template-columns:1fr 1fr;gap:60px;margin:28px 0}
.reveal .nsynth-cnn-add h3{font-size:28px;color:var(--accent);margin:0 0 14px}
.reveal .nsynth-cnn-add .two p{font-size:29px;line-height:1.3;margin:0 0 24px}
.reveal .nsynth-cnn-add .split{border-top:1px solid var(--rule);padding-top:20px;font-size:28px;display:flex;gap:36px}
.reveal .nsynth-cnn-add .split b{color:var(--accent)}
.reveal .nsynth-cnn-add .architecture{width:100%;font-size:23px;line-height:1.4;margin:6px 0 15px;border-collapse:collapse}
.reveal .nsynth-cnn-add td,.reveal .nsynth-cnn-add th{vertical-align:top;padding:10px 18px;border-bottom:1px solid var(--rule)}
.reveal .nsynth-cnn-add th{font-size:27px;color:var(--accent);text-align:left}
.reveal .nsynth-cnn-add .detail{font-size:22px;line-height:1.3;margin:12px 0;color:var(--muted)}
.reveal .nsynth-cnn-add .chart{margin:0;height:380px;display:flex;align-items:center;justify-content:center}
.reveal .nsynth-cnn-add .chart img{width:100%;height:100%;object-fit:contain;max-height:none;margin:0}
.reveal .nsynth-cnn-add .listen-grid{display:grid;grid-template-columns:190px repeat(3,1fr);gap:18px 16px;align-items:center;margin-top:18px}
.reveal .nsynth-cnn-add .listen-grid h3{text-align:center;font-size:23px;margin:0}
.reveal .nsynth-cnn-add .listen-grid figure{margin:0;text-align:center}
.reveal .nsynth-cnn-add .listen-grid img{width:100%;height:145px;object-fit:contain;margin:0;max-height:none}
.reveal .nsynth-cnn-add .note-label{font-size:23px;line-height:1.3}
.reveal .nsynth-cnn-add .note-label p{font-size:20px;color:var(--muted);margin:12px 0}
.reveal .nsynth-cnn-add .conclusions p{font-size:29px;line-height:1.35;margin:30px 0}
.reveal .nsynth-cnn-add .take{font-size:26px;line-height:1.28}
'''
    (P/'extra_nsynth_cnn.css').write_text(css)
    sections=re.findall(r'<section\b.*?</section>',updated,re.S)
    assert [s for s in sections if 'class="nsynth-cnn-add"' not in s]==old
    for name,digest in protected.items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest,name
    record={'base_commit':'9744710','base_sections':len(old),'new_sections':len(additions),'slides':len(sections),
            'all_previous_sections_and_notes_exact':True,'protected_files_unchanged':len(protected),
            'added_pages':list(range(58,65)),'same_audio_phase':'pghi','generation_relative_medians':medians,
            'source_base_sha256':hashlib.sha256(base.encode()).hexdigest()}
    (R/'verification.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))

if __name__=='__main__':build()
