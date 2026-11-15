"""Add exact architecture documentation without modifying any prior slide."""
from pathlib import Path
import re, json, hashlib

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'presentation'
def slide(title,body,take,notes):
    return f'<section class="arquitectura-formal"><h2>{title}</h2>{body}<p class="take">{take}</p><aside class="notes">{notes}</aside></section>'

formal=slide('Convolución 2D: definición y parámetros', '''
<p class="intro">Cada filtro combina una región local y comparte sus pesos en todas las posiciones.</p>
<div class="formula">Y<sub>i,j,o</sub> = b<sub>o</sub> + ∑<sub>u,v ∈ {0,1,2}</sub> ∑<sub>c=1…Cᵢₙ</sub> W<sub>u,v,c,o</sub> X̃<sub>si+u,sj+v,c</sub></div>
<p>X̃: entrada con padding cero. W: filtros aprendidos. b: un sesgo por canal de salida.</p>
<table><tr><th>Parámetro</th><th>Implementación usada</th></tr><tr><td>Kernel y padding</td><td>3 × 3, un píxel en cada borde</td></tr><tr><td>Stride s</td><td>2 en encoder, 1 en decoder</td></tr><tr><td>Salida espacial</td><td>H′ = ⌊(H + 2p − k) / s⌋ + 1, análogo para W′</td></tr></table>''',
'Con stride 2 reducimos la resolución. Los filtros y sus pesos se aprenden mediante backpropagation.',
'Fuente: src/nn/spatial.py, clase Conv3. Se omite el índice de minibatch. El código implementa correlación cruzada, llamada convolución en redes neuronales: no invierte el kernel. W tiene forma 3×3×Cin×Cout. Parámetros: (9 Cin + 1) Cout. ReLU después de convoluciones internas. Los logits de salida usan identidad y el modelo aplica sigmoid. No usamos max pooling ni convolución transpuesta. La localidad y el peso compartido favorecen patrones espaciales, pero no imponen reglas de geometría.')

enc=slide('Figuras: estructuras exactas del encoder', '''
<p class="intro">Dimensiones por ejemplo. H × W × C significa alto × ancho × canales.</p>
<table class="enc"><tr><th>Denso inicial</th><th>Primera CNN</th><th>CNN ancha y refinada</th></tr>
<tr><td>RGB 32 × 32 × 3<br>Flatten: 3072</td><td>RGB 32 × 32 × 3<br>Conv 3×3, s2, 8 canales<br>16 × 16 × 8</td><td>RGB 32 × 32 × 3<br>Conv 3×3, s2, 16 canales<br>16 × 16 × 16</td></tr>
<tr><td>Dense 256<br>Dense 128</td><td>Conv 3×3, s2, 16 canales<br>8 × 8 × 16<br>Flatten 1024, Dense 128</td><td>Conv 3×3, s2, 32 canales<br>8 × 8 × 32<br>Flatten 2048, Dense 256, 128</td></tr>
<tr><td colspan="3">Salida lineal de 16 valores: μ ∈ ℝ⁸ y log σ² ∈ ℝ⁸<br>z = μ + exp(½ log σ²) ⊙ ε, con ε ∼ N(0, I₈)</td></tr></table><p>ReLU después de cada capa interna. Las cabezas μ y log σ² son lineales.</p>''',
'Las tres variantes mantienen el mecanismo variacional. Cambian las capas que calculan sus parámetros.',
'Fuentes: results/ej2_width_art_vae_wide/base__seed=7/config.json; results/ej2_search_art_cnn4k_100/base__seed=7/config.json; results/ej2_search_art_cnn4k_wide100/base__seed=7/config.json; results/ej2_search_art_cnn4k_lowbeta_conv2_50/base__seed=7/config.json. ReLU en capas internas, cabezas lineales, precisión float32. La comparación histórica también cambia datos y presupuesto: no permite atribuir toda la diferencia a las convoluciones. La CNN ancha y refinada comparte encoder, no todos sus hiperparámetros.')

dec=slide('Figuras: decoder y refinamiento aprendido', '''
<table><tr><th>Denso inicial</th><th>CNN ancha / refinada</th></tr>
<tr><td>z: 8<br>Dense 128, Dense 256<br>Dense 3072 lineal<br>Sigmoid, RGB 32 × 32 × 3</td><td>z: 8<br>Dense 128, 256, 2048<br>Reshape 8 × 8 × 32<br>Upsample ×2, Conv 3×3: 32 a 16 canales<br>16 × 16 × 16<br>Upsample ×2: 32 × 32 × 16</td></tr>
<tr><td>Primera CNN<br>Dense 128, 1024<br>Reshape 8 × 8 × 16<br>Upsample, Conv 16 a 8<br>Upsample, Conv 8 a 3, Sigmoid</td><td><b>Refinamiento: 0, 1 o 2 Conv 3×3 de 16 a 16</b><br>Conv 3×3 final de 16 a 3 canales, lineal<br>Sigmoid: RGB 32 × 32 × 3</td></tr></table><p>ReLU interna. Upsample repite cada píxel 2× por eje y no tiene pesos.</p>''',
'El control pareado agrega una segunda capa de refinamiento: 2320 parámetros, entrenados con la misma pérdida.',
'Fuente: src/nn/spatial.py, spatial_networks. Upsample2 repite cada píxel 2× por eje y no tiene parámetros. Todas las capas Dense del decoder previas a logits y las convoluciones internas usan ReLU. El decoder básico tiene dos convoluciones, el refinado una o dos adicionales. En la CNN ancha las Dense son 128,256,2048. En la primera CNN son 128,1024. El control una vs dos capas parte del mismo checkpoint por semilla y entrena 50 épocas con Adam 0.0003 y beta 0.02, batch64. La capa nueva empieza como identidad. La pérdida sigue MSE más beta KL. Limpieza mejora de175/192 a182/192, sin garantizar formas correctas.')

audio=slide('Audio: arquitectura de la comparación CNN', '''
<p class="intro">Entrada: espectrograma mel 128 × 64 × 1. Latente de 16 dimensiones en ambos modelos.</p>
<table><tr><th>Encoder CNN</th><th>Decoder CNN</th></tr>
<tr><td>Conv 3×3, s2: 1 a 8 canales<br>64 × 32 × 8<br>Conv 3×3, s2: 8 a 16 canales<br>32 × 16 × 16<br>Flatten 8192, Dense 512, 128<br>Dense 32 lineal: μ y log σ²</td><td>z: 16, Dense 128, 512, 8192<br>Reshape 32 × 16 × 16<br>Upsample ×2, Conv 16 a 8<br>64 × 32 × 8<br>Upsample ×2, Conv 8 a 1 lineal<br>Sigmoid: mel 128 × 64 × 1</td></tr></table>
<p class="detail">Control denso: 8192, 512, 128, 32 en encoder y 16, 128, 512, 8192 en decoder.</p>''',
'Ambos usan MSE + 0,5 KL, warmup de 50 épocas y 200 épocas. La evaluación decidirá si mejora la interpolación.',
'Fuentes: configs/audio_conv/ej2_audio_conv_{mlp,conv}_{complete,omitted}.json y src/nn/spatial.py. ReLU interna, cabezas lineales, float32, Adam0.001, batch64. El eje mel representa frecuencia y el otro tiempo. No usamos capas de refinamiento RGB en audio. Cuatro condiciones por tres semillas7,17,27:12 corridas. MLP8535328 parámetros, CNN8537809. Mantener separado el número de épocas y beta de audio respecto del estudio RGB. No afirmar mejora hasta disponer de evaluación comparable. No cambia la representación mel ni la pérdida matemática.')

base=(P/'con_agregados_hilo_completo.html').read_text()
old=list(re.finditer(r'<section\b.*?</section>',base,re.S)); assert len(old)==69
assert 'Convolución: aprender' in old[51].group()
# Insert three pages after the original convolution introduction, and audio at the end.
new=base[:old[-1].end()]+audio+base[old[-1].end():]
at=old[51].end();new=new[:at]+formal+enc+dec+new[at:]
new=new.replace('</head>','<link rel="stylesheet" href="extra_arquitecturas.css"></head>')
(P/'con_agregados_arquitecturas.html').write_text(new)
(P/'extra_arquitecturas.css').write_text('''
.reveal .arquitectura-formal{text-align:left}
.reveal .arquitectura-formal h2{font-size:43px}
.reveal .arquitectura-formal p{font-size:24px;line-height:1.3;margin:14px 0}
.reveal .arquitectura-formal .intro{color:var(--muted);font-size:25px;margin-bottom:22px}
.reveal .arquitectura-formal table{width:100%;font-size:23px;line-height:1.4;border-collapse:collapse;margin:12px 0}
.reveal .arquitectura-formal td,.reveal .arquitectura-formal th{padding:12px 18px;border-bottom:1px solid #dbe4e9;vertical-align:top}
.reveal .arquitectura-formal th{color:#176a98;font-size:26px}
.reveal .arquitectura-formal .enc{font-size:22px}
.reveal .arquitectura-formal .formula{font-size:33px;background:#eef4f7;padding:22px 18px;margin:24px 0;white-space:nowrap}
.reveal .arquitectura-formal .formula sub,.reveal .arquitectura-formal .formula sup{font-size:55%}
.reveal .arquitectura-formal .take{font-size:25px;line-height:1.25}
''')
sections=re.findall(r'<section\b.*?</section>',new,re.S)
remaining=[s for s in sections if 'class="arquitectura-formal"' not in s]
assert remaining==[m.group() for m in old]
report=ROOT/'reports/figure_architectures';report.mkdir(parents=True,exist_ok=True)
(report/'verification.json').write_text(json.dumps({'slides':len(sections),'all_previous_69_unchanged':True,'added_pages':[53,54,55,73],'base_sha256':hashlib.sha256(base.encode()).hexdigest()},indent=2))
print('Created 73 slides; all previous 69 preserved exactly')
