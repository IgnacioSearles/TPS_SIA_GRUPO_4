"""Exporta activaciones de un VAE fijo: visualización, sin entrenar."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.data.art_capacity import file_hash,load_probe
from src.data.art_search import assert_loss_unchanged
from src.experiments.runner import load_run
from src.nn.spatial import Conv3

ROOT=Path(__file__).resolve().parents[1]


def export(root=ROOT):
    root=Path(root);assert_loss_unchanged(root)
    run=root/'results/ej2_search_art_cnn4k_lowbeta_conv2_50/base__seed=7'
    weight_hash=file_hash(run/'model.npz')
    model,_=load_run(run)
    samples,records=load_probe(root/'results/art_search/data','validation')
    indices=[0,1,2,3];chosen=2
    references=samples[indices].astype(model.precision)
    assert records[chosen]['family']=='triangle'
    reconstructed=model.reconstruct(references)
    encoder=[layer.outputs.copy() for layer in model._encoder.layers if isinstance(layer,Conv3)]
    refinement=[layer.outputs.copy() for layer in model._decoder.layers
                if isinstance(layer,Conv3) and layer.weights.shape[-2:]==(16,16)]
    assert [a.shape for a in encoder]==[(4,16,16,16),(4,8,8,32)]
    assert len(refinement)==2
    output=root/'reports/ej2_activaciones_cnn'
    output.mkdir(parents=True,exist_ok=True)
    maxima=[float(values.max()) for values in encoder]
    np.savez_compressed(output/'activaciones.npz',references=references,reconstructed=reconstructed,
                        encoder_1=encoder[0],encoder_2=encoder[1],
                        refinement_1=refinement[0],refinement_2=refinement[1])

    figure=plt.figure(figsize=(14,5.5))
    grid=figure.add_gridspec(2,10,width_ratios=[1.35,*([1]*8),.09],wspace=.16,hspace=.35)
    original=figure.add_subplot(grid[:,0])
    original.imshow(references[chosen].reshape(32,32,3),interpolation='nearest')
    original.set_title('Original RGB\nreferencia 2',fontsize=11)
    original.set_xticks([]);original.set_yticks([])
    for row,values in enumerate(encoder):
        for channel in range(8):
            axis=figure.add_subplot(grid[row,channel+1])
            shown=axis.imshow(values[chosen,:,:,channel],cmap='cividis',vmin=0,vmax=maxima[row],interpolation='nearest')
            axis.set_title(f'L{row+1}, canal {channel}',fontsize=9)
            axis.set_xticks([]);axis.set_yticks([])
        color_axis=figure.add_subplot(grid[row,9])
        figure.colorbar(shown,cax=color_axis,ticks=[0,maxima[row]])
        color_axis.tick_params(labelsize=8)
    figure.suptitle('Misma entrada; ocho primeros canales de cada capa, sin selección por apariencia',fontsize=13)
    figure.text(.54,.025,'Primera capa: 16×16; segunda: 8×8. Escala común dentro de cada capa, distinta entre capas.',
                ha='center',fontsize=10)
    figure.subplots_adjust(left=.025,right=.96,top=.86,bottom=.11)
    figure.savefig(output/'mapas_encoder.png',dpi=150);plt.close(figure)

    for level,values in enumerate(encoder,start=1):
        channels=values.shape[-1];rows=int(np.ceil(channels/8))
        figure,axes=plt.subplots(rows,8,figsize=(12,1.7*rows+1.1),squeeze=False)
        for channel,axis in enumerate(axes.ravel()):
            axis.imshow(values[chosen,:,:,channel],cmap='cividis',vmin=0,vmax=maxima[level-1],interpolation='nearest')
            axis.set_title(f'Canal {channel}',fontsize=10)
            axis.set_xticks([]);axis.set_yticks([])
        figure.suptitle(f'Encoder, capa {level}: los {channels} canales de la referencia 2')
        figure.text(.5,.015,f'Escala compartida 0 a {maxima[level-1]:.3f}. Los mapas oscuros también se conservan.',ha='center',fontsize=10)
        figure.tight_layout(rect=[0,.045,1,.93])
        figure.savefig(output/f'encoder_{level}_completo.png',dpi=150);plt.close(figure)

    figure,axes=plt.subplots(2,8,figsize=(12,4.1))
    for row,values in enumerate(refinement):
        for channel in range(8):
            axis=axes[row,channel]
            axis.imshow(values[chosen,:,:,channel],cmap='cividis',vmin=0,vmax=float(values.max()),interpolation='nearest')
            axis.set_title(f'R{row+1}, canal {channel}',fontsize=9)
            axis.set_xticks([]);axis.set_yticks([])
    figure.suptitle('Mapas internos de los dos bloques de refinamiento, 32×32')
    figure.text(.5,.02,'Escala por bloque. Un mismo número de canal no garantiza igual significado entre bloques.',ha='center',fontsize=9)
    figure.tight_layout(rect=[0,.05,1,.93]);figure.savefig(output/'mapas_refinamiento.png',dpi=150);plt.close(figure)
    assert file_hash(run/'model.npz')==weight_hash
    metadata={'source_run':str(run.relative_to(root)),'model_sha256':weight_hash,
              'config_sha256':file_hash(run/'config.json'),'code_sha256':file_hash(Path(__file__)),
              'reference_split':'validation','reference_indices':indices,'display_reference_index':chosen,
              'display_channels':[0,1,2,3,4,5,6,7],'encoder_shapes':[list(v.shape) for v in encoder],
              'encoder_common_maxima':maxima,'parameters_modified':False,'training_performed':False,
              'interpretation':'Deterministic post-ReLU feature maps. Not attention, semantic masks or proof of understanding. '
                               'Brightness comparable within each layer, not across layers.'}
    (output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (output/'README.md').write_text('''# Mapas de activación del VAE CNN

Visualización posterior al cierre del estudio; no se entrenó ni cambió el modelo.
Referencias fijas 0–3 de validación, y triángulo índice 2 para los gráficos.
El gráfico principal conserva los primeros ocho canales de cada capa; los
gráficos completos incluyen los 16 y 32 canales, incluso mapas oscuros.

Se usa una escala común por capa sobre las cuatro referencias. Un tono claro
significa activación mayor dentro de esa escala; no se comparan intensidades
entre capas. Son salidas posteriores a ReLU, no filtros, atención ni máscaras
de figuras. El refinamiento está en el decoder y se documenta por separado.

Las matrices completas, selección, escalas y hashes quedan en NPZ/manifest.
Los pesos originales conservan su hash. No se usan referencias finales.
''')
    print(json.dumps(metadata,indent=2))
    return output


if __name__=='__main__':export()
