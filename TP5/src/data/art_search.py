"""Configuraciones y referencias fijadas para buscar calidad sin cambiar el error."""

import copy
import json
from pathlib import Path

import numpy as np

from src.data.art_capacity import read_json, file_hash, load_probe
from src.data.factory import build_dataset
from src.evaluation.art_cleanliness import calibrate_cleanliness, cleanliness, clean_pass
from src.options import Options, ConfigurationError

TRAIN_SEED = 2026111600
PROBE_SEEDS = {'validation':2026111601, 'reserved_final':2026111602}
GENERATION_SEEDS = {'validation':2026111603, 'reserved_final':2026111604}
CANDIDATES = {
    'cnn1k_200': {'count':1024,'epochs':200},
    'cnn4k_100': {'count':4096,'epochs':100},
    'cnn4k_200': {'count':4096,'epochs':200},
    'cnn4k_lr003': {'count':4096,'epochs':200,'lr':.003},
    'cnn4k_leaky': {'count':4096,'epochs':200,'activation':'leaky_relu'},
    'cnn4k_z16': {'count':4096,'epochs':200,'latent':16},
    'cnn4k_400': {'count':4096,'epochs':400},
    'mlp4k_200': {'count':4096,'epochs':200,'architecture':'mlp'},
    'cnn4k_wide100': {'count':4096,'epochs':100,'channels':[16,32],'hidden':[256,128]},
    'cnn4k_z16_100': {'count':4096,'epochs':100,'latent':16},
    'cnn64_4k_50': {'count':4096,'epochs':50,'size':64},
    'cnn64_4k_100': {'count':4096,'epochs':100,'size':64},
    'cnn64_4k_lr003': {'count':4096,'epochs':100,'size':64,'lr':.003},
    'cnn4k_linear100': {'count':4096,'epochs':100,'output':'identity'},
    'cnn4k_lr003_100': {'count':4096,'epochs':100,'lr':.003},
    'mlp4k_wide100': {'count':4096,'epochs':100,'architecture':'mlp','hidden':[512,256]},
    'cnn4k_beta01_100': {'count':4096,'epochs':100,'beta':.1},
    'cnn4k_beta002_100': {'count':4096,'epochs':100,'beta':.02},
    'cnn4k_refine100': {'count':4096,'epochs':100,'refinement':True},
    'cnn4k_refine_beta01_100': {'count':4096,'epochs':100,'refinement':True,'beta':.1},
    'cnn4k_identity100': {'count':4096,'epochs':100,'refinement':True,'refinement_init':'identity'},
    'cnn4k_identity_beta01_100': {'count':4096,'epochs':100,'refinement':True,'refinement_init':'identity','beta':.1},
    'cnn64_tune50': {'count':4096,'epochs':50,'size':64,'source':'cnn64_4k_50','warmup':0,'lr':.0003},
    'cnn4k_wide_tune50': {'count':4096,'epochs':50,'channels':[16,32],'hidden':[256,128],
                         'source':'cnn4k_wide100','warmup':0,'lr':.0003,'beta':.1},
    'cnn4k_wide_identity_tune50': {'count':4096,'epochs':50,'channels':[16,32],'hidden':[256,128],
                         'source':'cnn4k_wide100','warmup':0,'lr':.0003,'beta':.1,
                         'refinement':True,'refinement_init':'identity','transfer':'identity_refinement'},
    'cnn4k_refined_more50': {'count':4096,'epochs':50,'channels':[16,32],'hidden':[256,128],
                         'source':'cnn4k_wide_identity_tune50','warmup':0,'lr':.0003,'beta':.1,
                         'refinement':True,'refinement_init':'identity'},
    'cnn4k_refined_beta002_50': {'count':4096,'epochs':50,'channels':[16,32],'hidden':[256,128],
                         'source':'cnn4k_wide_identity_tune50','warmup':0,'lr':.0003,'beta':.02,
                         'refinement':True,'refinement_init':'identity'},
    'cnn8k_refined25': {'count':8192,'epochs':25,'channels':[16,32],'hidden':[256,128],
                         'source':'cnn4k_wide_identity_tune50','warmup':0,'lr':.0003,'beta':.1,
                         'refinement':True,'refinement_init':'identity'},
    'cnn8k_refined_beta002_25': {'count':8192,'epochs':25,'channels':[16,32],'hidden':[256,128],
                         'source':'cnn4k_wide_identity_tune50','warmup':0,'lr':.0003,'beta':.02,
                         'refinement':True,'refinement_init':'identity'},
    'cnn4k_lowbeta_more25': {'count':4096,'epochs':25,'channels':[16,32],'hidden':[256,128],
                         'source':'cnn4k_refined_beta002_50','warmup':0,'lr':.0003,'beta':.02,
                         'refinement':True,'refinement_init':'identity'},
    'cnn4k_lowbeta0005_25': {'count':4096,'epochs':25,'channels':[16,32],'hidden':[256,128],
                         'source':'cnn4k_refined_beta002_50','warmup':0,'lr':.0003,'beta':.005,
                         'refinement':True,'refinement_init':'identity'},
    'cnn4k_lowbeta_more50': {'count':4096,'epochs':50,'channels':[16,32],'hidden':[256,128],
                         'source':'cnn4k_refined_beta002_50','warmup':0,'lr':.0003,'beta':.02,
                         'refinement':True,'refinement_init':'identity'},
    'cnn4k_lowbeta_dense50': {'count':4096,'epochs':50,'channels':[16,32],'hidden':[256,128],
                         'source':'cnn4k_refined_beta002_50','warmup':0,'lr':.0003,'beta':.02,
                         'refinement':True,'refinement_init':'identity','decoder_hidden':[128,256,256],
                         'transfer':'identity_dense','layer_index':2},
    'cnn8k_lowbeta_more25': {'count':8192,'epochs':25,'channels':[16,32],'hidden':[256,128],
                         'source':'cnn8k_refined_beta002_25','warmup':0,'lr':.0003,'beta':.02,
                         'refinement':True,'refinement_init':'identity'},
    'cnn4k_lowbeta_conv2_50': {'count':4096,'epochs':50,'channels':[16,32],'hidden':[256,128],
                         'source':'cnn4k_refined_beta002_50','warmup':0,'lr':.0003,'beta':.02,
                         'refinement':True,'refinement_init':'identity','refinement_layers':2,
                         'transfer':'identity_refinement'},
}


def assert_loss_unchanged(root):
    root=Path(root)
    local=root/'results/art_search/loss_guard.json'
    committed=root/'reports/ej2_search/loss_guard.json'
    guard=read_json(local if local.exists() else committed)
    if local.exists() and committed.exists() and guard!=read_json(committed):
        raise ConfigurationError('La protección local difiere de la registrada en Git')
    for path, expected in guard.items():
        if file_hash(root/path)!=expected:
            raise ConfigurationError(f'Cambió una fórmula protegida: {path}')
    return guard


def configuration(root, name, seeds=(7,)):
    spec=CANDIDATES[name]
    config=copy.deepcopy(read_json(root/'configs/ej2_width_art_vae_wide.json'))
    config['seeds']=list(seeds)
    size=spec.get('size',32)
    config['dataset'].update(train_count=spec['count'],test_count=64,data_seed=TRAIN_SEED,image_size=size)
    model=config['model']
    model.update(latent_size=spec.get('latent',8),activation=spec.get('activation','relu'))
    model['kl']['weight']=spec.get('beta',.5)
    model['kl']['warmup_epochs']=spec.get('warmup',50)
    if 'source' in spec:
        config['initial_parameters']=f"results/ej2_search_art_{spec['source']}/base__seed={{seed}}/model.npz"
        if 'transfer' in spec:
            config['initial_parameters']={'path':config['initial_parameters'],'transfer':spec['transfer']}
            if 'layer_index' in spec:config['initial_parameters']['layer_index']=spec['layer_index']
    if 'output' in spec:model['output_activation']=spec['output']
    if spec.get('architecture','conv')=='conv':
        model.update(architecture='conv',image_shape=[size,size,3],conv_channels=spec.get('channels',[8,16]),hidden_layers=spec.get('hidden',[128]))
        if spec.get('refinement'): model['conv_refinement']=True
        if spec.get('refinement_init'): model['conv_refinement_init']=spec['refinement_init']
        if 'refinement_layers' in spec:model['conv_refinement_layers']=spec['refinement_layers']
        if 'decoder_hidden' in spec:model['decoder_hidden_layers']=spec['decoder_hidden']
    elif 'hidden' in spec:
        model['hidden_layers']=spec['hidden']
    config['training'].update(epochs=spec['epochs'],log_every=10)
    config['training']['optimizer']['learning_rate']=spec.get('lr',.001)
    config['evaluation']={'generative':{'count':64,'test_examples':8,'audio_examples':0,'seed':1234},
                          'presentation':{'pairs':3,'steps':11}}
    return config


def write_config(root, name, seeds=(7,)):
    path=root/f'configs/ej2_search_art_{name}.json'
    expected=configuration(root,name,seeds)
    if path.exists():
        old=read_json(path)
        if {k:v for k,v in old.items() if k!='seeds'}!={k:v for k,v in expected.items() if k!='seeds'}:
            raise ConfigurationError(f'Cambió una configuración de búsqueda: {path}')
        expected['seeds']=sorted(set(old['seeds'])|set(seeds))
    path.write_text(json.dumps(expected,indent=2)+'\n',encoding='utf-8')
    return path


def prepare_search(root):
    root=Path(root)
    guard=assert_loss_unchanged(root)
    directory=root/'results/art_search/data'
    if (directory/'manifest.json').exists():
        manifest=read_json(directory/'manifest.json')
        if manifest['loss_guard']!=guard:
            raise ConfigurationError('Cambió el objetivo protegido')
        for split,item in manifest['partitions'].items():
            if file_hash(directory/f'{split}.npz')!=item['file_sha256']:
                raise ConfigurationError(f'Partición alterada: {split}')
        for name in CANDIDATES:write_config(root,name)
        return directory
    directory.mkdir(parents=True,exist_ok=True)
    base=configuration(root,'cnn4k_200')['dataset']
    train=build_dataset(Options(base,'dataset'),root)
    seen={r.tobytes() for r in np.vstack([train.samples,train.generative_study.test_samples])}
    partitions={}
    validation_images=None
    for split,seed in PROBE_SEEDS.items():
        data=build_dataset(Options({**base,'train_count':512,'test_count':4,'data_seed':seed},'probe'),root)
        samples=data.samples
        hashes={r.tobytes() for r in samples}
        if seen.intersection(hashes): raise ConfigurationError('Las referencias se solapan con entrenamiento')
        seen.update(hashes)
        records=[{**r,'label':f'search_{split}_{i:05d}'} for i,r in enumerate(data.generative_study.train_records)]
        path=directory/f'{split}.npz'
        np.savez_compressed(path,samples=samples,records_json=json.dumps(records))
        partitions[split]={'file_sha256':file_hash(path),'count':len(samples),'seed':seed}
        if split=='validation': validation_images=samples.reshape(-1,32,32,3)
    limits=calibrate_cleanliness(validation_images)
    original_rate=float(np.mean([clean_pass(cleanliness(a),limits) for a in validation_images]))
    if original_rate<.99: raise ConfigurationError('La métrica de limpieza rechaza originales válidos')
    manifest={'loss_guard':guard,'partitions':partitions,'cleanliness_limits':limits,
              'original_clean_rate':original_rate,'training_seed':TRAIN_SEED,
              'generation_seeds':GENERATION_SEEDS,
              'criteria':{'macro_iou_mean':.92,'macro_iou_min':.90,'polygon_iou_mean':.88,
                          'polygon_iou_min':.85,'clean_rate_mean':.90,'clean_rate_min':.85,
                          'manual_min_each_family_per_seed':5},
              'interpretation':'Same MSE and beta=0.5 KL; cleanliness is evaluation only. '
                               'No filtering or analytic renderer in decoded outputs.'}
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for name in CANDIDATES: write_config(root,name)
    print(f'Búsqueda preparada; originales limpios: {original_rate:.1%}',flush=True)
    return directory
