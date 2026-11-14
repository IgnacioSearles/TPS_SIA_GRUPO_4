"""Resume el control pareado, exigiendo igual fuente y presupuesto por semilla."""
import copy
import csv
import json
from pathlib import Path

import numpy as np

from src.data.art_capacity import file_hash
from src.data.art_search import assert_loss_unchanged
from src.evaluation.context import write_csv

ROOT=Path(__file__).resolve().parents[1]
NAMES=('cnn4k_lowbeta_more50','cnn4k_lowbeta_conv2_50')


def comparable(config):
    result=copy.deepcopy(config)
    result['model'].pop('conv_refinement_layers',None)
    initial=result['initial_parameters']
    result['initial_parameters']=initial['path'] if isinstance(initial,dict) else initial
    return result


def summarize(root=ROOT):
    root=Path(root);assert_loss_unchanged(root)
    directories=[root/f'results/art_search/validation/{name}/seeds_7_17_27' for name in NAMES]
    statuses=[json.loads((p/'status.json').read_text()) for p in directories]
    rows=[];provenance=[]
    for seed in (7,17,27):
        runs=[root/f'results/ej2_search_art_{name}/base__seed={seed}' for name in NAMES]
        configs=[json.loads((p/'config.json').read_text()) for p in runs]
        assert configs[0]['model'].get('conv_refinement_layers',1)==1
        assert configs[1]['model']['conv_refinement_layers']==2
        assert comparable(configs[0])==comparable(configs[1]),'El control no iguala fuente y presupuesto'
        training=[json.loads((p/'training.json').read_text()) for p in runs]
        parent_hashes=[item['initial_parameters']['sha256'] for item in training]
        assert parent_hashes[0]==parent_hashes[1],'Los pesos padres no son idénticos'
        for directory in runs:
            provenance.append({'run':str(directory.relative_to(root)),
                               'model_sha256':file_hash(directory/'model.npz')})
        with np.load(directories[0]/f'seed={seed}/generated.npz') as a, np.load(directories[1]/f'seed={seed}/generated.npz') as b:
            np.testing.assert_array_equal(a['coordinates'],b['coordinates'])
        clean=[];metrics=[];audits=[]
        for name,directory in zip(NAMES,directories):
            with (directory/f'seed={seed}/cleanliness.csv').open() as stream:
                clean.append({int(r['sample']):r['clean']=='True' for r in csv.DictReader(stream)})
            with (directory/'runs.csv').open() as stream:
                metrics.append(next(r for r in csv.DictReader(stream) if int(r['seed'])==seed))
            audits.append(json.loads((root/f'results/art_search/validation_audit/{name}/seed={seed}/summary.json').read_text()))
        assert set(clean[0])==set(clean[1])==set(range(64))
        row={'seed':seed,'one_block_clean':sum(clean[0].values()),'two_blocks_clean':sum(clean[1].values()),
             'became_clean':sum(not clean[0][i] and clean[1][i] for i in range(64)),
             'became_unclean':sum(clean[0][i] and not clean[1][i] for i in range(64)),
             'one_block_iou':float(metrics[0]['macro_silhouette_iou']),
             'two_blocks_iou':float(metrics[1]['macro_silhouette_iou']),
             'one_block_audit_clean':audits[0]['clean_count'],
             'two_blocks_audit_clean':audits[1]['clean_count']}
        row['shared_parent_sha256']=parent_hashes[0]
        row['clean_difference_pp']=100*(row['two_blocks_clean']-row['one_block_clean'])/64
        row['audit_difference_pp']=100*(row['two_blocks_audit_clean']-row['one_block_audit_clean'])/256
        rows.append(row)
    result={'candidates':list(NAMES),'rows':rows,'sources':provenance,
            'mean_clean_difference_pp':float(np.mean([r['clean_difference_pp'] for r in rows])),
            'mean_audit_difference_pp':float(np.mean([r['audit_difference_pp'] for r in rows])),
            'interpretation':'Three matched model seeds; descriptive paired differences. Shared generation coordinates; '
                             'no independence claim across all generated images. Cleanliness is not semantic validity.'}
    output=root/'reports/ej2_search/matched_control'
    output.mkdir(exist_ok=True)
    write_csv(output/'seeds.csv',rows)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':summarize()
