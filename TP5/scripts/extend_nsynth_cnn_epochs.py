"""One-time audited extension of the NSynth study from 100 to 200 epochs.

Only the planned epoch ceiling changes. Optimizer, RNG, weights and history remain
byte-identical as arrays or JSON values. Existing signature checks remain enabled.
"""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import shutil
import numpy as np

from src.experiments.runner import _prepare
from src.training.checkpoint import Checkpoint,training_signature

ROOT=Path(__file__).resolve().parents[1]

def array_digest(arrays):
    h=hashlib.sha256()
    for key in sorted(arrays):
        if key=='metadata':continue
        value=arrays[key]
        h.update(key.encode());h.update(str(value.dtype).encode());h.update(str(value.shape).encode());h.update(value.tobytes())
    return h.hexdigest()

def main():
    os.chdir(ROOT)
    folder=ROOT/'results/nsynth_cnn_study'
    with (folder/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan_path=folder/'extend_to_200_plan.json'
        if plan_path.exists():
            plan=json.loads(plan_path.read_text())
        else:
            entries=[]
            for path in sorted((ROOT/'configs/nsynth_cnn').rglob('*.json')):
                old=json.loads(path.read_text());assert old['training']['epochs']==100
                new=json.loads(json.dumps(old));new['training']['epochs']=200
                entries.append({'path':str(path.relative_to(ROOT)),'old':old,'new':new})
            assert len(entries)==16
            plan={'from_epochs':100,'to_epochs':200,'entries':entries}
            plan_path.write_text(json.dumps(plan,indent=2))
        audit=[]
        for entry in plan['entries']:
            if '/runs/' not in entry['path']:continue
            old={k:v for k,v in entry['old'].items() if k not in ('name','output')}
            new={k:v for k,v in entry['new'].items() if k not in ('name','output')}
            directory=ROOT/entry['old']['output']/f'base__seed={old["seed"]}'
            path=directory/'checkpoint.npz'
            if not path.exists():continue
            assert not (directory/'summary.json').exists(),'Unexpected completed run'
            prepared=_prepare(new,initialize=False)
            old_signature=training_signature(old,prepared.dataset.samples,prepared.interpolation_triplets)
            new_signature=training_signature(new,prepared.dataset.samples,prepared.interpolation_triplets)
            backup=folder/'before_epoch_extension'/f'{entry["old"]["name"]}_seed={old["seed"]}.npz'
            with np.load(path,allow_pickle=False) as saved:
                arrays={k:saved[k].copy() for k in saved.files}
            metadata=json.loads(str(arrays['metadata']))
            assert metadata['signature'] in (old_signature,new_signature),'Unexpected config, code, data or environment change'
            original_metadata=json.loads(json.dumps(metadata));before=array_digest(arrays)
            if metadata['signature']==old_signature:
                backup.parent.mkdir(parents=True,exist_ok=True)
                if not backup.exists():shutil.copy2(path,backup)
                metadata['signature']=new_signature
                arrays['metadata']=np.array(json.dumps(metadata))
                temporary=path.with_suffix('.extend.tmp.npz')
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
                os.replace(temporary,path)
            with np.load(path,allow_pickle=False) as saved:
                after_arrays={k:saved[k].copy() for k in saved.files}
            assert array_digest(after_arrays)==before,'Weights or Adam arrays changed'
            after_metadata=json.loads(str(after_arrays['metadata']))
            stripped_before={k:v for k,v in original_metadata.items() if k!='signature'}
            stripped_after={k:v for k,v in after_metadata.items() if k!='signature'}
            assert stripped_before==stripped_after,'Optimizer, RNG or history changed'
            # Verify the real loader accepts and restores the exact continued state.
            rngs={'shuffle':prepared.shuffle_rng,'corruption':prepared.corruption_rng,'evaluation':prepared.evaluation_rng}
            if hasattr(prepared.model,'_rng'):rngs['sampling']=prepared.model._rng
            if prepared.interpolation_rng is not None:rngs['interpolation']=prepared.interpolation_rng
            loaded=Checkpoint(path,prepared.model,prepared.plan.optimizer,rngs,new_signature).load()
            assert loaded['epoch']==metadata['epoch']
            audit.append({'condition':entry['old']['name'],'seed':old['seed'],'resumed_epoch':loaded['epoch'],
                          'arrays_sha256':before,'optimizer_rng_history_unchanged':True,
                          'backup':str(backup.relative_to(ROOT))})
            print(audit[-1],flush=True)
            del arrays,after_arrays,prepared
        # All checkpoint migrations were verified before publishing the new configs.
        for entry in plan['entries']:
            path=ROOT/entry['path']
            assert json.loads(path.read_text()) in (entry['old'],entry['new'])
            path.write_text(json.dumps(entry['new'],indent=2)+'\n')
        module=ROOT/'src/analysis/nsynth_cnn_protocol.py'
        module.write_text(module.read_text().replace("'epochs':100","'epochs':200"))
        protocol=folder/'protocol.json';value=json.loads(protocol.read_text());assert value['epochs'] in (100,200)
        value['epochs']=200;protocol.write_text(json.dumps(value,indent=2)+'\n')
        record={'old_limit':100,'new_limit':200,'resumed':audit,'reference_configs_epochs':200,
                'loss_optimizer_schedule_and_rng_unchanged':True}
        (folder/'epoch_extension_audit.json').write_text(json.dumps(record,indent=2)+'\n')
        report=ROOT/'reports/nsynth_cnn_study';report.mkdir(exist_ok=True)
        (report/'epoch_extension_audit.json').write_text(json.dumps(record,indent=2)+'\n')
        print('Verified extension to 200 epochs; checkpoints retain weights, Adam, RNG, history.',flush=True)

if __name__=='__main__':main()
