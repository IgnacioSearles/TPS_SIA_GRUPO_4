"""Estado completo por época, atómico y sin pickle; no cambia el entrenamiento."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from src.options import ConfigurationError


def training_signature(config, samples, triplets=None):
    root=Path(__file__).resolve().parents[2]
    paths=[p for folder in ('models','nn','training') for p in (root/'src'/folder).glob('*.py')]
    paths += [root/'src/options.py',root/'src/data/noise.py',root/'src/experiments/runner.py']
    digest=lambda data:hashlib.sha256(data).hexdigest()
    return {'config':digest(json.dumps(config,sort_keys=True).encode()),
            'samples':digest(samples.tobytes()),
            'triplets':None if triplets is None else digest(triplets.tobytes()),
            'numpy':np.__version__,
            'threads':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
            'sources':{str(p.relative_to(root)):digest(p.read_bytes()) for p in sorted(paths)}}


def _pack(value,arrays):
    if isinstance(value,np.ndarray):
        key=f'state_{len(arrays)}';arrays[key]=value
        return {'__checkpoint_array__':key}
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:_pack(v,arrays) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_pack(v,arrays) for v in value]
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise TypeError(f'Estado no soportado: {type(value).__name__}')


def _unpack(value,saved):
    if isinstance(value,dict):
        if set(value)=={'__checkpoint_array__'}:return saved[value['__checkpoint_array__']].copy()
        return {k:_unpack(v,saved) for k,v in value.items()}
    if isinstance(value,list):return [_unpack(v,saved) for v in value]
    return value


class Checkpoint:
    def __init__(self,path,model,optimizer,rngs,signature,extra=None):
        self.path=Path(path);self.model=model;self.optimizer=optimizer
        self.rngs=rngs;self.signature=signature;self.extra=extra or {}

    def save(self,epoch,history,elapsed,interval):
        arrays={f'parameter_{i}':p for i,p in enumerate(self.model.parameters())}
        metadata={'version':1,'signature':self.signature,'epoch':epoch,'history':history,
                  'elapsed':elapsed,'interval':interval,'extra':self.extra,
                  'optimizer_class':type(self.optimizer).__name__,
                  'optimizer':_pack(vars(self.optimizer),arrays),
                  'rngs':_pack({k:r.bit_generator.state for k,r in self.rngs.items()},arrays)}
        arrays['metadata']=np.array(json.dumps(metadata))
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temporary=self.path.with_suffix('.tmp.npz')
        try:
            with temporary.open('wb') as stream:
                np.savez_compressed(stream,**arrays)
                stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def load(self):
        with np.load(self.path,allow_pickle=False) as saved:
            metadata=json.loads(str(saved['metadata']))
            if metadata['version']!=1 or metadata['signature']!=self.signature:
                raise ConfigurationError('Checkpoint incompatible: cambiaron configuración, datos, código o entorno')
            if metadata['optimizer_class']!=type(self.optimizer).__name__:
                raise ConfigurationError('El optimizador del checkpoint no coincide')
            parameters=self.model.parameters()
            values=[saved[f'parameter_{i}'].copy() for i in range(len(parameters))]
            if any(p.shape!=v.shape or p.dtype!=v.dtype for p,v in zip(parameters,values)):
                raise ConfigurationError('Los parámetros del checkpoint no coinciden')
            optimizer=_unpack(metadata['optimizer'],saved)
            rngs=_unpack(metadata['rngs'],saved)
            if set(optimizer)!=set(vars(self.optimizer)) or set(rngs)!=set(self.rngs):
                raise ConfigurationError('Estado del optimizador o generadores incompatible')
            if any(rngs[k]['bit_generator']!=r.bit_generator.state['bit_generator'] for k,r in self.rngs.items()):
                raise ConfigurationError('Cambió el tipo de generador aleatorio')
        for p,v in zip(parameters,values):p[:]=v
        vars(self.optimizer).update(optimizer)
        for k,r in self.rngs.items():r.bit_generator.state=rngs[k]
        self.extra=metadata['extra']
        return metadata
