"""Inicialización opcional desde pesos terminados; no es un checkpoint de Adam."""

import hashlib
from pathlib import Path

from src.models.factory import load_parameters
from src.options import ConfigurationError, Options


def _load_transfer_arrays(path,destination):
    import numpy as np
    with np.load(path,allow_pickle=False) as saved:
        if len(saved.files)!=len(destination):raise ConfigurationError('Arquitecturas de transferencia incompatibles')
        arrays=[saved[f'parameter_{i}'] for i in range(len(destination))]
    if any(a.shape!=b.shape for a,b in zip(arrays,destination)):
        raise ConfigurationError('Dimensiones de transferencia incompatibles')
    return arrays


def _transfer_identity_dense(model,path,layer_index):
    import numpy as np
    from src.nn.network import DenseLayer
    decoder=model._decoder
    layers=getattr(decoder,'layers',getattr(decoder,'_layers',[]))
    if (type(layer_index) is not int or not 1<=layer_index<len(layers)
            or not all(isinstance(layer,DenseLayer) for layer in layers[:layer_index+1])):
        raise ConfigurationError('layer_index debe señalar una capa del prefijo denso del decoder')
    layer=layers[layer_index];previous=layers[layer_index-1]
    if (layer._activation.name!='relu' or previous._activation.name!='relu'
            or layer.weights.shape[0]!=layer.weights.shape[1]):
        raise ConfigurationError('La nueva capa identidad requiere ReLU y dimensiones iguales')
    offset=len(model._encoder.parameters())+2*layer_index
    parameters=model.parameters()
    destination=parameters[:offset]+parameters[offset+2:]
    arrays=_load_transfer_arrays(path,destination)
    layer.weights[:]=np.eye(layer.weights.shape[0],dtype=layer.weights.dtype)
    layer.biases.fill(0)
    for value,parameter in zip(arrays,destination):parameter[:]=value


def _transfer_identity_refinement(model, path):
    import numpy as np
    from src.nn.spatial import Conv3
    layers = getattr(model._decoder, 'layers', [])
    previous=next((layer for layer in reversed(layers[:-3]) if hasattr(layer,'weights')),None)
    if (len(layers)<5 or not isinstance(layers[-3],Conv3) or not isinstance(previous,Conv3)
            or layers[-3].activation.name!='relu' or previous.activation.name!='relu'):
        raise ConfigurationError('La transferencia requiere refinamiento ReLU con inicio identidad')
    layer=layers[-3]
    identity=np.zeros_like(layer.weights)
    count=identity.shape[2]
    if identity.shape!=(3,3,count,count):raise ConfigurationError('Refinamiento no cuadrado')
    identity[1,1,np.arange(count),np.arange(count)]=1
    if not np.array_equal(layer.weights,identity) or np.any(layer.biases):
        raise ConfigurationError('El refinamiento no está inicializado como identidad')
    parameters=model.parameters()
    destination=parameters[:-4]+parameters[-2:]
    arrays=_load_transfer_arrays(path,destination)
    for value,parameter in zip(arrays,destination):parameter[:]=value


def initialize_from_parameters(model, specification, root, seed):
    if specification is None:
        return None
    transfer='exact';layer_index=None
    if isinstance(specification,dict):
        options=Options(specification,'initial_parameters')
        specification=options.take('path',None)
        transfer=options.take('transfer','exact')
        layer_index=options.take('layer_index',None)
        options.ensure_consumed()
    if transfer not in ('exact','identity_refinement','identity_dense'):
        raise ConfigurationError('Transferencia inicial desconocida')
    if layer_index is not None and transfer!='identity_dense':
        raise ConfigurationError('layer_index solo aplica a identity_dense')
    if not isinstance(specification, str) or not specification:
        raise ConfigurationError('initial_parameters debe ser una ruta no vacía; admite {seed}')
    path = Path(root)/specification.replace('{seed}', str(seed))
    if not path.is_file():
        raise ConfigurationError(f'Faltan los pesos de inicialización: {path}')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if transfer=='exact':load_parameters(model, path)
    elif transfer=='identity_refinement':_transfer_identity_refinement(model,path)
    else:_transfer_identity_dense(model,path,layer_index)
    report={'path': str(path.resolve()), 'sha256': digest,
            'optimizer_state_restored': False, 'rng_states_restored': False, 'transfer':transfer,
            'interpretation': 'New optimization phase initialized from saved parameters; not exact resume.'}
    if layer_index is not None:report['layer_index']=layer_index
    return report
