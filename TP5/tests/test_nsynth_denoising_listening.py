import json
import numpy as np

from scripts import build_nsynth_denoising_listening as listening


def test_cache_canonicalizes_dtype_and_uses_complete_atomic_wav(tmp_path,monkeypatch):
    class Codec:
        image_shape=(2,2)
        sample_rate=16000
        calls=0
        def decode(self,values):
            assert values.dtype==np.float64
            self.calls+=1
            return np.zeros(16000)
    codec=Codec()
    monkeypatch.setattr(listening,'cached_codec',lambda description:codec)
    spec=np.array([.25,.5,.75,1],dtype=np.float32)
    description=json.dumps({'phase_initialization':'pghi'})
    one=listening.decode_one(('clean',spec,description,str(tmp_path)))
    two=listening.decode_one(('sigma0',spec.astype(np.float64),description,str(tmp_path)))
    assert one[1]==two[1] and codec.calls==1
    assert one[2]['codec_input_precision']=='float64'
    assert one[2]['cache_signature']==two[2]['cache_signature']
    assert one[2]['sample_rate']==16000 and one[2]['samples']==16000
    assert not list(tmp_path.glob('*.tmp'))


def test_cache_distinguishes_codec_phase(tmp_path,monkeypatch):
    class Codec:
        image_shape=(2,2)
        sample_rate=16000
        def decode(self,values):return np.zeros(16000)
    monkeypatch.setattr(listening,'cached_codec',lambda description:Codec())
    spec=np.zeros(4)
    one=listening.decode_one(('a',spec,json.dumps({'phase_initialization':'pghi'}),str(tmp_path)))
    two=listening.decode_one(('b',spec,json.dumps({'phase_initialization':'random'}),str(tmp_path)))
    assert one[2]['cache_signature']!=two[2]['cache_signature']
    assert len(list(tmp_path.glob('*.wav')))==2
