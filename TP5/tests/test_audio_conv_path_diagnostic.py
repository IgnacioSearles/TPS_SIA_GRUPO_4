import numpy as np

from src.analysis.audio_conv_path_diagnostic import generate_paths,summarize


class Affine:
    precision=np.float32
    def encode(self,x):return 2*x+1
    def decode(self,z):return 3*z-2


def test_affine_decoder_commutes_with_interpolation():
    neighbors=np.array([[1,2],[3,8]],dtype=np.float32)
    paths=generate_paths(Affine(),neighbors)
    np.testing.assert_allclose(paths['latent_mix'],paths['decoded_neighbor_mix'])
    np.testing.assert_allclose(paths['spectrogram_mix'][1],neighbors.mean(axis=0))


def test_nonlinear_decoder_separates_the_two_interpolations():
    class Nonlinear(Affine):
        def encode(self,x):return x
        def decode(self,z):return z*z
    paths=generate_paths(Nonlinear(),np.array([[0],[2]],dtype=np.float32))
    assert paths['latent_mix'][1,0]==1
    assert paths['decoded_neighbor_mix'][1,0]==2


def test_invalid_pitch_kept_as_failure_and_noncenters_unmeasured():
    base={'architecture':'conv','method':'latent_mix','spectral_hit':True,'spectral_mse':.1}
    rows=[dict(base,fraction=.5,pitch_evaluated=True,hit=False,valid=False),
          dict(base,fraction=.25,pitch_evaluated=False)]
    central=next(r for r in summarize(rows) if r['fraction']==.5)
    assert central['trials']==1 and central['invalid']==1 and central['joint_hits']==0
    assert next(r for r in summarize(rows) if r['fraction']==.25)['pitch_trials']==0
