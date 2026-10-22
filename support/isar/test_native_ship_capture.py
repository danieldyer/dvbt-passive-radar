import contextlib,hashlib,io,json,tempfile,unittest
from pathlib import Path
import numpy as np
from prepare_native_ship_capture import prepare
from recorded_injection_fields import write_profiles


class NativeShipCaptureTests(unittest.TestCase):
    def test_streamed_means_preserve_complex_moments_epochs_and_erasure_weights(self):
        rng=np.random.default_rng(941);n=35;ranges=96
        h=dict(range_count=ranges,first_range_sample=160,applied_channels=[dict(rf_hz=177.5e6),dict(rf_hz=184.5e6)])
        channels={}
        for c in range(2):
            rows=np.zeros((n,6));rows[:,1]=6.+np.arange(n)*.001088+c*1e-6;rows[:,4]=1.;rows[8:12,4]=0
            information=np.where(np.arange(n)%7<3,2.,5.)
            values=rng.normal(size=(2,n,ranges))+1j*rng.normal(size=(2,n,ranges))
            pulse=np.broadcast_to(np.sinc(np.arange(-64,65)*.87),(n,129)).copy()
            channels[c]=dict(values=values,pulse=pulse,rows=rows,information=information)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'source.bin';out=Path(tmp)/'cache';write_profiles(path,h,channels)
            identity=hashlib.sha256(path.read_bytes()).hexdigest()
            with contextlib.redirect_stdout(io.StringIO()):prepare(path,out)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),identity)
            for block in (4,16):
                for c,ch in channels.items():
                    d=out/f'block-{block}/channel-{c}';a=dict(np.load(d/'coordinates.npz'));actual=np.load(d/'moments.npy')
                    self.assertEqual(actual.shape,(2,32//block,ranges))
                    w=ch['information'][:32]*ch['rows'][:32,4]
                    t=ch['rows'][:32,1];source=ch['values'][:,:32].astype('<c8').astype(complex)
                    for j,start in enumerate(range(0,32,block)):
                        q=slice(start,start+block);total=w[q].sum()
                        expected=(source[:,q]*w[None,q,None]).sum(axis=1)/total if total else np.zeros((2,ranges),complex)
                        np.testing.assert_allclose(actual[:,j],expected,atol=1.3e-7,rtol=1e-7)
                        expected_time=float(w[q]@t[q]/total) if total else t[q].mean()
                        self.assertAlmostEqual(a['time_s'][j],expected_time,places=14)
                        self.assertEqual(a['information'][j],total)
                    np.testing.assert_array_equal(a['carrier_reference_path_m'],0.)
                    self.assertTrue(json.loads((d/'metadata.json').read_text())['complete'])
            for c,ch in channels.items():
                a=dict(np.load(out/f'block-16/channel-{c}/native-epochs.npz'))
                np.testing.assert_array_equal(a['time_s'],ch['rows'][:32,1])
                np.testing.assert_array_equal(a['information'],ch['information'][:32]*ch['rows'][:32,4])


if __name__=='__main__':unittest.main()
