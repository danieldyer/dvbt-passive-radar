import unittest
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np

from scene_dictionary import union_grid, fingerprint
from fit_cached_scene import selected_times, main as fit_main


class SceneDictionaryTests(unittest.TestCase):
    def test_extended_grid_preserves_the_core_and_shared_boundary(self):
        core=union_grid([[-80,160,-20,60]],2.,20.)
        extended=union_grid([[-80,160,-20,60],[-200,-80,-50,0]],2.,20.)
        self.assertEqual(len(core),4961)
        self.assertEqual(len(extended),6536)
        keep=(extended[:,0]>=-80)&(extended[:,0]<=160)&(extended[:,1]>=-20)&(extended[:,1]<=60)
        np.testing.assert_array_equal(extended[keep],core)
        self.assertTrue(np.all(extended[:,2]==20.))

    def test_temporal_blocks_use_recording_epochs_not_sample_counts(self):
        time=np.array([142.02,142.2,142.99,143.01,143.8,144.1,144.9,145.1])
        np.testing.assert_array_equal(selected_times(time,[142.,172.],1.),[1,1,1,0,0,1,1,0])
        np.testing.assert_array_equal(selected_times(time,[142.,172.],0.),[1,0,1,0,1,0,1,0])
        with self.assertRaises(ValueError):
            selected_times(time,[142.,172.],-1.)

    def test_cached_fit_preserves_phase_and_does_not_train_on_heldout_blocks(self):
        rng=np.random.default_rng(2831)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); cache=root/'cache'; cache.mkdir()
            points=np.c_[np.arange(10),np.zeros(10),np.full(10,20.)]
            time=np.arange(180)*.05
            coefficient=rng.normal(size=10)+1j*rng.normal(size=10)
            phase=np.exp(1j*np.array([.4,-.6,1.2,0.,-.3]))
            background=rng.normal(size=(540,3))+1j*rng.normal(size=(540,3))
            manifest={'complete':True,'source_motion':'synthetic','source_alignment':'synthetic',
                'points_body_m':points.tolist(),'interval_s':[0.,9.],'channels':[]}
            for c in range(5):
                a=rng.normal(size=(540,10))+1j*rng.normal(size=(540,10))
                y=phase[c]*(a@coefficient)+background@np.array([2+3j,-.2+.7j,.9j])+ .01*(rng.normal(size=540)+1j*rng.normal(size=540))
                path=cache/f'channel-{c}-design.npy';np.save(path,a)
                manifest['channels'].append({'channel':c,'shape':list(a.shape),'design_sha256':fingerprint(path)})
                np.savez(cache/f'channel-{c}-observation.npz',time=time,data=y.reshape(180,3),background=background)
            (cache/'dictionary.json').write_text(json.dumps(manifest))
            def run(name):
                output=root/name
                with patch.object(sys,'argv',['fit_cached_scene','--dictionary',str(cache),'--output',str(output),'--holdout-seconds','1']),contextlib.redirect_stdout(io.StringIO()):
                    fit_main()
                return json.loads((output/'fit.json').read_text())
            first=run('original')
            registered=first['trials'][2]
            self.assertGreater(registered['heldout_explained_fraction'],.999)
            np.testing.assert_allclose(np.exp(1j*np.radians(registered['phase_registration']['relative_phase_degrees'])),phase,atol=.002)
            for c in registered['channels']:
                self.assertLess(c['prediction_regeneration_relative_error'],1e-10)
            # Change only withheld observations. All fitted coefficients and
            # phase parameters must remain identical; only test scores change.
            keep=selected_times(time,[0.,9.],1.)
            for c in range(5):
                path=cache/f'channel-{c}-observation.npz'
                with np.load(path) as f: data=dict(f)
                data['data'][~keep]*=np.exp(.7j)
                np.savez(path,**data)
            second=run('modified-test')
            for before,after in zip(first['trials'],second['trials']):
                for c,d in zip(before['channels'],after['channels']):
                    np.testing.assert_array_equal(c['coefficient_real_imag'],d['coefficient_real_imag'])
                self.assertNotEqual(before['heldout_explained_fraction'],after['heldout_explained_fraction'])


if __name__=='__main__':
    unittest.main()
