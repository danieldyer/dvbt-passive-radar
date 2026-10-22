import unittest
import numpy as np
from coherent_arc import CoherentArc
from coherent_chirp_scene import ChirpScene
from test_recovery_experiment import problem
from block_pulse import BlockPulse
from waveform_kernels import KernelField


class ChirpSceneTests(unittest.TestCase):
    def test_profiled_derivative_and_excluded_prediction(self):
        f=problem().fields[0];rng=np.random.default_rng(714)
        kernels=BlockPulse(np.array([f.pulse.samples*(1+.02*np.sin(i)) for i in range(len(f.time))]))
        f=KernelField(f,kernels,np.arange(len(f.data['row_time']))//f.metadata['block_rows'])
        keep=np.arange(len(f.time))%4!=1;f.y[:]=1.
        arc=CoherentArc(f,keep,stationary_paths=[])
        truth=np.array([-.4,.3,.12,.6,-.2,-.07])
        h=np.column_stack([arc.template(p) for p in truth.reshape(2,3)])
        alpha=np.array([.7+.2j,-.3+.8j]);f.y=(h@alpha).reshape(f.y.shape)
        arc=CoherentArc(f,keep,stationary_paths=[]);scene=ChirpScene(arc,2)
        at=truth+rng.normal(size=6)*.002;value=scene.evaluate(at)
        for j in range(6):
            step=np.eye(6)[j]*1e-6
            finite=(scene.evaluate(at+step)['residual']-scene.evaluate(at-step)['residual'])/2e-6
            self.assertLess(np.linalg.norm(finite-value['jacobian'][:,j])/np.linalg.norm(finite),2e-5)
        # This checks inference and frozen prediction, without fitting any
        # coefficient to the omitted epoch. Acquisition is tested separately.
        fit=scene.fit(at,(-1.,1.))
        self.assertLess(fit['training_rss']/arc.energy,1e-14)
        self.assertGreater(fit['prediction']['explained_fraction'],1-1e-12)


if __name__=='__main__':unittest.main()
