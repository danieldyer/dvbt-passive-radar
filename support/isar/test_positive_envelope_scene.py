import unittest
import numpy as np
from positive_envelope_scene import solve


class EnvelopeSceneTests(unittest.TestCase):
    def test_varying_brightness_preserves_one_phase_per_physical_point(self):
        rng=np.random.default_rng(581);a=rng.normal(size=(300,12))+1j*rng.normal(size=(300,12))
        groups=np.repeat(np.arange(3),4);phase=np.array([.7,-1.2,2.1]);amplitude=np.array([1,0,.3,0,.2,1,.4,0,0,.7,.9,.2])
        y=a@(np.exp(1j*phase[groups])*amplitude);g=a.conj().T@a;b=a.conj().T@y;e=np.vdot(y,y).real
        r=solve(g,b,e,groups,phase)
        np.testing.assert_allclose(r['amplitude'],amplitude,atol=1e-12);self.assertLess(abs(r['cost']),1e-10)
        p=phase+np.array([.04,-.03,.02]);r=solve(g,b,e,groups,p);h=1e-5
        for j in range(3):
            d=np.eye(3)[j]*h;numeric=(solve(g,b,e,groups,p+d)['cost']-solve(g,b,e,groups,p-d)['cost'])/(2*h)
            self.assertAlmostEqual(r['phase_gradient'][j],numeric,places=6)

    def test_unresolved_envelope_columns_do_not_need_diagonal_loading(self):
        rng=np.random.default_rng(82);a=rng.normal(size=(30,4))+1j*rng.normal(size=(30,4));a[:,3]=a[:,2]
        groups=np.array([0,0,1,1]);p=np.array([.4,-.7]);y=a@(np.exp(1j*p[groups])*[1,.3,.2,.5])
        r=solve(a.conj().T@a,a.conj().T@y,np.vdot(y,y).real,groups,p)
        self.assertLess(np.linalg.norm(y-a@r['coefficient']),1e-11);self.assertEqual(r['numerical_rank'],3)


if __name__=='__main__':unittest.main()
