import unittest

import numpy as np

from shared_scene import solve,phase_coherent


class SharedSceneTests(unittest.TestCase):
    def test_common_phase_allows_brightness_changes_without_independent_phase_rotation(self):
        rng=np.random.default_rng(20);beta=np.array([.3,-1.,.6,0.,2.]);phi=np.linspace(-2,1,6)
        amplitude=np.exp(rng.normal(size=(5,6))*1.3);gamma=amplitude*np.exp(1j*(beta[:,None]+phi))
        gram=[];rhs=[];energy=[]
        for c in range(5):
            a=rng.normal(size=(67,6))+1j*rng.normal(size=(67,6));y=a@gamma[c]
            gram.append(a.conj().T@a);rhs.append(a.conj().T@y);energy.append(np.vdot(y,y).real)
        found=phase_coherent(gram,rhs,energy)
        self.assertLess(abs(found['cost'])/sum(energy),1e-12)
        np.testing.assert_allclose(found['coefficient'],gamma,rtol=2e-5,atol=2e-5)
        random_phase=gamma*np.exp(1j*rng.uniform(-np.pi,np.pi,gamma.shape))
        bad_rhs=[g@z for g,z in zip(gram,random_phase)]
        bad_energy=[np.vdot(z,g@z).real for g,z in zip(gram,random_phase)]
        bad=phase_coherent(gram,bad_rhs,bad_energy)
        self.assertGreater(bad['cost']/sum(bad_energy),.01)

    def test_unequal_information_and_correlated_columns_recover_complex_band_gains(self):
        rng=np.random.default_rng(246);scene=rng.normal(size=6)+1j*rng.normal(size=6)
        gain=np.array([.3+.5j,-.1+.2j,.8-.3j,1.,-.6+.4j])
        gram=[];rhs=[];energy=[]
        for c,g in enumerate(gain):
            a=(rng.normal(size=(71,6))+1j*rng.normal(size=(71,6)))@np.diag(np.linspace(.5,2,6))*(c+1)
            y=g*(a@scene);gram.append(a.conj().T@a);rhs.append(a.conj().T@y);energy.append(np.vdot(y,y).real)
        found=solve(gram,rhs,energy)
        np.testing.assert_allclose(found['gain'],gain,rtol=0,atol=2e-6)
        np.testing.assert_allclose(found['scene'],scene,rtol=0,atol=2e-6)
        self.assertLess(abs(found['cost'])/sum(energy),1e-12)


if __name__=='__main__':unittest.main()
