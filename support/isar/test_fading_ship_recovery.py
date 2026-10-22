import unittest
import numpy as np
from validate_fading_ship_recovery import nuisance_energy_scale


class TestRecoveryScenario(unittest.TestCase):
    def test_competing_field_matches_measured_energy_with_coherent_cross_term(self):
        rng=np.random.default_rng(826);target=rng.normal(size=(300,7))+1j*rng.normal(size=(300,7))
        nuisance=.3*target+rng.normal(size=target.shape)+1j*rng.normal(size=target.shape)
        desired=float(np.vdot(target+.7*nuisance,target+.7*nuisance).real)
        scale=nuisance_energy_scale(float(np.vdot(target,target).real),float(np.vdot(target,nuisance).real),float(np.vdot(nuisance,nuisance).real),desired)
        np.testing.assert_allclose(scale,.7,atol=3e-16,rtol=0)
        self.assertLess(abs(np.vdot(target+scale*nuisance,target+scale*nuisance).real-desired)/desired,1e-14)


if __name__=='__main__':unittest.main()
