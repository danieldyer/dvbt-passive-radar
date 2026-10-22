import unittest

import numpy as np
from scipy.optimize import nnls

from phase_gram import solve,fit


class PhaseGramTests(unittest.TestCase):
    def test_gram_solution_matches_original_complex_design_and_direct_nnls(self):
        rng=np.random.default_rng(9631);design=rng.normal(size=(117,9))+1j*rng.normal(size=(117,9))
        data=rng.normal(size=117)+1j*rng.normal(size=117)
        for phase in (-2.7,.4,2.1):
            a=design.copy();a[:,:3]*=np.exp(1j*phase)
            expected,_=nnls(np.r_[a.real,a.imag],np.r_[data.real,data.imag])
            actual,cost,rank=solve((a.conj().T@a).real,(a.conj().T@data).real,np.vdot(data,data).real)
            np.testing.assert_allclose(actual,expected,atol=2e-14)
            self.assertAlmostEqual(cost,np.linalg.norm(data-a@actual)**2,places=11)
            self.assertEqual(rank,9)

    def test_constant_phase_and_positive_coefficients_recover_known_scene(self):
        rng=np.random.default_rng(3764);design=rng.normal(size=(151,12))+1j*rng.normal(size=(151,12))
        strength=np.r_[rng.uniform(.3,1.,8),np.zeros(4)];phase=1.183
        truth=design.copy();truth[:,:5]*=np.exp(1j*phase);data=truth@strength
        result=fit(design,data,5)
        self.assertAlmostEqual(result['phase_rad'],phase,places=6)
        np.testing.assert_allclose(result['amplitude'],strength,atol=1e-7)
        self.assertLess(abs(result['cost']),1e-9)

    def test_dependent_nuisance_columns_do_not_require_a_ridge(self):
        rng=np.random.default_rng(4827);a=rng.normal(size=(51,4));a[:,3]=a[:,2]*2
        data=a@np.array([.2,.4,.1,.3])
        amplitude,cost,rank=solve(a.T@a,a.T@data,data@data)
        self.assertEqual(rank,3)
        np.testing.assert_allclose(a@amplitude,data,atol=1e-13)
        self.assertLess(abs(cost),1e-12)


if __name__=='__main__':unittest.main()
