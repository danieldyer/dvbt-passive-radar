import unittest

import numpy as np

from clutter_information import efficient_mean,pair_mean,pair_information
from stationary import Stationary
from test_physical_pair import fixture


class ClutterInformationTests(unittest.TestCase):
    def test_constant_clutter_phase_information_matches_exact_finite_aperture(self):
        time=np.linspace(-1.5,1.5,301);design=np.ones((len(time),1),complex)
        nuisance=Stationary(design,np.ones(len(time),bool))
        for frequency in (0.,.01,.05,.1,.3,1.):
            h=np.exp(2j*np.pi*frequency*time)
            known=efficient_mean((1j*h)[:,None],h[:,None])
            estimated=efficient_mean((1j*h)[:,None],h[:,None],nuisance)
            expected=2*np.sum(abs(h-h.mean())**2)
            np.testing.assert_allclose(2*np.sum(estimated**2),expected,atol=1e-10,rtol=1e-11)
            np.testing.assert_allclose(2*np.sum(known**2),2*len(time),atol=1e-10)

    def test_raw_mean_reproduces_existing_profiled_operator(self):
        model,truth=fixture()
        for pair,index in zip(model.pairs,model.indices):
            p=truth[index]
            expected=pair.evaluate(p)['mean_derivative']
            actual=pair_mean(pair,p,estimate_stationary=True)
            np.testing.assert_allclose(actual,expected,atol=2e-9,rtol=2e-9)

    def test_known_clutter_is_positive_semidefinite_information_upper_bound(self):
        model,truth=fixture()
        for pair,index in zip(model.pairs,model.indices):
            p=truth[index]
            known=pair_information(pair,p,estimate_stationary=False)
            estimated=pair_information(pair,p,estimate_stationary=True)
            difference=known-estimated
            self.assertGreater(np.linalg.eigvalsh(difference).min(),-1e-9*max(np.linalg.norm(known),1.))


if __name__=='__main__':unittest.main()
