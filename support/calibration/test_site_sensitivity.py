import unittest

import numpy as np
from geometry_evidence import pair_error
from scipy.optimize._numdiff import approx_derivative

from coherent_fit import OTHER
from site_sensitivity import phase_objective, profile_channel_phases


class FiniteGeometryTests(unittest.TestCase):
    def test_complex_gaussian_binary_error_against_sampled_classifier(self):
        random = np.random.default_rng(193)
        count = 100_000
        delta = np.array([1+1j, .2-.5j, .3j])
        truth = random.integers(0, 2, count)
        noise = (random.normal(size=(count, 3))+
                 1j*random.normal(size=(count, 3)))/np.sqrt(2)
        samples = noise+truth[:, None]*delta
        choose_one = (np.sum(abs(samples-delta)**2, axis=1) <
                      np.sum(abs(samples)**2, axis=1))
        observed = np.mean(choose_one != truth)
        expected = pair_error(np.vdot(delta, delta).real)
        self.assertLess(abs(observed-expected),
                        5*np.sqrt(expected*(1-expected)/count))
        self.assertEqual(pair_error(0), .5)
        for invalid in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                pair_error(invalid)

    def test_shared_phase_derivative(self):
        random = np.random.default_rng(728)
        a = random.normal(size=(7, 5))+1j*random.normal(size=(7, 5))
        information = a.T@a.conj()
        phases = np.array([.3, 2.7, -1.1, -3.])
        _, jacobian = phase_objective(phases, information)
        numeric = approx_derivative(lambda x: phase_objective(x, information)[0], phases).ravel()
        np.testing.assert_allclose(jacobian, numeric, rtol=1e-8, atol=1e-8)

    def test_known_phase_recovered_after_unknown_target_reflectivities(self):
        random = np.random.default_rng(744)
        energy = np.exp(random.normal(size=(8, 5)))
        target = random.normal(size=8)+1j*random.normal(size=8)
        truth = np.array([2.9, .3, -2.8, -1.2, .4])
        correlation = target[:, None]*energy*np.exp(1j*truth)
        recovered = profile_channel_phases(correlation, energy.sum(axis=1))
        error = np.angle(np.exp(1j*(recovered-(truth-truth[1]))))
        np.testing.assert_allclose(error[OTHER], 0, atol=1e-8)
        self.assertEqual(recovered[1], 0)


if __name__ == '__main__':
    unittest.main()
