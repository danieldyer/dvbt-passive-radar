import unittest

import numpy as np

from coherent_fit import OTHER
from vehicle_response import ridge_gcv, phase_information


class VehicleResponseTests(unittest.TestCase):
    def test_complex_regularized_fit_preserves_unitary_observation_coordinates(self):
        random = np.random.default_rng(712)
        a = random.normal(size=(30, 7))+1j*random.normal(size=(30, 7))
        a[:, -1] = a[:, 0]+a[:, 1]*1e-7
        truth = random.normal(size=7)+1j*random.normal(size=7)
        y = a@truth+.1*(random.normal(size=30)+1j*random.normal(size=30))
        rotation, _ = np.linalg.qr(random.normal(size=(30, 30))+1j*random.normal(size=(30, 30)))
        first, choices = ridge_gcv(a, y)
        second, other = ridge_gcv(rotation@a, rotation@y)
        np.testing.assert_allclose(first, second, atol=1e-10)
        self.assertAlmostEqual(choices['regularization_relative_to_max_gram'], other['regularization_relative_to_max_gram'])

    def test_point_phase_information_matches_analytic_common_phase_schur_complement(self):
        sizes = [10, 12, 11, 9, 13]
        amplitude = np.array([1+.3j])
        a = np.repeat(np.array([1., 2., .8, 1.1, .6]), sizes)[:, None].astype(complex)
        expected_energy = np.array(sizes)*np.array([1., 2., .8, 1.1, .6])**2*abs(amplitude[0])**2
        e = expected_energy[OTHER]
        expected = 2*(np.diag(e)-np.outer(e, e)/expected_energy.sum())
        result = phase_information(a, amplitude, sizes)
        np.testing.assert_allclose(result['fisher_information'], expected, atol=1e-12)

    def test_independent_band_reflectivities_cannot_identify_channel_phase(self):
        sizes = [10]*5
        a = np.zeros((50, 5), complex)
        for channel in range(5):
            a[10*channel:10*(channel+1), channel] = np.exp(.1j*np.arange(10))
        result = phase_information(a, np.array([1., 2j, .8+.3j, -.2j, -.6]), sizes)
        self.assertLess(max(result['singular_values_per_radian']), 1e-12)


if __name__ == '__main__':
    unittest.main()
