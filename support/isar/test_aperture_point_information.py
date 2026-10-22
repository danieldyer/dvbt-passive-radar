import unittest

import numpy as np

from aperture_point_information import combined_information, summarize
from compression_information import point_information


class AperturePointInformationTests(unittest.TestCase):
    def test_independent_amplitudes_match_explicit_nuisance_projection(self):
        rng = np.random.default_rng(917)
        responses = [(rng.normal(size=23)+1j*rng.normal(size=23),
                      rng.normal(size=(23, 2))+1j*rng.normal(size=(23, 2))) for _ in range(5)]
        energy, information = combined_information(responses)
        u = np.zeros((115, 5), complex)
        d = np.concatenate([v for _, v in responses])
        for c, (v, _) in enumerate(responses):
            u[c*23:(c+1)*23, c] = v
        q, _ = np.linalg.qr(u)
        residual = d-q@(q.conj().T@d)
        np.testing.assert_allclose(information['independent_complex_band_amplitudes'],
                                   2*np.real(residual.conj().T@residual), rtol=1e-13)
        improvement = information['known_relative_band_calibration']-information['independent_complex_band_amplitudes']
        self.assertGreaterEqual(np.linalg.eigvalsh(improvement).min(), -1e-11)
        self.assertAlmostEqual(energy, np.sum(abs(u)**2))

    def test_common_gain_and_coordinate_rotation_preserve_bounds(self):
        rng = np.random.default_rng(416)
        u = rng.normal(size=37)+1j*rng.normal(size=37)
        d = rng.normal(size=(37, 2))+1j*rng.normal(size=(37, 2))
        rotation = np.array([[.8, -.6], [.6, .8]])
        energy, j = point_information(u, d)
        e2, j2 = point_information((3+4j)*u, (3+4j)*d@rotation)
        np.testing.assert_allclose(j2/e2, rotation.T@j@rotation/energy, atol=1e-14)
        np.testing.assert_allclose(summarize(j, energy, 30)['principal_axis_standard_deviation_m'],
                                  summarize(j2, e2, 30)['principal_axis_standard_deviation_m'], rtol=1e-13)


if __name__ == '__main__':
    unittest.main()
