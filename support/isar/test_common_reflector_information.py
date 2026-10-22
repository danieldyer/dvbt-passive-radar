import unittest

import numpy as np
from scipy.linalg import block_diag

from antenna_motion import Antennas
from common_reflector_information import compact_roots, nested_information
from geometry_information import real_jacobian
from geometry_profile import GeometryProfile
import test_physical_profile as physical_tests


class CommonReflectorInformationTests(unittest.TestCase):
    def test_shared_and_independent_nuisance_match_dense_gaussian_elimination(self):
        rng = np.random.default_rng(167241)
        roots = [rng.normal(size=(53, 13)) for _ in range(3)]
        physical = 7
        interest, nuisance = np.arange(3), np.arange(3, physical)
        result = nested_information(roots, physical, interest, nuisance, cutoff=1e-12)
        mean = np.concatenate([r[:, :physical] for r in roots])
        points = [r[:, physical:] for r in roots]
        for name, extra in [('known', np.empty((len(mean), 0))),
                            ('shared', np.concatenate(points)),
                            ('independent', block_diag(*points))]:
            other = np.column_stack((mean[:, nuisance], extra))
            target = mean[:, interest]
            residual = target-other@np.linalg.lstsq(other, target, rcond=1e-12)[0]
            np.testing.assert_allclose(result[name], residual.T@residual, rtol=1e-12, atol=1e-12)
        for a, b in [('known', 'shared'), ('shared', 'independent')]:
            self.assertGreaterEqual(np.linalg.eigvalsh(result[a]-result[b]).min(), -1e-12)

    def test_single_band_has_no_artificial_gain_from_point_correspondence(self):
        root = np.random.default_rng(7816).normal(size=(47, 11))
        for cutoff in (1e-8, 1e-10, 1e-12, 1e-14):
            result = nested_information([root], 5, [0, 1], [2, 3, 4], cutoff=cutoff)
            np.testing.assert_allclose(result['shared'], result['independent'], atol=1e-12, rtol=0)
            self.assertEqual(result['additional_independent_point_rank'], 0)

    def test_physical_point_columns_include_range_carrier_and_doppler(self):
        band, truth = physical_tests.PhysicalProfileTests().fixture()
        joint = GeometryProfile([band], truth*.3, Antennas(np.eye(3), [0]))
        parameters = joint.initial.copy()
        parameters[joint.antenna_slice] = [.01, -.02, .003, -.01, .005, .002]
        roots = compact_roots(joint, parameters)
        mean, _ = real_jacobian(joint, parameters)
        response = joint.responses(parameters)[0]
        design = band.nuisance.remove(response['target'][band.selected])
        columns = []
        for point in range(len(band.points)):
            for axis in range(3):
                delta = np.zeros_like(band.points); delta[point, axis] = .0001
                values = [band.design(joint.expand(parameters), points=band.points+sign*delta,
                    antenna_offsets=joint.offsets(parameters, 0),
                    range_bias_m=parameters[joint.range_slice.start])[0]
                    for sign in (1, -1)]
                derivative = band.nuisance.remove(((values[0]-values[1])@response['coefficient'])[band.selected]/.0002)
                derivative -= design@np.linalg.lstsq(design, derivative, rcond=None)[0]
                columns.append(np.sqrt(2)*np.r_[derivative.real, derivative.imag])
        dense = np.column_stack((mean, np.column_stack(columns)))
        # Normalize columns so large site-coordinate derivatives cannot hide a
        # weak or incorrect height/point derivative in a global matrix norm.
        scale = np.linalg.norm(dense, axis=0)
        scale = np.where(scale > 0, scale, 1.)
        expected = (dense/scale).T@(dense/scale)
        actual = (roots[0]/scale).T@(roots[0]/scale)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
