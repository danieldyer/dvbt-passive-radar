import unittest

import numpy as np

from antenna_motion import Antennas
from geometry_information import expected
from geometry_profile import GeometryProfile
import test_physical_profile as physical_tests


class GeometryInformationTests(unittest.TestCase):
    def fixture(self):
        band, truth = physical_tests.PhysicalProfileTests().fixture()
        joint = GeometryProfile([band], truth*.3, Antennas(np.eye(3), [0]))
        p = joint.initial.copy()
        p[joint.antenna_slice] = [.01, -.02, .003, -.01, .005, .002]
        return joint, p

    def test_expected_information_matches_fixed_amplitude_mean_differences(self):
        joint, p = self.fixture()
        band = joint.bands[0]
        response = joint.responses(p)[0]
        design = band.nuisance.remove(response['target'][band.selected])
        columns = []
        for j in range(len(p)):
            delta = np.eye(len(p))[j]*1e-5
            values = []
            for q in (p+delta, p-delta):
                a, _ = band.design(joint.expand(q), antenna_offsets=joint.offsets(q, 0),
                    range_bias_m=q[joint.range_slice.start])
                values.append(band.nuisance.remove(a[band.selected]@response['coefficient']))
            columns.append((values[0]-values[1])/2e-5)
        derivative = np.column_stack(columns)
        derivative -= design@np.linalg.lstsq(design, derivative, rcond=None)[0]
        numeric = 2*np.real(derivative.conj().T@derivative)
        actual, _ = expected(joint, p)
        relative = np.linalg.norm(actual-numeric)/np.linalg.norm(numeric)
        self.assertLess(relative, 3e-5)

    def test_unknown_real_point_coordinates_cannot_add_geometry_information(self):
        joint, p = self.fixture()
        fixed, _ = expected(joint, p)
        unknown, ranks = expected(joint, p, point_coordinates=True)
        scale = np.linalg.norm(fixed, 2)
        self.assertGreaterEqual(np.linalg.eigvalsh(fixed-unknown).min(), -scale*1e-10)
        self.assertGreaterEqual(np.linalg.eigvalsh(unknown).min(), -scale*1e-10)
        self.assertGreater(np.linalg.norm(fixed-unknown), scale*.01)
        self.assertLessEqual(ranks[0], 3*len(joint.bands[0].points))


if __name__ == '__main__':
    unittest.main()
