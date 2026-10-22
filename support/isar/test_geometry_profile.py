import unittest

import numpy as np

from antenna_motion import Antennas
from fit_physical_profile import solve
from geometry_profile import GeometryProfile
from physical_profile import profile
import test_physical_profile as physical_tests


class GeometryProfileTests(unittest.TestCase):
    def test_profile_residual_derivatives_include_antenna_and_origin_changes(self):
        band, truth = physical_tests.PhysicalProfileTests().fixture()
        parameters, offsets, origin = truth*.4, np.array([[.7, -.6, .2], [-.3, .4, -.1]]), 1.1
        result = profile(band, parameters, antenna_offsets=offsets, range_origin=origin, derivatives=True)
        n = len(parameters)
        for j in (0, n-1, *range(n, n+7)):
            delta = np.eye(n+7)[j]*.002
            evaluations = []
            for sign in (1, -1):
                evaluations.append(profile(band, parameters+sign*delta[:n],
                    antenna_offsets=offsets+sign*delta[n:n+6].reshape(2, 3),
                    range_origin=origin+sign*delta[-1])['residual'])
            numeric = (evaluations[0]-evaluations[1])/.004
            error = np.linalg.norm(numeric-result['jacobian'][:, j])/np.linalg.norm(numeric)
            self.assertLess(error, 2e-4, (j, error))

    def test_correlated_geometry_mapping_and_prior_derivatives(self):
        band, truth = physical_tests.PhysicalProfileTests().fixture()
        antennas = Antennas(np.eye(3), [0])
        model = GeometryProfile([band], truth*.3, antennas,
            base_offsets=[[[1., -.4, .2], [-.2, .3, .1]]])
        parameters = model.initial.copy()
        parameters[model.antenna_slice] = [.01, -.02, .003, -.01, .005, .002]
        result = model.evaluate(parameters)
        for j in (0, model.nmotion-1, *range(model.antenna_slice.start, len(parameters))):
            delta = np.eye(len(parameters))[j]*.0002
            derivative = (model.objective(parameters+delta)[0]-model.objective(parameters-delta)[0])/.0004
            self.assertAlmostEqual(derivative, result['gradient'][j], delta=5e-6, msg=str(j))
        self.assertGreaterEqual(np.linalg.eigvalsh(result['gauss_newton']).min(), -1e-9)
        self.assertEqual(model.validation(parameters)[0]['stationary_validation_mode'], 'project')

    def test_known_motion_is_preserved_with_free_antenna_nuisance_at_nominal_truth(self):
        band, truth = physical_tests.PhysicalProfileTests().fixture(noise=0.)
        model = GeometryProfile([band], truth*.3, Antennas(np.eye(3), [0]))
        n = len(model.initial)
        lower, upper = np.full(n, -5.), np.full(n, 5.)
        parameters, result = solve(model, np.empty((0, n)), np.empty(0), np.empty(0),
            lower, upper, iterations=100)
        self.assertTrue(result['converged'], result)
        self.assertLess(result['objective'], 1e-12)
        self.assertLess(np.max(abs(parameters[model.antenna_slice])), 1e-4)
        self.assertLess(model.validation(parameters)[0]['conditional_efficiency_loss_db'], 1e-5)


if __name__ == '__main__':
    unittest.main()
