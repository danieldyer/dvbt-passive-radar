import unittest

import numpy as np
from scipy.optimize import minimize

from profiled_point_phases import ProfiledPointPhases
import test_coherent_points as point_fixture


class ProfiledPointPhaseTests(unittest.TestCase):
    def test_profile_gradient_recovery_and_repeatability(self):
        physical, truth = point_fixture.CoherentPointTests().fixture()
        seed = truth.copy(); seed[:4] = 0.; seed[4:] += [.1, -.1, .1, -.1]
        model = ProfiledPointPhases(physical, seed, coordinate_sweeps=3)
        x = truth[:4]+[.04, -.02, .03, -.05]
        value = model.evaluate(x)
        for j in range(4):
            step = np.eye(4)[j]*5e-4
            numeric = (model.objective(x+step)[0]-model.objective(x-step)[0])/.001
            self.assertAlmostEqual(value['gradient'][j], numeric, delta=2e-7)
        repeated = model.evaluate(x)
        self.assertAlmostEqual(repeated['value'], value['value'], places=13)
        np.testing.assert_allclose(repeated['gradient'], value['gradient'], atol=1e-11)
        fit = minimize(model.objective, model.initial, jac=True, method='L-BFGS-B', bounds=[(-1., 1.)]*4,
            options={'gtol': 1e-10, 'ftol': 1e-16, 'maxiter': 200, 'maxcor': 20})
        self.assertLess(fit.fun, 1e-12)
        np.testing.assert_allclose(fit.x, truth[:4], atol=4e-4)
        self.assertLess(model.evaluate(fit.x)['inner_phase_gradient'], 1e-7)

    def test_phase_updates_reuse_exact_physical_dictionary(self):
        physical, truth = point_fixture.CoherentPointTests().fixture()
        first = physical.evaluate(truth)
        prepared = physical.prepared
        changed = truth.copy(); changed[-1] += .1
        physical.evaluate(changed)
        self.assertIs(physical.prepared, prepared)
        self.assertEqual(physical.geometry_evaluations, 1)
        physical.evaluate(truth)
        np.testing.assert_allclose(physical.last['gradient'], first['gradient'], atol=1e-14)
        changed[0] += .01
        physical.evaluate(changed)
        self.assertEqual(physical.geometry_evaluations, 2)


if __name__ == '__main__':
    unittest.main()
