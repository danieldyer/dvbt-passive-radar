import unittest

import numpy as np

from delayed_path import DelayedPath
import test_physical_profile as physical_tests


class DelayedPathTests(unittest.TestCase):
    def test_delay_and_rate_derivatives_include_phase_envelope_and_symbol_moments(self):
        band, truth = physical_tests.PhysicalProfileTests().fixture()
        model = DelayedPath(band, truth)
        at = np.array([300., .17])
        value, derivative = model.response(at, derivatives=True)
        for j, step in enumerate([1e-3, 1e-5]):
            delta = np.eye(2)[j]*step
            numeric = (model.response(at+delta)-model.response(at-delta))/(2*step)
            np.testing.assert_allclose(numeric, derivative[:, j], rtol=1e-4, atol=1e-7)
        result = model.evaluate(at)
        for j, step in enumerate([1e-3, 1e-5]):
            delta = np.eye(2)[j]*step
            numeric = (model.objective(at+delta)[0]-model.objective(at-delta)[0])/(2*step)
            self.assertAlmostEqual(numeric, result['gradient'][j], delta=1e-6)

    def test_noiseless_amplitude_is_preserved_with_main_and_stationary_nuisance(self):
        band, truth = physical_tests.PhysicalProfileTests().fixture()
        model = DelayedPath(band, truth)
        at, amplitude = np.array([400., .11]), .7-.4j
        nuisance = model.nuisance.design@np.linspace(.1, .2j, model.nuisance.design.shape[1])
        band.field.y = (amplitude*model.response(at)+nuisance).reshape(band.field.y.shape)
        measured = DelayedPath(band, truth).evaluate(at)
        self.assertLess(measured['value'], 1e-19)
        self.assertLess(abs(measured['amplitude']-amplitude), 1e-10)


if __name__ == '__main__':
    unittest.main()
