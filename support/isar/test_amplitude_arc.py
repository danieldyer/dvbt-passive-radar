import unittest

import numpy as np

from amplitude_arc import AmplitudeArc, linear_basis
from stationary import make_design
from test_cluster_phase import synthetic


def problem():
    knots = np.array([164., 164.4, 164.8, 165.2, 165.7])
    strength = np.array([.2, 1.4, .1, .8, .3])
    original, truth = synthetic(row_amplitude=lambda t:np.interp(t, knots, strength))
    f = original.field
    f.data['centre_path_m'] = 1200.-.56*(f.data['row_time']-original.epoch)
    design, _ = make_design(f)
    rng = np.random.default_rng(149)
    beta = 4*(rng.normal(size=design.shape[1])+1j*rng.normal(size=design.shape[1]))
    f.y += (design@beta).reshape(f.y.shape)
    return AmplitudeArc(f, original.training, knots), truth


class AmplitudeArcTests(unittest.TestCase):
    def test_original_row_partition_and_constant_strength_limit(self):
        p, truth = problem()
        np.testing.assert_allclose(p.amplitude_basis.sum(axis=1), 1., atol=1e-15)
        self.assertTrue(np.all(p.amplitude_basis >= 0))
        raw, _ = p.response(truth)
        design = p.reduced(raw[:, :, None]*p.amplitude_basis[:, None, :])
        np.testing.assert_allclose(design.sum(axis=1), p.base.template(truth).ravel(), atol=2e-14)
        with self.assertRaises(ValueError):
            linear_basis(np.array([0., 2.]), np.array([0., 1.]))

    def test_profiled_gradient_recovery_and_frozen_prediction(self):
        p, truth = problem()
        x = truth+np.array([.03, .04, .015, -.01])
        _, gradient = p.evaluate(x)
        finite = [(p.evaluate(x+d)[0]-p.evaluate(x-d)[0])/2e-5 for d in np.eye(4)*1e-5]
        np.testing.assert_allclose(gradient, finite, atol=2e-8, rtol=2e-5)
        fit = p.fit(seeds=[x])
        np.testing.assert_allclose(fit.parameters, truth, atol=3e-4)
        self.assertLess(fit.gradient, 1e-6)
        d = p.diagnostics(fit)
        self.assertGreater(d['test_complex_prediction_fraction'], .99999)
        self.assertLess(abs(d['test_coherent_phase_error_deg']), .002)

    def test_withheld_data_cannot_change_fit_or_initialisation(self):
        p, truth = problem()
        initial = p.initial((-.8, .8), 2.)
        fit = p.fit(seeds=[truth])
        p.field.y[~p.training] *= np.exp(1.3j)
        other = AmplitudeArc(p.field, p.training, p.knots)
        np.testing.assert_array_equal(initial, other.initial((-.8, .8), 2.))
        np.testing.assert_array_equal(fit.parameters, other.fit(seeds=[truth]).parameters)
        with self.assertRaises(ValueError):
            p.diagnostics(fit, p.training)

    def test_phase_diagnostic_does_not_report_zero_at_zero_predicted_strength(self):
        p, truth = problem()
        fit = p.fit(seeds=[truth])
        evaluation = p.evaluate(fit.parameters, True)
        # Freeze the actual fit, then force a zero-strength predicted knot
        # vector. The phase diagnostic must still use the unit trajectory.
        frozen = list(evaluation); frozen[3] = np.zeros_like(frozen[3])
        p.evaluate = lambda parameters, full=False: tuple(frozen) if full else frozen[:2]
        result = p.diagnostics(fit)
        self.assertGreater(np.max(np.abs(result['phase_residual_deg'])), .1)


if __name__ == '__main__':
    unittest.main()
