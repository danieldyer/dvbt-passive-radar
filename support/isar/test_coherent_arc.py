import unittest

import numpy as np

from coherent_arc import CoherentArc
from stationary import make_design
from test_cluster_phase import synthetic


def problem():
    original, truth = synthetic()
    f = original.field
    f.data['centre_path_m'] = 1200.-.56*(f.data['row_time']-original.epoch)
    amplitude = .8+.2*np.cos(2*np.pi*(f.time-f.time[0])/(f.time[-1]-f.time[0]))
    f.y /= amplitude[:, None]
    design, _ = make_design(f)
    beta = np.random.default_rng(45).normal(size=design.shape[1])+1j*np.random.default_rng(46).normal(size=design.shape[1])
    f.y += (design@beta*4).reshape(f.y.shape)
    return CoherentArc(f, original.training), truth


class CoherentArcTests(unittest.TestCase):
    def test_exact_profiled_gradient_and_recovery_with_strong_clutter(self):
        p, truth = problem()
        x = truth[[0, 2, 3]]+np.array([.03, .02, -.01])
        _, gradient = p.evaluate(x)
        finite = [(p.evaluate(x+d)[0]-p.evaluate(x-d)[0])/2e-5 for d in np.eye(3)*1e-5]
        np.testing.assert_allclose(gradient, finite, atol=2e-8, rtol=2e-5)
        fit = p.fit(seeds=[x])
        np.testing.assert_allclose(fit.parameters, truth, atol=3e-4)
        diagnostic = p.diagnostics(fit)
        self.assertGreater(diagnostic['test_complex_prediction_fraction'], .99999)
        self.assertLess(abs(diagnostic['test_coherent_phase_error_deg']), .002)

    def test_test_data_cannot_change_training_fit_or_seeds(self):
        p, truth = problem()
        fit = p.fit(seeds=[truth[[0, 2, 3]]])
        initial = p.initial((-1., 1.), 2.)
        p.field.y[~p.training] *= np.exp(1.1j)
        # A freshly constructed object must also use training samples only.
        altered = CoherentArc(p.field, p.training)
        other = altered.fit(seeds=[truth[[0, 2, 3]]])
        np.testing.assert_array_equal(initial, altered.initial((-1., 1.), 2.))
        np.testing.assert_allclose(fit.parameters, other.parameters, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
