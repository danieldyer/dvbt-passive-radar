import unittest

import numpy as np

from gpga import Isolation, SplineLikelihood, fit, refine, remove_affine, phase_metrics
from matched_phase import history, prediction_error


class PhaseGradientTests(unittest.TestCase):
    def test_perfect_coherence_has_finite_zero_circular_variance(self):
        values = np.tile(np.exp(1j*np.linspace(-np.pi, np.pi, 101)), (7, 1))
        with np.errstate(invalid='raise'):
            detail = phase_metrics(np.arange(7), values, np.arange(1., 8.),
                np.ones(7, bool), values[0])
        np.testing.assert_allclose(detail['circular_std_degrees_by_point'], 0., atol=2e-6)

    def test_matched_history_is_sufficient_with_changing_reflector_strength(self):
        rng = np.random.default_rng(85)
        t = np.linspace(0, 20, 600)
        template = (1+.8*np.sin(t[:, None]*np.array([.3, .2, .5]))) * np.exp(1j*t[:, None]*np.array([.2, .3, -.4]))
        phase = .7*np.sin(.5*t)
        data = template*np.exp(1j*phase[:, None])+.02*(rng.normal(size=template.shape)+1j*rng.normal(size=template.shape))
        z, information = history(data, template)
        np.testing.assert_allclose(prediction_error(data, template, phase, np.ones(len(t), bool)),
            np.sum(abs(data)**2)+information.sum()-2*np.sum(information*np.real(z[:, 0]*np.exp(-1j*phase))), atol=1e-10)
        selected = np.arange(len(t))%2 == 0
        phase = remove_affine(t, phase, information*selected)
        clean = template*np.exp(1j*phase[:, None])
        z, subset_weight = history(clean, template, [0, 2])
        found, diagnostic = refine(t, z, subset_weight, selected, np.zeros(len(t)), knot_seconds=.5,
                                   gauge_information=information)
        self.assertTrue(diagnostic['converged'])
        self.assertLess(np.sqrt(np.mean((found-phase)**2)), .001)
        self.assertLess(prediction_error(clean, template, found, ~selected, [1]), .0001)

    def test_spline_gradient_with_nonuniform_information_and_holdout(self):
        rng = np.random.default_rng(314)
        t = np.linspace(0, 10, 200)
        z = rng.normal(size=(len(t), 4))+1j*rng.normal(size=(len(t), 4))
        model = SplineLikelihood(t, z, np.exp(rng.normal(size=len(t))), np.arange(len(t))%3 != 0, 1.,
                                 gauge_information=np.exp(rng.normal(size=len(t))))
        p = rng.normal(size=model.basis.shape[1])
        value, gradient = model.objective(p)
        step = 1e-5
        numeric = np.array([(model.objective(p+delta)[0]-model.objective(p-delta)[0])/(2*step)
                            for delta in np.eye(len(p))*step])
        np.testing.assert_allclose(gradient, numeric, atol=1e-10, rtol=1e-6)

    def test_smooth_phase_refinement_reaches_stationary_solution(self):
        t = np.linspace(0, 15, 800)
        w = 1+.2*np.sin(t)
        train = np.arange(len(t))%2 == 0
        phase = remove_affine(t, .6*np.sin(.6*t)+.15*np.cos(1.1*t), w*train)
        z = np.exp(1j*phase[:, None])*np.array([1, 2j, -.7+.3j])
        fitted, report = refine(t, z, w, train, np.zeros(len(t)), knot_seconds=.5)
        self.assertTrue(report['converged'], report['maximum_gradient'])
        self.assertLess(np.sqrt(np.mean((fitted-phase)**2)), .001)

    def test_isolation_preserves_constant_and_has_consistent_time_derivative(self):
        t = np.linspace(0, 20, 800)
        selected = np.arange(len(t)) % 2 == 0
        kernel = Isolation(t, np.ones(len(t)), selected, .8)
        z, dz = kernel.apply(np.ones((len(t), 2), complex)*(1+2j))
        np.testing.assert_allclose(z, 1+2j, atol=2e-14)
        np.testing.assert_allclose(dz, 0, atol=2e-14)
        signal = np.exp(.3j*t)[:, None]
        z, dz = kernel.apply(signal)
        expected = np.gradient(z, t, axis=0)
        np.testing.assert_allclose(dz[30:-30], expected[30:-30], atol=6e-5)

    def test_known_phase_with_doppler_contaminants_and_time_holdout(self):
        random = np.random.default_rng(817)
        t = np.linspace(0, 30, 1200)
        selected = np.arange(len(t)) % 2 == 0
        w = 1+.2*np.sin(t)
        phase = remove_affine(t, 1.2*np.sin(.6*t)+.5*np.cos(.35*t), w*selected)
        amplitude = np.array([1, 1j, -.7+.2j])
        interference = .18*np.exp(2j*np.pi*t[:, None]*np.array([3.2, -2.7, 4.1]))
        noise = .015*(random.normal(size=interference.shape)+1j*random.normal(size=interference.shape))
        z = (amplitude+interference)*np.exp(1j*phase[:, None])+noise
        result = fit(t, z, w, selected, bandwidths_hz=(1.5, .8, .4), iterations_per_bandwidth=20)
        error = remove_affine(t, result['phase_radians']-phase, w*selected)
        self.assertLess(np.sqrt(np.mean(error[60:-60]**2)), .09)
        self.assertGreater(result['heldout_coherent_power_gain_db'], 2)
        self.assertLess(result['after']['prediction_error'], result['before']['prediction_error']/4)

    def test_heldout_samples_never_change_estimated_phase(self):
        t = np.linspace(0, 10, 400)
        selected = np.arange(len(t)) % 2 == 0
        z = np.exp(1j*np.sin(t))[:, None]*np.array([1, 2j])
        first = fit(t, z, np.ones(len(t)), selected, bandwidths_hz=(1., .5))
        z[~selected] *= np.exp(5j*np.cos(t[~selected, None]))
        second = fit(t, z, np.ones(len(t)), selected, bandwidths_hz=(1., .5))
        np.testing.assert_allclose(first['phase_radians'], second['phase_radians'], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
