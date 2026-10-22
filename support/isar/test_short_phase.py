"""Known-phase controls for the local ship phase-transfer diagnostic."""
import unittest

import numpy as np

from short_phase import Patch, fit_rate


class ShortPhaseTests(unittest.TestCase):
    def make(self, seed=3, rates=None, noise=0.):
        rng = np.random.default_rng(seed)
        patches = []
        for c, f in enumerate([177.5, 184.5, 191.625, 219.5, 226.5]):
            t = np.linspace(0, 2, 101, endpoint=False)+c*.0003
            k = 2*np.pi*f/299.792458
            path = 1.1*(t-1)+.07*(t-1)**2
            rate = .273 if rates is None else rates[c]
            # Independent constant phases, unknown clutter, nonuniform weights.
            a = (rng.normal(size=7)+1j*rng.normal(size=7))*np.exp(1j*c*.79)
            b = 10*(rng.normal(size=7)+1j*rng.normal(size=7))
            w = rng.uniform(.5, 2, len(t))
            y = b+np.exp(-1j*k*(path+rate*(t-1)))[:, None]*a
            y += noise*(rng.normal(size=y.shape)+1j*rng.normal(size=y.shape))/np.sqrt(2*w[:, None])
            patches.append(Patch(t, y, w, path, k, 1., t < 1))
        return patches

    def test_unknown_constant_band_phases_and_strong_clutter_transfer(self):
        patches = self.make()
        rate = fit_rate(patches)
        self.assertAlmostEqual(rate, .273, places=7)
        for p in patches:
            e = p.evaluate(rate)
            self.assertLess(e['test_phase_rms_deg'], 1e-5)
            self.assertLess(e['test_error'], 1e-12)

    def test_band_specific_drift_is_detected_on_future_samples(self):
        patches = self.make(rates=[.273, .273, .273, .273, .8])
        common = fit_rate(patches)
        common_error = sum(p.evaluate(common)['test_error'] for p in patches)
        independent_error = sum(p.evaluate(fit_rate([p]))['test_error'] for p in patches)
        self.assertGreater(common_error, 1e3)
        self.assertLess(independent_error, 1e-10)

    def test_training_does_not_use_held_out_samples(self):
        patches = self.make(noise=.1)
        before = fit_rate(patches)
        brightness = [p.brightness() for p in patches]
        for p in patches:
            p.value[~p.training] *= np.exp(1j*1.3)
        self.assertEqual(before, fit_rate(patches))
        self.assertEqual(brightness, [p.brightness() for p in patches])

    def test_exactly_zero_doppler_is_declared_unidentifiable(self):
        p = self.make()[0]
        p.path[:] = 0
        self.assertEqual(p.coefficients(0)[3], 0)

    def test_reversible_known_phase_deramp_leaves_inference_unchanged(self):
        patches = self.make(noise=.1)
        before = fit_rate(patches)
        errors = [p.evaluate(before)['test_error'] for p in patches]
        for p in patches:
            rotation = np.exp(1j*p.k*p.path)
            p.value *= rotation[:, None]
            p.background = rotation
            p.path[:] = 0
        after = fit_rate(patches)
        self.assertAlmostEqual(after, before, places=7)
        np.testing.assert_allclose([p.evaluate(after)['test_error'] for p in patches], errors, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
