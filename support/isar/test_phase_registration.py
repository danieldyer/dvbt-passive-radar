import unittest

import numpy as np

from phase_registration import PhaseRegistration


class PhaseRegistrationTests(unittest.TestCase):
    def test_profile_gradient_and_known_relative_phase_recovery(self):
        rng = np.random.default_rng(325)
        a = [rng.normal(size=(100, 12))+1j*rng.normal(size=(100, 12)) for _ in range(5)]
        alpha = rng.normal(size=12)+1j*rng.normal(size=12)
        phase = np.exp(1j*np.array([.4, -.9, 1.7, 0., -.3]))
        y = [g*b@alpha for g, b in zip(phase, a)]
        model = PhaseRegistration(a, y, 1e-8)
        p = np.array([.1, -.2, .3, -.1])
        value, gradient = model.objective(p)
        for i in range(4):
            e = np.eye(4)[i]*1e-5
            numeric = (model.objective(p+e)[0]-model.objective(p-e)[0])/2e-5
            self.assertLess(abs(numeric-gradient[i]), 1e-9)
        g, fitted, result = model.fit()
        self.assertTrue(result['converged'])
        np.testing.assert_allclose(g, phase, atol=2e-7)
        np.testing.assert_allclose(fitted, alpha, atol=2e-7)
        direct = sum(np.sum(abs(b@fitted*factor-data)**2) for b, factor, data in zip(a, g, y))+1e-8*np.vdot(fitted, fitted).real
        self.assertAlmostEqual(result['objective'], direct/model.energy, places=13)

    def test_ill_conditioned_scene_does_not_corrupt_phase_derivatives(self):
        rng = np.random.default_rng(135)
        u = np.linalg.qr(rng.normal(size=(400, 15))+1j*rng.normal(size=(400, 15)))[0]
        v = np.linalg.qr(rng.normal(size=(15, 15))+1j*rng.normal(size=(15, 15)))[0]
        a = np.split((u*np.geomspace(1., 1e-9, 15))@v.conj().T, 5)
        phase = np.exp(1j*np.array([.4, -.7, 1.2, 0., -.3]))
        coefficient = rng.normal(size=15)+1j*rng.normal(size=15)
        y = [g*b@coefficient for g, b in zip(phase, a)]
        model = PhaseRegistration(a, y, 1e-13)
        p = np.array([.1, -.2, .1, -.1])
        gradient = model.objective(p)[1]
        for i in range(4):
            e = np.eye(4)[i]*1e-5
            numeric = (model.objective(p+e)[0]-model.objective(p-e)[0])/2e-5
            self.assertLess(abs(numeric-gradient[i]), 1e-9)
        fitted, _, result = model.fit()
        self.assertTrue(result['converged'])
        np.testing.assert_allclose(fitted, phase, atol=1e-6)
        self.assertGreater(min(result['curvature_eigenvalues']), 0.)

    def test_profiled_information_matches_independent_nuisance_elimination(self):
        rng = np.random.default_rng(136)
        a = [rng.normal(size=(50, 10))+1j*rng.normal(size=(50, 10)) for _ in range(5)]
        coefficient = rng.normal(size=10)+1j*rng.normal(size=10)
        model = PhaseRegistration(a, [b@coefficient for b in a], 1e-4, retain_basis=True)
        derivative = np.zeros((250, 4), complex)
        for column, band in enumerate([0, 1, 2, 4]):
            derivative[band*50:(band+1)*50, column] = 1j*a[band]@coefficient
        full = np.concatenate(a)
        residual = derivative-full@np.linalg.lstsq(full, derivative, rcond=None)[0]
        expected = 2*np.real(residual.conj().T@residual)
        detail = model.phase_information(coefficient)
        for d in detail['profiled_scene']:
            np.testing.assert_allclose(d['information_matrix'], expected, atol=1e-10)
            self.assertGreater(min(d['fraction_of_fixed_scene_information_eigenvalues']), .1)

    def test_disjoint_scene_modes_cannot_calibrate_channel_phases(self):
        rng = np.random.default_rng(137)
        a = []
        for c in range(5):
            block = np.zeros((40, 30), complex)
            block[:, c*6:(c+1)*6] = rng.normal(size=(40, 6))+1j*rng.normal(size=(40, 6))
            a.append(block)
        coefficient = rng.normal(size=30)+1j*rng.normal(size=30)
        model = PhaseRegistration(a, [b@coefficient for b in a], .01, retain_basis=True)
        detail = model.phase_information(coefficient)
        self.assertGreater(np.linalg.eigvalsh(detail['fixed_scene_information']).min(), 1.)
        for d in detail['profiled_scene']:
            self.assertLess(max(d['fraction_of_fixed_scene_information_eigenvalues']), 1e-25)


if __name__ == '__main__':
    unittest.main()
