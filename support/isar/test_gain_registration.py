import unittest

import numpy as np

from gain_registration import GainRegistration


class GainRegistrationTests(unittest.TestCase):
    def test_profile_gradient_and_gain_recovery(self):
        rng = np.random.default_rng(325)
        a = [rng.normal(size=(60, 8))+1j*rng.normal(size=(60, 8)) for _ in range(5)]
        coefficient = rng.normal(size=8)+1j*rng.normal(size=8)
        gains = np.exp(np.array([.4, -.9, .7, 0., -.3])+1j*np.array([.6, -.4, 1.2, 0., -2.]))
        y = [g*(b@coefficient) for g, b in zip(gains, a)]
        model = GainRegistration(a, y, 1e-9, orthogonal_energy=17.)
        p = rng.normal(size=8)*.2
        gradient = model.objective(p)[1]
        for i in range(8):
            delta = np.eye(8)[i]*1e-5
            numerical = (model.objective(p+delta)[0]-model.objective(p-delta)[0])/2e-5
            self.assertLess(abs(numerical-gradient[i]), 1e-9)
        fitted, scene, result = model.fit(random_starts=1)
        self.assertTrue(result['converged'])
        self.assertFalse(result['gain_bound_hit'])
        np.testing.assert_allclose(fitted, gains, atol=1e-6)
        np.testing.assert_allclose(scene, coefficient, atol=1e-6)
        energy = sum(np.sum(abs(g*(b@scene)-z)**2) for g, b, z in zip(fitted, a, y))
        energy += 17.+1e-9*np.vdot(scene, scene).real
        self.assertAlmostEqual(energy/model.energy, result['objective'], places=13)

    def test_ill_conditioned_scene_gradient(self):
        rng = np.random.default_rng(136)
        u = np.linalg.qr(rng.normal(size=(200, 12))+1j*rng.normal(size=(200, 12)))[0]
        v = np.linalg.qr(rng.normal(size=(12, 12))+1j*rng.normal(size=(12, 12)))[0]
        a = np.split((u*np.geomspace(1., 1e-10, 12))@v.conj().T, 5)
        coefficient = rng.normal(size=12)+1j*rng.normal(size=12)
        model = GainRegistration(a, [b@coefficient for b in a], 1e-12)
        p = rng.normal(size=8)*.1
        gradient = model.objective(p)[1]
        for i in range(8):
            delta = np.eye(8)[i]*1e-5
            numerical = (model.objective(p+delta)[0]-model.objective(p-delta)[0])/2e-5
            self.assertLess(abs(numerical-gradient[i]), 1e-8)


if __name__ == '__main__':
    unittest.main()
