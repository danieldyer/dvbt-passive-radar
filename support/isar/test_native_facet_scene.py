import unittest

import numpy as np

from native_facet_scene import FacetBand, fit
from test_recovery_experiment import problem


class NativeFacetTest(unittest.TestCase):
    def band(self, c=1, quadrature=24):
        ex = problem()
        from finite_facet import geometry
        local_tx = (ex.transmitters[2]-ex.receiver)@ex.rotation.T
        g, _ = geometry(ex.state[0], local_tx, np.zeros(3))
        tangent = np.array([g[1], -g[0], 0.]); tangent /= np.linalg.norm(tangent)
        return FacetBand(ex.fields[c], ex.training[c], ex.designs[c], ex.receiver, ex.rotation,
            ex.state, ex.midpoint, tangent, np.array([.5, .3, 0.]), quadrature=quadrature)

    def test_physical_material_velocity_and_six_derivatives(self):
        b = self.band(); p = np.array([2., .03, 1., .15, 20., 120.])
        for i in range(6):
            delta = np.eye(6)[i]*1e-4
            a, av, dp, dv = b.material_state(p, .32)
            up, uv, _, _ = b.material_state(p+delta, .32)
            down, dvel, _, _ = b.material_state(p-delta, .32)
            np.testing.assert_allclose((up-down)/(2*delta[i]), dp[:, :, i], atol=3e-7, rtol=2e-6)
            np.testing.assert_allclose((uv-dvel)/(2*delta[i]), dv[:, :, i], atol=1e-8, rtol=2e-6)

    def test_native_profile_and_gain_nuisance_derivatives(self):
        b = self.band(); p = np.array([2., .03, 1., .15, 20., 120.])
        h, j = b.template(p)
        for i in range(6):
            step = 2e-4 if i in [0, 4, 5] else 2e-6
            d = np.eye(6)[i]*step
            finite = (b.template(p+d, derivatives=False)-b.template(p-d, derivatives=False))/(2*step)
            self.assertLess(np.linalg.norm(finite-j[:, i])/np.linalg.norm(finite), 3e-5)
        rng = np.random.default_rng(12)
        b.field.y[:] = ((.7+.2j)*h+.02*(rng.normal(size=len(h))+1j*rng.normal(size=len(h)))).reshape(b.field.y.shape)
        b.clean = b.nuisance.remove(b.field.y.ravel()[b.selected]); b.last = None
        r = b.evaluate(p)
        for i in range(6):
            step = 2e-4 if i in [0, 4, 5] else 2e-6; d = np.eye(6)[i]*step
            numeric = (b.evaluate(p+d)['residual']-b.evaluate(p-d)['residual'])/(2*step)
            self.assertLess(np.linalg.norm(numeric-r['jacobian'][:, i])/np.linalg.norm(numeric), 3e-5)

    def test_unknown_spin_and_motion_recovery_with_finer_synthesis(self):
        truth = np.array([2., .03, 1., .15, 20., 120.])
        bands = []
        for c in [0, 2, 4]:
            source = self.band(c, quadrature=64)
            h = source.template(truth, derivatives=False)
            source.field.y[:] = ((.8+.15j)*h).reshape(source.field.y.shape)
            b = FacetBand(source.field, source.training, source.design, source.receiver,
                source.rotation, source.state, source.midpoint, source.tangent, source.direction, quadrature=24)
            bands.append(b)
        seed = truth+np.array([1., .02, .2, .03, 0., 0.])
        result = fit(bands, seed, free=(0, 1, 2, 3), max_evaluations=70)
        self.assertTrue(result['success'], result)
        self.assertLess(result['objective'], 1e-13)
        np.testing.assert_allclose(result['parameters'][:4], truth[:4], atol=2e-5)
        for b in bands:
            self.assertLess(b.prediction(result['parameters'])['test_rss']/b.energy, 1e-13)


if __name__ == '__main__':
    unittest.main()
