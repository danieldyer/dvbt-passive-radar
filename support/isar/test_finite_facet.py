import unittest

import numpy as np

from finite_facet import Facet, C, field, path, local_signature, stationary_coordinates, backproject


class FacetTest(unittest.TestCase):
    def setUp(self):
        self.rx = np.zeros(3)
        self.tx = np.array([31643., -3064., 680.])
        self.p = np.array([-5321., -6846., -42.])
        from finite_facet import geometry
        g, _ = geometry(self.p, self.tx, self.rx)
        u = np.array([g[1], -g[0], 0.]); u /= np.linalg.norm(u)
        self.f = Facet(self.p, [-1.98, 4.45, .003], u, np.radians(.227), 300.)

    def test_zero_length_is_exact_point(self):
        f = Facet(self.p, self.f.velocity, self.f.tangent, self.f.spin_radps, 0.)
        offsets = np.arange(-90., 91., 3.)
        actual = field(f, [-2., 0., 2.], offsets, self.tx, self.rx, 191.625e6)
        np.testing.assert_allclose(actual, np.tile(np.sinc(offsets*6.65625e6/C), (3, 1)), atol=3e-14)

    def test_quadrature_against_independent_dense_midpoints(self):
        t = np.array([-2.1, -.3, 1.7])
        offsets = np.array([-25., 0., 35.])
        actual = field(self.f, t, offsets, self.tx, self.rx, 226.5e6, quadrature=128)
        q = (np.arange(32768)+.5)/32768*self.f.length_m-self.f.length_m/2
        positions = self.f.points(t, q)
        difference = path(positions, self.tx, self.rx)-path(self.f.points(t, [0.])[:, 0], self.tx, self.rx)[:, None]
        reference = np.mean(np.sinc((offsets[None, :, None]-difference[:, None, :])*6.65625e6/C)*
            np.exp(-2j*np.pi*226.5e6/C*difference[:, None, :]), axis=-1)
        np.testing.assert_allclose(actual, reference, atol=3e-8, rtol=3e-8)

    def test_stationary_motion_curvature(self):
        signature = local_signature(self.f, self.tx, self.rx)
        step = .01
        t = np.array([-step, 0., step])
        q = stationary_coordinates(self.f, t, self.tx, self.rx)
        numerical_speed = (q[2]-q[0])/(2*step)
        self.assertAlmostEqual(numerical_speed, signature['glint_speed_mps'], delta=2e-4)
        positions = self.f.points(t, [0.])[:, 0]+q[:, None]*self.f.frame(t)
        l = path(positions, self.tx, self.rx)
        acceleration = (l[2]-2*l[1]+l[0])/step**2
        expected = signature['material_origin_acceleration_mps2']+signature['glint_extra_acceleration_at_specular_mps2']
        self.assertAlmostEqual(acceleration, expected, delta=2e-6)

    def test_point_adjoint_phase_and_multiband_sum(self):
        f = Facet(self.p, self.f.velocity, self.f.tangent, self.f.spin_radps, 0.)
        times, offsets = np.linspace(-1., 1., 31), np.arange(-90., 90.1, .5)
        tx = np.array([self.tx, self.tx+[100., 200., -20.]])
        rf = [177.5e6, 226.5e6]
        profiles = [field(f, times, offsets, antenna, self.rx, frequency) for antenna, frequency in zip(tx, rf)]
        pixels = np.array([[0., 0., 0.], [40., 40., 0.]])
        images = backproject(f, times, offsets, profiles, pixels, tx, self.rx, rf)
        np.testing.assert_allclose(images[:, 0], 1., atol=3e-14)
        self.assertAlmostEqual(abs(images[:, 0].sum())**2, 4., places=12)
        self.assertLess(abs(images[:, 1].sum())**2, .04)


if __name__ == '__main__':
    unittest.main()
