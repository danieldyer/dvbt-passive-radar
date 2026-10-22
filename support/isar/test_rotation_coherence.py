import unittest

import numpy as np

from rotation_coherence import path_change


class RotationCoherenceTests(unittest.TestCase):
    def test_exact_path_change_matches_independent_distances_and_local_derivative(self):
        rng = np.random.default_rng(31411)
        centre = rng.normal(size=(25, 3))*100+[1500., -13000., 40.]
        frame = np.tile(np.eye(3), (len(centre), 1, 1))
        point = np.array([150., -30., 12.])
        tx, rx = np.array([32000., -3000., 700.]), np.array([4., 5., 40.])
        angle = np.linspace(-.003, .003, len(centre))
        c, s = np.cos(angle), np.sin(angle)
        moved = np.c_[c*point[0]-s*point[1], s*point[0]+c*point[1], np.full(len(angle), point[2])]
        direct = np.linalg.norm(tx-rx)
        expected = (sum(np.linalg.norm(centre+moved-a, axis=1) for a in (tx, rx))-direct
                    -(sum(np.linalg.norm(centre+point-a, axis=1) for a in (tx, rx))-direct))
        actual = path_change(centre, frame, point, angle, tx, rx)
        np.testing.assert_allclose(actual, expected, atol=2e-11)
        step = np.full(len(centre), 1e-6)
        numeric = (path_change(centre, frame, point, step, tx, rx)
                   -path_change(centre, frame, point, -step, tx, rx))/(2e-6)
        position = centre+point
        gradient = sum((position-a)/np.linalg.norm(position-a, axis=1)[:, None] for a in (tx, rx))
        np.testing.assert_allclose(numeric, gradient@np.cross([0., 0., 1.], point), rtol=1e-9)
        np.testing.assert_array_equal(path_change(centre, frame, np.zeros(3), angle, tx, rx), 0.)

    def test_uniform_aperture_phase_correlation_agrees_with_sinc_loss(self):
        nodes, weights = np.polynomial.legendre.leggauss(256)
        for cycles in (.01, .2, .7, 1.2, 2.3):
            exact = abs((weights/2)@np.exp(-1j*np.pi*cycles*nodes))**2
            self.assertAlmostEqual(exact, np.sinc(cycles)**2, places=13)


if __name__ == '__main__':
    unittest.main()
