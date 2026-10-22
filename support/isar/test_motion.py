"""Geometry conventions which directly affect bistatic carrier phase."""
import unittest

import numpy as np

from motion import C, WGS84_A, body_basis, ecef, enu_basis, excess_path, path_difference


class MotionTests(unittest.TestCase):
    def test_geodetic_axes_and_broadcast(self):
        np.testing.assert_allclose(ecef(0, np.array([0, np.pi/2]), 10),
                                   [[WGS84_A+10, 0, 0], [0, WGS84_A+10, 0]], atol=1e-9)
        lat, lon = np.radians([-37.81, 144.99])
        basis = enu_basis(lat, lon)
        np.testing.assert_allclose(basis.T@basis, np.eye(3), atol=1e-15)
        self.assertAlmostEqual(np.linalg.det(basis), 1)

    def test_body_heading_is_forward_port_up(self):
        lat, lon = np.radians([-37.81, 144.99])
        enu = enu_basis(lat, lon)
        north = enu.T@body_basis(lat, lon, 0.)
        east = enu.T@body_basis(lat, lon, np.pi/2)
        np.testing.assert_allclose(north, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-15)
        np.testing.assert_allclose(east, np.eye(3), atol=1e-15)
        with self.assertRaises(ValueError):
            body_basis(lat, lon, np.nan)

    def test_excess_path_and_doppler_share_the_same_derivative(self):
        # Finite differences of an independent moving ECEF point. Excess path
        # is the sum of two legs minus the direct path, with no extra factor 2.
        tx = ecef(*np.radians([-37.8375, 145.3466]), 760)
        rx = ecef(*np.radians([-37.8108, 144.9873]), 48)
        point = ecef(*np.radians([-37.94, 144.92]), 30)
        velocity = enu_basis(*np.radians([-37.94, 144.92]))@np.array([0.8, 6., 0.])
        path, gradient = excess_path(point, tx, rx)
        h = .01
        numerical = (excess_path(point+h*velocity, tx, rx)[0]-excess_path(point-h*velocity, tx, rx)[0])/(2*h)
        self.assertAlmostEqual(numerical, gradient@velocity, delta=1e-7)
        rf = 226.5e6
        samples = np.exp(-2j*np.pi*rf*np.array([
            excess_path(point-h*velocity, tx, rx)[0],
            excess_path(point+h*velocity, tx, rx)[0]])/C)
        fd_from_phase = np.angle(samples[1]*samples[0].conjugate())/(4*np.pi*h)
        self.assertAlmostEqual(fd_from_phase, -rf/C*(gradient@velocity), delta=1e-7)
        self.assertGreater(path, 0)

    def test_exact_offset_path_at_earth_scale_preserves_phase(self):
        tx = ecef(*np.radians([-37.8375, 145.3466]), 760)
        rx = ecef(*np.radians([-37.8108, 144.9873]), 48)
        centre = ecef(*np.radians([-37.94, 144.92]), 30)
        basis = body_basis(*np.radians([-37.94, 144.92]), np.radians(8.))
        offsets = np.array([[140, 20, 15], [-140, -20, 0], [.0001, 0, 0]])@basis.T
        # Direct double norm calculation is independent of the stable quotient.
        expected = (np.linalg.norm(centre+offsets-tx, axis=1)-np.linalg.norm(centre-tx)
                    +np.linalg.norm(centre+offsets-rx, axis=1)-np.linalg.norm(centre-rx))
        difference = path_difference(centre, offsets, tx, rx)
        phase_error_degrees = (difference-expected)*360*226.5e6/C
        self.assertLess(np.max(abs(phase_error_degrees)), 1e-6)


if __name__ == '__main__':
    unittest.main()
