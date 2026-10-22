import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from heading_seed import Band, fit
from motion import body_basis, ecef, enu_basis, excess_path


def fixture(channel=0):
    t = np.linspace(0, 12, 180)
    lat, lon = np.radians([-37.87, 144.93])
    origin, local = ecef(lat, lon, 5), enu_basis(lat, lon)
    centre = origin+t[:, None]*(local@np.array([.4, 4., 0.]))
    frame = body_basis(np.full(len(t), lat), np.full(len(t), lon), -.1-.003*t)
    tx = origin+local@np.array([35000+200*channel, 15000., 700.])
    rx = origin+local@np.array([1000., 8000., 50.])
    _, g = excess_path(centre, tx, rx)
    direction = g/np.sum(g*g, axis=1)[:, None]
    points = np.array([[-90, -12, 20], [-25, 7, 10], [45, -8, 18], [110, 12, 25]])
    return Band(t, np.zeros((len(t), 4), complex), 1+.2*np.sin(t), points,
                centre, frame, tx, rx, direction, (180+20*channel)*1e6, [0, 12],
                path_spacing=4, heading_spacing=6)


class HeadingSeedTests(unittest.TestCase):
    def test_phase_matches_independent_rigid_rotation_and_bistatic_ranges(self):
        b = fixture()
        p = np.linspace(-.15, .25, b.count)
        phase, _, _ = b.phase(p, True)
        distance = b.path_basis@p[:b.npath]
        yaw = b.heading_basis@p[b.npath:]
        frame = b.frame@Rotation.from_euler('z', yaw[:, None], degrees=True).as_matrix()
        # Express point columns explicitly to avoid sharing the implementation's
        # stable difference or Euler rotation evaluation.
        original = b.centre[:, None, :]+np.einsum('nij,pj->npi', b.frame, b.pixels)
        moved = (b.centre+distance[:, None]*b.direction)[:, None, :]+np.einsum('nij,pj->npi', frame, b.pixels)
        reference = sum(np.linalg.norm(moved-a, axis=-1)-np.linalg.norm(original-a, axis=-1) for a in (b.tx, b.rx))
        np.testing.assert_allclose(phase, -b.k*reference, atol=1e-7)

    def test_likelihood_gradient_with_free_heading_offset(self):
        rng = np.random.default_rng(148)
        b = fixture()
        b.value = rng.normal(size=b.value.shape)+1j*rng.normal(size=b.value.shape)
        p = rng.normal(size=b.count)*.05
        _, derivative = b.evaluate(p)
        step = 1e-4
        numeric = np.array([(b.evaluate(p+delta)[0]-b.evaluate(p-delta)[0])/(2*step)
                            for delta in np.eye(b.count)*step])
        np.testing.assert_allclose(derivative, numeric, rtol=1e-4, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
