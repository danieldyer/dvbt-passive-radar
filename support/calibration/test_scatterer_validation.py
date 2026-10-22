import unittest
from types import SimpleNamespace

import numpy as np

from scatterer_validation import project_points, RigidEcho, body_frames
from geometry import C, RF, central_jacobian
from range_fit import Field, Pulse
from road_geometry import RoadCurve
from road_model import RoadTrajectory, RoadProblem


class PointProjectionTest(unittest.TestCase):
    def test_exact_complex_points(self):
        rng = np.random.default_rng(19)
        u = rng.normal(size=(40, 2))+1j*rng.normal(size=(40, 2))
        a = np.array([1+.7j, -.2+2j])
        residual, actual = project_points(u@a, u)
        np.testing.assert_allclose(actual, a, atol=1e-14)
        np.testing.assert_allclose(residual, 0, atol=1e-14)

    def test_variable_projection_jacobian_including_residual(self):
        rng = np.random.default_rng(20)
        u = rng.normal(size=(60, 2))+1j*rng.normal(size=(60, 2))
        y = rng.normal(size=60)+1j*rng.normal(size=60)
        du = rng.normal(size=(60, 2, 4))+1j*rng.normal(size=(60, 2, 4))
        _, jacobian, _ = project_points(y, u, du)
        h = 1e-5
        for k in range(4):
            numeric = (project_points(y, u+h*du[:, :, k])[0]-
                       project_points(y, u-h*du[:, :, k])[0])/(2*h)
            np.testing.assert_allclose(jacobian[:, k], numeric, rtol=2e-8, atol=2e-9)

    def test_coincident_points_have_no_extra_information(self):
        rng = np.random.default_rng(21)
        u = rng.normal(size=50)+1j*rng.normal(size=50)
        y = 2j*u+rng.normal(size=50)*.01
        one, _ = project_points(y, u[:, None])
        two, _ = project_points(y, np.column_stack([u, u]))
        np.testing.assert_allclose(one, two, atol=5e-15)


class RigidModelTest(unittest.TestCase):
    def setUp(self):
        road = RoadCurve({'enu_m': [[0, -2000, 0], [300, -1950, 2], [600, -1800, 3],
                                   [900, -1700, 4], [1200, -1650, 7]],
                          'directions': [-1, 1], 'posted_speed_mps': 20.})
        seed = {'epoch': 50., 'scale': 15., 'duration_s': 14., 'path': [1500., 70.]}
        model = RoadTrajectory(seed, road, [600, 70, 1, 0], [0, 2, 1, 0, 2], np.zeros((5, 3)))
        model.receiver_enu_m = np.array([.4, -.3, 1.2])
        sites = np.array([[31000., -3000., 700.], [31200., -2800., 720.], [31100., -2700., 710.]])
        self.state = np.r_[np.zeros(model.count), np.linspace(-.05, .05, 9)]
        self.time = np.linspace(43.1, 56.9, 101)
        rng = np.random.default_rng(311)
        spectrum = np.zeros(8192)
        spectrum[np.arange(-3408, 3409)%8192] = 1
        pulse = Pulse(np.fft.ifft(spectrum)[np.arange(-64, 65)%8192])
        fields = []
        for ch in range(5):
            path = model.paths(self.state, self.time, np.full(len(self.time), ch), sites, True)[0]
            coefficients = np.polynomial.polynomial.polyfit((self.time-50)/15, path, 4)
            local_seed = seed | {'rf_hz': RF[ch], 'path': coefficients.tolist()}
            data = rng.normal(size=(len(self.time), 9))+1j*rng.normal(size=(len(self.time), 9))
            fields.append(Field({'time': self.time, 'offset_samples': np.arange(-4, 5),
                                 'value': data, 'variance': np.ones(data.shape)}, pulse, local_seed, .2))
        self.problem = SimpleNamespace(fields=fields, sites=sites, model=model)
        self.beta = np.array([.2, 0, -.3, .4, .5])

    def test_vehicle_frame_is_orthonormal_and_follows_motion(self):
        frame = body_frames(self.problem.model, self.state, self.time)
        np.testing.assert_allclose(np.einsum('nji,njk->nik', frame, frame),
                                   np.broadcast_to(np.eye(3), frame.shape), atol=5e-16)
        model = self.problem.model
        velocity = (model.positions(self.state, self.time+.0001, self.problem.sites, True)[0]-
                    model.positions(self.state, self.time-.0001, self.problem.sites, True)[0])/.0002
        velocity /= np.linalg.norm(velocity, axis=1)[:, None]
        np.testing.assert_allclose(frame[:, :, 0], velocity, atol=2e-9)

    def test_zero_offset_preserves_original_point_phase_reference(self):
        echo = RigidEcho(self.problem, self.state, self.beta, 1)
        actual, _ = echo.design(np.zeros(7))
        expected = []
        for ch, field in enumerate(self.problem.fields):
            path = self.problem.model.paths(self.state, self.time, np.full(len(self.time), ch),
                                            self.problem.sites, True)[0]
            value, _ = field.template(path)
            expected.append(np.exp(1j*self.beta[ch])*value)
        np.testing.assert_allclose(actual[:, 0], np.concatenate(expected), atol=1e-13)

    def test_full_two_point_phase_and_projection_derivative(self):
        echo = RigidEcho(self.problem, self.state, self.beta, 2)
        p = np.array([-.8, .2, .1, 1.3, -.4, -.2, .1, -.2, .3, -.1])
        actual = echo.residual(p, True)
        numeric = central_jacobian(echo.residual, p, 1e-4)
        self.assertLess(np.linalg.norm(actual-numeric)/np.linalg.norm(actual), 2e-7)


if __name__ == '__main__':
    unittest.main()
