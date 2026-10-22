import unittest

import numpy as np
from scipy.optimize import minimize

from fit_scene import Scene
from stationary import make_design
from test_rigid_motion import fixture


class SceneTests(unittest.TestCase):
    def test_subpixel_position_gradient_and_heldout_complex_prediction(self):
        field = fixture()
        d = field.data
        state = {'position_ecef_m': d['centre_ecef_m'], 'velocity_ecef_mps': d['centre_velocity_ecef_mps'],
            'body_to_ecef': d['body_to_ecef'],
            'frame_velocity': np.gradient(d['body_to_ecef'], d['row_time'], axis=0, edge_order=2)}
        points = np.array([[-65., 3., 14.], [80., -5., 35.]])
        offsets = np.array([[2., -3., 1.], [-1., 1., 2.]])
        selected = np.arange(len(field.time))%2 == 0
        truth = np.array([.04, -.03, -.05, .06])
        physical_points = points.copy()
        physical_points[:, :2] += truth.reshape(2, 2)*10.
        target = np.column_stack([field.point(p, motion=state, transmitter_offset_ecef_m=offsets[0],
            receiver_offset_ecef_m=offsets[1]).ravel() for p in physical_points])
        stationary, _ = make_design(field)
        field.y = (target@np.array([.7-.2j, -.3+.9j])+
                   stationary[:, 10]*(3+4j)).reshape(field.y.shape)
        scene = Scene(field, state, points, selected, offsets)
        p = truth*.5
        _, gradient = scene.objective(p)
        for i in range(len(p)):
            step = np.eye(len(p))[i]*1e-5
            numeric = (scene.objective(p+step)[0]-scene.objective(p-step)[0])/2e-5
            self.assertAlmostEqual(numeric, gradient[i], delta=2e-5)
        fitted = minimize(scene.objective, np.zeros(4), jac=True, method='BFGS',
                          options={'gtol': 1e-9, 'maxiter': 150})
        check = scene.validation(fitted.x, [field.time[0], field.time[-1]])
        self.assertLess(fitted.fun, 1e-10)
        self.assertLess(check['prediction_error']/check['stationary_only_error'], 1e-9)
        self.assertLess(check['prediction_phase_rms_degrees'], 1e-3)


if __name__ == '__main__':
    unittest.main()
