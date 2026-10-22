import unittest

import numpy as np
from scipy.optimize import minimize

from fit_scatterers import projected


class ContinuousScattererTests(unittest.TestCase):
    def test_profiled_gradient_and_off_grid_two_point_recovery(self):
        time = np.linspace(-1, 1, 300)
        directions = np.column_stack([4*time, 3*time**2])
        truth = np.array([[-1.3, .7], [2.5, -.9]])
        amplitude = np.array([.6+.2j, -.4+.7j])

        def model(points):
            design = np.exp(-1j*(directions@points.T))
            derivative = -1j*design[:, :, None]*directions[:, None, :]
            return design, derivative

        data = model(truth)[0]@amplitude
        seed = truth+np.array([[.13, -.07], [-.11, .06]])

        def objective(p):
            design, derivative = model(p.reshape(2, 2))
            value, gradient, _ = projected(design[::2], derivative[::2], data[::2])
            return value, gradient.ravel()

        _, jacobian = objective(seed.ravel())
        for axis in range(4):
            delta = np.eye(4)[axis]*1e-5
            numeric = (objective(seed.ravel()+delta)[0]-objective(seed.ravel()-delta)[0])/2e-5
            self.assertAlmostEqual(numeric, jacobian[axis], places=7)
        result = minimize(objective, seed.ravel(), method='BFGS', jac=True,
                          options={'gtol': 1e-10})
        np.testing.assert_allclose(result.x.reshape(2, 2), truth, atol=1e-7)
        design, derivative = model(result.x.reshape(2, 2))
        _, _, recovered = projected(design[::2], derivative[::2], data[::2])
        np.testing.assert_allclose(design[1::2]@recovered, data[1::2], atol=1e-7)


if __name__ == '__main__':
    unittest.main()
