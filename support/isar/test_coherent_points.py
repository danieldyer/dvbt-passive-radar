import unittest

import numpy as np
from scipy.optimize import least_squares, minimize

from channel_alignment import AlignmentBand
from coherent_points import CoherentPoints
from field import FS
from motion import C
from test_rigid_motion import fixture


class CoherentPointTests(unittest.TestCase):
    def fixture(self):
        points = np.array([[-60., 5., 20.], [70., -9., 20.]])
        truth_points = points+np.array([[.35, -.2, 0.], [-.25, .3, 0.]])
        point_phase = np.array([.3, -.6])
        band_phase = np.array([.4, -.2, 0.])
        strengths = np.array([[.8, 1.4], [1.3, .6], [1., 1.1]])
        bands = []
        for c, rf_channel in enumerate([0, 2, 3]):
            field = fixture(rf_channel)
            d = field.data
            state = {'position_ecef_m': d['centre_ecef_m'], 'velocity_ecef_mps': d['centre_velocity_ecef_mps'],
                'body_to_ecef': d['body_to_ecef'],
                'frame_velocity': np.gradient(d['body_to_ecef'], d['row_time'], axis=0, edge_order=2)}
            target = np.column_stack([field.trajectory(state['position_ecef_m']+state['body_to_ecef']@p,
                state['velocity_ecef_mps']+state['frame_velocity']@p).ravel() for p in truth_points])
            coefficient = strengths[c]*np.exp(1j*(point_phase+band_phase[c]))
            stationary_path = (np.round(d['centre_path_m'].mean()*FS/C)+2)*C/FS
            stationary = field.stationary(float(stationary_path)).ravel()
            field.y = (target@coefficient+(15+7j)*stationary).reshape(field.y.shape)
            selected = np.floor(field.time).astype(int)%2 == 0
            bands.append(AlignmentBand(field, state, points, selected))
        model = CoherentPoints(bands, points, [0., 0., 0.], reference=2)
        truth = model.initial(truth_points, point_phase, band_phase)
        return model, truth

    def test_joint_coordinate_and_phase_profile_gradient(self):
        model, truth = self.fixture()
        check = truth+np.array([.03, -.02, -.04, .02, .04, -.03, .02, -.03])
        result = model.evaluate(check)
        for j in range(len(check)):
            step = 3e-4 if j < 4 else 1e-5
            delta = np.eye(len(check))[j]*step
            numerical = (model.objective(check+delta)[0]-model.objective(check-delta)[0])/(2*step)
            self.assertAlmostEqual(numerical, result['gradient'][j], delta=3e-6)
        raw = sum(np.linalg.norm(b.clean-b.nuisance.remove(o['target'][b.selected])@o['coefficient'])**2
            for b, o in zip(model.bands, result['bands']))/model.energy
        self.assertAlmostEqual(raw, result['value'], places=12)

    def test_known_position_and_channel_phase_recovery(self):
        model, truth = self.fixture()
        initial = truth.copy(); initial[:4] = 0.; initial[4:] += np.array([.05, -.04, .06, -.05])
        result = minimize(model.objective, initial, jac=True, method='L-BFGS-B',
            bounds=[(-1., 1.)]*4+[(None, None)]*4,
            options={'gtol': 1e-10, 'ftol': 1e-16, 'maxiter': 500, 'maxcor': 30})
        recovered = model.evaluate(result.x)
        self.assertLess(recovered['value'], 1e-12)
        np.testing.assert_allclose(result.x[:4], truth[:4], atol=4e-4)
        np.testing.assert_allclose(np.exp(1j*result.x[-2:]), np.exp(1j*truth[-2:]), atol=1e-5)

    def test_profile_residual_jacobian_includes_refitted_positive_strengths(self):
        model, truth = self.fixture()
        check = truth+np.array([.03, -.02, -.04, .02, .04, -.03, .02, -.03])
        result = model.evaluate(check, jacobian=True)
        np.testing.assert_allclose(2*result['residual_jacobian'].T@result['real_residual'], result['gradient'], atol=1e-11)
        for j in range(len(check)):
            step = 3e-4 if j < 4 else 1e-5
            delta = np.eye(len(check))[j]*step
            plus = model.evaluate(check+delta, jacobian=True)['real_residual']
            minus = model.evaluate(check-delta, jacobian=True)['real_residual']
            numerical = (plus-minus)/(2*step)
            self.assertLess(np.linalg.norm(numerical-result['residual_jacobian'][:, j])/np.linalg.norm(numerical), 3e-5)
        initial = truth.copy(); initial[:4] = 0.; initial[4:] += [.05, -.04, .06, -.05]
        fit = least_squares(lambda x: model.evaluate(x, jacobian=True)['real_residual'], initial,
            jac=lambda x: model.evaluate(x, jacobian=True)['residual_jacobian'],
            bounds=(np.r_[np.full(4, -1.), np.full(4, -np.inf)], np.r_[np.ones(4), np.full(4, np.inf)]),
            x_scale='jac', gtol=1e-11, ftol=1e-14, xtol=1e-12)
        self.assertLess(2*fit.cost, 1e-14)
        np.testing.assert_allclose(fit.x[:4], truth[:4], atol=1e-5)

    def test_inactive_strengths_and_expected_information(self):
        model, truth = self.fixture()
        exact = model.evaluate(truth, jacobian=True)
        expected = 2*model.energy*(exact['residual_jacobian'].T@exact['residual_jacobian'])
        np.testing.assert_allclose(exact['expected_information'], expected, rtol=2e-8, atol=1e-5)
        wrong = truth.copy(); wrong[4] += np.pi
        result = model.evaluate(wrong, jacobian=True)
        self.assertTrue(any(abs(b['coefficient'][0]) == 0 for b in result['bands']))
        np.testing.assert_allclose(2*result['residual_jacobian'].T@result['real_residual'], result['gradient'], atol=1e-11)
        delta = np.eye(len(wrong))[0]*3e-4
        numerical = (model.evaluate(wrong+delta, jacobian=True)['real_residual']-
                     model.evaluate(wrong-delta, jacobian=True)['real_residual'])/(6e-4)
        np.testing.assert_allclose(numerical, result['residual_jacobian'][:, 0], atol=2e-7)


if __name__ == '__main__':
    unittest.main()
