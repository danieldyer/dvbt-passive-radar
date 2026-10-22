import unittest

import numpy as np

from stationarity import check, polish
from fit_physical_profile import solve


class StationarityTests(unittest.TestCase):
    def test_search_curvature_cannot_replace_exact_objective_or_stationarity(self):
        hessian = np.array([[3., .2], [.2, 2.]])
        truth = np.array([.4, -.2])
        def objective(p):
            d = p-truth
            return .5*d@hessian@d, hessian@d
        # A badly displaced surrogate optimum is irrelevant: only its
        # curvature guides a step. True values and gradients accept it.
        approximate = lambda p: 1.001*hessian@(p-truth-np.array([20., -10.]))
        fitted, report = polish(objective, truth+np.array([.01, -.01]), np.eye(2),
            np.empty((0, 2)), np.empty(0), np.empty(0), np.full(2, -3.), np.full(2, 3.),
            curvature_gradient=approximate)
        self.assertTrue(report['accepted'])
        self.assertLess(np.max(abs(objective(fitted)[1])), 1e-8)
        np.testing.assert_allclose(fitted, truth, atol=1e-9)

    def test_fixed_coordinates_are_eliminated_from_optimizer_constraints(self):
        class Quadratic:
            initial = np.array([0., 2., 0.])
            evaluations = 0

            def evaluate(self, p):
                self.evaluations += 1
                delta = p-np.array([1., 2., -1.])
                return {'value': float(delta@delta), 'gradient': 2*delta,
                        'gauss_newton': 2*np.eye(3), 'scene_conditions': [1.]}

            def objective(self, p):
                r = self.evaluate(p)
                return r['value'], r['gradient']
        # The first linear row and the middle box coordinate are fixed and
        # feasible. Their zero Jacobian rows must not reach the optimizer.
        matrix = np.array([[0., 1., 0.], [1., 0., 1.]])
        fitted, report = solve(Quadratic(), matrix, np.array([2., -3.]), np.array([2., 3.]),
            np.array([-3., 2., -3.]), np.array([3., 2., 3.]))
        self.assertTrue(report['converged'], report['termination'])
        np.testing.assert_allclose(fitted, [1., 2., -1.], atol=1e-8)

    def test_solver_accepts_checked_box_and_linear_constraint_optimum(self):
        hessian = np.array([[6., 1., .3], [1., 3., -.2], [.3, -.2, 2.]])
        truth = np.array([1., -.4, .2])
        normal = np.array([1., 2., 0.])
        active_gradient = -.2*normal-np.array([.4, 0., 0.])
        class Quadratic:
            def __init__(self, initial):
                self.initial, self.evaluations = initial.copy(), 0

            def evaluate(self, p):
                self.evaluations += 1
                delta = p-truth
                return {'value': .5*delta@hessian@delta+active_gradient@delta,
                    'gradient': hessian@delta+active_gradient, 'gauss_newton': hessian,
                    'scene_conditions': [1.]}

            def objective(self, p):
                value = self.evaluate(p)
                return value['value'], value['gradient']

        bounds = (normal[None], np.array([-2.]), np.array([normal@truth]),
                  np.full(3, -3.), np.array([1., 3., 3.]))
        parameters, report = solve(Quadratic(np.zeros(3)), *bounds, polish_stationarity=False)
        self.assertTrue(report['converged'])
        self.assertIn('KKT gate', report['termination'])
        np.testing.assert_allclose(parameters, truth, atol=1e-6)
        self.assertLess(report['maximum_constraint_violation'], 1e-10)
        repeated, report = solve(Quadratic(parameters), *bounds, polish_stationarity=False)
        self.assertEqual(report['iterations'], 0)
        np.testing.assert_array_equal(repeated, parameters)

    def test_polish_recovers_constrained_minimum_despite_rounded_objective(self):
        hessian = np.array([[6., 1., .3], [1., 3., -.2], [.3, -.2, 2.]])
        truth = np.array([1., -.4, .2])
        normal = np.array([1., 2., 0.])
        matrix = normal[None]
        lo, hi = np.array([-2.]), np.array([normal@truth])
        lower, upper = np.full(3, -3.), np.full(3, 3.)
        def objective(p):
            delta = p-truth
            value = .5*delta@hessian@delta-.2*normal@delta
            return np.round(value, 8), hessian@delta-.2*normal
        initial = truth+np.array([2e-5, -1e-5, 3e-5])
        result, diagnostics = polish(objective, initial, np.eye(3), matrix, lo, hi, lower, upper)
        self.assertTrue(diagnostics['accepted'])
        np.testing.assert_allclose(result, truth, atol=1e-12)
        self.assertLess(check(result, objective(result)[1], matrix, lo, hi, lower, upper)[0], 1e-12)

    def test_negative_tangent_curvature_is_rejected(self):
        objective = lambda p: (-.5*(p@p), -p)
        initial = np.ones(2)*1e-5
        lower, upper = np.full(2, -3.), np.full(2, 3.)
        result, diagnostics = polish(objective, initial, np.eye(2), np.zeros((1, 2)), np.array([-1.]), np.array([1.]), lower, upper)
        self.assertFalse(diagnostics['accepted'])
        np.testing.assert_array_equal(result, initial)

    def test_polish_reaches_new_constraint_face_and_rebuilds_tangent(self):
        hessian = np.array([[2., .4], [.4, 3.]])
        truth = np.array([1., .2])
        def objective(p):
            delta = p-truth
            return .5*delta@hessian@delta-delta[0], hessian@delta-np.array([1., 0.])
        # The full first Newton step crosses x=1. Both box and general linear
        # forms must stop at that face, then optimize the remaining direction.
        for box in (False, True):
            lower, upper = np.full(2, -3.), np.array([1. if box else 3., 3.])
            matrix = np.empty((0, 2)) if box else np.array([[1., 0.]])
            lo, hi = (np.empty(0), np.empty(0)) if box else (np.array([-3.]), np.array([1.]))
            fitted, report = polish(objective, np.array([.9, .3]), np.eye(2), matrix, lo, hi, lower, upper)
            self.assertTrue(report['accepted'], report)
            self.assertLess(report['steps'][0]['feasible_step_fraction'], 1.)
            self.assertTrue(all(s['violation'] <= 1e-7 for s in report['steps']))
            np.testing.assert_allclose(fitted, truth, atol=1e-10)
            self.assertLess(report['final_kkt_residual'], 1e-10)

    def test_iteration_limit_does_not_suppress_a_checked_newton_correction(self):
        class Quadratic:
            initial = np.array([.1, -.2])
            evaluations = 0

            def evaluate(self, p):
                self.evaluations += 1
                delta = p-np.array([.2, .1])
                return {'value': float(delta@delta), 'gradient': 2*delta,
                        'gauss_newton': 2*np.eye(2), 'scene_conditions': [1.]}

            def objective(self, p):
                r = self.evaluate(p)
                return r['value'], r['gradient']
        fitted, report = solve(Quadratic(), np.empty((0, 2)), np.empty(0), np.empty(0),
            np.full(2, -3.), np.full(2, 3.), iterations=0)
        self.assertTrue(report['converged'], report)
        self.assertTrue(report['stationarity_polish']['accepted'])
        np.testing.assert_allclose(fitted, [.2, .1], atol=1e-11)


if __name__ == '__main__':
    unittest.main()
