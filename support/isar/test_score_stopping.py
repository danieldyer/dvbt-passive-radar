import unittest

import numpy as np

from score_stopping import fit


class Quadratic:
    def __init__(self):
        self.matrix = np.diag([.03, .2, 4., 10.])
        self.truth = np.array([.01, -.02, .03, -.005])
        self.evaluations = 0
    def evaluate(self, p):
        self.evaluations += 1
        d = p-self.truth
        return {'value': float(.5*d@self.matrix@d), 'gradient': self.matrix@d,
                'gauss_newton': self.matrix}
    def objective(self, p):
        r = self.evaluate(p)
        return r['value'], r['gradient']


class ScoreStoppingTests(unittest.TestCase):
    def test_callback_final_point_and_original_score_agree_for_both_solvers(self):
        for method in ('BFGS', 'trust-exact'):
            problem = Quadratic()
            transform = np.diag(1/np.sqrt(np.diag(problem.matrix)))
            p, result = fit(problem, np.ones(4), transform, iterations=50, method=method)
            score = np.linalg.norm(transform.T@problem.objective(p)[1])
            self.assertLessEqual(score, .005)
            self.assertAlmostEqual(score, result['information_normalized_score_norm'])
            self.assertLess(np.linalg.norm(np.linalg.solve(transform, p-problem.truth)), .005)
            self.assertTrue(result['score_converged'])

    def test_initial_acceptance_and_iteration_exhaustion_remain_distinct(self):
        problem = Quadratic()
        transform = np.diag(1/np.sqrt(np.diag(problem.matrix)))
        initial = problem.truth+transform@np.full(4, .001)
        p, result = fit(problem, initial, transform, iterations=20, method='trust-exact')
        np.testing.assert_array_equal(p, initial)
        self.assertEqual(result['iterations'], 0)
        self.assertTrue(result['score_converged'])
        _, failed = fit(problem, np.full(4, 100.), transform, iterations=1, method='trust-exact')
        self.assertFalse(failed['score_converged'])


if __name__ == '__main__':
    unittest.main()
