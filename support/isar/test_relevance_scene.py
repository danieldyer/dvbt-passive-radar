import unittest

import numpy as np
from scipy.linalg import block_diag

from relevance_scene import RelevanceScene
from scene_statistics import compress


class RelevanceSceneTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(16225)
        designs = [(rng.normal(size=(18, 4))+1j*rng.normal(size=(18, 4)))/4 for _ in range(3)]
        data = [rng.normal(size=18)+1j*rng.normal(size=18) for _ in designs]
        return designs, data, [compress(a, y) for a, y in zip(designs, data)]

    def test_evidence_posterior_and_all_scores_match_dense_covariance(self):
        designs, data, records = self.fixture()
        for shared in (False, True):
            model = RelevanceScene(records, shared=shared, reference=1)
            p = np.linspace(-.4, .7, model.count)
            root, gains = model.unpack(p)
            columns = [a*g*root[0 if shared else c]/model.scale for c, (a, g) in enumerate(zip(designs, gains))]
            design = np.concatenate(columns) if shared else block_diag(*columns)
            y = np.concatenate(data)
            covariance = np.eye(len(y))+design@design.conj().T
            cholesky = np.linalg.cholesky(covariance)
            inverse_y = np.linalg.solve(covariance, y)
            expected = 2*np.log(cholesky.diagonal().real).sum()+np.vdot(y, inverse_y).real
            value = model.evaluate(p)
            self.assertAlmostEqual(value['quadratic']+value['logdet'], expected, places=10)
            actual = np.concatenate([a@c for a, c in zip(designs, value['coefficients'])])
            np.testing.assert_allclose(actual, y-inverse_y, atol=2e-13)
            for i in range(len(p)):
                step = np.eye(len(p))[i]*1e-5
                numeric = (model.objective(p+step)[0]-model.objective(p-step)[0])/2e-5
                self.assertAlmostEqual(numeric, value['gradient'][i], delta=2e-8)

    def test_learned_point_variances_preserve_the_phase_quadratic(self):
        _, _, records = self.fixture()
        model = RelevanceScene(records, shared=True, reference=1)
        first = np.linspace(-.4, .7, model.count)
        second = first.copy(); second[-2:] = [-.9, .3]
        matrix = model.phase_matrix(first)
        z, w = [np.exp(-1j*np.angle(model.unpack(p)[1])) for p in (first, second)]
        self.assertAlmostEqual(model.objective(first)[0]-model.objective(second)[0],
            (np.vdot(w, matrix@w)-np.vdot(z, matrix@z)).real, places=10)

    def test_orthogonal_relevance_recovers_the_analytic_evidence_optimum(self):
        y = np.array([3+4j, .2j, 2+1j, .1, -3j])
        record = {'design': np.eye(len(y)), 'data': y, 'outside': 0., 'observations': len(y)}
        for shared in (False, True):
            model = RelevanceScene([record], shared=shared, reference=0)
            coefficients, fit = model.fit()
            variance = np.maximum(abs(y)**2-1., 0.)
            expected = variance/(1+variance)*y
            np.testing.assert_allclose(coefficients[0], expected, atol=2e-5)
            self.assertTrue(fit['selected']['stationary'])

    def test_unknown_relevance_retains_a_known_common_band_phase(self):
        y = np.array([10+4j, 0., 20-10j, 0., -3+8j, 5j, 0., 10.])
        records = [{'design': np.eye(len(y)), 'data': g*y, 'outside': 0., 'observations': len(y)}
                   for g in (1., .6*np.exp(.8j))]
        model = RelevanceScene(records, shared=True, reference=0)
        _, fit = model.fit()
        phase = np.angle(complex(*fit['band_gains_real_imag'][1]))
        self.assertAlmostEqual(phase, .8, places=7)
        self.assertTrue(fit['selected']['stationary'])
        self.assertLess(fit['phase_certificate']['retained_solution_gap'], 1e-5)


if __name__ == '__main__':
    unittest.main()
