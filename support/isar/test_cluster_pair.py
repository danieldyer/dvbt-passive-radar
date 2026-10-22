import unittest

import numpy as np
from scipy.optimize import nnls

from cluster_pair import Pair, strengths
from test_cluster_phase import synthetic


class PairTests(unittest.TestCase):
    def test_strengths_against_independent_active_set_solver(self):
        rng = np.random.default_rng(927)
        h = rng.normal(size=(70, 5, 2))+1j*rng.normal(size=(70, 5, 2))
        h[0, :, 1] = h[0, :, 0]
        h[1, :, 1] = 0
        y = rng.normal(size=(70, 5))+1j*rng.normal(size=(70, 5))
        found = strengths(h, y)
        for i in range(len(y)):
            reference = nnls(np.r_[h[i].real, h[i].imag], np.r_[y[i].real, y[i].imag])[0]
            np.testing.assert_allclose(h[i]@found[i], h[i]@reference, atol=2e-14)

    def test_gradient_and_noiseless_two_echo_recovery(self):
        single, primary = synthetic(neighbour=2.)
        p = Pair(single.field, single.training)
        truth = np.r_[primary, 2.7, -1.1, .7*single.scale, 0.]
        self.assertLess(p.evaluate(truth)[0], 2e-8)
        start = truth+np.array([.05, .06, .03, -.02, -.06, -.04, -.03, .02])
        _, gradient = p.evaluate(start)
        differences = [(p.evaluate(start+d)[0]-p.evaluate(start-d)[0])/2e-5 for d in np.eye(8)*1e-5]
        np.testing.assert_allclose(gradient, differences, rtol=2e-5, atol=1e-9)
        fit = p.fit([start], [(-.8,.8), (2.,3.4)])
        np.testing.assert_allclose(fit.parameters, truth, atol=4e-4)
        self.assertEqual(fit.information_rank, 8)
        self.assertLess(fit.objective, 2e-8)

    def test_withheld_data_cannot_change_fit_or_prediction(self):
        single, primary = synthetic(neighbour=2., noise=.1)
        p = Pair(single.field, single.training)
        seed = np.r_[primary, 2.7, -1.1, .7*single.scale, 0.]
        fit = p.fit([seed], [(-.8,.8), (2.,3.4)])
        prediction = p.prediction(fit.parameters)
        before = p.diagnostics(fit)['test_complex_prediction_fraction']
        p.field.y[~p.training] *= np.exp(1.5j)
        again = p.fit([seed], [(-.8,.8), (2.,3.4)])
        np.testing.assert_allclose(again.parameters, fit.parameters, atol=1e-12)
        np.testing.assert_allclose(p.prediction(fit.parameters), prediction, atol=1e-13)
        self.assertGreater(before, .99)
        self.assertLess(p.diagnostics(fit)['test_complex_prediction_fraction'], .1)


if __name__ == '__main__':
    unittest.main()
