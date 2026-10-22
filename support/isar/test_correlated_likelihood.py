import unittest

import numpy as np

from correlated_likelihood import FourierMetric, MetricExperiment, sample_covariance
from test_recovery_experiment import problem


class CorrelatedLikelihoodTests(unittest.TestCase):
    def test_metric_matches_explicit_complex_quadratic_form(self):
        rng = np.random.default_rng(7); n, r = 16, 3
        a = rng.normal(size=(2,r,r))+1j*rng.normal(size=(2,r,r))
        cov = a@a.conj().transpose(0,2,1)+np.eye(r)
        labels = np.arange(n)%2; metric = FourierMetric(labels, cov)
        x = rng.normal(size=(n,r,2))+1j*rng.normal(size=(n,r,2))
        actual = metric.apply(x.reshape(n*r,2))
        z = np.fft.fft(x, axis=0, norm='ortho')
        expected = sum(np.vdot(z[i,:,k], np.linalg.solve(cov[labels[i]], z[i,:,k])).real
            for i in range(n) for k in range(2))
        self.assertAlmostEqual(np.vdot(actual,actual).real, expected, places=11)

    def test_complex_covariance_convention_and_thermal_floor(self):
        rng = np.random.default_rng(2)
        a = np.array([[2., .3j], [.4+.7j, 3.]])
        x = ((rng.normal(size=(100000,2))+1j*rng.normal(size=(100000,2)))/2**.5)@a.T
        cov, amount = sample_covariance(x, shrink=False)
        np.testing.assert_allclose(cov, a@a.conj().T, atol=.04)
        small, _ = sample_covariance(x*.01)
        np.testing.assert_allclose(small, np.eye(2), atol=1e-12)
        self.assertGreaterEqual(amount, 0.); self.assertLessEqual(amount, 1.)

    def test_unitary_metric_preserves_profile_and_all_derivatives(self):
        base = problem(); metric = []
        for f, keep in zip(base.fields, base.training):
            metric.append(FourierMetric(np.zeros(keep.sum(),int), np.eye(len(f.offsets))[None]))
        ex = MetricExperiment(base, metric)
        p = np.array([.02,-.03,.04,-.02,.01,.005])
        y = [base.independent_signal(p,c,[.8+.3j]) for c in range(5)]
        at = p+np.array([.003,-.002,.001,-.001,.001,.0005])
        r,j,alpha,_ = ex.profile(at,y); original = base.profile(at,y)
        self.assertAlmostEqual(r@r, original[0]@original[0], places=5)
        np.testing.assert_allclose(j.T@r, original[1].T@original[0], rtol=2e-9, atol=1e-5)
        for a,b in zip(alpha,original[2]): np.testing.assert_allclose(a,b,rtol=2e-10)
        for i in range(6):
            delta = np.eye(6)[i]*2e-5
            finite = (ex.profile(at+delta,y,derivative=False)[0]-ex.profile(at-delta,y,derivative=False)[0])/4e-5
            self.assertLess(np.linalg.norm(finite-j[:,i])/np.linalg.norm(finite),3e-4)


if __name__ == '__main__': unittest.main()
