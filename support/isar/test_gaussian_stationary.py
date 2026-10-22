import unittest

import numpy as np

from gaussian_stationary import GaussianStationary


class GaussianStationaryTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(427052)
        normal = lambda shape: (rng.normal(size=shape)+1j*rng.normal(size=shape))/np.sqrt(2)
        b, target = normal((70, 9)), normal((31, 9))
        u, v, y = normal(70), normal(31), normal(70)
        return rng, b, target, u, v, y

    def test_restricted_likelihood_equals_independent_dense_contrast_covariance(self):
        _, b, _, u, _, y = self.fixture()
        model = GaussianStationary(y, u, b)
        q = np.linalg.qr(u[:, None], mode='complete')[0][:, 1:]
        for eta in (0., .3, 8., 1e4):
            cov = q.conj().T@(np.eye(len(u))+eta/model.scale*(b@b.conj().T))@q
            z = q.conj().T@y
            noise = np.vdot(z, np.linalg.solve(cov, z)).real/len(z)
            expected = len(z)*np.log(noise)+2*np.log(np.linalg.cholesky(cov).diagonal().real).sum()
            self.assertAlmostEqual(model.profile(eta)['value'], expected, delta=3e-10)
        eta = 2.3
        numeric = (model.profile(eta+1e-4)['value']-model.profile(eta-1e-4)['value'])/2e-4
        self.assertAlmostEqual(numeric, model.profile(eta)['derivative_eta'], delta=2e-8)

    def test_training_variance_fit_reaches_an_independently_checked_score_root(self):
        rng, b, _, u, _, y = self.fixture()
        data = .6*u+b@(rng.normal(size=b.shape[1])+1j*rng.normal(size=b.shape[1]))+y
        model = GaussianStationary(data, u, b)
        result = model.fit()
        self.assertTrue(result['stationary'])
        self.assertGreater(result['eta'], 0.)
        eta, step = result['eta'], 1e-3
        numeric = (model.profile(eta*np.exp(step))['value']-model.profile(eta*np.exp(-step))['value'])/(2*step)
        self.assertLess(abs(numeric), 2e-6)

    def test_predictor_and_uncertainty_match_dense_gaussian_conditioning(self):
        _, b, target, u, v, y = self.fixture()
        model = GaussianStationary(y, u, b)
        sigma2, tau2 = .7, 1.4
        eta = tau2*model.scale/sigma2
        prediction, stats = model.predict(target, v, eta, sigma2=sigma2)
        covariance = sigma2*np.eye(len(u))+tau2*b@b.conj().T
        inverse = np.linalg.inv(covariance)
        amplitude = np.vdot(u, inverse@y)/np.vdot(u, inverse@u).real
        beta = tau2*b.conj().T@inverse@(y-amplitude*u)
        np.testing.assert_allclose(prediction, amplitude*v+target@beta, atol=1e-13)
        posterior = tau2*np.eye(b.shape[1])-tau2**2*b.conj().T@inverse@b
        direction = v-tau2*target@b.conj().T@inverse@u
        variance = sigma2*np.eye(len(v))+target@posterior@target.conj().T
        variance += np.outer(direction, direction.conj())/np.vdot(u, inverse@u).real
        self.assertAlmostEqual(stats['expected_prediction_error'], np.trace(variance).real, delta=2e-11)

    def test_fixed_hyperparameter_prediction_error_attains_its_conditional_bound(self):
        rng, b, target, u, v, _ = self.fixture()
        normal = lambda shape: (rng.normal(size=shape)+1j*rng.normal(size=shape))/np.sqrt(2)
        sigma2, tau2, amplitude = .7, 1.4, .6-.3j
        # A dense expression is independent of the low-rank implementation.
        covariance = sigma2*np.eye(len(u))+tau2*b@b.conj().T
        invu = np.linalg.solve(covariance, u)
        information = np.vdot(u, invu).real
        gain = tau2*target@b.conj().T@np.linalg.inv(covariance)
        operator = gain+np.outer(v-gain@u, invu.conj()/information)
        count = 6000
        beta = np.sqrt(tau2)*normal((b.shape[1], count))
        train = amplitude*u[:, None]+b@beta+np.sqrt(sigma2)*normal((len(u), count))
        test = amplitude*v[:, None]+target@beta+np.sqrt(sigma2)*normal((len(v), count))
        model = GaussianStationary(train[:, 0], u, b)
        _, stats = model.predict(target, v, tau2*model.scale/sigma2, sigma2=sigma2)
        observed = np.mean(np.sum(abs(test-operator@train)**2, axis=0))
        self.assertLess(abs(observed/stats['expected_prediction_error']-1), .02)


if __name__ == '__main__':
    unittest.main()
