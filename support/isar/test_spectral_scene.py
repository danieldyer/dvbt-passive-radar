import unittest

import numpy as np
from scipy.linalg import block_diag

from phase_quadratic import solve
from spectral_scene import SpectralScene, compress, prediction


class SpectralSceneTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(16219)
        designs = [(rng.normal(size=(20, 5))+1j*rng.normal(size=(20, 5)))/5 for _ in range(3)]
        y = [rng.normal(size=20)+1j*rng.normal(size=20) for _ in designs]
        return designs, y

    def test_qr_and_latent_solve_equal_the_full_observation_covariance(self):
        designs, observations = self.fixture()
        records = [compress(a, y) for a, y in zip(designs, observations)]
        y = np.concatenate(observations)
        for shared in (False, True):
            model = SpectralScene(records, shared=shared, reference=0)
            p = np.r_[[.2, -.3, .5], [.4, -.8] if shared else []]
            g = model.gains(p)/model.scale
            matrix = (np.concatenate([z*a for z, a in zip(g, designs)]) if shared
                      else block_diag(*[z*a for z, a in zip(g, designs)]))
            covariance = np.eye(len(y))+matrix@matrix.conj().T
            root = np.linalg.cholesky(covariance)
            expected = 2*np.log(root.diagonal().real).sum()+np.vdot(y, np.linalg.solve(covariance, y)).real
            result = model.evaluate(p)
            self.assertAlmostEqual(result['negative_log_likelihood_without_pi_constant'], expected, places=10)
            for j in range(len(p)):
                delta = np.eye(len(p))[j]*1e-5
                numeric = (model.objective(p+delta)[0]-model.objective(p-delta)[0])/2e-5
                self.assertAlmostEqual(numeric, result['gradient'][j], delta=1e-7)
            for a, yy, r, coefficient in zip(designs, observations, records, result['coefficients']):
                actual = prediction(r, coefficient)
                self.assertAlmostEqual(actual['prediction_error'], np.linalg.norm(yy-a@coefficient)**2, places=10)
                self.assertAlmostEqual(actual['matched_energy'], abs(np.vdot(a@coefficient, yy))**2/
                                       np.linalg.norm(a@coefficient)**2, places=10)

    def test_phase_quadratic_has_the_exact_marginal_likelihood_difference(self):
        designs, y = self.fixture()
        model = SpectralScene([compress(a, b) for a, b in zip(designs, y)], shared=True, reference=0)
        logs = np.array([.2, -.3, .5])
        matrix = model.phase_matrix(logs)
        first, second = np.array([0., .4, -.8]), np.array([0., -.2, .1])
        values = [model.objective(np.r_[logs, phase[1:]])[0] for phase in (first, second)]
        z, w = np.exp(-1j*first), np.exp(-1j*second)
        self.assertAlmostEqual(values[0]-values[1], (np.vdot(w, matrix@w)-np.vdot(z, matrix@z)).real, places=10)
        phase, certificate = solve(matrix, reference=0)
        self.assertTrue(certificate['globally_certified'])
        self.assertLessEqual(model.objective(np.r_[logs, -np.angle(phase)[1:]])[0], min(values)+1e-9)

    def test_nearly_degenerate_scene_at_high_power_preserves_likelihood_and_score(self):
        rng = np.random.default_rng(16224)
        unitary = lambda rows, cols: np.linalg.qr(
            rng.normal(size=(rows, cols))+1j*rng.normal(size=(rows, cols)))[0]
        left, right = unitary(72, 12), unitary(12, 12)
        singular = np.geomspace(1., 1e-12, 12)
        matrix = (left*singular)@right.conj().T
        designs = np.split(matrix, 3)
        observations = [20*(rng.normal(size=24)+1j*rng.normal(size=24)) for _ in designs]
        records = [compress(a, y) for a, y in zip(designs, observations)]
        for shared in (False, True):
            model = SpectralScene(records, shared=shared, reference=0, scale=1.)
            parameters = np.r_[[11.2, 10.4, 11.8], [.4, -.7] if shared else []]
            gains = model.gains(parameters)
            full = (np.concatenate([g*a for g, a in zip(gains, designs)]) if shared
                    else block_diag(*[g*a for g, a in zip(gains, designs)]))
            u, s, vh = np.linalg.svd(full, full_matrices=False)
            y = np.concatenate(observations)
            z = u.conj().T@y
            quadratic = np.linalg.norm(y-u@z)**2+np.sum(abs(z)**2/(1+s*s))
            expected = quadratic+np.log1p(s*s).sum()
            result = model.evaluate(parameters)
            self.assertAlmostEqual(result['negative_log_likelihood_without_pi_constant'], expected, delta=2e-7)
            for j in range(len(parameters)):
                delta = np.eye(len(parameters))[j]*1e-4
                numeric = (model.objective(parameters+delta)[0]-model.objective(parameters-delta)[0])/2e-4
                self.assertAlmostEqual(numeric, result['gradient'][j], delta=2e-5)
            coefficients = (vh.conj().T*(s/(1+s*s)))@z
            predicted = np.concatenate([a@c for a, c in zip(designs, result['coefficients'])])
            np.testing.assert_allclose(predicted, full@coefficients, atol=2e-7, rtol=2e-8)

    def test_gaussian_posterior_risk_matches_its_conditional_bound(self):
        rng = np.random.default_rng(16220)
        a = (rng.normal(size=(35, 6))+1j*rng.normal(size=(35, 6)))/4
        truth = (rng.normal(size=(6, 10000))+1j*rng.normal(size=(6, 10000)))/np.sqrt(2)
        noise = (rng.normal(size=(35, 10000))+1j*rng.normal(size=(35, 10000)))/np.sqrt(2)
        covariance = np.linalg.inv(np.eye(6)+a.conj().T@a)
        fitted = covariance@a.conj().T@(a@truth+noise)
        error = fitted-truth
        measured = error@error.conj().T/error.shape[1]
        self.assertLess(np.linalg.norm(measured-covariance)/np.linalg.norm(covariance), .055)
        # Test the implementation on the same known-prior posterior, including
        # its arbitrary global design-unit scale.
        y = a@truth[:, 0]+noise[:, 0]
        model = SpectralScene([compress(a, y)], shared=True, reference=0)
        actual = model.evaluate([np.log(model.scale)])['coefficients'][0]
        np.testing.assert_allclose(actual, fitted[:, 0], atol=2e-14)

    def test_phase_mle_attains_the_conditional_gaussian_phase_crlb(self):
        rng = np.random.default_rng(16221)
        samples, trials, power, truth = 128, 4000, 2., .7
        signal = np.sqrt(power/2)*(rng.normal(size=(trials, samples))+1j*rng.normal(size=(trials, samples)))
        noise = (rng.normal(size=(2, trials, samples))+1j*rng.normal(size=(2, trials, samples)))/np.sqrt(2)
        y0, y1 = signal+noise[0], np.exp(1j*truth)*signal+noise[1]
        estimate = np.angle(np.sum(y0.conj()*y1, axis=1))
        error = np.angle(np.exp(1j*(estimate-truth)))
        rho = power/(1+power)
        bound = (1-rho**2)/(2*samples*rho**2)
        self.assertLess(abs(np.mean(error**2)/bound-1), .08)
        self.assertLess(abs(np.mean(error))/np.sqrt(bound), .06)
        records = [{'design': np.eye(samples), 'data': y, 'outside': 0., 'observations': samples}
                   for y in (y0[0], y1[0])]
        model = SpectralScene(records, shared=True, reference=0)
        phase, _ = solve(model.phase_matrix(np.full(2, .5*np.log(power))), reference=0)
        self.assertAlmostEqual(np.angle(phase[1])+estimate[0], 0., places=9)

    def test_gain_fit_recovers_the_phase_mle_and_reports_its_bound(self):
        rng = np.random.default_rng(16222)
        signal = (rng.normal(size=60)+1j*rng.normal(size=60))*2
        data = [g*signal+(rng.normal(size=60)+1j*rng.normal(size=60))/np.sqrt(2)
                for g in (1., np.exp(.7j))]
        records = [{'design': np.eye(60), 'data': y, 'outside': 0., 'observations': 60} for y in data]
        model = SpectralScene(records, shared=True, reference=0)
        _, fit = model.fit()
        actual = fit['selected']['parameters'][2]
        expected = np.angle(np.vdot(data[0], data[1]))
        self.assertLess(abs(np.angle(np.exp(1j*(actual-expected)))), 1e-6)
        self.assertLess(fit['selected']['maximum_kkt_residual'], 1e-5)
        self.assertLess(fit['phase_certificate']['retained_solution_gap'], 1e-5)


if __name__ == '__main__':
    unittest.main()
