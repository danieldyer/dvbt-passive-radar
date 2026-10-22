import json
import unittest

import numpy as np
from scipy.linalg import solve_triangular

from cluster_phase import Cluster, SplinePath
from field import Field
from motion import C


def synthetic(seed=4, noise=.0, neighbour=0., fade=False, rf=219.5e6, gaps=False, row_amplitude=None):
    """Independent Fourier pulse and numerical OFDM quadrature, before binning."""
    rng = np.random.default_rng(seed)
    block, bins = 16, 96
    row_time = 164+np.arange(block*bins)*.001088
    time = row_time.reshape(-1, block).mean(axis=1)
    epoch, scale = (time[0]+time[-1])/2, (time[-1]-time[0])/2
    u = (row_time-epoch)/scale
    parameters = np.array([.23, .7, -.29, .055])
    increment = parameters[2]*u+parameters[3]*u**2
    rate = (parameters[2]+2*parameters[3]*u)/scale
    k = 2*np.pi*rf/C
    offsets = np.arange(-3, 4)
    frequencies = np.fft.fftfreq(256)
    spectrum = (abs(frequencies) < .39)*(1+.2*np.sin(8*frequencies))
    spectrum /= spectrum.sum()
    def pulse(lag):
        return (np.exp(2j*np.pi*np.asarray(lag).reshape(-1, 1)*frequencies)@spectrum).reshape(np.shape(lag))
    samples = pulse(np.arange(-64, 65))
    centre_rate = -.56
    centre_omega = np.full(len(row_time), -k*.001024*centre_rate)
    inner = (np.arange(128)+.5)/128-.5
    def echo(origin, path, derivative, phase):
        omega = -k*.001024*(centre_rate+derivative)
        tone = np.exp(1j*omega[:, None]*inner)
        factor = tone.mean(axis=1)-1j*centre_omega*(tone*inner).mean(axis=1)
        lag = offsets[None, :]-origin-path[:, None]*8e6/C
        return pulse(lag)*factor[:, None]*np.exp(1j*phase-1j*k*path[:, None])
    amplitude = .8+.2*np.cos(2*np.pi*(time-time[0])/(time[-1]-time[0]))
    if fade:
        amplitude *= 1-.999*np.exp(-((time-epoch)/.16)**8)
    strength = np.repeat(amplitude, block) if row_amplitude is None else row_amplitude(row_time)
    raw = echo(parameters[0], increment, rate, parameters[1])*strength[:, None]
    if neighbour:
        raw += neighbour*echo(2.7, .7*(row_time-epoch), np.full(len(row_time), .7), -1.1)
    row_info = np.ones(len(row_time))/8192
    if gaps:
        row_info[36*block:45*block] = 0
    information = row_info.reshape(-1, block).sum(axis=1)
    value = np.zeros((bins, len(offsets)), complex)
    for j in range(bins):
        use = slice(j*block, (j+1)*block)
        if information[j]:
            value[j] = np.sum(raw[use]*row_info[use, None], axis=0)/information[j]
    covariance = samples[offsets[:, None]-offsets[None, :]+64]
    chol = np.linalg.cholesky((covariance+covariance.conj().T)/2)
    epsilon = (rng.standard_normal(value.shape)+1j*rng.standard_normal(value.shape))/np.sqrt(2)
    # Conditional unit whitened variance at noise=1, zero weight for erasures.
    value += noise*(epsilon@chol.T)/np.sqrt(np.maximum(8192*information, 1))[:, None]
    data = {'time': time, 'value': value, 'information': information,
            'row_time': row_time, 'row_information': row_info,
            'centre_omega': centre_omega, 'offset_samples': offsets,
            'metadata_json': json.dumps({'channel': 0, 'rf_hz': rf, 'block_rows': block})}
    field = Field(data, samples)
    training = (np.floor((time-time[0])/.16).astype(int)%2 == 0)&(information > 0)
    return Cluster(field, training), parameters


class ClusterPhaseTests(unittest.TestCase):
    def test_analytic_gradient_and_independent_forward_model(self):
        problem, truth = synthetic()
        value, gradient = problem.evaluate(truth)
        self.assertLess(value, 1e-7)
        p = truth+np.array([.08, .15, .04, -.025])
        _, gradient = problem.evaluate(p)
        difference = []
        for delta in np.eye(len(p))*1e-5:
            difference.append((problem.evaluate(p+delta)[0]-problem.evaluate(p-delta)[0])/2e-5)
        np.testing.assert_allclose(gradient, difference, atol=1e-8, rtol=2e-6)

    def test_recovers_continuous_delay_phase_and_rate_at_two_frequencies(self):
        for rf in (177.5e6, 226.5e6):
            problem, truth = synthetic(rf=rf, gaps=True)
            fit = problem.fit()
            np.testing.assert_allclose(fit.parameters, truth, atol=3e-4)
            self.assertEqual(fit.information_rank, len(truth))
            self.assertLess(fit.gradient, 2e-7)

    def test_training_only_fit_does_not_see_withheld_samples(self):
        problem, truth = synthetic()
        fit = problem.fit(seeds=[truth])
        seed = problem.phase_seed(.2)
        problem.field.y[~problem.training] *= np.exp(1.2j)
        np.testing.assert_array_equal(seed, problem.phase_seed(.2))
        other = problem.fit(seeds=[truth])
        np.testing.assert_allclose(other.parameters, fit.parameters, atol=1e-12)
        self.assertGreater(problem.diagnostics(other)['test_phase_rms_deg'], 65)

    def test_fade_carries_little_phase_information(self):
        problem, truth = synthetic(noise=.15, fade=True)
        fit = problem.fit(seeds=[truth+.02])
        self.assertLess(np.max(abs(fit.parameters[1:]-truth[1:])), .025)
        self.assertLess(problem.diagnostics(fit)['test_phase_rms_deg'], 4)

    def test_competing_echo_is_visible_in_prediction_failure(self):
        clean, truth = synthetic(noise=.05)
        dirty, _ = synthetic(noise=.05, neighbour=2.)
        good = clean.diagnostics(clean.fit(seeds=[truth]))
        bad = dirty.diagnostics(dirty.fit(seeds=[truth]))
        self.assertGreater(good['test_complex_prediction_fraction'], .99)
        self.assertLess(bad['test_complex_prediction_fraction'], .5)

    def test_spline_preserves_physical_path_and_exact_adjoint(self):
        polynomial, truth = synthetic(gaps=True)
        field = polynomial.field
        curve = SplinePath(field.time[0], field.time[-1], .4)
        spline = Cluster(field, polynomial.training, curve=curve)
        coefficients = np.linalg.lstsq(spline.basis, polynomial.basis@truth[2:], rcond=None)[0]
        parameters = np.r_[truth[:2], coefficients]
        np.testing.assert_allclose(spline.template(parameters), polynomial.template(truth), rtol=1e-12, atol=1e-12)
        p = parameters+np.linspace(.01, .04, len(parameters))
        value, gradient = spline.objective(p)
        explicit = spline.evaluate(p, True)
        self.assertAlmostEqual(value, explicit[0], places=14)
        np.testing.assert_allclose(gradient, explicit[1], atol=2e-14)
        direction = np.linspace(-.3, .5, len(p))
        difference = (spline.objective(p+1e-5*direction)[0]-spline.objective(p-1e-5*direction)[0])/2e-5
        self.assertAlmostEqual(difference, float(gradient@direction), places=8)
        residual, jacobian = spline.residual_jacobian(p)
        self.assertAlmostEqual(float(residual@residual), value, places=13)
        np.testing.assert_allclose(2*residual@jacobian, gradient, atol=2e-13)
        difference = (spline.residual_jacobian(p+1e-5*direction)[0]-
                      spline.residual_jacobian(p-1e-5*direction)[0])/2e-5
        np.testing.assert_allclose(difference, jacobian@direction, atol=1e-9, rtol=1e-5)

    def test_prediction_mask_cannot_include_training(self):
        problem, truth = synthetic()
        fit = problem.fit(seeds=[truth])
        with self.assertRaises(ValueError):
            problem.diagnostics(fit, testing=np.ones(len(problem.field.time), bool))


if __name__ == '__main__':
    unittest.main()
