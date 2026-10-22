import unittest

import numpy as np

from cluster_phase import Cluster
from field import Field
from motion import C
from road_arc import observation, observables, seed_path


class RoadArcTests(unittest.TestCase):
    def test_independent_symbol_synthesis_retains_delay_phase_and_original_rows(self):
        time = 49.8+np.arange(1536)*.001088
        weights = (1+.3*np.cos(np.arange(len(time))*.017))/8192
        seed = {'epoch': 50., 'scale': 15., 'path': [1638.3, 411.6, -15.7, 5.6], 'rf_hz': 184.5e6}
        offsets = np.arange(-2, 3)
        bins = [(np.average(time[i:i+16], weights=weights[i:i+16])) for i in range(0, len(time), 16)]
        epoch, scale = (bins[0]+bins[-1])/2, (bins[-1]-bins[0])/2
        truth = np.array([.19, -.6, -.22, .04])
        u = (time-epoch)/scale
        extra = truth[2]*u+truth[3]*u**2
        extra_rate = (truth[2]+2*truth[3]*u)/scale
        k = 2*np.pi*seed['rf_hz']/C
        frequency = np.fft.fftfreq(256)
        spectrum = (abs(frequency) < .39)*(1+.13*np.cos(5*frequency)); spectrum /= spectrum.sum()
        def pulse(lag):
            return (np.exp(2j*np.pi*np.asarray(lag).reshape(-1, 1)*frequency)@spectrum).reshape(np.shape(lag))
        samples = pulse(np.arange(-64, 65))
        # Independent symbol quadrature and the extraction's actual carrier
        # reference. A large absolute phase is not silently set to zero.
        seed_rate = seed_path(seed, time, 1)
        omega = -k*.001024*(seed_rate+extra_rate)
        omega0 = -k*.001024*seed_rate
        inner = (np.arange(2048)+.5)/2048-.5
        tone = np.exp(1j*omega[:, None]*inner)
        symbol = tone.mean(axis=1)-1j*omega0*(tone*inner).mean(axis=1)
        physical = seed_path(seed, time)+extra
        scatter_phase = truth[1]+k*seed['path'][0]
        raw = pulse(offsets[None]-truth[0]-extra[:, None]*8e6/C)*symbol[:, None]
        raw *= np.exp(1j*scatter_phase-1j*k*physical)[:, None]
        raw *= np.exp(1j*k*(seed_path(seed, time)-seed['path'][0]))[:, None]
        data = observation({'row_time': time, 'row_information': weights, 'row_value': raw,
                            'offset_samples': offsets}, seed, samples)
        field = Field(data, samples)
        training = ((field.time-field.time[0])//.125).astype(int)%2 == 0
        problem = Cluster(field, training)
        self.assertLess(problem.objective(truth)[0], 1e-7)
        fit = problem.fit(seeds=[truth+.02])
        np.testing.assert_allclose(fit.parameters, truth, atol=1e-4)
        cropped = field.windowed(50., 51.)
        self.assertGreater(cropped.time[0], 50.)
        np.testing.assert_allclose(cropped.data['centre_path_m'], seed_path(seed, cropped.data['row_time']), atol=1e-12)
        with self.assertRaisesRegex(ValueError, 'empirical path'):
            cropped.trajectory(np.zeros((len(cropped.data['row_time']), 3)), np.zeros((len(cropped.data['row_time']), 3)))

    def test_absolute_observables_use_same_polynomial_and_full_covariance(self):
        seed = {'epoch': 50., 'scale': 15., 'path': [1638.3, 411.6, -15.7, 5.6], 'rf_hz': 184.5e6}
        factor = np.array([[1., .1, .02, .03], [.2, 1., .1, .01], [.01, .2, .1, .04], [.01, .02, .01, .1]])
        row = {'epoch_s': 51., 'scale_s': .5, 'degree': 2, 'parameters': [.19, -.6, -.22, .04],
               'conditional_parameter_covariance': (factor@factor.T).tolist()}
        mid = 51.1
        value, covariance = observables(row, seed, mid)
        def path(t):
            u = (t-row['epoch_s'])/row['scale_s']
            return seed_path(seed, t)+row['parameters'][0]*C/8e6+row['parameters'][2]*u+row['parameters'][3]*u*u
        h = .001
        expected = [path(mid), (path(mid+h)-path(mid-h))/(2*h), (path(mid+h)-2*path(mid)+path(mid-h))/h**2]
        np.testing.assert_allclose(value, expected, atol=5e-7)
        self.assertGreater(abs(covariance[0, 2]), .1)
        self.assertGreater(np.linalg.eigvalsh(covariance).min(), 0.)


if __name__ == '__main__':
    unittest.main()
