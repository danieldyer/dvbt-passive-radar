import unittest

import numpy as np

from autofocus import design, evaluate, fit


class AutofocusTests(unittest.TestCase):
    def test_common_path_phase_and_independent_complex_amplitudes(self):
        random = np.random.default_rng(3921)
        interval, degree = (0., 120.), 5
        coefficients = np.array([.45, -.12, .07, .03])
        groups = []
        for channel, k in enumerate((3.7, 4.6)):
            t = np.linspace(.01+channel*.003, 119.99, 300)
            phase = k*design(t, interval, degree)@coefficients
            amplitude = random.normal(size=4)+1j*random.normal(size=4)
            noise = .03*(random.normal(size=(len(t), 4))+1j*random.normal(size=(len(t), 4)))
            groups.append({'channel': channel, 'time': t, 'value': np.exp(1j*phase[:, None])*amplitude+noise,
                           'information': 1+.2*np.sin(t), 'phase_scale': k, 'train': (t//2)%2 == 0})
        result = fit(groups, interval, degree=degree)
        np.testing.assert_allclose(result['coefficients'], coefficients, atol=.004)
        self.assertTrue(result['converged'])
        self.assertTrue(all(d['held_out_prediction_error_reduction_percent'] > 99 for d in result['diagnostics']))

    def test_held_out_signal_cannot_drive_phase_estimate(self):
        t = np.linspace(.01, 29.99, 200)
        train = (t//2)%2 == 0
        true = {'interval_s': [0, 30], 'degree': 4, 'coefficients': [.35, -.1, .08]}
        z = np.exp(1j*evaluate(true, t))[:, None]*np.array([1., 2j])
        group = {'time': t, 'value': z.copy(), 'information': np.ones(len(t)), 'phase_scale': 1., 'train': train}
        first = fit([group], (0, 30), degree=4)
        group['value'][~train] *= np.exp(2.3j*np.sin(t[~train]))[:, None]
        second = fit([group], (0, 30), degree=4)
        np.testing.assert_allclose(first['coefficients'], second['coefficients'], atol=1e-7)
        self.assertLess(second['diagnostics'][0]['held_out_prediction_error_reduction_percent'], 99)


if __name__ == '__main__':
    unittest.main()
