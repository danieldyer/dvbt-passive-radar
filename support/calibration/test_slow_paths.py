import unittest

import numpy as np
from scipy.optimize._numdiff import approx_derivative

from complex_likelihood import projected_profile
from cosited_paths import PathHistory
from phase_history import C, FS
from range_fit import Field, Pulse
from slow_paths import Observation, residual


class SlowPathTests(unittest.TestCase):
    def setUp(self):
        time = np.arange(300)*.02+.01
        self.seed = {'epoch': 3., 'scale': 15., 'duration_s': 6.,
                     'path': [1000., 7.5], 'rf_hz': 184.5e6}
        self.model = PathHistory(self.seed, 2.)
        self.pulse = Pulse(np.sinc(.75*np.arange(-64, 65)))
        self.offsets = np.arange(-4, 5)
        self.seed_path = 1000.+.5*(time-3.)
        k = 2*np.pi*self.seed['rf_hz']/C

        def raw(path):
            return (np.exp(-1j*k*(path-(self.seed_path-1000.)))[:, None]
                    *self.pulse(self.offsets[None, :]
                                -(path-self.seed_path)[:, None]*FS/C))
        self.alpha = .8+.3j
        self.delays = 1000.+np.array([-1., 0., 1.])*C/FS
        background = np.array([30.+12j, -50j, -26.+11j])
        values = self.alpha*raw(self.seed_path)
        for delay, coefficient in zip(self.delays, background):
            values += coefficient*raw(np.full(len(time), delay))
        self.source = Field({'time': time, 'value': values,
                             'variance': np.ones(values.shape),
                             'offset_samples': self.offsets}, self.pulse, self.seed)
        mask = np.floor(time).astype(int)%2 == 0
        self.train = Observation(self.source, self.model, mask, self.delays)
        self.test = Observation(self.source, self.model, ~mask, self.delays)

    def test_joint_stationary_fit_preserves_slow_target_phase_and_predicts_holdout(self):
        p = np.zeros(self.model.count)
        u, _ = self.train.template(p)
        _, amplitude = projected_profile(self.train.y, u)
        self.assertAlmostEqual(abs(amplitude-self.alpha), 0, delta=1e-10)
        result = self.train.evaluate(p, self.test)
        self.assertLess(result['heldout_prediction_error'], 1e-16)
        self.assertGreater(result['stationary_only_heldout_error'], 1.)
        self.assertLess(result['training_target_information_loss_db'], 1.)

    def test_projected_path_derivative(self):
        p = np.linspace(-.01, .02, self.model.count)
        actual = residual([self.train], p, True)
        numeric = approx_derivative(lambda x: residual([self.train], x), p,
                                    method='3-point', abs_step=1e-5)
        np.testing.assert_allclose(actual, numeric, rtol=2e-6, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
