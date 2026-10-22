import unittest

import numpy as np

from block_pulse import BlockPulse, RowPulse


def direct(samples, lag):
    delta = np.asarray(lag)[..., None]-np.arange(-64, 65)
    weights = np.sinc(delta)*np.i0(8*np.sqrt(np.maximum(0, 1-(delta/32)**2)))/np.i0(8)
    weights *= abs(delta) <= 32
    weights /= weights.sum(axis=-1)[..., None]
    return weights@samples


class BlockPulseTests(unittest.TestCase):
    def test_matches_independent_interpolation_and_fractional_delay_derivative(self):
        rng = np.random.default_rng(72513)
        frequency = np.arange(-3408, 3409)/8192
        spectra = rng.uniform(.3, 1.7, size=(3, len(frequency)))
        samples = np.exp(2j*np.pi*np.outer(np.arange(-64, 65), frequency))@spectra.T/8192
        kernels = BlockPulse(samples.T)
        lag = rng.uniform(-31.95, 31.95, size=(3, 100))
        index = np.arange(3)[:, None]
        expected = np.array([direct(sample, values) for sample, values in zip(samples.T, lag)])
        np.testing.assert_allclose(kernels.at(lag, index), expected, atol=3e-10, rtol=1e-8)
        step = 1e-5
        derivative = np.array([(direct(sample, values+step)-direct(sample, values-step))/(2*step)
            for sample, values in zip(samples.T, lag)])
        np.testing.assert_allclose(kernels.at(lag, index, 1), derivative, atol=2e-8, rtol=2e-6)

    def test_row_identity_phase_linearity_and_boundaries(self):
        samples = np.zeros((2, 129), complex); samples[:, 64] = [1., .3+.7j]
        kernels = BlockPulse(samples)
        pulse = RowPulse(kernels, [1, 0, 1], samples.mean(axis=0))
        lag = np.tile([-32., -.4, 0., 1.9, 32.], (3, 1))
        values = pulse(lag)
        np.testing.assert_allclose(values[0], values[1]*(.3+.7j), atol=1e-14)
        np.testing.assert_array_equal(values[0], values[2])
        with self.assertRaises(ValueError): pulse(lag[:2])
        with self.assertRaises(ValueError): kernels.at(np.array([32.001]), 0)


if __name__ == '__main__':
    unittest.main()
