import unittest
import numpy as np

from mirror_diagnostic import spectra, channel_rate


class MirrorCovarianceTests(unittest.TestCase):
    def test_iq_epochs_use_channel_rate_after_decimation(self):
        self.assertEqual(channel_rate({'corrected_rate_hz': 64e6, 'decimation': 8}), 8e6)

    def test_conjugate_pair_keeps_complex_phase_and_negative_frequency_index(self):
        rng = np.random.default_rng(712)
        x = rng.normal(size=64*500)+1j*rng.normal(size=64*500)
        angle = .72
        count, cross, power = spectra(np.array([x, np.exp(1j*angle)*x.conj()]), [[0, 1]], 64, 4)
        self.assertEqual(count, 500)
        np.testing.assert_allclose(cross[0]/np.sqrt(power[0, 0]*power[0, 1]), np.exp(1j*angle), atol=2e-15)

    def test_frequency_dependent_image_is_visible_when_broadband_covariance_cancels(self):
        rng = np.random.default_rng(912)
        x = np.exp(1j*rng.uniform(-np.pi, np.pi, (500, 64)))
        phase = np.repeat(np.array([1., -1.]*8), 4)
        mirror = (-np.arange(64)) % 64
        y = (x.conj()*phase)[:, mirror]
        iq = np.array([np.fft.ifft(x).ravel(), np.fft.ifft(y).ravel()])
        self.assertLess(abs(np.mean(iq[0]*iq[1])), 1e-17)
        _, cross, power = spectra(iq, [[0, 1]], 64, 4)
        np.testing.assert_allclose(abs(cross)/np.sqrt(power[:, 0]*power[:, 1]), 1., atol=2e-15)


if __name__ == '__main__':
    unittest.main()
