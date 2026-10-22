from pathlib import Path
import tempfile
import unittest

import numpy as np

from render_movie import Channel, calibrated_images


class MovieTests(unittest.TestCase):
    def test_declared_channel_phases_enter_the_coherent_sum(self):
        gains = np.exp(2j*np.pi*np.arange(5)/5)*np.array([1., .8, 2., .4, .9])
        signal = np.array([[1.+2j, -.5j], [2., -.7+.1j]])
        weights = np.array([1., 2., 3., 4., 5.])
        samples = [(g*signal, w, [0., 3.]) for g, w in zip(gains, weights)]
        power = calibrated_images(samples, gains)
        expected = 8192*np.sum(weights*abs(gains)**2)*abs(signal)**2
        np.testing.assert_allclose(power[-1], expected, rtol=1e-14)
        np.testing.assert_allclose(power[:-1].sum(axis=0), expected, rtol=1e-14)
        self.assertLess(calibrated_images(samples, np.ones(5))[-1].sum(), .2*expected.sum())

    def test_fixed_apertures_preserve_complex_phase_and_information(self):
        rng = np.random.default_rng(952)
        rows, block, period = 1057, 16, .001088
        first = np.arange(0, rows, block)
        last = np.minimum(first+block-1, rows-1)
        weight = rng.uniform(.3, 2., rows)
        signal = np.exp(1j*np.arange(rows)[:, None, None]*.004)*np.exp(1j*np.arange(6).reshape(2, 3))
        signal += .1*(rng.normal(size=signal.shape)+1j*rng.normal(size=signal.shape))
        sums = np.add.reduceat(signal*weight[:, None, None], first)
        prefix = np.concatenate([np.zeros((1, 2, 3)), np.cumsum(sums, axis=0)]).astype('complex64')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.save(root/'channel-0-prefix.npy', prefix)
            np.savez(root/'channel-0-timing.npz', first_time=first*period, last_time=(last+1)*period,
                     rows_per_block=last-first+1, block_rows=block, symbol_period_s=period,
                     weight_prefix=np.r_[0, np.cumsum(np.add.reduceat(weight, first))],
                     along_m=np.arange(3), port_m=np.arange(2))
            channel = Channel(root, 0)
            lengths = []
            for centre in np.linspace(.25, .85, 30):
                measured, information, interval = channel.image(centre, .3)
                begin, end = np.rint(np.array(interval)/period).astype(int)
                reference = np.sum(signal[begin:end]*weight[begin:end, None, None], axis=0)/weight[begin:end].sum()
                np.testing.assert_allclose(measured, reference, atol=5e-7)
                self.assertLess(np.max(abs(np.degrees(np.angle(measured/reference)))), .0001)
                self.assertAlmostEqual(information, weight[begin:end].sum())
                lengths.append(interval[1]-interval[0])
            self.assertLess(np.ptp(lengths), 1e-12)
            with self.assertRaises(ValueError):
                channel.image(.5, 2.)


if __name__ == '__main__':
    unittest.main()
