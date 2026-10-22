import unittest

import numpy as np

from phase_noise import phase_variance


class PhaseNoiseTests(unittest.TestCase):
    def test_uniform_phase_and_high_snr_limits(self):
        self.assertAlmostEqual(phase_variance(0),np.pi**2/3,places=14)
        self.assertLess(abs(phase_variance(1e6)*2e6-1),1e-5)
        for value in (-1.,np.inf,np.nan):
            with self.assertRaises(ValueError):phase_variance(value)

    def test_matches_independently_sampled_complex_gaussian_phases(self):
        rng=np.random.default_rng(2399)
        noise=(rng.normal(size=400000)+1j*rng.normal(size=400000))/np.sqrt(2)
        for rho in (.05,.5,5.,50.):
            measured=np.mean(np.angle(np.sqrt(rho)+noise)**2)
            self.assertLess(abs(measured/phase_variance(rho)-1),.015)


if __name__=='__main__':unittest.main()
