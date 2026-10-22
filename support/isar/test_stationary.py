import unittest

import numpy as np

from stationary import Stationary


class StationaryNuisanceTests(unittest.TestCase):
    def test_joint_static_and_chirped_target_crossing_zero_doppler(self):
        time = np.linspace(-10, 10, 1201)
        profiles = np.array([[1., .4j], [.3+.2j, 1.], [-.1j, .2]])
        design = np.tile(profiles, (len(time), 1))
        moving = (np.exp(-1j*np.pi*.15*time**2)[:, None]*np.array([1., .2j, .6])).ravel()
        amplitude = .7-.3j
        nuisance_amplitude = np.array([30.+20j, -40.+11j])
        data = design@nuisance_amplitude+moving*amplitude
        selected = np.repeat((np.floor(time+10)%2) == 0, 3)
        nuisance = Stationary(design, selected)
        signal = nuisance.remove(moving[selected])
        clean = nuisance.remove(data[selected])
        recovered = np.vdot(signal, clean)/np.vdot(signal, signal)
        self.assertAlmostEqual(abs(recovered-amplitude), 0, places=12)
        beta = nuisance.solve(data[selected]-moving[selected]*recovered)
        np.testing.assert_allclose(design[~selected]@beta+moving[~selected]*recovered,
                                   data[~selected], atol=1e-11)
        loss_db = -10*np.log10(np.vdot(signal, signal).real/np.vdot(moving[selected], moving[selected]).real)
        self.assertLess(loss_db, .5)
        self.assertEqual(nuisance.rank, 2)


if __name__ == '__main__':
    unittest.main()
