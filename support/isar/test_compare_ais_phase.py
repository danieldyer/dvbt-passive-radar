"""Independent physical path/phase controls for the direct AIS comparison."""
import unittest

import numpy as np

from compare_ais_phase import C, path_geometry, phase_to_path


class AisPhaseTests(unittest.TestCase):
    def test_unequal_carriers_and_constant_offsets_do_not_create_path_difference(self):
        grid = np.linspace(0, 1, 51)
        anchor = grid < .5
        errors = []
        for channel, rf in enumerate([177.5e6, 184.5e6, 191.625e6, 219.5e6, 226.5e6]):
            time = np.linspace(-.02, 1.02, 400)+channel*.0001
            k = 2*np.pi*rf/C
            prior = 14000-.6*time+.015*time**2
            actual = prior+.2*time-.01*time**2
            raw_signal = np.exp(-1j*k*actual+1j*(channel+1)*1.3)
            aligned = raw_signal*np.exp(1j*k*prior)
            observed, expected, _ = phase_to_path(time, aligned, k, time, prior, grid, anchor)
            truth = np.interp(grid, time, actual)
            truth -= truth[anchor].mean()
            errors.append(observed-truth)
        self.assertLess(np.max(abs(np.asarray(errors))), 1e-10)

    def test_separate_transmitter_motion_difference_survives_constant_calibration(self):
        time = np.linspace(0, 1, 101)
        position = np.array([15000., -3000., 20.])+time[:, None]*np.array([1., 4., 0.])
        velocity = np.array([1., 4., 0.])
        receiver = np.array([0., 0., 20.])
        transmitters = [np.array([30000., 25000., 600.]), np.array([30000., 25350., 600.])]
        rf = [177.5e6, 226.5e6]
        paths, recovered = [], []
        for c in range(2):
            path, rate = path_geometry(position, velocity, transmitters[c], receiver)
            k = 2*np.pi*rf[c]/C
            # A deliberately common (wrong for band 2) carrier deramp.
            deramp = path_geometry(position, velocity, transmitters[0], receiver)[0]
            aligned = np.exp(-1j*k*(path-deramp)+1j*c*2.1)
            measured, _, _ = phase_to_path(time, aligned, k, time, deramp, time, time < .5)
            paths.append(path-path[time < .5].mean())
            recovered.append(measured)
            numerical = np.gradient(path, time, edge_order=2)
            np.testing.assert_allclose(numerical, rate, atol=2e-8)
        np.testing.assert_allclose(np.subtract(*recovered), np.subtract(*paths), atol=2e-10)
        self.assertGreater(abs((paths[1]-paths[0])[-1]), .001)


if __name__ == '__main__':
    unittest.main()
