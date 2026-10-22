"""A weakly observed stationary mode must not invalidate target transfer."""
from types import SimpleNamespace
import unittest

import numpy as np

from stationary import Stationary
from validate_motion import validate


class TransferValidationTests(unittest.TestCase):
    def test_stationary_projection_prevents_extrapolation_but_preserves_phase_error(self):
        time = np.arange(64, dtype=float)
        selected = time < 32
        t = np.repeat(time, 3)
        train = np.repeat(selected, 3)
        weak_mode = np.where(train, 1e-6, 1.)*np.exp(.07j*t)
        background = np.c_[np.ones(len(t)), weak_mode]
        target = np.exp(1j*(.81*t+np.tile([.1, -.2, .4], len(time))))[:, None]
        amplitude = np.array([.8+.3j])
        observations = target@amplitude+background@np.array([3+2j, -4+1j])
        # Small training noise lies in a nuisance direction barely seen there,
        # but becomes enormous if its fitted coefficient is extrapolated.
        observations[train] += 1e4*weak_mode[train]
        field = SimpleNamespace(time=time, y=observations.reshape(-1, 3))
        nuisance = Stationary(background, train)
        extrapolated = validate(field, target, nuisance, selected, [0, 64])
        projected = validate(field, target, nuisance, selected, [0, 64], stationary_mode='project')
        self.assertGreater(extrapolated['prediction_error'], 1000*extrapolated['raw_heldout_energy'])
        self.assertLess(projected['prediction_error'], 1e-16)
        self.assertGreater(projected['test_target_information_retained_fraction'], .95)
        self.assertLessEqual(projected['test_target_information_retained_fraction'], 1.)
        self.assertLess(projected['prediction_phase_rms_degrees'], 1e-6)
        wrong_phase = validate(field, target, nuisance, selected, [0, 64], stationary_mode='project',
                               coefficients=amplitude*np.exp(.4j))
        self.assertAlmostEqual(wrong_phase['prediction_phase_rms_degrees'], np.degrees(.4), places=8)
        self.assertGreater(wrong_phase['prediction_error'], 1.)


if __name__ == '__main__':
    unittest.main()
