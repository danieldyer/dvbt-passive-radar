import unittest

import numpy as np

from companion_transfer import matched_change, prediction_score


class CompanionTransferTests(unittest.TestCase):
    def test_zero_scene_has_finite_error_and_undefined_gain(self):
        data = np.array([1+2j, 3-1j])
        before = prediction_score(data, np.zeros_like(data))
        after = prediction_score(data, .5*data)
        self.assertAlmostEqual(before['prediction_error'], 15.)
        self.assertEqual(before['matched_energy'], 0.)
        self.assertIsNone(matched_change(before, after))
        self.assertAlmostEqual(matched_change(after, after), 0.)


if __name__ == '__main__':
    unittest.main()
