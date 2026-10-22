import unittest

import numpy as np

from antenna_motion import Antennas
from test_shared_target_geometry import road_fixture
from validate_shared_geometry import RoadMotion, fit_target


class SharedGeometryTransferTests(unittest.TestCase):
    def test_transfer_refits_motion_without_mutating_or_refitting_antenna_coordinates(self):
        road, truth = road_fixture(Antennas(np.eye(3), [0]), noise=0.)
        antenna = np.zeros(road.antennas.count)
        initial = truth*.3
        problem = RoadMotion(road, antenna, initial)
        before = problem.evaluate(initial)
        antenna[:] = 7.
        # A caller cannot change the supposedly frozen calibration in-place.
        np.testing.assert_array_equal(problem.antenna, np.zeros_like(antenna))
        for j in (0, 4, len(initial)-1):
            delta = np.eye(len(initial))[j]*1e-5
            numeric = (problem.objective(initial+delta)[0]-problem.objective(initial-delta)[0])/2e-5
            self.assertAlmostEqual(numeric, before['gradient'][j], delta=1e-6)
        result = fit_target(road, problem.antenna, [initial], 180)
        fitted = result['starts'][result['selected_start']]
        self.assertTrue(fitted['fit']['converged'])
        self.assertLess(fitted['fit']['objective'], before['value'])
        self.assertEqual(len(fitted['parameters']), road.nmotion)
        np.testing.assert_array_equal(problem.antenna, np.zeros_like(antenna))
        self.assertTrue(np.isfinite(result['prediction'][0]['prediction_error']))


if __name__ == '__main__':
    unittest.main()
