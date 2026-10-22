import unittest

import numpy as np

from antenna_motion import Antennas
from geometry_profile import GeometryProfile
from shared_origin_transfer import FrozenOriginRoad, subset
from shared_target_geometry import SharedGeometry
from test_shared_target_geometry import road_fixture
import test_physical_profile as physical_tests
from validate_shared_geometry import RoadMotion


class OriginTransferTests(unittest.TestCase):
    def test_excluded_road_samples_cannot_change_the_calibration_likelihood(self):
        antennas = Antennas(np.eye(3), [0])
        band, truth = physical_tests.PhysicalProfileTests().fixture()
        ship = GeometryProfile([band], truth*.3, antennas)
        roads = [road_fixture(antennas)[0] for _ in range(2)]
        legacy = SharedGeometry(ship, roads)
        n = len(legacy.initial)
        bounds = (np.zeros((0, n)), np.zeros(0), np.zeros(0), np.full(n, -10.), np.full(n, 10.))
        _, reduced, _ = subset(legacy, ship, bounds, legacy.initial, [1])
        point = reduced.initial.copy(); point[reduced.range_slice] = 1.5
        before = reduced.evaluate(point)
        roads[0].training[0].y[:] = 1e6+1e6j
        reduced.last = None
        after = reduced.evaluate(point)
        self.assertEqual(before['value'], after['value'])
        np.testing.assert_array_equal(before['gradient'], after['gradient'])
        self.assertEqual(len(point), legacy.ship_count+roads[1].nmotion)

    def test_frozen_origin_transfer_keeps_physical_derivatives_and_never_refits_the_origin(self):
        original, truth = road_fixture(Antennas(np.eye(3), [0]))
        origin = np.array([2.4]); road = FrozenOriginRoad(original, origin)
        antenna = np.zeros(original.antennas.count); motion = truth*.7
        problem = RoadMotion(road, antenna, motion)
        before = problem.evaluate(motion)
        origin[:] = -100.
        np.testing.assert_array_equal(road.origins, [2.4])
        for i in (0, 4, len(motion)-1):
            delta = np.eye(len(motion))[i]*1e-5
            numeric = (problem.objective(motion+delta)[0]-problem.objective(motion-delta)[0])/2e-5
            self.assertAlmostEqual(numeric, before['gradient'][i], delta=1e-6)
        direct = road.road.predict(motion, antenna, [2.4])
        self.assertEqual(road.predict(motion, antenna), direct)
        self.assertEqual(len(problem.initial), original.nmotion)


if __name__ == '__main__':
    unittest.main()
