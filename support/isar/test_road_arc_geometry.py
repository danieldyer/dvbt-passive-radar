import unittest

import numpy as np

from road_arc_geometry import road_state
from road_geometry import RoadCurve


class RoadGeometryTests(unittest.TestCase):
    def test_position_velocity_and_acceleration_consistent_on_curved_road(self):
        x = np.linspace(0., 1000., 21)
        road = RoadCurve({'enu_m': np.column_stack([x, 30*np.sin(x/180), 3*np.sin(x/190)]).tolist()})
        state = np.array([432.7, 11.2, -.8, 3.2, 2.4])
        p, v, a = road_state(road, state)
        dt = .0003
        def position(t):
            q = state.copy(); q[0] += state[1]*t+.5*state[2]*t*t
            return road_state(road, q)[0]
        np.testing.assert_allclose(v, (position(dt)-position(-dt))/(2*dt), atol=1e-8)
        np.testing.assert_allclose(a, (position(dt)-2*p+position(-dt))/dt**2, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
