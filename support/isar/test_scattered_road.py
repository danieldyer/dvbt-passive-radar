import copy
import unittest

import numpy as np

from antenna_motion import Antennas
from scattered_road import ScatteredRoad
from shared_origin_geometry import OriginRoad
from shared_target_geometry import RoadTarget
from test_shared_target_geometry import road_fixture


def fixture():
    original, truth = road_fixture(Antennas(np.eye(3), [0, 1]))
    problem = copy.copy(original.problem); problem.range_origins = np.zeros(2)
    train, test = [], []
    for frequency in (177.5e6, 219.5e6):
        for target, source in ((train, original.training[0]), (test, original.validation[0])):
            field = copy.copy(source); field.seed = {**source.seed, 'rf_hz': frequency}
            field.k = 2*np.pi*frequency/299792458.
            target.append(field)
    road = RoadTarget(problem, train, test, original.initial,
        np.array([[31000., -3000., 700.], [31200., -2850., 730.]]), original.antennas)
    return road, truth


class ScatteredRoadTests(unittest.TestCase):
    def test_zero_offsets_are_exactly_the_original_common_point_model(self):
        road, truth = fixture(); origins = np.array([2., -1.])
        model = ScatteredRoad(road, origins, reference=0)
        p = np.r_[truth, np.zeros(3)]; antenna = np.arange(road.antennas.count)*.001
        previous = OriginRoad(road)
        for c, field in enumerate(model.training):
            value, jacobian = model.template(p, antenna, c, field)
            expected, derivative = previous.template(truth, antenna, origins, c, field)
            np.testing.assert_allclose(value, expected, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(jacobian[:, :road.nmotion], derivative[:, :road.nmotion], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(jacobian[:, model.nmotion:], derivative[:, road.nmotion:-2], rtol=1e-12, atol=1e-12)

    def test_physical_offsets_and_antenna_derivatives_match_independent_path(self):
        road, truth = fixture(); model = ScatteredRoad(road, [2., -1.], reference=0)
        motion = np.r_[truth, [1.2, -.4, .7]]; antenna = np.arange(road.antennas.count)*.0004
        field = model.training[1]
        value, derivative = model.template(motion, antenna, 1, field)
        position = road.problem.model.positions(truth, field.time, road.problem.sites)[0]
        position += model.axes@motion[-3:]
        tx_delta, rx = (road.mappings[1]@antenna).reshape(2, 3)
        tx = road.transmitters[1]+tx_delta
        path = np.linalg.norm(position-tx, axis=1)+np.linalg.norm(position-rx, axis=1)-np.linalg.norm(tx-rx)
        expected = field.template(path, range_bias=-1.)[0]
        np.testing.assert_allclose(value, expected, rtol=1e-12, atol=1e-12)
        parameters = np.r_[motion, antenna]
        for j in range(len(parameters)):
            step = 2e-5; delta = np.eye(len(parameters))[j]*step
            a, b = [model.template(q[:model.nmotion], q[model.nmotion:], 1, field)[0]
                    for q in (parameters+delta, parameters-delta)]
            np.testing.assert_allclose((a-b)/(2*step), derivative[:, j], atol=4e-7, rtol=4e-5)

    def test_profiled_objective_preserves_motion_priors_and_reference_gauge(self):
        road, truth = fixture(); model = ScatteredRoad(road, [2., -1.], reference=0)
        p = np.r_[truth*.3, [.8, -.3, .4]]; antenna = np.zeros(road.antennas.count)
        result = model.evaluate(p, antenna)
        for j in [0, road.nmotion-1, road.nmotion, road.nmotion+1, road.nmotion+2]:
            step = 2e-5; delta = np.eye(len(p))[j]*step
            numeric = (model.evaluate(p+delta, antenna)['value']-model.evaluate(p-delta, antenna)['value'])/(2*step)
            self.assertAlmostEqual(numeric, result['gradient'][j], delta=2e-5+abs(numeric)*3e-5)
        a = model.template(p, antenna, 0, model.training[0])[0]
        p[-3:] += [1., -.5, .2]
        b = model.template(p, antenna, 0, model.training[0])[0]
        np.testing.assert_array_equal(a, b)
        lower, upper = model.problem.model.bounds()
        np.testing.assert_array_equal(lower[-3:], [-3., -2., -2.])
        np.testing.assert_array_equal(upper[-3:], [3., 2., 2.])


if __name__ == '__main__':
    unittest.main()
