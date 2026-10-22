import unittest

import numpy as np

from antenna_motion import Antennas
from shared_target_geometry import complex_gain
from stationary_road_control import StationaryRoad
from test_shared_target_geometry import road_fixture


class StationaryRoadTests(unittest.TestCase):
    def fixture(self):
        target, truth = road_fixture(Antennas(np.eye(3), [0]), noise=0.)
        field = target.training[0]
        delays = [np.mean(field.seed_path)+np.array([-1., 0., 1.])*299792458/8e6]
        temporary = StationaryRoad(target, delays=delays)
        beta = np.array([20+3j, -40j, -16+10j])
        for field in (*temporary.training, *temporary.validation):
            field.original.y += field.stationary@beta
        target.energy = float(np.vdot(target.training[0].y, target.training[0].y).real)
        return StationaryRoad(target, delays=delays), truth

    def test_exact_joint_nuisance_fit_recovers_moving_amplitude_and_predicts_test_rows(self):
        target, truth = self.fixture()
        field = target.training[0]
        u, du = target.template(truth, np.zeros(target.antennas.count), 0, field)
        _, _, amplitude, _ = complex_gain(field.y, u, du)
        self.assertAlmostEqual(abs(amplitude-(.7-.4j)), 0., delta=1e-11)
        metrics = target.predict(truth, np.zeros(target.antennas.count))[0]
        self.assertLess(metrics['prediction_error'], 1e-17)
        self.assertLess(metrics['moving_prediction_error_after_test_nuisance'], 1e-17)
        self.assertGreater(metrics['stationary_only_full_prediction_error'], 1.)

    def test_physical_motion_and_antenna_derivatives_after_stationary_projection(self):
        target, truth = self.fixture()
        p = np.r_[truth*.4, np.linspace(-.003, .003, target.antennas.count)]
        result = target.evaluate(p[:target.nmotion], p[target.nmotion:])
        for i in (0, 4, target.nmotion-1, target.nmotion, len(p)-1):
            delta = np.eye(len(p))[i]*2e-5
            a, b = [target.evaluate(q[:target.nmotion], q[target.nmotion:])['value'] for q in (p+delta, p-delta)]
            numeric = (a-b)/4e-5
            self.assertAlmostEqual(numeric, result['gradient'][i], delta=3e-5+abs(numeric)*4e-5)

    def test_amplitude_variance_matches_the_unknown_stationary_nuisance_crlb(self):
        target, truth = self.fixture()
        field = target.training[0]
        u, _ = target.template(truth, np.zeros(target.antennas.count), 0, field)
        energy = np.vdot(u, u).real
        rng = np.random.default_rng(90231)
        noise = (rng.normal(size=(len(u), 6000))+1j*rng.normal(size=(len(u), 6000)))/np.sqrt(2)
        error = u.conj()@field.remove(noise)/energy
        self.assertLess(abs(np.mean(abs(error)**2)*energy-1.), .04)
        self.assertLess(abs(error.mean())*np.sqrt(energy), .04)


if __name__ == '__main__':
    unittest.main()
