import unittest

import numpy as np

from antenna_motion import Antennas
from road_branch_control import branch, rebase_motion
from test_shared_target_geometry import road_fixture


class RoadBranchControlTests(unittest.TestCase):
    def test_reparameterizing_a_road_preserves_physical_motion_and_likelihood(self):
        target, truth = road_fixture(Antennas(np.eye(3), [0]))
        old = target.problem.model
        description = old.road.description
        choice = {'route': 0, 'coarse_state': (old.coarse+[1.3, -.7, .3, -.05]).tolist(),
                  'state': np.zeros(old.count).tolist()}
        updated = branch(target, choice, [description])
        parameters = rebase_motion(truth, old.coarse, updated.problem.model)
        time = np.linspace(-old.half, old.half, 137)
        first = old.positions(truth, time, target.problem.sites)[0]
        second = updated.problem.model.positions(parameters, time, target.problem.sites)[0]
        np.testing.assert_allclose(first, second, atol=3e-12)
        antenna = np.linspace(-.01, .01, target.antennas.count)
        a, b = target.evaluate(truth, antenna), updated.evaluate(parameters, antenna)
        self.assertAlmostEqual(a['value'], b['value'], places=8)
        np.testing.assert_allclose(a['gradient'], b['gradient'], atol=1e-8)
        np.testing.assert_allclose(a['expected_real_jacobian'], b['expected_real_jacobian'], atol=1e-8)
        for fold in ('training', 'validation'):
            for old_field, new_field in zip(getattr(target, fold), getattr(updated, fold)):
                self.assertIs(old_field, new_field)

    def test_alternative_road_changes_only_geometry_not_observations_or_origins(self):
        target, _ = road_fixture(Antennas(np.eye(3), [0]))
        original = target.problem.model.road.description
        alternative = {**original, 'enu_m': (np.asarray(original['enu_m'])+[0., 40., 3.]).tolist()}
        choice = {'route': 0, 'coarse_state': target.problem.model.coarse.tolist(),
                  'state': target.initial.tolist()}
        updated = branch(target, choice, [alternative])
        self.assertIs(updated.training, target.training)
        self.assertIs(updated.validation, target.validation)
        self.assertEqual(updated.energy, target.energy)
        np.testing.assert_array_equal(updated.origins, target.origins)
        np.testing.assert_array_equal(updated.transmitters, target.transmitters)
        a = target.problem.model.positions(target.initial, np.array([0.]), target.problem.sites)[0]
        b = updated.problem.model.positions(target.initial, np.array([0.]), target.problem.sites)[0]
        np.testing.assert_allclose(b-a, [[0., 40., 3.]], atol=1e-12)
        np.testing.assert_array_equal(target.problem.model.road.description['enu_m'], original['enu_m'])


if __name__ == '__main__':
    unittest.main()
