import unittest

import numpy as np

from companion_motion import add_companion
from delayed_path import DelayedPath
from physical_profile import profile
from marginal_motion import MarginalMotion
import test_physical_profile as physical_tests


class CompanionMotionTests(unittest.TestCase):
    def test_delayed_return_is_jointly_eliminated_without_changing_observations(self):
        band, truth = physical_tests.PhysicalProfileTests().fixture(noise=0.)
        component = DelayedPath(band, truth)
        component.epoch = float(band.field.time.mean())
        parameters = [420., .03]
        contaminated = band.field.y+50*component.response(parameters).reshape(band.field.y.shape)
        band.field.y = contaminated.copy()
        records = [{'interval_s': [band.field.time[0]-1., band.field.time[-1]+1.],
            'phase_epoch_s': component.epoch, 'parameters': parameters}]
        diagnostic = add_companion(band, truth, records)
        np.testing.assert_array_equal(band.field.y, contaminated)
        self.assertEqual(diagnostic['new_rank'], diagnostic['old_rank']+1)
        self.assertTrue(np.all(np.array(diagnostic['point_information_retained']) <= 1.+1e-10))
        modes = np.array(diagnostic['scene_mode_information_retained'])
        self.assertTrue(np.all((modes > 0.)&(modes <= 1.+1e-10)))
        result = profile(band, truth)
        self.assertLess(result['value']/band.energy, 1e-20)

    def test_motion_derivatives_include_the_same_fixed_nuisance_projection(self):
        band, truth = physical_tests.PhysicalProfileTests().fixture()
        component = DelayedPath(band, truth)
        parameters = [430., -.02]
        band.field.y += (2-3j)*component.response(parameters).reshape(band.field.y.shape)
        add_companion(band, truth, [{'interval_s': [band.field.time[0]-1., band.field.time[-1]+1.],
            'phase_epoch_s': component.epoch, 'parameters': parameters}])
        joint = MarginalMotion([band], truth*.3)
        at = joint.initial.copy()
        result = joint.evaluate(at)
        for j in range(len(at)):
            delta = np.eye(len(at))[j]*1e-4
            numeric = (joint.objective(at+delta)[0]-joint.objective(at-delta)[0])/2e-4
            self.assertAlmostEqual(numeric, result['gradient'][j], delta=3e-6)

    def test_missing_or_overlapping_time_support_is_rejected(self):
        band, truth = physical_tests.PhysicalProfileTests().fixture()
        record = {'interval_s': [band.field.time[0]+.1, band.field.time[-1]+1.],
            'phase_epoch_s': 8., 'parameters': [420., .03]}
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            add_companion(band, truth, [record])
        record['interval_s'][0] -= 1.
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            add_companion(band, truth, [record, record])


if __name__ == '__main__':
    unittest.main()
