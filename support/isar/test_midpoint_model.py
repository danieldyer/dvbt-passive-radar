import unittest

import numpy as np

from field import Field
from full_motion import Band
from midpoint_model import discrepancy, representative
from physical_profile import profile
from spline_motion import SplineMotion, basis
from test_rigid_motion import fixture


class MidpointModelTests(unittest.TestCase):
    def test_fixed_observer_range_and_symbol_reference_remain_bitwise_constant(self):
        original=fixture();data=dict(original.data)
        for key in ('centre_ecef_m','centre_path_m','centre_velocity_ecef_mps','centre_omega'):
            data[key]=np.broadcast_to(data[key][0],data[key].shape).copy()
        original=Field(data,original.pulse.samples);fast=representative(original)
        for key in ('centre_ecef_m','centre_path_m','centre_velocity_ecef_mps','centre_omega'):
            np.testing.assert_array_equal(fast.data[key],np.broadcast_to(original.data[key][0],fast.data[key].shape))

    def test_observations_and_erased_bins_are_unchanged(self):
        original = fixture()
        data = dict(original.data)
        rng = np.random.default_rng(52)
        data['value'] = rng.normal(size=data['value'].shape)+1j*rng.normal(size=data['value'].shape)
        original = Field(data, original.pulse.samples)
        fast = representative(original)
        np.testing.assert_array_equal(fast.y, original.y)
        self.assertTrue(np.isfinite(fast.data['centre_ecef_m']).all())
        for p in ([0., 0., 20.], [110., -12., 20.]):
            d = discrepancy(original.point(p), fast.point(p))
            self.assertLess(d['matched_snr_loss_db'], .001)
            self.assertLess(abs(d['constant_phase_difference_degrees']), .01)

    def test_free_path_origin_has_no_artificial_gauge_removal(self):
        interval = [0., 30.]
        n = basis([0.], interval, 6.).shape[1]
        m = basis([0.], interval, 8.).shape[1]
        model = {'interval_s': interval, 'path_spacing_seconds': 6., 'heading_spacing_seconds': 8.,
            'parameters': np.r_[np.full(n, 3.), np.zeros(m)], 'anchor_path': False}
        curve = SplineMotion(model)
        parameters, active = curve.gauge_fixed_parameters()
        self.assertEqual(len(active), n+m)
        np.testing.assert_array_equal(parameters, model['parameters'])
        np.testing.assert_allclose(curve.curves(np.arange(31.))[0], 3., atol=1e-14)
        np.testing.assert_allclose(curve.curves(np.arange(31.), 1)[0], 0., atol=1e-14)

    def test_profiled_free_path_origin_derivative_moves_delay_and_phase(self):
        self.check_profiled_translation_derivative(transverse=False)

    def test_profiled_transverse_derivatives_include_range_carrier_and_doppler(self):
        self.check_profiled_translation_derivative(transverse=True)

    def check_profiled_translation_derivative(self, *, transverse):
        field = representative(fixture())
        d = field.data
        interval = [0., float(d['row_time'][-1])+.1]
        model = {'interval_s': interval, 'path_spacing_seconds': 6., 'heading_spacing_seconds': 8., 'anchor_path': False}
        n = basis([8.], interval, 6.).shape[1]
        count = n+basis([8.], interval, 8.).shape[1]
        if transverse:
            model.update(transverse_spacing_seconds=6., horizontal_translation=True)
            count += n
        prior = {'position_ecef_m': d['centre_ecef_m'], 'velocity_ecef_mps': d['centre_velocity_ecef_mps'],
            'body_to_ecef': d['body_to_ecef'], 'frame_velocity': np.gradient(d['body_to_ecef'], d['row_time'], axis=0, edge_order=2)}
        band = Band(field, prior, d['transmitter_ecef_m'], d['receiver_ecef_m'],
            [[-70., 4., 20.], [90., -7., 20.]], model, np.ones(len(field.time), bool))
        truth = np.r_[np.full(n, 1.4), np.zeros(count-n)]
        if transverse:
            truth[-n:] = np.linspace(-25., 30., n)
        field.y = (band.design(truth)[0]@np.array([.8+.3j, -.5+.4j])).reshape(field.y.shape)
        band.clean = band.nuisance.remove(field.y.ravel())
        band.energy = float(np.vdot(band.clean, band.clean).real)
        candidate = truth*.6
        analytic = profile(band, candidate, derivatives=True)
        expected = analytic['jacobian'][:, :n].sum(axis=1)
        # The constant path direction largely profiles out as a complex phase.
        # Sub-mm differences amplify ECEF round-off in its small remaining
        # envelope derivative. Check two cm-scale steps in the accuracy plateau.
        for size in (.03, .01):
            step = np.r_[np.full(n, size), np.zeros(count-n)]
            numeric = (profile(band, candidate+step)['residual']-profile(band, candidate-step)['residual'])/(2*size)
            self.assertLess(np.linalg.norm(numeric-expected)/np.linalg.norm(numeric), 5e-6)
        if transverse:
            for i in range(count-n, count):
                step = np.eye(count)[i]*.6
                coarse = (profile(band, candidate+step)['residual']-profile(band, candidate-step)['residual'])/1.2
                fine = (profile(band, candidate+step/2)['residual']-profile(band, candidate-step/2)['residual'])/.6
                numeric = (4*fine-coarse)/3
                expected = analytic['jacobian'][:, i]
                self.assertLess(np.linalg.norm(numeric-expected)/np.linalg.norm(numeric), 1e-5)


if __name__ == '__main__':
    unittest.main()
