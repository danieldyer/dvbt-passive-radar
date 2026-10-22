"""Independent fluctuating-scatterer synthesis and all profile derivatives."""
import unittest

import numpy as np

from amplitude_motion import AmplitudeMotion
from full_motion import Band
from spline_motion import SplineMotion
from test_attitude_motion import independent_echo, setup
from test_rigid_motion import fixture


def synthetic(*, independent_magnitudes=False, fit_path_rates=False):
    points = np.array([[-105., -12., 8.], [15., 14., 35.], [115., -6., 45.]])
    phases = np.exp(1j*np.array([.5, -1.3, 2.1]))
    gains = np.array([.8*np.exp(.7j), 1., .7*np.exp(-1.2j)])
    origins = [12., -4., -7.5]
    rates = [.012, 0., -.016] if fit_path_rates else [0., 0., 0.]
    magnitudes = np.array([[1., .3, .2], [.2, .4, 1.1], [.3, 1.2, .7], [.9, .6, .2], [.8, .3, .6]])
    reference = fixture(3)
    tx, rx = reference.data['transmitter_ecef_m'], reference.data['receiver_ecef_m']
    bands = []
    for c, channel in enumerate((0, 3, 4)):
        field = fixture(channel)
        model, prior = setup(field)
        curve = SplineMotion(model)
        model.pop('rock_spacing_seconds')
        model['parameters'] = curve.parameters[:sum(curve.counts[:2])]
        if fit_path_rates:
            model['channel_path_rate_bias_mps'] = rates
        curve = SplineMotion(model)
        state = curve.apply(field.data['row_time'], prior, tx, rx)
        block = np.floor(field.time/4.).astype(int)
        scale = np.array([.5+.5*c, 1.5-.4*c, .8+.1*c]) if independent_magnitudes else np.ones(3)
        field.y = sum(scale[p]*gains[c]*phases[p]*magnitudes[block, p, None]*
            independent_echo(field, state, [point], [1.], range_bias_m=origins[c], path_rate_bias_mps=rates[c],
                phase_epoch_s=float(np.mean(curve.interval))) for p, point in enumerate(points))
        bands.append(Band(field, prior, tx, rx, points, model, np.arange(len(field.time))%2 == 0,
                          range_bias_m=origins[c]))
    joint = AmplitudeMotion(bands, curve.parameters, gains, phases, 4., reference_channel=1,
        independent_magnitudes=independent_magnitudes, fit_path_rates=fit_path_rates)
    return joint, magnitudes


class AmplitudeMotionTests(unittest.TestCase):
    def test_channel_path_rate_couples_envelope_phase_and_moment_response(self):
        model, _ = synthetic(independent_magnitudes=True, fit_path_rates=True)
        self.assertLess(model.objective(model.initial)[0], 2e-8)
        probe = model.initial.copy()
        probe[model.rate_slice] += [.004, -.003]
        _, gradient = model.objective(probe)
        for i in range(len(probe)):
            delta = np.eye(len(probe))[i]*2e-5
            numeric = (model.objective(probe+delta)[0]-model.objective(probe-delta)[0])/4e-5
            self.assertLess(abs(numeric-gradient[i]), 5e-6, i)
        information = model.information(probe)
        self.assertGreater(np.linalg.eigvalsh(information).min(), -information.max()*1e-11)
        for check in model.validation(model.initial):
            self.assertLess(check['prediction_error']/check['stationary_only_error'], 1e-8)

    def test_independent_strengths_keep_frequency_phases_and_all_derivatives(self):
        model, magnitudes = synthetic(independent_magnitudes=True)
        fit = model.project(model.initial)
        for c, measured in enumerate(fit['magnitude']):
            np.testing.assert_allclose(measured, magnitudes*np.array([.5+.5*c, 1.5-.4*c, .8+.1*c]), atol=8e-5)
        probe = model.initial.copy()
        probe[:model.nmotion] *= .7
        probe[model.phase_slice] += [.02, -.03, .04]
        probe[model.range_slice] += [1., -.5, .6]
        _, gradient = model.objective(probe)
        for i in range(len(probe)):
            delta = np.eye(len(probe))[i]*.0002
            numeric = (model.objective(probe+delta)[0]-model.objective(probe-delta)[0])/.0004
            self.assertLess(abs(numeric-gradient[i]), 3e-6, i)
        gauge = np.arange(model.gain_slice.start, model.gain_slice.start+model.ngain)
        information = model.information(probe)
        self.assertLess(abs(information[gauge]).max()/information.max(), 1e-12)
        for check in model.validation(model.initial):
            self.assertLess(check['prediction_error']/check['stationary_only_error'], 1e-8)

    def test_shared_point_coordinates_update_phase_range_and_symbol_doppler(self):
        model, magnitudes = synthetic()
        physical, origins, gains, phases = model.unpack(model.initial)
        refined = AmplitudeMotion(model.bands, physical, gains, phases, 4., reference_channel=1, refine_points=True)
        np.testing.assert_allclose(refined.project(refined.initial)['magnitude'], magnitudes, atol=8e-5)
        probe = refined.initial.copy()
        probe[refined.point_slice] += [.1, -.05, -.04, .08]
        _, gradient = refined.objective(probe)
        for i in range(len(probe)):
            delta = np.eye(len(probe))[i]*.0002
            numeric = (refined.objective(probe+delta)[0]-refined.objective(probe-delta)[0])/.0004
            self.assertLess(abs(numeric-gradient[i]), 3e-6, i)
        information = refined.information(probe)
        self.assertGreater(np.linalg.eigvalsh(information).min(), -information.max()*1e-11)

    def test_physical_forward_and_profile_gradient_with_changing_strengths(self):
        model, magnitudes = synthetic()
        fit = model.project(model.initial)
        np.testing.assert_allclose(fit['magnitude'], magnitudes, atol=8e-5)
        self.assertLess(model.objective(model.initial)[0], 2e-8)
        probe = model.initial.copy()
        probe[:model.nmotion] *= .6
        probe[model.gain_slice] += [.03, -.02, .04, -.03]
        probe[model.phase_slice] += [.03, -.02, .04]
        probe[model.range_slice] += [1., -2., .5]
        _, gradient = model.objective(probe)
        for i in range(len(probe)):
            delta = np.eye(len(probe))[i]*.0002
            numeric = (model.objective(probe+delta)[0]-model.objective(probe-delta)[0])/.0004
            self.assertLess(abs(numeric-gradient[i]), 3e-6, i)
        information = model.information(model.initial)
        self.assertGreater(np.linalg.eigvalsh(information).min(), -information.max()*1e-11)
        for check in model.validation(model.initial):
            self.assertLess(check['prediction_phase_rms_degrees'], .003)
            self.assertLess(check['prediction_error']/check['stationary_only_error'], 1e-8)


if __name__ == '__main__':
    unittest.main()
