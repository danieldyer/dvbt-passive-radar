import unittest

import numpy as np

from fixed_frame_motion import geographic_frame, point_centred_start, restrict_interval, rotating_control_seed
from imaging import motion_state
from motion import body_basis, ecef, enu_basis
from spline_motion import SplineMotion, basis


def example():
    t = np.linspace(0, 45, 1001)
    model = {'interval_s': [0., 45.], 'path_spacing_seconds': 3., 'heading_spacing_seconds': 8.,
        'transverse_spacing_seconds': 20., 'anchor_path': False, 'anchor_transverse': True,
        'horizontal_translation': True, 'nominal_start_utc': '1970-01-01T00:00:00Z',
        'linear_reference': {'utc_seconds': 0., 'position_ecef_m': ecef(-.65, 2.5, 5.).tolist(),
            'velocity_ecef_mps': (enu_basis(-.65, 2.5)@np.array([1., 4., 0.])).tolist(),
            'body_to_ecef': body_basis(-.65, 2.5, -.4).tolist()},
        'selected_points_body_m': [[[12., -4., 20.], [-50., 7., 20.]]]*5}
    values = [.3*(t-22.5)+.002*(t-22.5)**2, 4.+.03*(t-22.5), .2*(t-22.5)]
    model['parameters'] = np.concatenate([np.linalg.lstsq(basis(t, model['interval_s'], spacing,
        fixed_midpoint=anchored), value, rcond=None)[0] for spacing, anchored, value in
        zip((3., 8., 20.), (False, False, True), values)]).tolist()
    return model


class FixedFrameTests(unittest.TestCase):
    def test_rotating_control_contains_the_fitted_fixed_frame_model(self):
        free = example()
        fixed = geographic_frame(free)
        curve = SplineMotion(fixed)
        p = curve.parameters.copy()
        p[curve.sections[0]] += np.linspace(-.1, .1, curve.sections[0].stop)
        fixed['parameters'] = p.tolist()
        control = rotating_control_seed(free, fixed)
        tx, rx = ecef(-.64, 2.51, 600.), ecef(-.651, 2.501, 30.)
        t = np.linspace(0, 45, 101)
        a = SplineMotion(fixed).apply(t, motion_state(None, t, fixed), tx, rx)
        b = SplineMotion(control).apply(t, motion_state(None, t, control), tx, rx)
        for x, y in zip(fixed['selected_points_body_m'][0], control['selected_points_body_m'][0]):
            np.testing.assert_allclose(a['position_ecef_m']+a['body_to_ecef']@x,
                b['position_ecef_m']+b['body_to_ecef']@y, atol=1e-9, rtol=0.)
            np.testing.assert_allclose(a['velocity_ecef_mps']+a['frame_velocity']@x,
                b['velocity_ecef_mps']+b['frame_velocity']@y, atol=1e-12, rtol=0.)

    def test_fixed_ne_conversion_preserves_centres_and_midpoint_reflectors(self):
        source = example()
        fixed = geographic_frame(source)
        tx, rx = ecef(-.64, 2.51, 600.), ecef(-.65, 2.5, 30.)
        t = np.linspace(0, 45, 101)
        old = SplineMotion(source).apply(t, motion_state(None, t, source), tx, rx)
        new = SplineMotion(fixed).apply(t, motion_state(None, t, fixed), tx, rx)
        np.testing.assert_allclose(new['position_ecef_m'], old['position_ecef_m'], atol=1e-10, rtol=0)
        np.testing.assert_allclose(new['velocity_ecef_mps'], old['velocity_ecef_mps'], atol=1e-12, rtol=0)
        np.testing.assert_allclose(new['body_to_ecef'], np.broadcast_to(enu_basis(-.65, 2.5), (len(t), 3, 3)), atol=1e-15)
        np.testing.assert_array_equal(new['frame_velocity'], np.zeros((len(t), 3, 3)))
        before = np.asarray(source['selected_points_body_m'][0])@old['body_to_ecef'][50].T
        after = np.asarray(fixed['selected_points_body_m'][0])@new['body_to_ecef'][50].T
        np.testing.assert_allclose(before, after, atol=1e-13)
        self.assertGreater(np.linalg.norm(old['frame_velocity']), 0.)

    def test_shorter_aperture_preserves_polynomial_initial_motion_and_gauges(self):
        model = example()
        shorter = restrict_interval(model, [7.5, 37.5])
        time = np.linspace(7.5, 37.5, 101)
        before, after = SplineMotion(model), SplineMotion(shorter)
        for derivative in (0, 1):
            np.testing.assert_allclose(after.components(time, derivative), before.components(time, derivative), atol=1e-12)
        self.assertLess(max(shorter['initial_curve_projection_maximum_component_error']), 1e-12)

    def test_offset_start_follows_the_selected_point_without_attitude_motion(self):
        source = example()
        fixed = geographic_frame(source)
        tx, rx = ecef(-.64, 2.51, 600.), ecef(-.651, 2.501, 30.)
        point = np.array(source['selected_points_body_m'][0][1])
        seeded = point_centred_start(source, fixed, point, tx, rx)
        t = np.linspace(0, 45, 101)
        before = SplineMotion(source).apply(t, motion_state(None, t, source), tx, rx)
        after = SplineMotion(seeded).apply(t, motion_state(None, t, seeded), tx, rx)
        a = before['position_ecef_m']+before['body_to_ecef']@point
        b = after['position_ecef_m']+after['body_to_ecef']@np.array(fixed['selected_points_body_m'][0][1])
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=0)
        self.assertLess(seeded['translation_seed_maximum_projection_error_m'], 1e-6)
        np.testing.assert_array_equal(after['frame_velocity'], np.zeros((len(t), 3, 3)))

    def test_near_receiver_spline_projection_error_is_reported(self):
        source = example()
        fixed = geographic_frame(source)
        tx, rx = ecef(-.64, 2.51, 600.), ecef(-.65, 2.5, 30.)
        point = np.array(source['selected_points_body_m'][0][1])
        seeded = point_centred_start(source, fixed, point, tx, rx)
        t = np.linspace(0, 45, 1001)
        before = SplineMotion(source).apply(t, motion_state(None, t, source), tx, rx)
        after = SplineMotion(seeded).apply(t, motion_state(None, t, seeded), tx, rx)
        a = before['position_ecef_m']+before['body_to_ecef']@point
        b = after['position_ecef_m']+after['body_to_ecef']@np.array(fixed['selected_points_body_m'][0][1])
        error = np.max(np.linalg.norm(a-b, axis=1))
        self.assertGreater(error, .01)
        self.assertAlmostEqual(error, seeded['translation_seed_maximum_projection_error_m'], delta=1e-8)


if __name__ == '__main__':
    unittest.main()
