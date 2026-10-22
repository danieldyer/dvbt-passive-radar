import json
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from field import Field, OFFSETS, average
from fit_rigid_motion import Band, fit
from motion import body_basis, ecef, enu_basis, excess_path
from rigid_motion import RigidCorrection, rotation


def fixture(channel=3):
    latitude, longitude = np.radians([-37.87, 144.93])
    origin = ecef(latitude, longitude, 5.)
    local = enu_basis(latitude, longitude)
    time = np.arange(512)*.032
    velocity = local@np.array([.3, 3., 0.])
    centre = origin+time[:, None]*velocity
    frame = body_basis(np.full(len(time), latitude), np.full(len(time), longitude),
                        .1+.002*time)
    transmitter = origin+local@np.array([38000.+200*channel, 12000., 700.])
    receiver = origin+local@np.array([0., 8000., 50.])
    path, gradient = excess_path(centre, transmitter, receiver)
    rf = [177.5, 184.5, 191.625, 219.5, 226.5][channel]*1e6
    omega = -2*np.pi*rf/299792458.*np.sum(gradient*velocity, axis=1)*.001024
    frequency = np.fft.fftfreq(256)
    spectrum = (abs(frequency) < .38)*(1+.3*np.sin(9*frequency))
    spectrum /= spectrum.sum()
    pulse = np.exp(2j*np.pi*np.arange(-64, 65)[:, None]*frequency)@spectrum
    weight = 1+.2*np.sin(time)
    weight[16:20] = 0
    mean_time, information = average(time, weight, 4)
    mean_time[4] = time[16:20].mean()
    data = {'time': mean_time, 'information': information,
            'value': np.zeros((len(mean_time), len(OFFSETS)), complex),
            'row_time': time, 'row_information': weight, 'offset_samples': OFFSETS,
            'centre_ecef_m': centre, 'body_to_ecef': frame, 'centre_path_m': path,
            'centre_velocity_ecef_mps': np.broadcast_to(velocity, centre.shape),
            'centre_omega': omega, 'transmitter_ecef_m': transmitter, 'receiver_ecef_m': receiver,
            'metadata_json': json.dumps({'rf_hz': rf, 'channel': channel, 'block_rows': 4})}
    return Field(data, pulse)


class RigidMotionTests(unittest.TestCase):
    def test_rotation_and_original_row_motion_derivatives(self):
        field = fixture()
        model = RigidCorrection(field, [0, field.data['row_time'][-1]], degree=2)
        parameters = np.array([.13, -.04, -.03, .02, .02, -.05, .11, -.03, -.08, .07, .15, -.1])
        state = model.state(parameters)
        angles = state['angles_radians']
        computed, _ = rotation(angles)
        np.testing.assert_allclose(computed, Rotation.from_euler('xyz', angles).as_matrix(), atol=1e-14)
        np.testing.assert_allclose(computed@computed.transpose(0, 2, 1),
                                   np.broadcast_to(np.eye(3), computed.shape), atol=1e-14)
        point = np.array([110., -12., 23.])
        value, jacobian = model.point(state, point)
        zero = model.state(np.zeros(12))
        np.testing.assert_allclose(model.point(zero, point, derivatives=False), field.point(point),
                                   rtol=1e-10, atol=1e-10)
        for parameter in range(12):
            # Too small a step loses the weak vertical derivative when adding
            # a sub-millimetre displacement to a 4,000-km ECEF coordinate.
            step = np.eye(12)[parameter]*3e-4
            numeric = (model.point(model.state(parameters+step), point, derivatives=False)-
                       model.point(model.state(parameters-step), point, derivatives=False))/(6e-4)
            self.assertLess(np.linalg.norm(numeric-jacobian[:, :, parameter])/np.linalg.norm(numeric), 5e-5)
        self.assertTrue(np.all(value[4] == 0))

    def test_profiled_likelihood_gradient_and_withheld_motion_recovery(self):
        field = fixture()
        interval = [0, field.data['row_time'][-1]]
        model = RigidCorrection(field, interval, degree=1)
        points = np.array([[-70., 4., 14.], [90., -7., 35.]])
        truth = np.array([.04, -.025, .01, .018, -.012, .03])
        # Generate the physical trajectory independently with scipy rotations.
        time = field.data['row_time']
        u = (time-np.mean(interval))/(np.diff(interval)[0]/2)
        scales = model.scale*truth
        frame = field.data['body_to_ecef']@Rotation.from_euler('xyz', u[:, None]*scales[3:]).as_matrix()
        centre = field.data['centre_ecef_m']+(u[:, None]*scales[:3])@model.reference_frame.T
        velocity = field.data['centre_velocity_ecef_mps']+(scales[:3]/(np.diff(interval)[0]/2))@model.reference_frame.T
        frame_rate = np.gradient(frame, time, axis=0, edge_order=2)
        target = np.column_stack([field.trajectory(centre+frame@point, velocity+frame_rate@point).ravel()
                                  for point in points])
        stationary = field.stationary(float(field.data['centre_path_m'].mean()+80)).ravel()
        field.y = (target@np.array([.8+.3j, -.5+.4j])+(15+8j)*stationary).reshape(field.y.shape)
        train = ((field.time-interval[0])//2)%2 == 0
        band = Band(model, points, train)
        check = truth*.6
        _, gradient = band.evaluate(check)
        for parameter in range(6):
            step = np.eye(6)[parameter]*1e-5
            numeric = (band.evaluate(check+step)[0]-band.evaluate(check-step)[0])/2e-5
            self.assertLess(abs(numeric-gradient[parameter])/max(abs(numeric), 1), 5e-4)
        result = fit([band], iterations=100)
        self.assertTrue(result['converged'], result['termination'])
        before, after = result['initial'][0], result['fitted'][0]
        self.assertLess(after['prediction_error'], before['prediction_error']*1e-3)
        self.assertLess(after['conditional_efficiency_loss_db'], .001)


if __name__ == '__main__':
    unittest.main()
