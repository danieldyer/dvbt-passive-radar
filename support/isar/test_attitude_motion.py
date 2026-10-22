"""Independent three-dimensional geometry, Doppler and likelihood checks."""
import unittest

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.spatial.transform import Rotation

from field import FS, OFFSETS, average
from full_motion import Band
from motion import C
from rigid_motion import rotation_state
from spline_motion import SplineMotion, basis
from test_rigid_motion import fixture


def setup(field):
    d = field.data
    interval = [d['row_time'][0], d['row_time'][-1]]
    model = {'interval_s': interval, 'path_spacing_seconds': 6.,
             'heading_spacing_seconds': 8., 'rock_spacing_seconds': 6.}
    counts = [basis([np.mean(interval)], interval, spacing).shape[1] for spacing in (6., 8., 6., 6.)]
    p = np.random.default_rng(27349).normal(size=sum(counts))*.03
    curve = SplineMotion({**model, 'parameters': p})
    parameters, _ = curve.gauge_fixed_parameters()
    model['parameters'] = parameters
    prior = {'position_ecef_m': d['centre_ecef_m'], 'velocity_ecef_mps': d['centre_velocity_ecef_mps'],
        'body_to_ecef': d['body_to_ecef'],
        'frame_velocity': np.gradient(d['body_to_ecef'], d['row_time'], axis=0, edge_order=2)}
    return model, prior


def independent_echo(field, state, points, amplitudes, *, range_bias_m=0., path_rate_bias_mps=0., phase_epoch_s=0.,
                     transmitter=None, receiver=None):
    """Direct norms, Fourier pulse and quadrature; never calls the fitted template."""
    d = field.data
    transmitter = d['transmitter_ecef_m'] if transmitter is None else transmitter
    receiver = d['receiver_ecef_m'] if receiver is None else receiver
    frequency = np.fft.fftfreq(256)
    power = (abs(frequency) < .38)*(1+.3*np.sin(9*frequency))
    power /= power.sum()
    u, weight = leggauss(32)
    u, weight = u/2, weight/2
    total = np.zeros((len(d['row_time']), len(OFFSETS)), complex)
    for point, amplitude in zip(points, amplitudes):
        position = state['position_ecef_m']+state['body_to_ecef']@point
        velocity = state['velocity_ecef_mps']+state['frame_velocity']@point
        lengths, radial_rates = [], []
        for antenna in (transmitter, receiver):
            vector = position-antenna
            distance = np.sqrt(np.sum(vector**2, axis=1))
            lengths.append(distance)
            radial_rates.append(np.sum(vector*velocity, axis=1)/distance)
        path = sum(lengths)-np.linalg.norm(transmitter-receiver)
        delta = path-d['centre_path_m']+path_rate_bias_mps*(d['row_time']-phase_epoch_s)
        lag = OFFSETS[None, :]-(delta[:, None]+range_bias_m)*FS/C
        pulse = (np.exp(2j*np.pi*lag.reshape(-1, 1)*frequency)@power).reshape(lag.shape)
        omega = -field.k*.001024*(sum(radial_rates)+path_rate_bias_mps)
        tone = np.exp(1j*omega[:, None]*u)
        factor = tone@weight-1j*d['centre_omega']*(tone@(weight*u))
        total += amplitude*factor[:, None]*pulse*np.exp(-1j*field.k*delta[:, None])
    return field.apply(average(total, d['row_information'], field.metadata['block_rows'])[0])


class AttitudeMotionTests(unittest.TestCase):
    def test_rotation_rates_and_all_mixed_second_partials(self):
        time = np.linspace(0., 17., 61)
        angles = np.column_stack([.04*np.sin(.8*time), .03*np.cos(.4*time), .1+.01*time])
        rates = np.column_stack([.032*np.cos(.8*time), -.012*np.sin(.4*time), np.full(len(time), .01)])
        value, rate, partial, rate_partial = rotation_state(angles, rates)
        expected = Rotation.from_euler('xyz', angles).as_matrix()
        np.testing.assert_allclose(value, expected, atol=1e-14)
        step = 1e-5
        plus = Rotation.from_euler('xyz', angles+step*rates).as_matrix()
        minus = Rotation.from_euler('xyz', angles-step*rates).as_matrix()
        np.testing.assert_allclose(rate, (plus-minus)/(2*step), atol=4e-11)
        for axis in range(3):
            delta = np.eye(3)[axis]*step
            vp, rp, _, _ = rotation_state(angles+delta, rates)
            vm, rm, _, _ = rotation_state(angles-delta, rates)
            np.testing.assert_allclose(partial[axis], (vp-vm)/(2*step), atol=5e-11)
            np.testing.assert_allclose(rate_partial[axis], (rp-rm)/(2*step), atol=3e-12)

    def test_full_attitude_zero_rock_equals_existing_yaw_model(self):
        field = fixture()
        model, prior = setup(field)
        curve = SplineMotion(model)
        p = curve.parameters.copy()
        p[sum(curve.counts[:2]):] = 0
        full = SplineMotion({**model, 'parameters': p})
        old = {k: v for k, v in model.items() if k != 'rock_spacing_seconds'}
        old['parameters'] = p[:sum(curve.counts[:2])]
        old = SplineMotion(old)
        args = field.data['row_time'], prior, field.data['transmitter_ecef_m'], field.data['receiver_ecef_m']
        for key in ('position_ecef_m', 'velocity_ecef_mps', 'body_to_ecef', 'frame_velocity'):
            np.testing.assert_allclose(full.apply(*args)[key], old.apply(*args)[key], atol=1e-14)
        anchored = curve.components([np.mean(curve.interval)])[0]
        np.testing.assert_array_equal(anchored[[0, 2, 3]], 0)

    def test_rotation_velocity_matches_independently_differentiated_points(self):
        field = fixture()
        model, prior = setup(field)
        curve = SplineMotion(model)
        time = field.data['row_time'][1:-1]
        dt = .005
        d = field.data
        # The fixture's frame rotates uniformly: recover that analytical prior,
        # not a finite difference of the spline implementation under test.
        initial = prior['body_to_ecef'][0]
        velocity = d['centre_velocity_ecef_mps'][0]
        def at(t):
            frame = initial@Rotation.from_rotvec(np.c_[t*0, t*0, -.002*t]).as_matrix()
            generator = np.array([[0, .002, 0], [-.002, 0, 0], [0, 0, 0]])
            return {'position_ecef_m': d['centre_ecef_m'][0]+t[:, None]*velocity,
                'velocity_ecef_mps': np.broadcast_to(velocity, (len(t), 3)),
                'body_to_ecef': frame, 'frame_velocity': frame@generator}
        def state(t):
            return curve.apply(t, at(t), d['transmitter_ecef_m'], d['receiver_ecef_m'])
        point = np.array([139., -17., 41.])
        actual = state(time)
        def position(t):
            s = state(t)
            return s['position_ecef_m']+s['body_to_ecef']@point
        numeric = (-position(time+2*dt)+8*position(time+dt)-
                   8*position(time-dt)+position(time-2*dt))/(12*dt)
        np.testing.assert_allclose(actual['velocity_ecef_mps']+actual['frame_velocity']@point, numeric, atol=1e-6)

    def test_three_dimensional_profiled_gradient_against_independent_echo(self):
        field = fixture()
        model, prior = setup(field)
        curve = SplineMotion(model)
        points = np.array([[-100., -12., 8.], [20., 16., 35.], [115., -6., 45.]])
        args = field.data['row_time'], prior, field.data['transmitter_ecef_m'], field.data['receiver_ecef_m']
        field.y = independent_echo(field, curve.apply(*args), points, [.8+.3j, -.5+.4j, .4-.7j])
        selected = np.arange(len(field.time))%2 == 0
        band = Band(field, prior, args[2], args[3], points, model, selected)
        p = curve.parameters*.4
        value, gradient = band.evaluate(p)
        for index in range(len(p)):
            delta = np.eye(len(p))[index]*.0003
            numeric = (band.evaluate(p+delta)[0]-band.evaluate(p-delta)[0])/.0006
            self.assertLess(abs(numeric-gradient[index])/max(1., abs(numeric)), 1e-4)
        # Independent pulse synthesis also tests the interpolation model, so its
        # nonzero floor must not be hidden by regenerating data with Band.design.
        self.assertLess(band.evaluate(curve.parameters)[0]/band.energy, 2e-8)
        check = band.validation(curve.parameters)
        self.assertLess(check['prediction_phase_rms_degrees'], .01)
        self.assertLess(check['conditional_efficiency_loss_db'], 1e-6)
        self.assertGreater(value/band.energy, 1e-4)


if __name__ == '__main__':
    unittest.main()
