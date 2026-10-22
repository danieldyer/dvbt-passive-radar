import json
import unittest

import numpy as np

from block_pulse import BlockPulse
from field import Field, average
from motion import C, excess_path
from waveform_kernels import KernelField


class KernelFieldTests(unittest.TestCase):
    def field(self):
        time = np.arange(64)*.001088; block = 16
        tx = np.array([35000., 18000., 700.]); rx = np.array([0., 0., 25.])
        velocity = np.broadcast_to([3., 1., 0.], (len(time), 3)).copy()
        centre = np.array([3000., 10000., 15.])+time[:, None]*velocity
        path, gradient = excess_path(centre, tx, rx); rf = 191.625e6
        row_information = 1+.1*np.sin(time)
        t, information = average(time, row_information, block)
        frequency = np.arange(-160, 161)/512
        spectra = np.array([(1+.2*np.sin(17*frequency+i)) for i in range(4)])
        samples = (np.exp(2j*np.pi*np.outer(np.arange(-64, 65), frequency))@spectra.T/len(frequency)).T
        mean = samples.mean(axis=0)
        data = {'row_time': time, 'row_information': row_information, 'time': t,
            'information': information, 'value': np.zeros((4, 17), complex),
            'offset_samples': np.arange(-8, 9), 'centre_ecef_m': centre,
            'centre_path_m': path, 'centre_velocity_ecef_mps': velocity,
            'centre_omega': -2*np.pi*rf/C*np.sum(gradient*velocity, axis=1)*.001024,
            'body_to_ecef': np.broadcast_to(np.eye(3), (len(time), 3, 3)),
            'transmitter_ecef_m': tx, 'receiver_ecef_m': rx,
            'metadata_json': json.dumps({'rf_hz': rf, 'block_rows': block})}
        return KernelField(Field(data, mean), BlockPulse(samples), np.arange(64)//block)

    def test_physical_phase_range_and_doppler_derivatives_survive_row_binding(self):
        field = self.field(); d = field.data
        position = d['centre_ecef_m']+np.array([31., -14., 8.])
        velocity = d['centre_velocity_ecef_mps']+np.array([.2, -.1, 0.])
        jac = np.broadcast_to(np.eye(3), (len(position), 3, 3))
        value, derivative = field.trajectory(position, velocity, position_jacobian=jac,
            velocity_jacobian=np.zeros_like(jac), range_bias_m=4., range_bias_derivative=True)
        step = 1e-3
        for axis in range(4):
            delta = np.eye(3)[axis]*step if axis < 3 else np.zeros(3)
            shift = step if axis == 3 else 0.
            upper = field.trajectory(position+delta, velocity, range_bias_m=4.+shift)
            lower = field.trajectory(position-delta, velocity, range_bias_m=4.-shift)
            numerical = (upper-lower)/(2*step)
            self.assertLess(np.linalg.norm(numerical-derivative[:, :, axis])/
                np.linalg.norm(derivative[:, :, axis]), 2e-5)
        self.assertGreater(np.linalg.norm(value), 0)

    def test_windowing_preserves_block_identity_and_adjoint(self):
        field = self.field(); d = field.data
        position = d['centre_ecef_m']+np.array([11., 12., 6.])
        velocity = d['centre_velocity_ecef_mps']
        whole = field.trajectory(position, velocity)
        crop = field.windowed(field.time[1], field.time[3])
        np.testing.assert_allclose(crop.trajectory(position[16:48], velocity[16:48]), whole[1:3], atol=1e-12)
        rng = np.random.default_rng(871)
        rows = rng.normal(size=(64, 17))+1j*rng.normal(size=(64, 17))
        residual = rng.normal(size=(4, 17))+1j*rng.normal(size=(4, 17))
        observed = field.apply(average(rows, d['row_information'], 16)[0])
        np.testing.assert_allclose(np.vdot(observed, residual), np.vdot(rows, field.raw_adjoint(residual)), rtol=2e-13)


if __name__ == '__main__':
    unittest.main()
