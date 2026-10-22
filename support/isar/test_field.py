import json
import unittest

import numpy as np

from ais import AisTrack
from field import Field, observation, average, FS
from imaging import attitude_basis
from motion import C, ecef, enu_basis, excess_path


class FieldTests(unittest.TestCase):
    def test_exact_row_binning_preserves_weight_and_erasures(self):
        values = np.array([1+2j, 9-4j, 3+6j, 4+8j, 5+10j])
        mean, weight = average(values, np.array([1., 0., 3., 0., 0.]), 2)
        np.testing.assert_allclose(mean, [1+2j, 3+6j, 0])
        np.testing.assert_allclose(weight, [1, 3, 0])

    def test_two_scatterers_through_range_alignment_phase_and_covariance(self):
        latitude, longitude = np.radians([-37.87, 144.93])
        seconds = np.array([0., 30., 60.])
        # North-going smooth AIS prior, explicitly used as a heading hypothesis.
        missing = np.full(3, np.nan)
        track = AisTrack(seconds, latitude+seconds*3/6359500, np.full(3, longitude),
                         np.full(3, 3.), np.zeros(3), missing.copy(), missing.copy(),
                         missing.copy(), missing.copy(), 563052200, 5.)
        time = 12+np.arange(49)*.001088
        state = track.sample(time, max_gap_s=31)
        centre = state['position_ecef_m']
        reference = ecef(latitude, longitude, 5.)
        local = enu_basis(latitude, longitude)
        transmitter = reference+local@np.array([38000., 12000., 700.])
        receiver = reference+local@np.array([0., 8000., 50.])
        path, _ = excess_path(centre, transmitter, receiver)
        frame = attitude_basis(track, state, 'course-prior')
        first = int(np.floor(path.min()*FS/C))-96
        rf = 219.5e6
        k = 2*np.pi*rf/C
        frequencies = np.fft.fftfreq(256)
        spectrum = (abs(frequencies) < .37)*(1+.4*np.sin(9*frequencies))
        spectrum /= spectrum.sum()
        pulse = np.exp(2j*np.pi*np.arange(-64, 65)[:, None]*frequencies)@spectrum
        positions = np.array([[-50., 10., 20.], [70., -20., 20.]])
        amplitudes = np.array([.4+.6j, -.2+.3j])
        moments = np.zeros((2, len(time), 192), complex)
        for point, amplitude in zip(positions, amplitudes):
            x = centre+np.einsum('nij,j->ni', frame, point)
            length, gradient = excess_path(x, transmitter, receiver)
            velocity = np.gradient(x, time, axis=0, edge_order=2)
            omega = -k*np.sum(gradient*velocity, axis=1)*.001024
            inner = (np.arange(128)+.5)/128-.5
            tone = np.exp(1j*omega[:, None]*inner)
            m0, m1 = tone.mean(axis=1), (tone*inner).mean(axis=1)
            lag = (first+np.arange(192)[None, :])-length[:, None]*FS/C
            kernel = (np.exp(2j*np.pi*lag.reshape(-1, 1)*frequencies)@spectrum).reshape(lag.shape)
            signal = amplitude*np.exp(-1j*k*length[:, None])*kernel
            moments[0] += m0[:, None]*signal
            moments[1] += m1[:, None]*signal
        stationary_path = float(path.mean()+80.)
        stationary_amplitude = 20+5j
        stationary_lag = first+np.arange(192)-stationary_path*FS/C
        stationary_pulse = np.exp(2j*np.pi*stationary_lag[:, None]*frequencies)@spectrum
        moments[0] += stationary_amplitude*np.exp(-1j*k*stationary_path)*stationary_pulse
        information = 1+.2*np.sin(time)
        valid = np.ones(len(time), bool);valid[16:32] = False;valid[3] = False
        aperture = {'time_s': time, 'moments': moments.astype('<c8'), 'information': information,
                    'valid': valid, 'metadata_json': json.dumps({'channel': 3, 'first_range_sample': first,
                                           'channel_configuration': {'rf_hz': rf}})}
        data = observation(aperture, track, 0, transmitter, receiver, 'course-prior')
        field = Field(data, pulse)
        design = np.column_stack([*(field.point(p).ravel() for p in positions), field.stationary(stationary_path).ravel()])
        amplitudes = np.r_[amplitudes, stationary_amplitude*np.exp(-1j*k*stationary_path)]
        recovered = np.linalg.lstsq(design, field.y.ravel(), rcond=None)[0]
        np.testing.assert_allclose(recovered, amplitudes, atol=5e-4)
        error = np.linalg.norm(design@amplitudes-field.y.ravel())/np.linalg.norm(field.y)
        self.assertLess(error, 2e-4)
        self.assertTrue(np.all(np.diff(field.time) > 0))
        self.assertEqual(data['information'][1], 0)
        self.assertTrue(np.all(field.y[1] == 0))
        alternate = observation(aperture, track, 0, transmitter, receiver, 'course-prior',
                                offsets=np.arange(-6, 11), block=8)
        alternate_field = Field(alternate, pulse)
        alternate_design = np.column_stack([*(alternate_field.point(p).ravel() for p in positions),
                                            alternate_field.stationary(stationary_path).ravel()])
        alternate_amplitude = np.linalg.lstsq(alternate_design, alternate_field.y.ravel(), rcond=None)[0]
        np.testing.assert_allclose(alternate_amplitude, amplitudes, atol=5e-4)
        combined, _ = average(alternate['value'], alternate['information'], 2)
        np.testing.assert_allclose(combined[:, :15], data['value'][:, 2:], rtol=1e-13, atol=1e-13)
        cropped = field.windowed(field.time[1], field.time[3])
        np.testing.assert_allclose(cropped.y, field.y[1:3])
        np.testing.assert_allclose(cropped.point(positions[0]), field.point(positions[0])[1:3], rtol=1e-7, atol=1e-7)
        shift = np.array([23., -7., 2.])
        displaced_motion = {'position_ecef_m': centre+frame@shift,
                            'velocity_ecef_mps': state['velocity_ecef_mps']+np.gradient(frame, time, axis=0, edge_order=2)@shift,
                            'body_to_ecef': frame}
        reanchored = field.point(positions[0]-shift, motion=displaced_motion)
        np.testing.assert_allclose(reanchored, field.point(positions[0]), rtol=1e-6, atol=1e-6)
        # A different body origin and heading convention must leave both the
        # physical point histories and the AIS antenna history unchanged.
        # Include frame velocity: relabelling a rotating body changes its
        # origin velocity even though the physical scatterers stay put.
        angle = np.radians(15.)
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                             [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        frame_rate = np.gradient(frame, time, axis=0, edge_order=2)
        relabelled = {**displaced_motion, 'body_to_ecef': frame@rotation,
                      'frame_velocity': frame_rate@rotation}
        original = {'position_ecef_m': centre, 'velocity_ecef_mps': state['velocity_ecef_mps'],
                    'body_to_ecef': frame, 'frame_velocity': frame_rate}
        antenna = relabelled['position_ecef_m']+relabelled['body_to_ecef']@(-rotation.T@shift)
        np.testing.assert_allclose(antenna, centre, rtol=0, atol=1e-9)
        for point in positions:
            changed = field.point(rotation.T@(point-shift), motion=relabelled)
            reference = field.point(point, motion=original)
            self.assertLess(np.linalg.norm(changed-reference)/np.linalg.norm(reference), 2e-8)
        _, jacobian = field.point(positions[0], derivatives=True)
        for axis in range(3):
            delta = np.eye(3)[axis]*.0002
            difference = (field.point(positions[0]+delta)-field.point(positions[0]-delta))/.0004
            relative = np.linalg.norm(difference-jacobian[:, :, axis])/np.linalg.norm(difference)
            self.assertLess(relative, 2e-5)


if __name__ == '__main__':
    unittest.main()
