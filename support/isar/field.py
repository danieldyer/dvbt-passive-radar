"""Complex range/slow-time observations for a rigid ship, before image pixels.

The calibration crop is a declared linear operation: symmetric range
interpolation, reversible centre-path deramp, then weighted 16-symbol means.
The forward model uses the same original row epochs and averaging operator.
No DC subtraction, fitted phase correction or local power normalization occurs.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import solve_triangular

from ais import read_csv, utc_seconds
from imaging import attitude_basis, sites
from motion import C, excess_path, path_difference
from symbol_moments import moments as symbol_moments

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'calibration'))
from geographic_io import read_profiles
from range_fit import Pulse

FS = 8e6
OFFSETS = np.arange(-8, 9)
BLOCK = 16


def interpolate(moments, coordinates, offsets=OFFSETS):
    centre = np.floor(coordinates).astype(int)
    tap = np.arange(-31, 33)
    lag = coordinates[:, None]-centre[:, None]-tap
    weight = np.sinc(lag)*np.i0(8*np.sqrt(np.maximum(0, 1-(lag/32)**2)))/np.i0(8)
    weight /= weight.sum(axis=1)[:, None]
    indices = centre[:, None, None]+offsets[None, :, None]+tap[None, None, :]
    if indices.min() < 0 or indices.max() >= moments.shape[2]:
        raise ValueError('Profile crop needs more interpolation context')
    return np.einsum('mnot,nt->mno', moments[:, np.arange(len(coordinates))[:, None, None], indices], weight)


def average(values, information, block=BLOCK):
    start = np.arange(0, len(information), block)
    weight = np.add.reduceat(information, start)
    shape = (-1,)+(1,)*(values.ndim-1)
    summed = np.add.reduceat(values*information.reshape(shape), start, axis=0)
    return np.divide(summed, weight.reshape(shape), out=np.zeros_like(summed), where=weight.reshape(shape) > 0), weight


def observation(aperture, track, epoch, transmitter, receiver, attitude, *, offsets=OFFSETS, block=BLOCK, max_ais_gap_s=180):
    meta = json.loads(str(aperture['metadata_json']))
    time = aperture['time_s']
    state = track.sample(epoch+time, max_gap_s=max_ais_gap_s)
    centre = state['position_ecef_m']
    path, gradient = excess_path(centre, transmitter, receiver)
    rf = meta['channel_configuration']['rf_hz']
    k = 2*np.pi*rf/C
    omega = -k*np.sum(gradient*state['velocity_ecef_mps'], axis=1)*.001024
    information = aperture['information']*aperture['valid']
    offsets = np.asarray(offsets)
    if offsets.ndim != 1 or not len(offsets) or not np.issubdtype(offsets.dtype, np.integer) or np.any(np.diff(offsets) <= 0):
        raise ValueError('Range offsets must be strictly increasing integer sample coordinates')
    if not isinstance(block, (int, np.integer)) or block <= 0:
        raise ValueError('A positive integer averaging block is required')
    values = np.empty((len(time), len(offsets)), complex)
    moments = aperture['moments']
    for first in range(0, len(time), 512):
        section = slice(first, first+512)
        coordinate = path[section]*FS/C-meta['first_range_sample']
        zero, one = interpolate(moments[:, section], coordinate, offsets)
        values[section] = (zero-1j*omega[section, None]*one)*np.exp(1j*k*path[section, None])
    mean, weight = average(values, information, block)
    mean_time, _ = average(time, information, block)
    mean_time[weight == 0] = average(time, np.ones(len(time)), block)[0][weight == 0]
    frame = attitude_basis(track, state, attitude)
    metadata = {'channel': meta['channel'], 'rf_hz': rf, 'block_rows': int(block),
                'attitude_hypothesis': attitude, 'centre_prior': 'AIS Hermite position with SOG/COG tangents',
                'operator': f'64-tap symmetric range interpolation; centre C0/C1 Doppler combination; exp(+j k Lcentre); information-weighted {block}-row mean',
                'no_aperture_dc_projection': True,
                'covariance': 'Conditional separable thermal model from averaged reconstructed waveform PSF and original information; to be checked against innovations'}
    return {'time': mean_time, 'value': mean, 'information': weight, 'offset_samples': offsets,
            'row_time': time, 'row_information': information,
            'centre_ecef_m': centre, 'body_to_ecef': frame,
            'centre_path_m': path, 'centre_velocity_ecef_mps': state['velocity_ecef_mps'],
            'centre_omega': omega, 'transmitter_ecef_m': transmitter, 'receiver_ecef_m': receiver,
            'metadata_json': json.dumps(metadata)}


class Field:
    def __init__(self, data, pulse_samples):
        self.data = data
        self.metadata = json.loads(str(data['metadata_json']))
        self.k = 2*np.pi*self.metadata['rf_hz']/C
        self.pulse = Pulse(pulse_samples)
        self.offsets = np.asarray(data['offset_samples'])
        if self.offsets.ndim != 1 or not len(self.offsets) or not np.issubdtype(self.offsets.dtype, np.integer) or np.any(np.diff(self.offsets) <= 0):
            raise ValueError('Field range offsets must be strictly increasing integers')
        difference = self.offsets[:, None]-self.offsets[None, :]
        if abs(difference).max() > 64 or data['value'].shape != (len(data['time']), len(self.offsets)):
            raise ValueError('Field crop exceeds covariance support or disagrees with its data')
        covariance = pulse_samples[difference+64]
        covariance = (covariance+covariance.conj().T)/2
        self.range_eigenvalues = np.linalg.eigvalsh(covariance)
        self.whiten = solve_triangular(np.linalg.cholesky(covariance), np.eye(len(self.offsets)), lower=True)
        self.time = data['time']
        self.root_information = np.sqrt(8192*data['information'])
        self.y = self.apply(data['value'])

    def windowed(self, first_s, last_s):
        """Select complete retained bins and their exact original-row support."""
        keep = np.flatnonzero((self.time >= first_s)&(self.time < last_s))
        if len(keep) < 2 or np.any(np.diff(keep) != 1):
            raise ValueError('Need a contiguous nonempty field interval')
        bins = slice(keep[0], keep[-1]+1)
        block = self.metadata['block_rows']
        rows = slice(keep[0]*block, min((keep[-1]+1)*block, len(self.data['row_time'])))
        selected = dict(self.data)
        for key in ('time', 'value', 'information'):
            selected[key] = selected[key][bins]
        for key in ('row_time', 'row_information', 'centre_ecef_m', 'body_to_ecef', 'centre_path_m',
                    'centre_velocity_ecef_mps', 'centre_omega', 'carrier_reference_path_m'):
            if key in selected:
                selected[key] = selected[key][rows]
        return Field(selected, self.pulse.samples)

    def apply(self, values):
        return (values@self.whiten.T)*self.root_information[:, None]

    def raw_adjoint(self, residual):
        """Adjoint of original-row averaging and conditional range whitening."""
        d = self.data
        bin_index = np.arange(len(d['row_time']))//self.metadata['block_rows']
        scale = np.divide(self.root_information, d['information'],
                          out=np.zeros_like(self.root_information), where=d['information'] > 0)
        unwhitened = (residual@self.whiten.conj())*scale[:, None]
        return unwhitened[bin_index]*d['row_information'][:, None]

    def stationary(self, path_m):
        """A constant-delay echo, through the same fixed observation operator.

        Its constant RF phase is absorbed into its unknown complex amplitude.
        The exp(+j k Lcentre) term reverses only the data's declared deramp.
        """
        d = self.data
        lag = self.offsets[None, :]-(path_m-d['centre_path_m'][:, None])*FS/C
        carrier=d.get('carrier_reference_path_m',d['centre_path_m'])
        raw = self.pulse(lag)*np.exp(1j*self.k*carrier[:, None])
        value, _ = average(raw, d['row_information'], self.metadata['block_rows'])
        return self.apply(value)

    def point(self, position_body_m, *, transmitter_offset_ecef_m=None, receiver_offset_ecef_m=None,
              derivatives=False, motion=None, range_bias_m=0., range_bias_derivative=False):
        """Template uses all original rows, then exactly the recorded binning.

        Both moment responses use the physical point velocity. The data-side
        centre Doppler is part of the fixed observation operator, not a fit.
        """
        d = self.data
        frame = d['body_to_ecef'] if motion is None else motion['body_to_ecef']
        centre = d['centre_ecef_m'] if motion is None else motion['position_ecef_m']
        velocity = d['centre_velocity_ecef_mps'] if motion is None else motion['velocity_ecef_mps']
        displacement = np.einsum('nij,j->ni', frame, position_body_m)
        tx, rx = d['transmitter_ecef_m'], d['receiver_ecef_m']
        tx = tx+(0 if transmitter_offset_ecef_m is None else transmitter_offset_ecef_m)
        rx = rx+(0 if receiver_offset_ecef_m is None else receiver_offset_ecef_m)
        frame_velocity = (motion['frame_velocity'] if motion is not None and 'frame_velocity' in motion
                          else np.gradient(frame, d['row_time'], axis=0, edge_order=2))
        point_velocity = velocity+np.einsum('nij,j->ni', frame_velocity, position_body_m)
        return self.trajectory(centre+displacement, point_velocity,
                               position_jacobian=frame if derivatives else None,
                               velocity_jacobian=frame_velocity if derivatives else None,
                               transmitter=tx, receiver=rx, range_bias_m=range_bias_m,
                               range_bias_derivative=range_bias_derivative)

    def trajectory(self, position, velocity, *, position_jacobian=None, velocity_jacobian=None,
                   transmitter=None, receiver=None, range_bias_m=0., range_bias_derivative=False,
                   path_rate_bias_mps=0., phase_epoch_s=0., path_rate_derivative=False,
                   position_offset=None):
        """A physical point history through the unchanged observation operator.

        Jacobians have shape (original rows, ECEF axis, real parameter). They
        change both range/phase and symbol-moment Doppler before row averaging.
        """
        raw = self.raw_trajectory(position, velocity, position_jacobian=position_jacobian,
                                  velocity_jacobian=velocity_jacobian, transmitter=transmitter, receiver=receiver,
                                  range_bias_m=range_bias_m, range_bias_derivative=range_bias_derivative,
                                  path_rate_bias_mps=path_rate_bias_mps, phase_epoch_s=phase_epoch_s,
                                  path_rate_derivative=path_rate_derivative, position_offset=position_offset)
        if position_jacobian is None and not range_bias_derivative and not path_rate_derivative:
            value, _ = average(raw, self.data['row_information'], self.metadata['block_rows'])
            return self.apply(value)
        value, derivative = raw
        value, _ = average(value, self.data['row_information'], self.metadata['block_rows'])
        derivative, _ = average(derivative, self.data['row_information'], self.metadata['block_rows'])
        return self.apply(value), np.stack([self.apply(derivative[:, :, axis])
                                           for axis in range(derivative.shape[-1])], axis=-1)

    def raw_trajectory(self, position, velocity, *, position_jacobian=None, velocity_jacobian=None,
                       transmitter=None, receiver=None, antenna_derivatives=False,
                       range_bias_m=0., range_bias_derivative=False,
                       path_rate_bias_mps=0., phase_epoch_s=0., path_rate_derivative=False,
                       position_offset=None):
        """Original-row response and derivatives before averaging/whitening."""
        d = self.data
        if 'centre_ecef_m' not in d:
            raise ValueError('An empirical path reference has no ECEF trajectory; use path_response with its saved path and phase origin')
        tx = d['transmitter_ecef_m'] if transmitter is None else transmitter
        rx = d['receiver_ecef_m'] if receiver is None else receiver
        offset = position-d['centre_ecef_m'] if position_offset is None else position_offset
        delta, omega, gradient = self.path_kinematics(position, velocity, transmitter=tx, receiver=rx,
            position_offset=offset)
        elapsed = d['row_time']-phase_epoch_s
        delta = delta+path_rate_bias_mps*elapsed
        omega = omega-self.k*.001024*path_rate_bias_mps
        point_velocity = velocity
        if position_jacobian is None and not antenna_derivatives and not range_bias_derivative and not path_rate_derivative:
            return self.path_response(delta, omega, range_bias_m=range_bias_m)
        raw, response_derivative, factor_derivative, envelope_derivative = self.path_response(
            delta, omega, range_bias_m=range_bias_m, derivatives=True)
        if position_jacobian is None:
            position_jacobian = np.empty((len(position), 3, 0))
            velocity_jacobian = np.empty_like(position_jacobian)
        if velocity_jacobian is None or velocity_jacobian.shape != position_jacobian.shape:
            raise ValueError('Position and velocity Jacobians must have matching shapes')
        path_jacobian = np.einsum('ni,nij->nj', gradient, position_jacobian)
        gradient_jacobian = np.zeros_like(position_jacobian)
        for antenna in (tx, rx):
            vector = (d['centre_ecef_m']-antenna)+offset
            distance = np.linalg.norm(vector, axis=1)
            unit = vector/distance[:, None]
            projected = np.einsum('ni,nij->nj', unit, position_jacobian)
            gradient_jacobian += (position_jacobian-unit[:, :, None]*projected[:, None, :])/distance[:, None, None]
        omega_jacobian = -self.k*.001024*(np.einsum('ni,nij->nj', gradient, velocity_jacobian)+
                                          np.einsum('ni,nij->nj', point_velocity, gradient_jacobian))
        if antenna_derivatives:
            # Include the direct-path baseline in the excess range derivative.
            # It has no Doppler derivative because both antenna sites are static.
            baseline = (tx-rx)/np.linalg.norm(tx-rx)
            antenna_path, antenna_omega = [], []
            for antenna, sign in ((tx, -1), (rx, 1)):
                vector = (d['centre_ecef_m']-antenna)+offset
                distance = np.linalg.norm(vector, axis=1)
                unit = vector/distance[:, None]
                antenna_path.append(-unit+sign*baseline)
                hessian_velocity = (point_velocity-unit*np.sum(unit*point_velocity, axis=1)[:, None])/distance[:, None]
                antenna_omega.append(self.k*.001024*hessian_velocity)
            path_jacobian = np.concatenate([path_jacobian, *antenna_path], axis=1)
            omega_jacobian = np.concatenate([omega_jacobian, *antenna_omega], axis=1)
        jacobian = (response_derivative[:, :, None]*path_jacobian[:, None, :]+
                    factor_derivative[:, :, None]*omega_jacobian[:, None, :])
        if range_bias_derivative:
            jacobian = np.concatenate([jacobian, envelope_derivative[:, :, None]], axis=-1)
        if path_rate_derivative:
            rate = elapsed[:, None]*response_derivative-self.k*.001024*factor_derivative
            jacobian = np.concatenate([jacobian, rate[:, :, None]], axis=-1)
        return raw, jacobian

    def path_kinematics(self, position, velocity, *, transmitter=None, receiver=None, position_offset=None):
        """Geometric path relative to the fixed data deramp, and symbol Doppler.

        An optional offset from the archived centre avoids rounding sub-metre
        fitted corrections through Earth-sized ECEF coordinates. It denotes
        the same physical position; it introduces no phase-only correction.
        """
        d = self.data
        tx = d['transmitter_ecef_m'] if transmitter is None else transmitter
        rx = d['receiver_ecef_m'] if receiver is None else receiver
        original = d['centre_ecef_m']
        offset = position-original if position_offset is None else position_offset
        delta = path_difference(original, offset, tx, rx)
        if not (np.array_equal(tx, d['transmitter_ecef_m']) and np.array_equal(rx, d['receiver_ecef_m'])):
            delta += excess_path(original, tx, rx)[0]-d['centre_path_m']
        gradient = np.zeros_like(original)
        for antenna in (tx, rx):
            vector = (original-antenna)+offset
            gradient += vector/np.linalg.norm(vector, axis=1)[:, None]
        omega = -self.k*np.sum(gradient*velocity, axis=1)*.001024
        return delta, omega, gradient

    def path_response(self, delta, omega, *, range_bias_m=0., derivatives=False):
        """Exact symbol response and path, Doppler, envelope-bias derivatives.

        A range-origin bias shifts the compressed envelope only. A separate
        complex channel gain represents the residual carrier phase; coupling
        that free phase to the range bias would introduce an exact redundancy.
        """
        lag = self.offsets[None, :]-(delta[:, None]+range_bias_m)*FS/C
        m0, m1, dm0, dm1 = symbol_moments(omega)
        factor = m0-1j*self.data['centre_omega']*m1
        phase_path=delta
        if 'carrier_reference_path_m' in self.data:
            phase_path=delta-(self.data['carrier_reference_path_m']-self.data['centre_path_m'])
        phase = np.exp(-1j*self.k*phase_path[:, None])
        pulse = self.pulse(lag)
        raw = factor[:, None]*pulse*phase
        if not derivatives:
            return raw
        envelope = -FS/C*self.pulse(lag, 1)*factor[:, None]*phase
        doppler = (dm0-1j*self.data['centre_omega']*dm1)[:, None]*pulse*phase
        return raw, envelope-1j*self.k*raw, doppler, envelope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aperture', type=Path, required=True)
    parser.add_argument('--ais', type=Path, required=True)
    parser.add_argument('--geometry-directory', type=Path, required=True)
    parser.add_argument('--templates', type=Path, required=True)
    parser.add_argument('--nominal-start-utc', required=True)
    parser.add_argument('--attitude', choices=['course-prior', 'reported-course-prior'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tx, rx, _ = sites(args.geometry_directory)
    track = read_csv(args.ais, reference_height_ellipsoid_m=5, mmsi=563052200)
    epoch = utc_seconds(args.nominal_start_utc)
    _, records = read_profiles(args.templates)
    args.output.mkdir(parents=True, exist_ok=True)
    for channel in range(5):
        destination = args.output/f'channel-{channel}.npz'
        if destination.exists():
            raise ValueError(f'Output already exists: {destination}')
        with np.load(args.aperture/f'channel-{channel}.npz') as aperture:
            data = observation(aperture, track, epoch, tx[channel], rx, args.attitude)
        keep = ((records['channel'] == channel)&(records['time'] >= data['row_time'][0])&
                (records['time'] <= data['row_time'][-1])&(records['information'] > 0))
        if keep.sum() < 100:
            raise ValueError('Insufficient reconstructed waveform template coverage')
        pulse = np.average(records[keep]['psf'], axis=0, weights=records[keep]['information'])
        field = Field(data, pulse)
        np.savez(destination, **data, pulse_samples=pulse)
        print(json.dumps({'channel': channel, 'bins': len(data['time']),
                          'time_s': [float(data['time'][0]), float(data['time'][-1])],
                          'range_covariance_eigenvalues': field.range_eigenvalues.tolist()}), flush=True)


if __name__ == '__main__':
    main()
