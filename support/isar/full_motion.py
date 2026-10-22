"""Full complex-data likelihood for a shared physical translation/yaw curve.

Range interpolation, carrier phase and moment Doppler all change together.
Original-row derivatives are accumulated with the exact averaging/whitening
adjoint. Independent per-band point responses and stationary nuisance responses
are profiled analytically on training rows only.
"""
import numpy as np

from field import average
from spline_motion import SplineMotion, basis
from stationary import Stationary, make_design
from validate_motion import validate


class Band:
    def __init__(self, field, prior, reference_transmitter, receiver, points, model, selected_time, *, range_bias_m=0.):
        self.field, self.prior = field, prior
        self.reference_transmitter, self.receiver = reference_transmitter, receiver
        self.points, self.model = np.asarray(points), model
        self.range_bias = range_bias_m
        self.time = field.data['row_time']
        sizes = [basis([self.time[0]], model['interval_s'], model[key]).shape[1]
                 for key in ('path_spacing_seconds', 'heading_spacing_seconds')]
        if model.get('rock_spacing_seconds') is not None:
            sizes += [basis([self.time[0]], model['interval_s'], model['rock_spacing_seconds']).shape[1]]*2
        if model.get('transverse_spacing_seconds') is not None:
            sizes += [basis([self.time[0]], model['interval_s'], model['transverse_spacing_seconds']).shape[1]]
        curve = SplineMotion({**model, 'parameters': np.zeros(sum(sizes))})
        self.bases, self.rate_bases = curve.coefficient_bases(self.time), curve.coefficient_bases(self.time, 1)
        self.npath, self.components = sizes[0], len(sizes)
        self.selected_time = selected_time
        self.selected = np.repeat(selected_time, field.y.shape[1])
        stationary, _ = make_design(field)
        self.nuisance = Stationary(stationary, self.selected)
        self.clean = self.nuisance.remove(field.y.ravel()[self.selected])
        self.energy = float(np.vdot(self.clean, self.clean).real)

    def design(self, parameters, *, derivatives=False, antenna_offsets=None,
               range_bias_m=None, range_derivative=False, points=None, point_derivatives=False,
               path_rate_bias_mps=0., path_rate_derivative=False):
        motion = SplineMotion({**self.model, 'parameters': parameters})
        state = motion.apply(self.time, self.prior, self.reference_transmitter, self.receiver, derivatives=derivatives)
        frame, frame_rate = state['body_to_ecef'], state['frame_velocity']
        tx, rx = self.field.data['transmitter_ecef_m'], self.field.data['receiver_ecef_m']
        if antenna_offsets is not None:
            tx, rx = tx+antenna_offsets[0], rx+antenna_offsets[1]
        range_bias = self.range_bias if range_bias_m is None else range_bias_m
        rate_options = {'path_rate_bias_mps': path_rate_bias_mps, 'phase_epoch_s': float(np.mean(self.model['interval_s']))}
        design, jacobians = [], []
        for point in self.points if points is None else points:
            position = state['position_ecef_m']+frame@point
            offset = (self.prior['position_ecef_m']-self.field.data['centre_ecef_m']+
                state['translation_offset_m']+frame@point)
            velocity = state['velocity_ecef_mps']+frame_rate@point
            if not derivatives:
                value = self.field.trajectory(position, velocity, transmitter=tx, receiver=rx,
                    range_bias_m=range_bias, position_offset=offset, **rate_options)
            else:
                directions = np.stack([state['translation_direction'],
                    *(partial@point for partial in state['attitude_partials'])], axis=-1)
                direction_rates = np.stack([state['translation_direction_rate'],
                    *(partial@point for partial in state['attitude_rate_partials'])], axis=-1)
                if self.model.get('transverse_spacing_seconds') is not None:
                    directions = np.concatenate([directions, state['transverse_direction'][:, :, None]], axis=-1)
                    direction_rates = np.concatenate([direction_rates, state['transverse_direction_rate'][:, :, None]], axis=-1)
                dp = np.concatenate([directions, np.zeros_like(directions)], axis=-1)
                dv = np.concatenate([direction_rates, directions], axis=-1)
                if point_derivatives:
                    dp = np.concatenate([dp, frame[:, :, :2]], axis=-1)
                    dv = np.concatenate([dv, frame_rate[:, :, :2]], axis=-1)
                raw, derivative = self.field.raw_trajectory(position, velocity,
                    position_jacobian=dp, velocity_jacobian=dv, transmitter=tx, receiver=rx,
                    antenna_derivatives=antenna_offsets is not None, range_bias_m=range_bias,
                    range_bias_derivative=range_derivative, path_rate_derivative=path_rate_derivative,
                    position_offset=offset, **rate_options)
                binned, _ = average(raw, self.field.data['row_information'], self.field.metadata['block_rows'])
                value = self.field.apply(binned)
                jacobians.append(derivative)
            design.append(value.ravel())
        return np.column_stack(design), jacobians

    def evaluate(self, parameters, antenna_offsets=None):
        target, jacobians = self.design(parameters, derivatives=True, antenna_offsets=antenna_offsets)
        design = self.nuisance.remove(target[self.selected])
        amplitudes = np.linalg.lstsq(design, self.clean, rcond=None)[0]
        residual = self.clean-design@amplitudes
        gradient = self.profile_gradient(jacobians, amplitudes, residual, antenna_derivatives=antenna_offsets is not None)
        return float(np.vdot(residual, residual).real), gradient

    def profile_gradient(self, jacobians, amplitudes, residual, *, antenna_derivatives=False, range_derivative=False):
        """Original-row adjoint for either shared or independently fitted amplitudes."""
        full_residual = np.zeros(self.field.y.size, complex)
        full_residual[self.selected] = residual
        adjoint = self.field.raw_adjoint(full_residual.reshape(self.field.y.shape))
        n = self.components
        row_gradient = np.zeros((len(self.time), 2*n+(6 if antenna_derivatives else 0)+int(range_derivative)))
        for derivative, amplitude in zip(jacobians, amplitudes):
            scale = amplitude if np.ndim(amplitude) == 0 else np.asarray(amplitude)[:, None]
            row_gradient -= 2*np.real(scale*np.einsum('nr,nrq->nq', adjoint.conj(), derivative))
        gradient = np.concatenate([b.T@row_gradient[:, i]+db.T@row_gradient[:, n+i]
                                   for i, (b, db) in enumerate(zip(self.bases, self.rate_bases))])
        if antenna_derivatives or range_derivative:
            gradient = np.r_[gradient, row_gradient[:, 2*n:].sum(axis=0)]
        return gradient

    def validation(self, parameters, antenna_offsets=None):
        target, _ = self.design(parameters, antenna_offsets=antenna_offsets)
        return validate(self.field, target, self.nuisance, self.selected_time, self.model['interval_s'])

    def information(self, parameters, antenna_offsets):
        """Conditional Gaussian information after profiling all complex responses."""
        target, jacobians = self.design(parameters, derivatives=True, antenna_offsets=antenna_offsets)
        design = self.nuisance.remove(target[self.selected])
        amplitudes = np.linalg.lstsq(design, self.clean, rcond=None)[0]
        raw = sum(a*j for a, j in zip(amplitudes, jacobians))
        del jacobians
        columns = []
        for j, (b, db) in enumerate(zip(self.bases, self.rate_bases)):
            v, dv = raw[:, :, j], raw[:, :, self.components+j]
            for i in range(b.shape[1]):
                value, _ = average(v*b[:, i, None]+dv*db[:, i, None],
                    self.field.data['row_information'], self.field.metadata['block_rows'])
                columns.append(self.field.apply(value).ravel()[self.selected])
        for axis in range(6):
            value, _ = average(raw[:, :, 2*self.components+axis], self.field.data['row_information'], self.field.metadata['block_rows'])
            columns.append(self.field.apply(value).ravel()[self.selected])
        jacobian = self.nuisance.remove(np.column_stack(columns))
        jacobian -= design@np.linalg.lstsq(design, jacobian, rcond=None)[0]
        return 2*np.real(jacobian.conj().T@jacobian)
