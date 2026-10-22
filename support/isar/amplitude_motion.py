"""Physical autofocus with changing nonnegative reflector magnitudes.

Each physical point has one constant complex phase and a nonnegative magnitude
in each declared time block. Channel gains and envelope origins are explicit.
The convex magnitude problem is profiled exactly at each motion candidate.
Range, carrier phase and symbol Doppler still use the original-row operator.
"""
import numpy as np
from scipy.optimize import nnls

from field import average
from spline_motion import SplineMotion
from temporal_scene import block_design
from validate_motion import validate


class AmplitudeMotion:
    def __init__(self, bands, motion, gains, amplitudes, duration, *, reference_channel=3, refine_points=False,
                 independent_magnitudes=False, fit_path_rates=False):
        self.bands = bands
        curve = SplineMotion({**bands[0].model, 'parameters': motion})
        if curve.rock_spacing is not None:
            raise ValueError('This experiment keeps roll and pitch fixed')
        self.full_initial, self.active = curve.gauge_fixed_parameters()
        self.nmotion, self.points = len(self.active), len(bands[0].points)
        self.other = np.array([c for c in range(len(bands)) if c != reference_channel])
        self.ngain = len(self.other)
        self.range_slice = slice(self.nmotion, self.nmotion+len(bands))
        offset = self.range_slice.stop
        self.gain_slice = slice(offset, offset+2*self.ngain)
        self.phase_slice = slice(self.gain_slice.stop, self.gain_slice.stop+self.points)
        self.point_slice = slice(self.phase_slice.stop, self.phase_slice.stop+2*(self.points-1)*refine_points)
        self.refine_points = refine_points
        self.independent_magnitudes = independent_magnitudes
        self.fit_path_rates = fit_path_rates
        if not fit_path_rates and np.any(bands[0].model.get('channel_path_rate_bias_mps', 0.)):
            raise ValueError('Nonzero channel path-rate registration must be explicitly retained')
        self.rate_slice = slice(self.point_slice.stop, self.point_slice.stop+self.ngain*fit_path_rates)
        self.initial = np.r_[self.full_initial[self.active], [b.range_bias for b in bands],
            np.log(abs(np.asarray(gains)[self.other])), np.angle(np.asarray(gains)[self.other]), np.angle(amplitudes)]
        if refine_points:
            self.initial = np.r_[self.initial, bands[0].points[1:, :2].ravel()]
        if fit_path_rates:
            self.initial = np.r_[self.initial, np.asarray(bands[0].model.get('channel_path_rate_bias_mps', np.zeros(len(bands))))[self.other]]
        self.duration = duration
        self.count = int(np.ceil(np.diff(curve.interval)[0]/duration-1e-10))
        self.blocks = [np.floor((b.field.time-curve.interval[0])/duration).astype(int) for b in bands]
        if any(np.any(x < 0) or np.any(x >= self.count) for x in self.blocks):
            raise ValueError('Data exceed the amplitude interval')
        self.row_blocks = [block[np.arange(len(b.time))//b.field.metadata['block_rows']] for b, block in zip(bands, self.blocks)]
        self.energy = sum(b.energy for b in bands)
        self.last = None

    def unpack(self, parameters):
        motion = np.zeros_like(self.full_initial)
        motion[self.active] = parameters[:self.nmotion]
        gains = np.ones(len(self.bands), complex)
        p = parameters[self.gain_slice]
        gains[self.other] = np.exp(p[:self.ngain]+1j*p[self.ngain:])
        return motion, np.asarray(parameters[self.range_slice]), gains, np.exp(1j*parameters[self.phase_slice])

    def point_positions(self, parameters):
        points = self.bands[0].points.copy()
        if self.refine_points:
            points[1:, :2] = parameters[self.point_slice].reshape(-1, 2)
        return points

    def path_rates(self, parameters):
        rates = np.zeros(len(self.bands))
        if self.fit_path_rates:
            rates[self.other] = parameters[self.rate_slice]
        return rates

    def project(self, parameters, *, derivatives=False):
        motion, origins, gains, phases = self.unpack(parameters)
        designs, partials, linear = [], [], []
        for b, origin, gain, block, rate in zip(self.bands, origins, gains, self.blocks, self.path_rates(parameters)):
            design, derivative = b.design(motion, derivatives=derivatives, range_bias_m=origin, range_derivative=True,
                points=self.point_positions(parameters), point_derivatives=self.refine_points,
                path_rate_bias_mps=rate, path_rate_derivative=self.fit_path_rates)
            designs.append(design)
            partials.append(derivative)
            index = np.repeat(block, b.field.y.shape[1])[b.selected]
            linear.append(b.nuisance.remove(block_design(gain*design[b.selected], index, self.count))*np.tile(phases, self.count))
        def solve(design, y):
            real_design, real_y = np.r_[design.real, design.imag], np.r_[y.real, y.imag]
            q, r = np.linalg.qr(real_design, mode='reduced')
            rhs = q.T@real_y
            magnitude, _ = nnls(r, rhs, maxiter=10*r.shape[1])
            linear_gradient = r.T@(r@magnitude-rhs)
            linear_gradient[(magnitude == 0)&(linear_gradient > 0)] = 0
            return magnitude.reshape(self.count, self.points), float(abs(linear_gradient).max()/self.energy)
        if self.independent_magnitudes:
            solutions = [solve(a, b.clean) for a, b in zip(linear, self.bands)]
            magnitude = np.array([s[0] for s in solutions])
            optimality = max(s[1] for s in solutions)
            by_channel = magnitude
        else:
            magnitude, optimality = solve(np.concatenate(linear), np.concatenate([b.clean for b in self.bands]))
            by_channel = np.broadcast_to(magnitude, (len(self.bands), *magnitude.shape))
        if optimality > 1e-9:
            raise ValueError(f'Inner amplitude solve is not stationary: {optimality:g}')
        return {'magnitude': magnitude, 'magnitudes_by_channel': by_channel, 'point_phases': phases,
            'gains': gains, 'designs': designs, 'partials': partials, 'linear': linear,
            'linear_optimality': optimality}

    def objective(self, parameters):
        fit = self.project(parameters, derivatives=True)
        gradient, cost = np.zeros(len(parameters)), 0.
        for c, (band, gain, phases, design, derivative, block, row_block) in enumerate(zip(
                self.bands, fit['gains'], [fit['point_phases']]*len(self.bands), fit['designs'], fit['partials'], self.blocks, self.row_blocks)):
            magnitude = fit['magnitudes_by_channel'][c]
            amplitude = magnitude*phases
            prediction = fit['linear'][c]@magnitude.ravel()
            residual = band.clean-prediction
            cost += np.vdot(residual, residual).real
            envelope = 2*band.components+2*self.refine_points
            motion_derivative = [np.concatenate([d[:, :, :2*band.components], d[:, :, envelope:envelope+1]], axis=-1) for d in derivative]
            local = band.profile_gradient(motion_derivative, (gain*amplitude[row_block]).T, residual, range_derivative=True)
            gradient[:self.nmotion] += local[self.active]
            gradient[self.range_slice.start+c] = local[-1]
            if c in self.other:
                index = int(np.flatnonzero(self.other == c)[0])
                cross = np.vdot(residual, prediction)
                gradient[self.gain_slice.start+index] = -2*cross.real
                gradient[self.gain_slice.start+self.ngain+index] = 2*cross.imag
            selected_block = np.repeat(block, band.field.y.shape[1])[band.selected]
            point_predictions = gain*design[band.selected]*amplitude[selected_block]
            # The residual is orthogonal to the stationary nuisance, so the
            # unprojected point columns give the same exact profile gradient.
            gradient[self.phase_slice] += 2*np.imag(residual.conj()@point_predictions)
            if self.refine_points or self.fit_path_rates:
                full = np.zeros(band.field.y.size, complex)
                full[band.selected] = residual
                adjoint = band.field.raw_adjoint(full.reshape(band.field.y.shape))
            if self.refine_points:
                gradient[self.point_slice] += np.array([-2*np.real(np.einsum('nr,nrq,n->q',
                    adjoint.conj(), d[:, :, 2*band.components:2*band.components+2], gain*amplitude[row_block, p]))
                    for p, d in enumerate(derivative) if p > 0]).ravel()
            if self.fit_path_rates and c in self.other:
                index = int(np.flatnonzero(self.other == c)[0])
                raw_rate = sum(gain*amplitude[row_block, p, None]*d[:, :, -1] for p, d in enumerate(derivative))
                gradient[self.rate_slice.start+index] = -2*np.vdot(adjoint, raw_rate).real
        self.last = np.asarray(parameters).copy(), float(cost/self.energy), gradient/self.energy
        return self.last[1:]

    def information(self, parameters):
        fit = self.project(parameters, derivatives=True)
        jacobians = []
        for c, (band, gain, design, derivatives, block, row_block) in enumerate(zip(
                self.bands, fit['gains'], fit['designs'], fit['partials'], self.blocks, self.row_blocks)):
            amplitude = fit['magnitudes_by_channel'][c]*fit['point_phases']
            raw = sum((gain*amplitude[row_block, p])[:, None, None]*d for p, d in enumerate(derivatives))
            def observation(value):
                value, _ = average(value, band.field.data['row_information'], band.field.metadata['block_rows'])
                return band.field.apply(value).ravel()[band.selected]
            motion_columns = []
            for j, (base, rate) in enumerate(zip(band.bases, band.rate_bases)):
                for k in range(base.shape[1]):
                    motion_columns.append(observation(raw[:, :, j]*base[:, k, None]+raw[:, :, band.components+j]*rate[:, k, None]))
            local = np.zeros((band.selected.sum(), len(parameters)), complex)
            local[:, :self.nmotion] = np.column_stack(motion_columns)[:, self.active]
            local[:, self.range_slice.start+c] = observation(raw[:, :, 2*band.components+2*self.refine_points])
            selected_block = np.repeat(block, band.field.y.shape[1])[band.selected]
            point_prediction = gain*design[band.selected]*amplitude[selected_block]
            prediction = point_prediction.sum(axis=1)
            if c in self.other:
                index = int(np.flatnonzero(self.other == c)[0])
                local[:, self.gain_slice.start+index] = prediction
                local[:, self.gain_slice.start+self.ngain+index] = 1j*prediction
            local[:, self.phase_slice] = 1j*point_prediction
            if self.refine_points:
                local[:, self.point_slice] = np.column_stack([observation(
                    gain*amplitude[row_block, p, None]*d[:, :, 2*band.components+axis])
                    for p, d in enumerate(derivatives) if p > 0 for axis in range(2)])
            if self.fit_path_rates and c in self.other:
                index = int(np.flatnonzero(self.other == c)[0])
                local[:, self.rate_slice.start+index] = observation(raw[:, :, -1])
            jacobians.append(band.nuisance.remove(local))
        def profiled_information(jacobian, design, magnitude):
            real_jacobian = np.r_[jacobian.real, jacobian.imag]
            design = design[:, magnitude.ravel() > 0]
            real_design = np.r_[design.real, design.imag]
            real_jacobian -= real_design@np.linalg.lstsq(real_design, real_jacobian, rcond=1e-10)[0]
            return 2*(real_jacobian.T@real_jacobian)
        if self.independent_magnitudes:
            return sum(profiled_information(j, d, m) for j, d, m in zip(jacobians, fit['linear'], fit['magnitude']))
        return profiled_information(np.concatenate(jacobians), np.concatenate(fit['linear']), fit['magnitude'])

    def validation(self, parameters):
        fit = self.project(parameters)
        checks = []
        for band, design, gain, block, magnitude in zip(self.bands, fit['designs'], fit['gains'], self.blocks, fit['magnitudes_by_channel']):
            coefficients = (magnitude*fit['point_phases']).ravel()
            target = block_design(gain*design, np.repeat(block, band.field.y.shape[1]), self.count)
            checks.append(validate(band.field, target, band.nuisance, band.selected_time, band.model['interval_s'],
                coefficients=coefficients, stationary_mode='project'))
        return checks
