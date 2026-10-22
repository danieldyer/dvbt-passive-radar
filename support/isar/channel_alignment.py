"""Fit constant range origins and complex channel gains for a shared ship scene.

The reference channel fixes the common phase, gain and range-origin gauges.
Independent complex reflectivity for every band is a control, not evidence for
channel phase calibration. Stationary nuisance responses are profiled using
training data only. No measured samples are normalized or altered.
"""
import numpy as np
from scipy.optimize import minimize

from field import average
from scatterers import normal, shared
from stationary import Stationary, make_design


class AlignmentBand:
    def __init__(self, field, state, points, selected_time, *, transmitter=None, receiver=None):
        self.field, self.state, self.points = field, state, np.asarray(points)
        self._cache = None
        self.selected_time = np.asarray(selected_time, bool)
        self.selected = np.repeat(self.selected_time, field.y.shape[1])
        self.paths = [field.path_kinematics(state['position_ecef_m']+state['body_to_ecef']@point,
            state['velocity_ecef_mps']+state['frame_velocity']@point,
            transmitter=transmitter, receiver=receiver)[:2] for point in self.points]
        nuisance, _ = make_design(field)
        self.nuisance = Stationary(nuisance, self.selected)
        self.clean = self.nuisance.remove(field.y.ravel()[self.selected])
        self.energy = float(np.vdot(self.clean, self.clean).real)

    def design(self, range_bias_m, *, derivatives=False, path_rate_bias_mps=None, phase_epoch_s=0.):
        key = range_bias_m, path_rate_bias_mps, phase_epoch_s
        if self._cache is not None:
            bias, values, partials = self._cache
            if bias == key and (not derivatives or partials is not None):
                return values, partials
        values, partials = [], []
        field = self.field
        def observed(raw):
            binned, _ = average(raw, field.data['row_information'], field.metadata['block_rows'])
            return field.apply(binned).ravel()
        for path, omega in self.paths:
            time = field.data['row_time']-phase_epoch_s
            if path_rate_bias_mps is not None:
                path = path+path_rate_bias_mps*time
                omega = omega-field.k*.001024*path_rate_bias_mps
            result = field.path_response(path, omega, range_bias_m=range_bias_m, derivatives=derivatives)
            values.append(observed(result[0] if derivatives else result))
            if derivatives:
                derivative = observed(result[3])
                if path_rate_bias_mps is not None:
                    rate = observed(result[1]*time[:, None]-field.k*.001024*result[2])
                    derivative = np.column_stack([derivative, rate])
                partials.append(derivative)
        values = np.column_stack(values)
        partials = ((np.column_stack(partials) if path_rate_bias_mps is None else np.stack(partials, axis=1))
                    if derivatives else None)
        self._cache = key, values, partials
        return values, partials

    def validation(self, design, amplitude, interval, *, test_time=None):
        y = self.field.y.ravel()
        prediction = design@amplitude
        nuisance = self.nuisance
        beta = nuisance.solve(y[self.selected]-prediction[self.selected])
        observed = y-nuisance.design@beta
        test_time = ~self.selected_time if test_time is None else np.asarray(test_time, bool)
        if np.any(test_time&self.selected_time) or not test_time.any():
            raise ValueError('Validation requires nonempty disjoint held-out observations')
        test = np.repeat(test_time, self.field.y.shape[1])
        error = float(np.sum(abs(observed[test]-prediction[test])**2))
        reference_beta = nuisance.solve(y[self.selected])
        stationary_error = float(np.sum(abs(y[test]-nuisance.design[test]@reference_beta)**2))
        projected = np.linalg.lstsq(design[test], observed[test], rcond=None)[0]
        upper = float(np.sum(abs(design[test]@projected)**2))
        energy = float(np.sum(abs(prediction[test])**2))
        cross = np.vdot(prediction[test], observed[test])
        phases = []
        for first in np.arange(interval[0], interval[1], 1.):
            keep = np.repeat(test_time&(self.field.time >= first)&(self.field.time < first+1),
                             self.field.y.shape[1])
            if keep.any():
                value = np.vdot(prediction[keep], observed[keep])
                weight = float(np.sum(abs(prediction[keep])**2))
                phases.append({'time_s': float(first+.5), 'phase_degrees': float(np.degrees(np.angle(value))),
                    'prediction_energy': weight})
        rms = np.sqrt(np.average([p['phase_degrees']**2 for p in phases],
                                  weights=[p['prediction_energy'] for p in phases]))
        return {'prediction_error': error, 'stationary_only_error': stationary_error,
                'heldout_error_reduction_fraction': 1-error/stationary_error,
                'prediction_energy': energy, 'prediction_cross_real_imag': [float(cross.real), float(cross.imag)],
                'independent_test_energy': upper, 'prediction_phase_rms_degrees': float(rms),
                'prediction_phase_by_one_second_block': phases}


class Alignment:
    def __init__(self, bands, *, reference=3, range_scale_m=10., path_rate_scale_mps=None, phase_epoch_s=None):
        self.bands, self.reference, self.scale = bands, reference, range_scale_m
        self.other = [c for c in range(len(bands)) if c != reference]
        self.count = len(self.other)
        self.rate_scale = path_rate_scale_mps
        self.groups = 3+(path_rate_scale_mps is not None)
        self.parameter_count = self.groups*self.count
        self.phase_epoch = (float(np.mean(bands[0].field.data['row_time'][[0, -1]]))
                            if phase_epoch_s is None else phase_epoch_s)
        if not 0 <= reference < len(bands) or range_scale_m <= 0:
            raise ValueError('Need a reference channel and positive parameter scale')
        if path_rate_scale_mps is not None and (not np.isfinite(path_rate_scale_mps) or path_rate_scale_mps <= 0):
            raise ValueError('Path-rate scale must be positive and finite')
        if not all(np.array_equal(b.points, bands[0].points) for b in bands):
            raise ValueError('Coherent alignment needs the same physical scattering support in all bands')
        self.energy = sum(b.energy for b in bands)

    def unpack(self, p):
        parameters = np.asarray(p).reshape(self.groups, self.count)
        bias, gains = np.zeros(len(self.bands)), np.ones(len(self.bands), complex)
        bias[self.other] = parameters[0]*self.scale
        gains[self.other] = np.exp(parameters[1]+1j*parameters[2])
        return bias, gains

    def rates(self, p):
        rates = np.zeros(len(self.bands))
        if self.rate_scale is not None:
            rates[self.other] = np.asarray(p)[3*self.count:]*self.rate_scale
        return rates

    def project(self, parameters, *, derivatives=False):
        bias, gains = self.unpack(parameters)
        rates = self.rates(parameters)
        values, partials, full = [], [], []
        for band, delay, gain, rate in zip(self.bands, bias, gains, rates):
            design, derivative = band.design(delay, derivatives=derivatives,
                path_rate_bias_mps=None if self.rate_scale is None else rate, phase_epoch_s=self.phase_epoch)
            full.append(design)
            values.append(gain*band.nuisance.remove(design[band.selected]))
            if derivatives:
                if self.rate_scale is None:
                    partials.append(gain*self.scale*band.nuisance.remove(derivative[band.selected]))
                else:
                    d = np.stack([band.nuisance.remove(derivative[band.selected, :, i]) for i in range(2)], axis=-1)
                    partials.append(gain*d*np.array([self.scale, self.rate_scale]))
        design = np.concatenate(values)
        clean = np.concatenate([band.clean for band in self.bands])
        amplitude = np.linalg.lstsq(design, clean, rcond=None)[0]
        return amplitude, values, partials, full

    def objective(self, parameters):
        amplitude, designs, partials, _ = self.project(parameters, derivatives=True)
        value, gradient = 0., np.zeros((self.groups, self.count))
        for channel, (band, design, derivative) in enumerate(zip(self.bands, designs, partials)):
            prediction = design@amplitude
            residual = band.clean-prediction
            value += np.vdot(residual, residual).real
            if channel != self.reference:
                column = self.other.index(channel)
                columns = [derivative@amplitude, prediction, 1j*prediction] if self.rate_scale is None else [
                    derivative[:, :, 0]@amplitude, prediction, 1j*prediction, derivative[:, :, 1]@amplitude]
                jacobian = np.column_stack(columns)
                gradient[:, column] = -2*np.real(residual.conj()@jacobian)
        return float(value/self.energy), gradient.ravel()/self.energy

    def initial(self):
        equations = []
        for band in self.bands:
            design, _ = band.design(0.)
            equations.append(normal(band.nuisance.remove(design[band.selected]), band.clean))
        fit = shared(equations, reference=self.reference)
        gains = fit['gains'][self.other]
        return np.r_[np.zeros(self.count), np.log(abs(gains)), np.angle(gains),
                     np.zeros(self.count if self.rate_scale is not None else 0)]

    def fit(self, initial=None, *, vary_range=True, iterations=180, range_bound_m=75.):
        seed = self.initial() if initial is None else np.asarray(initial).copy()
        if not vary_range:
            seed[:self.count] = 0
        active = np.arange(0 if vary_range else self.count, self.parameter_count)
        def expand(x):
            p = seed.copy(); p[active] = x
            return p
        def objective(x):
            value, gradient = self.objective(expand(x))
            return value, gradient[active]
        radius = range_bound_m/self.scale
        bounds = ([(-radius, radius)]*self.count+[(-3., 3.)]*self.count+[(None, None)]*self.count)
        if self.rate_scale is not None:
            bounds += [(-.2/self.rate_scale, .2/self.rate_scale)]*self.count
        trace = []
        result = minimize(objective, seed[active], jac=True, bounds=[bounds[i] for i in active],
            method='L-BFGS-B', callback=lambda x: trace.append(float(objective(x)[0])),
            options={'maxiter': iterations, 'ftol': 1e-14, 'gtol': 1e-8, 'maxcor': 30})
        p = expand(result.x)
        projected = result.jac.copy()
        for i, index in enumerate(active):
            lo, hi = bounds[index]
            if (lo is not None and p[index] <= lo+1e-7 and projected[i] > 0) or (
                hi is not None and p[index] >= hi-1e-7 and projected[i] < 0):
                projected[i] = 0
        bias, gain = self.unpack(p)
        return p, {'objective': float(result.fun), 'converged': bool(result.success and abs(projected).max() < 1e-7),
            'maximum_projected_gradient': float(abs(projected).max()), 'iterations': result.nit,
            'termination': str(result.message), 'trace': trace,
            'range_bias_m': bias.tolist(), 'gain_amplitude': abs(gain).tolist(),
            'gain_phase_degrees': np.degrees(np.angle(gain)).tolist(),
            'phase_epoch_s': self.phase_epoch, 'path_rate_bias_mps': self.rates(p).tolist(),
            'path_rate_bound_hit': bool(np.any(abs(self.rates(p)) >= .2-1e-7)),
            'range_bound_m': range_bound_m, 'gain_bound_hit': bool(np.any(abs(p[self.count:2*self.count]) >= 3-1e-5)),
            'range_bound_hit': bool(np.any(abs(bias) >= range_bound_m-1e-5))}

    def validation(self, parameters, interval, *, independent=False, test_masks=None):
        amplitude, _, _, designs = self.project(parameters)
        _, gains = self.unpack(parameters)
        results = []
        for channel, (band, design, gain) in enumerate(zip(self.bands, designs, gains)):
            coefficients = (np.linalg.lstsq(band.nuisance.remove(design[band.selected]), band.clean, rcond=None)[0]
                            if independent else gain*amplitude)
            results.append(band.validation(design, coefficients, interval,
                test_time=None if test_masks is None else test_masks[channel]))
        cross = sum(complex(*r['prediction_cross_real_imag']) for r in results)
        energy = sum(r['prediction_energy'] for r in results)
        upper = sum(r['independent_test_energy'] for r in results)
        captured = abs(cross)**2/energy
        return {'channels': results, 'prediction_error': sum(r['prediction_error'] for r in results),
            'stationary_only_error': sum(r['stationary_only_error'] for r in results),
            'matched_energy': float(captured), 'independent_test_energy': float(upper),
            'conditional_coherent_loss_db': float(-10*np.log10(captured/upper)),
            'scope': 'Fixed training scene, channel gains and ranges predict untouched bins. The conditional dictionary comparison includes scattering mismatch and noise; it is not a global SNR bound.'}

    def information(self, parameters):
        """Conditional local information after eliminating complex scene amplitudes."""
        amplitude, designs, partials, _ = self.project(parameters, derivatives=True)
        pieces = []
        for channel, (design, derivative) in enumerate(zip(designs, partials)):
            jacobian = np.zeros((len(design), self.parameter_count), complex)
            if channel != self.reference:
                index = self.other.index(channel)
                prediction = design@amplitude
                values = [derivative@amplitude, prediction, 1j*prediction] if self.rate_scale is None else [
                    derivative[:, :, 0]@amplitude, prediction, 1j*prediction, derivative[:, :, 1]@amplitude]
                jacobian[:, index+np.arange(self.groups)*self.count] = np.column_stack(values)
            pieces.append(jacobian)
        jacobian, design = np.concatenate(pieces), np.concatenate(designs)
        jacobian -= design@np.linalg.lstsq(design, jacobian, rcond=None)[0]
        information = 2*np.real(jacobian.conj().T@jacobian)
        values = np.linalg.eigvalsh(information)
        return {'parameter_scaling': {'range_m': self.scale, 'log_gain': 1., 'phase_radians': 1.,
                                     'path_rate_mps': self.rate_scale},
            'information_eigenvalues': values.tolist(),
            'rank_at_relative_tolerance_1e_10': int(np.sum(values > values[-1]*1e-10)),
            'condition_number': float(values[-1]/values[0]) if values[0] > 0 else None,
            'scope': 'Conditional Gaussian information for fixed scattering coordinates and motion, with complex scene amplitudes and stationary nuisance eliminated. Does not account for scattering mismatch, covariance uncertainty or motion/site errors.'}
