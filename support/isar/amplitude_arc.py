"""Continuous point phase, changing real strength, and explicit static clutter.

The strength is nonnegative and piecewise linear at declared original-row
epochs. Its coefficients are profiled by NNLS after eliminating the complex
stationary coefficients. This changes amplitude, never the phase trajectory.
The exact range/phase/symbol operator and averaging precede range whitening.
"""
import numpy as np
from scipy.linalg import qr
from scipy.optimize import minimize, nnls

from cluster_phase import Fit
from coherent_arc import CoherentArc
from field import average, FS
from motion import C


def qr_nnls(matrix,right):
    """Exact Householder reduction of tall NNLS without normal equations."""
    if matrix.shape[0]<=matrix.shape[1]:return nnls(matrix,right,maxiter=40*matrix.shape[1])[0]
    q,r=qr(matrix,mode='economic',check_finite=False)
    return nnls(r,q.T@right,maxiter=40*matrix.shape[1])[0]


def linear_basis(time, knots):
    time, knots = np.asarray(time), np.asarray(knots)
    if (knots.ndim != 1 or len(knots) < 2 or np.any(np.diff(knots) <= 0)
            or time.min() < knots[0] or time.max() > knots[-1]):
        raise ValueError('Strictly increasing amplitude knots must cover every original row')
    index = np.minimum(np.searchsorted(knots, time, side='right')-1, len(knots)-2)
    fraction = (time-knots[index])/(knots[index+1]-knots[index])
    result = np.zeros((len(time), len(knots)))
    result[np.arange(len(time)), index] = 1-fraction
    result[np.arange(len(time)), index+1] = fraction
    return result


class AmplitudeArc(CoherentArc):
    def __init__(self, field, training, knot_times, *, stationary_paths=None, stationary_design=None):
        super().__init__(field, training, stationary_paths=stationary_paths,stationary_design=stationary_design)
        self.knots = np.asarray(knot_times)
        self.amplitude_basis = linear_basis(field.data['row_time'], self.knots)

    def reduced(self, raw):
        f = self.field
        value = average(raw, f.data['row_information'], f.metadata['block_rows'])[0]
        return np.einsum('nrj,sr,n->nsj', value, f.whiten, f.root_information).reshape(-1, value.shape[-1])

    def response(self, parameters):
        f, base = self.field, self.base
        delta = base.basis@parameters[2:]
        omega = f.data['centre_omega']-f.k*.001024*(base.rate_basis@parameters[2:])
        raw, path, rate, envelope = f.path_response(delta, omega, range_bias_m=parameters[0]*C/FS, derivatives=True)
        rotation = np.exp(1j*parameters[1])
        derivative = np.concatenate([(envelope*C/FS)[:, :, None], (1j*raw)[:, :, None],
            path[:, :, None]*base.basis[:, None, :]-
            rate[:, :, None]*(f.k*.001024*base.rate_basis[:, None, :])], axis=2)
        return raw*rotation, derivative*rotation

    def evaluate(self, parameters, full=False):
        raw, derivative = self.response(parameters)
        design = self.reduced(raw[:, :, None]*self.amplitude_basis[:, None, :])
        projected = self.nuisance.remove(design[self.selected])
        real_design = np.r_[projected.real, projected.imag]
        coefficients, _ = nnls(real_design, np.r_[self.clean.real, self.clean.imag], maxiter=20*len(self.knots))
        residual = self.clean-projected@coefficients
        mean_derivative = self.reduced(derivative*(self.amplitude_basis@coefficients)[:, None, None])
        jacobian = self.nuisance.remove(mean_derivative[self.selected])
        value = float(np.vdot(residual, residual).real)/self.energy
        gradient = -2*np.real(jacobian.conj().T@residual)/self.energy
        if full:
            return value, gradient, design, coefficients, projected, jacobian
        return value, gradient

    def fit(self, origin_bounds=(-.8, .8), rate_bound=4., seeds=None):
        if seeds is None:
            reduced = super().initial(origin_bounds, rate_bound)
            seeds = []
            for p in reduced:
                # Only the initial constant phase comes from the simpler model.
                _, _, _, amplitude = CoherentArc.evaluate(self, p, True)
                seeds.append(np.r_[p[0], np.angle(amplitude), p[1:]])
        bound = rate_bound*self.scale
        bounds = [origin_bounds, (None, None), (-bound, bound), (-bound, bound)]
        trials = [minimize(self.evaluate, p, jac=True, method='L-BFGS-B', bounds=bounds,
            options={'gtol': 2e-9, 'ftol': 2e-13, 'maxiter': 250, 'maxcor': 12}) for p in seeds]
        best = min(trials, key=lambda f:f.fun)
        _, gradient, _, coefficients, design, derivative = self.evaluate(best.x, True)
        active = coefficients > 0
        real_design = np.r_[design.real, design.imag][:, active]
        real_derivative = np.r_[derivative.real, derivative.imag]
        real_derivative -= real_design@np.linalg.lstsq(real_design, real_derivative, rcond=1e-10)[0]
        information = 2*real_derivative.T@real_derivative
        eigen = np.linalg.eigvalsh(information)
        rank = int(np.count_nonzero(eigen > eigen[-1]*1e-10))
        return Fit(best.x, float(best.fun), float(abs(gradient).max()), [],
            [{'objective': float(t.fun), 'parameters': t.x.tolist(), 'termination': str(t.message)} for t in trials],
            np.linalg.pinv(information, rcond=1e-10), rank)

    def diagnostics(self, fit, testing=None):
        test = ~self.training if testing is None else np.asarray(testing, bool)
        if test.shape != self.training.shape or np.any(test & self.training) or not np.any(test):
            raise ValueError('Prediction requires unused samples')
        _, _, design, amplitude, projected, _ = self.evaluate(fit.parameters, True)
        y = self.field.y.ravel(); prediction = design@amplitude
        beta = self.nuisance.solve(y[self.selected]-prediction[self.selected])
        static = self.design@beta
        clean = (y-static).reshape(self.field.y.shape)
        prediction = prediction.reshape(clean.shape)
        use = np.repeat(test, len(self.field.offsets))
        baseline = self.nuisance.solve(y[self.selected])
        error = float(np.sum(abs(clean[test]-prediction[test])**2))
        baseline_error = float(np.sum(abs(y[use]-self.design[use]@baseline)**2))
        v, z = prediction[test].ravel(), clean[test].ravel()
        product = np.vdot(v, z); power = np.vdot(v, v).real
        # The fitted strength can be zero during a fade. Its phase is then
        # undefined: multiplying by it would manufacture zero phase errors.
        # Compare against the unit-strength trajectory for diagnostic plots.
        unit = self.base.template(fit.parameters)
        row_product = np.sum(unit.conj()*clean, axis=1)
        row_information = np.sum(abs(unit)**2, axis=1)
        before = np.vdot(prediction[self.training], prediction[self.training]).real
        after = np.vdot(projected@amplitude, projected@amplitude).real
        return {'time_s': self.field.time.tolist(), 'training': self.training.tolist(), 'testing': test.tolist(),
            'phase_residual_deg': np.degrees(np.angle(row_product)).tolist(),
            'matched_information': row_information.tolist(),
            'model_residual_phase_rad': (fit.parameters[1]-self.field.k*self.base.path(fit.parameters)).tolist(),
            'amplitude_knots_s': self.knots.tolist(), 'amplitude_coefficients': amplitude.tolist(),
            'amplitude': (linear_basis(self.field.time, self.knots)@amplitude).tolist(),
            'test_complex_prediction_fraction': float(1-error/baseline_error),
            'test_coherent_phase_error_deg': float(np.degrees(np.angle(product))),
            'test_matched_energy_conditional_noise_units': float(abs(product)**2/max(power, 1e-300)),
            'training_target_information_fraction_after_stationary': float(after/max(before, 1e-300)),
            'stationary_rank': self.nuisance.rank,
            'scope': 'Nonnegative original-row amplitudes and static coefficients frozen from training; conditional thermal covariance only. No unconstrained time-dependent phase.'}
