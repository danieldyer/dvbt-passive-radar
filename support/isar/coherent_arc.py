"""Short coherent point arc with explicit stationary complex nuisance.

Profile one constant complex target coefficient and fixed-delay stationary
coefficients. This is the complex Gaussian profile likelihood, not per-bin
normalization or a Doppler notch. Physical delay and phase use Cluster's exact
original-row and OFDM-moment operator. Prediction freezes all coefficients.
"""
import numpy as np
from scipy.optimize import minimize

from cluster_phase import Cluster, Fit
from shared_target_geometry import complex_gain
from stationary import Stationary, make_design


class CoherentArc:
    def __init__(self, field, training, *, stationary_paths=None, stationary_design=None):
        self.base = Cluster(field, training)
        self.field, self.training = field, np.asarray(training, bool)
        self.epoch, self.scale = self.base.epoch, self.base.scale
        self.selected = np.repeat(self.training, len(field.offsets))
        if stationary_design is not None:
            if stationary_paths is not None:raise ValueError('Choose a stationary design or physical delay columns')
            self.design=np.asarray(stationary_design,complex)
            if self.design.ndim!=2 or self.design.shape[0]!=field.y.size or not np.isfinite(self.design).all():
                raise ValueError('Stationary design must match the observed complex coordinates')
            self.delays=np.empty(0)
        elif stationary_paths is None:
            self.design, self.delays = make_design(field)
        else:
            self.delays = np.asarray(stationary_paths, float)
            if self.delays.ndim != 1 or not np.isfinite(self.delays).all():
                raise ValueError('Stationary paths must be a finite vector')
            self.design = (np.column_stack([field.stationary(path).ravel() for path in self.delays])
                           if len(self.delays) else np.empty((field.y.size,0),complex))
        self.nuisance = Stationary(self.design, self.selected)
        self.clean = self.nuisance.remove(field.y.ravel()[self.selected])
        self.energy = float(np.vdot(self.clean, self.clean).real)
        self.indices = [0, 2, 3]

    def template(self, parameters, derivatives=False):
        full = np.r_[parameters[0], 0., parameters[1:]]
        if not derivatives:
            return self.base.template(full).ravel()
        value, derivative = self.base.template(full, True)
        return value.ravel(), derivative[:, :, self.indices].reshape(-1, 3)

    def evaluate(self, parameters, full=False):
        value, derivative = self.template(parameters, True)
        h = self.nuisance.remove(value[self.selected])
        dh = self.nuisance.remove(derivative[self.selected])
        residual, jacobian, amplitude, _ = complex_gain(self.clean, h, dh)
        cost = float(np.vdot(residual, residual).real)/self.energy
        gradient = 2*np.real(jacobian.conj().T@residual)/self.energy
        return (cost, gradient, value, amplitude) if full else (cost, gradient)

    def initial(self, origin_bounds, rate_bound, count=6):
        elapsed = self.field.time-self.epoch
        spacing = np.pi/(3*self.field.k*(elapsed[-1]-elapsed[0]))
        rates = np.linspace(-rate_bound, rate_bound, max(9, int(np.ceil(2*rate_bound/spacing))+1))
        rotations = np.repeat(np.exp(-1j*self.field.k*elapsed[:, None]*rates), len(self.field.offsets), axis=0)
        candidates = []
        for origin in np.linspace(*origin_bounds, 5):
            base = self.template([origin, 0., 0.])
            h = self.nuisance.remove((base[:, None]*rotations)[self.selected])
            score = abs(self.clean.conj()@h)**2/np.maximum(np.sum(abs(h)**2, axis=0), 1e-300)
            maxima = np.r_[0, np.flatnonzero((score[1:-1] >= score[:-2])&(score[1:-1] >= score[2:]))+1, len(rates)-1]
            candidates.extend((float(score[i]), np.array([origin, rates[i]*self.scale, 0.])) for i in maxima)
        return [p for _, p in sorted(candidates, key=lambda q: -q[0])[:count]]

    def fit(self, origin_bounds=(-1.2, 1.2), rate_bound=4., seeds=None):
        if seeds is None:
            seeds = self.initial(origin_bounds, rate_bound)
        bound = rate_bound*self.scale
        bounds = [origin_bounds, (-bound, bound), (-bound, bound)]
        trials = [minimize(self.evaluate, p, jac=True, method='L-BFGS-B', bounds=bounds,
            options={'gtol': 2e-9, 'ftol': 2e-13, 'maxiter': 200, 'maxcor': 12}) for p in seeds]
        best = min(trials, key=lambda f: f.fun)
        _, gradient, value, amplitude = self.evaluate(best.x, True)
        parameters = np.r_[best.x[0], np.angle(amplitude), best.x[1:]]
        h, derivative = self.base.template(parameters, True)
        h = self.nuisance.remove(h.reshape(-1)[self.selected])
        derivative = self.nuisance.remove(derivative.reshape(-1, 4)[self.selected])*abs(amplitude)
        # Keep the carrier-phase coordinate; eliminate only one real amplitude
        # after eliminating the independent complex stationary coefficients.
        derivative -= h[:, None]*np.real(h.conj()@derivative/np.vdot(h, h).real)[None, :]
        information = 2*np.real(derivative.conj().T@derivative)
        eigen = np.linalg.eigvalsh(information)
        rank = int(np.count_nonzero(eigen > eigen[-1]*1e-10))
        return Fit(parameters, float(best.fun), float(abs(gradient).max()), [],
            [{'objective': float(t.fun), 'parameters_reduced': t.x.tolist(),
              'termination': str(t.message)} for t in trials], np.linalg.pinv(information, rcond=1e-10), rank)

    def diagnostics(self, fit, testing=None):
        test = ~self.training if testing is None else np.asarray(testing, bool)
        if test.shape != self.training.shape or np.any(test & self.training) or not np.any(test):
            raise ValueError('Prediction requires unused samples')
        parameters = fit.parameters[self.indices]
        _, _, h, amplitude = self.evaluate(parameters, True)
        y = self.field.y.ravel()
        beta = self.nuisance.solve(y[self.selected]-amplitude*h[self.selected])
        static = self.design@beta
        clean = (y-static).reshape(self.field.y.shape)
        prediction = (amplitude*h).reshape(self.field.y.shape)
        normalized = h.reshape(self.field.y.shape)*np.exp(1j*np.angle(amplitude))
        norm = np.sum(abs(normalized)**2, axis=1)
        q = np.sum(normalized.conj()*clean, axis=1)/norm
        weights = abs(q[test])**2*norm[test]
        phase = np.angle(q)
        baseline_beta = self.nuisance.solve(y[self.selected])
        use = np.repeat(test, len(self.field.offsets))
        baseline_error = float(np.sum(abs(y[use]-self.design[use]@baseline_beta)**2))
        residual = clean-prediction
        error = float(np.sum(abs(residual[test])**2))
        v = prediction[test].ravel(); z = clean[test].ravel()
        product = np.vdot(v, z); information = np.vdot(v, v).real
        projected_norm = np.vdot(self.nuisance.remove(h[self.selected]), self.nuisance.remove(h[self.selected])).real
        return {'time_s': self.field.time.tolist(), 'training': self.training.tolist(), 'testing': test.tolist(),
            'phase_residual_deg': np.degrees(phase).tolist(), 'amplitude': abs(q).tolist(),
            'matched_information': norm.tolist(),
            'model_path_increment_m': self.base.path(fit.parameters).tolist(),
            'model_residual_phase_rad': (fit.parameters[1]-self.field.k*self.base.path(fit.parameters)).tolist(),
            'test_phase_rms_deg': float(np.degrees(np.sqrt(np.average(phase[test]**2, weights=weights)))),
            'test_complex_prediction_fraction': float(1-error/baseline_error),
            'test_complex_prediction_improvement': baseline_error-error,
            'test_coherent_phase_error_deg': float(np.degrees(np.angle(product))),
            'test_matched_energy_conditional_noise_units': float(abs(product)**2/information),
            'test_coherence_squared': float(abs(product)**2/(information*np.vdot(z, z).real)),
            'conditional_residual_power_per_complex_cell': float(np.mean(abs(residual[test])**2)),
            'training_target_information_fraction_after_stationary': float(projected_norm/np.vdot(h[self.selected], h[self.selected]).real),
            'stationary_columns': len(self.delays), 'stationary_rank': self.nuisance.rank,
            'stationary_singular_values': self.nuisance.singular_values.tolist(),
            'phase_scope': 'Point amplitude and stationary coefficients fit on training only; both frozen for prediction. Per-bin phase is noisy at weak target SNR and is not an acceptance gate.',
            'covariance_scope': 'Conditional thermal covariance, including nuisance elimination. Unmodelled nonstationary interference remains in the residual; no local noise renormalization.'}
