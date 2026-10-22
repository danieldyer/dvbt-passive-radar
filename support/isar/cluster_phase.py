"""Joint delay/phase estimation for one local prominent-point hypothesis.

The observations are unchanged complex range vectors. A physical range PSF,
its range covariance, the original OFDM row epochs, symbol moments and recorded
averaging operator all enter the likelihood. One polynomial path increment
drives BOTH envelope migration and carrier phase. A constant carrier phase is
free, as is one nonnegative real amplitude per retained time bin. The latter
allows scintillation without giving the fit an arbitrary time-dependent phase.

This is a conditional complex-Gaussian maximum-likelihood estimator, not a
claim that a distributed ship is a point or that the conditional covariance
describes all interference. Prediction on withheld time blocks tests that
model. No phase is unwrapped in the optimization.
"""
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import least_squares, minimize

from field import Field, average, FS
from motion import C


def crop(data, first, last, offsets):
    """Use a covariance submatrix, not selected coordinates of whitened data."""
    selected = dict(data)
    columns = np.array([np.flatnonzero(data['offset_samples'] == x)[0] for x in offsets])
    selected['value'] = data['value'][:, columns]
    selected['offset_samples'] = np.asarray(offsets)
    return Field(selected, data['pulse_samples']).windowed(first, last)


@dataclass
class Fit:
    parameters: np.ndarray
    objective: float
    gradient: float
    iterations: list
    trials: list
    covariance: np.ndarray
    information_rank: int


class PolynomialPath:
    def __init__(self, first, last, degree):
        self.epoch, self.scale = (first+last)/2, (last-first)/2
        self.degree = degree

    def __call__(self, time, derivative=0):
        u = (np.asarray(time)-self.epoch)/self.scale
        columns = []
        for j in range(1, self.degree+1):
            coefficient = np.polynomial.polynomial.polyder(np.r_[np.zeros(j), 1.], derivative)
            columns.append(np.polynomial.polynomial.polyval(u, coefficient)/self.scale**derivative)
        return np.column_stack(columns)


class SplinePath:
    """Cubic path with a removed constant gauge; spacing selected by prediction."""
    def __init__(self, first, last, spacing):
        self.epoch, self.scale = (first+last)/2, (last-first)/2
        knots = np.r_[np.repeat(first, 4), np.arange(first+spacing, last-1e-10, spacing), np.repeat(last, 4)]
        n = len(knots)-4
        self.spline = BSpline(knots, np.eye(n), 3)
        self.degree = n-1
        self.at_epoch = self.spline(self.epoch)[1:]

    def __call__(self, time, derivative=0):
        values = self.spline(np.asarray(time), nu=derivative)[:, 1:]
        return values-self.at_epoch if derivative == 0 else values


class Cluster:
    """Parameters: envelope origin (samples), constant phase, path coefficients.

    Path coefficients are metres multiplying u, u**2, ... with
    u=(t-epoch)/scale. Removing the path intercept from the carrier avoids its
    exact redundancy with unknown complex reflectivity. It remains measurable
    through the range envelope. All bands keep their own wavenumber.
    """
    def __init__(self, field, training, degree=2, curve=None):
        self.field = field
        self.training = np.asarray(training, bool)
        self.curve = PolynomialPath(field.time[0], field.time[-1], degree) if curve is None else curve
        degree = self.curve.degree
        if self.training.shape != field.time.shape or np.count_nonzero(self.training) < degree+3:
            raise ValueError('Insufficient training epochs for the requested polynomial')
        self.degree = degree
        self.epoch, self.scale = self.curve.epoch, self.curve.scale
        t = field.data['row_time']
        self.basis = self.curve(t)
        self.rate_basis = self.curve(t, 1)
        self.energy = float(np.sum(abs(field.y[self.training])**2))
        if self.energy <= 0:
            raise ValueError('No training energy')

    def path(self, parameters, time=None, derivative=0):
        t = self.field.time if time is None else np.asarray(time)
        return self.curve(t, derivative)@parameters[2:]

    def template(self, parameters, derivatives=False):
        f = self.field
        delta = self.basis@parameters[2:]
        omega = f.data['centre_omega']-f.k*.001024*(self.rate_basis@parameters[2:])
        response = f.path_response(delta, omega, range_bias_m=parameters[0]*C/FS,
                                   derivatives=derivatives)
        rotation = np.exp(1j*parameters[1])
        if derivatives:
            raw, path_jacobian, omega_jacobian, envelope_jacobian = response
            jacobian = np.concatenate([
                (envelope_jacobian*C/FS)[:, :, None],
                (1j*raw)[:, :, None],
                path_jacobian[:, :, None]*self.basis[:, None, :]-
                omega_jacobian[:, :, None]*(f.k*.001024*self.rate_basis[:, None, :])], axis=2)
        else:
            raw = response
        value = average(raw, f.data['row_information'], f.metadata['block_rows'])[0]
        value = f.apply(value)*rotation
        if not derivatives:
            return value
        derivative = average(jacobian, f.data['row_information'], f.metadata['block_rows'])[0]
        derivative = np.stack([f.apply(derivative[:, :, j]) for j in range(derivative.shape[2])], axis=2)
        return value, derivative*rotation

    def evaluate(self, parameters, full=False):
        if not full:
            return self.objective(parameters)
        h, jacobian = self.template(parameters, True)
        norm = np.sum(abs(h)**2, axis=1)
        product = np.sum(h.conj()*self.field.y, axis=1)
        amplitude = np.maximum(0., product.real)/np.maximum(norm, np.finfo(float).tiny)
        residual = self.field.y-amplitude[:, None]*h
        use = self.training
        value = float(np.sum(abs(residual[use])**2)/self.energy)
        gradient = -2*np.real(np.einsum('nt,ntj,n->j', residual[use].conj(),
                                      jacobian[use], amplitude[use]))/self.energy
        if full:
            q = np.divide(product, norm, out=np.zeros_like(product), where=norm > 0)
            return value, gradient, h, jacobian, amplitude, q, residual
        return value, gradient

    def objective(self, parameters):
        """Exact adjoint avoids a rows × ranges × coefficients Jacobian."""
        f = self.field
        delta = self.basis@parameters[2:]
        omega = f.data['centre_omega']-f.k*.001024*(self.rate_basis@parameters[2:])
        raw, path_jacobian, omega_jacobian, envelope_jacobian = f.path_response(
            delta, omega, range_bias_m=parameters[0]*C/FS, derivatives=True)
        rotation = np.exp(1j*parameters[1])
        value = average(raw, f.data['row_information'], f.metadata['block_rows'])[0]
        h = f.apply(value)*rotation
        norm = np.sum(abs(h)**2, axis=1)
        product = np.sum(h.conj()*f.y, axis=1)
        amplitude = np.maximum(0., product.real)/np.maximum(norm, 1e-300)
        residual = (f.y-amplitude[:, None]*h)*self.training[:, None]
        objective = float(np.sum(abs(residual)**2)/self.energy)
        adjoint = f.raw_adjoint(residual*amplitude[:, None]*rotation.conjugate()).conj()
        gradient = np.r_[np.sum(adjoint*envelope_jacobian)*C/FS, np.sum(adjoint*(1j*raw)),
                         np.sum(adjoint*path_jacobian, axis=1)@self.basis-
                         (f.k*.001024*np.sum(adjoint*omega_jacobian, axis=1))@self.rate_basis]
        return objective, -2*gradient.real/self.energy

    def phase_seed(self, origin):
        """Training-only unwrapping is a seed, never an accepted cycle solution."""
        p = np.zeros(self.degree+2);p[0] = origin
        h = self.template(p)[self.training]
        q = np.sum(h.conj()*self.field.y[self.training], axis=1)
        phase = np.unwrap(np.angle(q))
        norm = np.sum(abs(h)**2, axis=1)
        weight = abs(q)/np.sqrt(np.maximum(norm, 1e-300))
        design = np.column_stack([np.ones(len(phase)), -self.field.k*self.curve(self.field.time[self.training])])
        coefficients = np.linalg.lstsq(design*weight[:, None], phase*weight, rcond=None)[0]
        p[1:] = coefficients
        return p

    def residual_jacobian(self, parameters):
        """Exact derivative of the amplitude-profiled real residual vector."""
        h, jacobian = self.template(parameters, True)
        y = self.field.y[self.training]
        h, jacobian = h[self.training], jacobian[self.training]
        norm = np.sum(abs(h)**2, axis=1)
        cross = np.sum(h.conj()*y, axis=1).real
        amplitude = np.maximum(cross, 0)/np.maximum(norm, 1e-300)
        dnorm = 2*np.real(np.einsum('nt,ntj->nj', h.conj(), jacobian))
        dcross = np.real(np.einsum('ntj,nt->nj', jacobian.conj(), y))
        da = (dcross-amplitude[:, None]*dnorm)/np.maximum(norm[:, None], 1e-300)
        da[cross <= 0] = 0
        residual = y-amplitude[:, None]*h
        derivative = -amplitude[:, None, None]*jacobian-h[:, :, None]*da[:, None, :]
        residual = np.r_[residual.real.ravel(), residual.imag.ravel()]/np.sqrt(self.energy)
        derivative = np.concatenate([derivative.real.reshape(-1, len(parameters)),
                                     derivative.imag.reshape(-1, len(parameters))])/np.sqrt(self.energy)
        return residual, derivative

    def initial(self, origin_bounds, rate_bound=2., count=5):
        """Training-only envelope/rate search; retain competing tone peaks."""
        use = self.training
        elapsed = self.field.time[use]-self.epoch
        spacing = np.pi/(3*self.field.k*(self.field.time[-1]-self.field.time[0]))
        rates = np.linspace(-rate_bound, rate_bound, max(9, int(np.ceil(2*rate_bound/spacing))+1))
        rotation = np.exp(1j*self.field.k*elapsed[:, None]*rates[None, :])
        candidates = []
        for origin in np.linspace(*origin_bounds, 5):
            p = np.zeros(self.degree+2);p[0] = origin
            h = self.template(p)[use]
            cross = np.sum(h.conj()*self.field.y[use], axis=1)
            norm = np.sum(abs(h)**2)
            score = abs(cross@rotation)**2/max(norm, np.finfo(float).tiny)
            maxima = np.r_[0, np.flatnonzero((score[1:-1] >= score[:-2]) &
                                            (score[1:-1] >= score[2:]))+1, len(rates)-1]
            for index in maxima:
                seed = p.copy()
                seed[1] = np.angle(cross@rotation[:, index])
                seed[2:] = np.linalg.lstsq(self.basis, rates[index]*(self.field.data['row_time']-self.epoch), rcond=None)[0]
                candidates.append((float(score[index]), seed))
        return [p for _, p in sorted(candidates, key=lambda x:-x[0])[:count]]

    def fit(self, origin_bounds=(-.8, .8), rate_bound=2., seeds=None):
        starts = self.initial(origin_bounds, rate_bound) if seeds is None else seeds
        bound = rate_bound*self.scale
        bounds = [origin_bounds, (None, None)]+[(-bound, bound)]*self.degree
        trials = []
        for start in starts:
            iterations = []
            def record(p):
                iterations.append({'parameters': p.tolist(), 'objective': self.evaluate(p)[0]})
            result = minimize(self.evaluate, start, jac=True, method='L-BFGS-B', bounds=bounds,
                callback=record, options={'gtol': 2e-9, 'ftol': 2e-13, 'maxiter': 180, 'maxcor': 12})
            trials.append((result, iterations))
        best, iterations = min(trials, key=lambda x:x[0].fun)
        if np.max(abs(best.jac)) > 1e-6:
            # A direct least-squares step handles the correlated spline basis
            # more reliably than endlessly extending a limited-memory update.
            # Cache the full residual/Jacobian between scipy's two callbacks.
            cache = [None, None]
            def evaluate(p):
                if cache[0] is None or not np.array_equal(cache[0], p):
                    cache[:] = [p.copy(), self.residual_jacobian(p)]
                    iterations.append({'parameters': p.tolist(), 'objective': float(cache[1][0]@cache[1][0]),
                                       'method': 'exact projected least squares'})
                return cache[1]
            lower = np.array([-np.inf if b[0] is None else b[0] for b in bounds])
            upper = np.array([np.inf if b[1] is None else b[1] for b in bounds])
            polished = least_squares(lambda p:evaluate(p)[0], best.x, jac=lambda p:evaluate(p)[1],
                                    bounds=(lower, upper), x_scale='jac', max_nfev=100,
                                    ftol=2e-12, xtol=2e-12, gtol=2e-9)
            value, gradient = self.evaluate(polished.x)
            if value < best.fun:
                best.x, best.fun, best.jac = polished.x, value, gradient
                best.message = 'Projected least squares: '+str(polished.message)
        _, gradient, h, jacobian, amplitude, _, _ = self.evaluate(best.x, True)
        use = self.training
        # Real Schur complement eliminates the per-bin real amplitude nuisance.
        derivative = jacobian[use]*amplitude[use, None, None]
        norm = np.sum(abs(h[use])**2, axis=1)
        radial = np.real(np.einsum('nt,ntj->nj', h[use].conj(), derivative))/np.maximum(norm[:, None], 1e-300)
        derivative -= h[use, :, None]*radial[:, None, :]
        information = 2*np.real(np.einsum('ntj,ntk->jk', derivative.conj(), derivative))
        singular = np.linalg.eigvalsh(information)
        rank = int(np.count_nonzero(singular > singular[-1]*1e-10))
        return Fit(best.x, float(best.fun), float(np.max(abs(gradient))), iterations,
                   [{'objective': float(r.fun), 'parameters': r.x.tolist(), 'termination': str(r.message)}
                    for r, _ in trials], np.linalg.pinv(information, rcond=1e-10), rank)

    def diagnostics(self, fit, testing=None):
        _, _, h, _, amplitude, q, residual = self.evaluate(fit.parameters, True)
        use = ~self.training if testing is None else np.asarray(testing, bool)
        if np.any(use & self.training):
            raise ValueError('Prediction samples must be withheld from fitting')
        phase = np.angle(q)
        # Descriptive circular residual, not a claim of prediction of amplitude.
        information = np.sum(abs(h)**2, axis=1)
        weights = abs(q[use])**2*information[use]
        rms = np.sqrt(np.average(phase[use]**2, weights=weights)) if weights.sum() else np.nan
        coherent = abs(np.sum(q[use]*np.sqrt(information[use])))**2
        coherent /= max(np.sum(abs(q[use])**2*information[use])*np.count_nonzero(use), 1e-300)
        # Interpolate only the training amplitudes to predict held-out complex data.
        prediction_amplitude = np.interp(self.field.time, self.field.time[self.training], amplitude[self.training])
        prediction = prediction_amplitude[:, None]*h
        error = np.sum(abs(self.field.y[use]-prediction[use])**2)
        baseline = np.sum(abs(self.field.y[use])**2)
        return {'time_s': self.field.time.tolist(), 'training': self.training.tolist(),
                'phase_residual_deg': np.degrees(phase).tolist(), 'amplitude': abs(q).tolist(),
                'matched_information': information.tolist(),
                'model_path_increment_m': self.path(fit.parameters).tolist(),
                'model_residual_phase_rad': (fit.parameters[1]-self.field.k*self.path(fit.parameters)).tolist(),
                'envelope_offset_m': (fit.parameters[0]*C/FS+self.path(fit.parameters)).tolist(),
                'test_phase_rms_deg': float(np.degrees(rms)),
                'test_complex_prediction_fraction': float(1-error/baseline),
                'test_coherence_squared': float(coherent),
                'conditional_residual_power_per_complex_cell': float(np.mean(abs(residual[use])**2)),
                'test_positive_amplitude_fraction': float(np.mean(q[use].real > 0)),
                'phase_scope': 'Held-out phase; fitted delay PSF and training phase frozen. Amplitude is not refitted for complex prediction.',
                'covariance_scope': 'Conditional thermal range covariance only; does not include unresolved echoes, interpolation errors or correlated slow-time interference.'}
