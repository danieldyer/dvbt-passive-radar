"""Training-only selection of one physical template variant per reflector.

Reflectors share phase across bands, with positive independent band strengths
and one constant phase per band. Exact single-reflector conditional updates
search every eligible variant. Continuous phases and all strengths are then
jointly profiled. Convergence certifies coordinate stationarity, not a global
optimum or physical identifiability of the selected variants.
"""
import numpy as np
from scipy.optimize import minimize

from phase_coupling import PhaseCoupling
from positive_phase import maximum


class VariantPhase:
    def __init__(self, designs, observations, allowed, *, reference=3):
        self.allowed = np.asarray(allowed, bool)
        if self.allowed.ndim != 2 or not self.allowed.any(axis=1).all():
            raise ValueError('Every reflector needs at least one eligible variant')
        self.points, self.variants = self.allowed.shape
        self.designs = [np.asarray(a) for a in designs]
        self.observations = [np.asarray(y) for y in observations]
        self.reference = reference
        self.other = [c for c in range(len(designs)) if c != reference]
        if not 0 <= reference < len(designs) or len(designs) != len(observations):
            raise ValueError('Invalid channels or reference')
        for a, y in zip(self.designs, self.observations):
            if a.ndim != 2 or a.shape[1] != self.allowed.size or y.shape != (len(a),):
                raise ValueError('Expected point-major variant dictionaries and matching observations')
            if not np.isfinite(a).all() or not np.isfinite(y).all():
                raise ValueError('Nonfinite observation')
        self.gram = np.array([a.conj().T@a for a in self.designs])
        self.rhs = np.array([a.conj().T@y for a, y in zip(self.designs, self.observations)])
        self.root = np.sqrt(self.gram.diagonal(axis1=1, axis2=2).real)
        if (self.root <= 0).any():
            raise ValueError('Zero-energy template')
        self.energy = sum(float(np.vdot(y, y).real) for y in self.observations)

    def columns(self, variants):
        variants = np.asarray(variants, int)
        if variants.shape != (self.points,) or (variants < 0).any() or (variants >= self.variants).any():
            raise ValueError('One variant index is required per reflector')
        if not self.allowed[np.arange(self.points), variants].all():
            raise ValueError('Ineligible physical template variant')
        return np.arange(self.points)*self.variants+variants

    def model(self, variants):
        columns = self.columns(variants)
        return PhaseCoupling([a[:, columns] for a in self.designs], self.observations, reference=self.reference)

    def coordinate(self, variants, parameters, *, sweeps=4):
        variants, parameters = np.array(variants, copy=True), np.array(parameters, copy=True)
        model = self.model(variants)
        fitted = model.evaluate(parameters)
        coefficients = fitted['coefficients'].copy()
        trace, changed = [fitted['value']], []
        for _ in range(sweeps):
            columns = self.columns(variants)
            _, band = model.phases(parameters)
            fixed = np.exp(1j*band)
            residual_correlation = self.rhs-np.array([g[:, columns]@a for g, a in zip(self.gram, coefficients)])
            switches = 0
            for j in range(self.points):
                old = columns[j]
                candidates = j*self.variants+np.flatnonzero(self.allowed[j])
                z = residual_correlation[:, candidates]+self.gram[:, candidates, old]*coefficients[:, j, None]
                z *= fixed.conj()[:, None]/self.root[:, candidates]
                options = [maximum(z[:, h]) for h in range(len(candidates))]
                best = int(np.argmax([np.vdot(strength, strength).real for _, strength in options]))
                phase, strength = options[best]
                new = candidates[best]
                value = fixed*np.exp(1j*phase)*strength/self.root[:, new]
                residual_correlation += self.gram[:, :, old]*coefficients[:, j, None]-self.gram[:, :, new]*value[:, None]
                switches += int(new != old)
                variants[j], columns[j], parameters[j] = new%self.variants, new, phase
                coefficients[:, j] = value
            # Each band rotation has a closed-form minimizer; restore the
            # reference-band gauge without changing the predicted samples.
            change = np.array([np.angle(np.vdot(a[:, columns]@c, y))
                for a, c, y in zip(self.designs, coefficients, self.observations)])
            parameters[:self.points] += change[self.reference]
            parameters[self.points:] = (band+change-change[self.reference])[self.other]
            model = self.model(variants)
            fitted = model.evaluate(parameters)
            if fitted['value'] > trace[-1]+1e-10:
                raise ValueError('Conditional variant update increased the exact likelihood')
            trace.append(fitted['value']); changed.append(switches)
            coefficients = fitted['coefficients'].copy()
            if trace[-2]-trace[-1] < 1e-12:
                break
        return variants, parameters, {'objective': trace, 'variant_changes': changed}

    def fit(self, starts, *, maximum_cycles=20, callback=None):
        trials = []
        for index, start in enumerate(starts):
            variants, anchor, *initial = start
            variants = np.asarray(variants, int).copy()
            parameters = np.asarray(initial[0]).copy() if initial else self.model(variants).initial(anchor)
            trace, converged, gradient = [], False, np.inf
            for cycle in range(maximum_cycles):
                variants, parameters, detail = self.coordinate(variants, parameters)
                model = self.model(variants)
                result = minimize(model.objective, parameters, jac=True, method='L-BFGS-B',
                    options={'gtol': 1e-10, 'ftol': 1e-15, 'maxiter': 2000, 'maxcor': 40, 'maxls': 40})
                parameters = result.x
                fitted = model.evaluate(parameters)
                gradient = float(abs(fitted['gradient']).max())
                # Test all eligible single-reflector alternatives after the
                # continuous solve, using its actual residual.
                probe_variants, probe_parameters, probe = self.coordinate(variants, parameters, sweeps=1)
                improvement = fitted['value']-probe['objective'][-1]
                trace.append({'coordinate': detail, 'continuous_objective': fitted['value'],
                    'continuous_gradient': gradient, 'continuous_iterations': result.nit,
                    'continuous_termination': str(result.message), 'next_coordinate_improvement': improvement})
                if improvement < 1e-10 and gradient < 1e-7:
                    converged = True
                    break
                variants, parameters = probe_variants, probe_parameters
            # Record the same state for parameters, variants and coefficients,
            # including the iteration-limit case.
            model = self.model(variants)
            fitted = model.evaluate(parameters)
            _, band = model.phases(parameters)
            trial = {'start': index, 'anchor': anchor, 'converged': converged,
                'objective': fitted['value'], 'gradient': float(abs(fitted['gradient']).max()),
                'variants': variants.tolist(), 'parameters': parameters.tolist(),
                'band_phase_radians': band.tolist(), 'cycles': trace}
            trials.append(trial)
            if callback is not None:
                callback(trial)
        best = min(trials, key=lambda t: t['objective'])
        model = self.model(best['variants'])
        fitted = model.evaluate(np.asarray(best['parameters']))
        return fitted['coefficients'], {'best': best, 'trials': trials,
            'reference_channel': self.reference, 'scope': __doc__}
