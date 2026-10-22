"""Four channel phase offsets, with a shared distributed scene eliminated exactly."""
import numpy as np
from scipy.optimize import minimize


class PhaseRegistration:
    def __init__(self, designs, data, strength, *, reference=3, orthogonal_energy=0., retain_basis=False):
        self.designs, self.data = designs, data
        self.other = np.array([c for c in range(len(designs)) if c != reference])
        self.energy = sum(np.vdot(y, y).real for y in data)+orthogonal_energy
        # Input designs already contain the initial complex channel gains.
        # Unit-modulus phase updates rotate rows of U but leave S,V unchanged.
        # Work in these orthogonal coordinates: explicit normal equations lose
        # phase accuracy when the distributed-scene dictionary is nearly singular.
        u, singular, vh = np.linalg.svd(np.concatenate(designs), full_matrices=False)
        edges = np.cumsum([len(a) for a in designs])[:-1]
        self.score = [q.conj().T@y for q, y in zip(np.split(u, edges), data)]
        self.singular, self.right_vectors = singular, vh.conj().T
        self.left_vectors = u if retain_basis else None
        self.regularize(strength)

    def regularize(self, strength):
        if not np.isfinite(strength) or strength <= 0:
            raise ValueError('The profiled scene requires a positive finite regularization strength')
        self.shrinkage = self.singular**2/(self.singular**2+strength)
        self.coefficient_filter = self.singular/(self.singular**2+strength)
        self.strength = strength

    def rotated_score(self, parameters):
        phase = np.ones(len(self.designs), complex)
        phase[self.other] = np.exp(1j*parameters)
        score = sum(g.conjugate()*s for g, s in zip(phase, self.score))
        return phase, score

    def project(self, parameters):
        phase, score = self.rotated_score(parameters)
        return phase, self.right_vectors@(self.coefficient_filter*score), score

    def phase_information(self, coefficient, tolerances=(1e-7, 1e-9, 1e-11)):
        """Conditional data information with the complex scene treated as free.

        A fixed ridge prior can make phase curvature look convincing when the
        data alone provide little phase information. Report both. Truncation
        thresholds here diagnose the numerical-rank dependence, not SNR cuts.
        Unit channel phases cancel after rotating each band's observations back.
        """
        if self.left_vectors is None:
            raise ValueError('Construct with retain_basis=True for information diagnostics')
        rows = np.cumsum([0]+[len(a) for a in self.designs])
        derivative = np.zeros((rows[-1], len(self.other)), complex)
        for column, channel in enumerate(self.other):
            derivative[rows[channel]:rows[channel+1], column] = 1j*(self.designs[channel]@coefficient)
        fixed = 2*np.real(derivative.conj().T@derivative)
        value, vector = np.linalg.eigh(fixed)
        whitening = vector/np.sqrt(np.maximum(value, np.finfo(float).tiny))
        result = []
        for tolerance in tolerances:
            rank = int(np.sum(self.singular > self.singular[0]*tolerance))
            u = self.left_vectors[:, :rank]
            score = (derivative.conj().T@u).conj().T
            residual = derivative-u@score
            information = 2*np.real(residual.conj().T@residual)
            fraction, directions = np.linalg.eigh(whitening.T@information@whitening)
            weak = whitening@directions[:, 0]
            weak /= max(abs(weak))
            pattern = np.zeros(len(self.designs))
            pattern[self.other] = weak
            result.append({'relative_rank_tolerance': tolerance, 'scene_rank': rank,
                'information_matrix': information.tolist(),
                'fraction_of_fixed_scene_information_eigenvalues': fraction.tolist(),
                'weakest_phase_direction_by_channel': pattern.tolist()})
        return {'fixed_scene_information': fixed.tolist(), 'profiled_scene': result,
            'scope': 'Conditional complex-Gaussian data information after eliminating arbitrary complex scene coefficients. Numerical rank sensitivity is shown explicitly; the ridge penalty is excluded.'}

    def objective(self, parameters):
        phase, score = self.rotated_score(parameters)
        weighted = self.shrinkage*score
        value = (self.energy-np.vdot(score, weighted).real)/self.energy
        gradient = np.array([-2*np.imag(phase[c].conjugate()*np.vdot(weighted, self.score[c])) for c in self.other])/self.energy
        return float(value), gradient

    def fit(self):
        seeds = [np.zeros(len(self.other)), *np.random.default_rng(731).uniform(-np.pi, np.pi, (8, len(self.other)))]
        trials = [minimize(self.objective, s, jac=True, method='BFGS',
            options={'gtol': 1e-10, 'maxiter': 250}) for s in seeds]
        best = min(trials, key=lambda r:r.fun)
        phase, alpha, _ = self.project(best.x)
        step = 1e-5
        hessian = np.column_stack([(self.objective(best.x+e*step)[1]-self.objective(best.x-e*step)[1])/(2*step)
                                   for e in np.eye(len(best.x))])
        return phase, alpha, {'converged': bool(abs(best.jac).max() < 1e-7),
            'maximum_gradient': float(abs(best.jac).max()), 'objective': float(best.fun),
            'relative_phase_degrees': np.degrees(np.angle(phase)).tolist(),
            'curvature_eigenvalues': np.linalg.eigvalsh((hessian+hessian.T)/2).tolist(),
            'trials': [{'objective': float(r.fun), 'gradient': float(abs(r.jac).max()), 'termination': str(r.message)} for r in trials],
            'regularization_strength': self.strength,
            'scope': 'Channel gain magnitudes and geometry fixed. One shared complex reflectivity vector is profiled with fixed training-GCV Tikhonov strength. Local phase fit is conditional on the declared spatial support and regularizer.'}
