"""A Gaussian scattering response shared across measured frequency bands.

Each supplied design contains the physical space/delay response. A single
proper-complex latent vector has unit covariance; band gains are fitted from
the training likelihood. The independent control gives each band its own
latent vector with the same marginal prior. QR removes only a parameter-free
orthogonal complement; its energy remains in every reported likelihood.

This is a conditional scattering hypothesis, not a calibration assumption.
The posterior mean is MMSE only under its declared Gaussian prior and noise.
"""
import numpy as np
from scipy.linalg import cho_solve
from scipy.optimize import minimize

from phase_quadratic import solve as solve_phases
from scene_statistics import compress
from stationarity import polish


class SpectralScene:
    def __init__(self, records, *, shared, reference=3, scale=None):
        self.shared, self.reference = shared, reference
        self.records = records
        self.channels = len(records)
        self.columns = records[0]['design'].shape[1]
        if not 0 <= reference < self.channels or any(
                r['design'].shape[1] != self.columns or r['data'].shape != (len(r['design']),)
                for r in records):
            raise ValueError('Bands must use the same physical scene basis and a valid reference')
        # One scalar changes latent coefficient units, not the observations,
        # relative band weights or physical phase. All gain outputs undo it.
        self.scale = (np.sqrt(sum(np.linalg.norm(r['design'])**2 for r in records)
                             /(self.channels*self.columns)) if scale is None else float(scale))
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('The design scale must be finite and positive')
        self.designs = [r['design']/self.scale for r in records]
        self.data = [r['data'] for r in records]
        # Rotate the isotropic latent prior into the right singular basis once.
        # Keeping every singular direction is important: this is a numerical
        # reparameterization, not a data-driven rank cutoff or changed prior.
        groups = [list(range(self.channels))] if shared else [[c] for c in range(self.channels)]
        self.groups = []
        for channels in groups:
            u, s, vh = np.linalg.svd(np.concatenate([self.designs[c] for c in channels]), full_matrices=False)
            split = np.cumsum([len(self.designs[c]) for c in channels])[:-1]
            directions = np.split(u, split)
            self.groups.append({'channels': channels, 'singular': s, 'right': vh.conj().T,
                'directions': directions,
                'grams': np.array([a.conj().T@a for a in directions]),
                'cross': np.array([a.conj().T@self.data[c] for a, c in zip(directions, channels)])})
        self.energy = sum(float(np.vdot(y, y).real)+r['outside'] for y, r in zip(self.data, records))
        self.other = [c for c in range(self.channels) if c != reference] if shared else []
        self.last = None

    def gains(self, parameters):
        phase = np.zeros(self.channels)
        phase[self.other] = parameters[self.channels:]
        return np.exp(np.asarray(parameters[:self.channels])+1j*phase)

    def precision(self, group, gains):
        """Equilibrated latent precision without squaring its dynamic range."""
        diagonal = np.hypot(1., max(abs(gains))*group['singular'])
        response = group['singular']/diagonal
        gram = np.einsum('c,cij->ij', abs(gains)**2, group['grams'])
        matrix = (gram*response[:, None])*response[None, :]
        matrix = (matrix+matrix.conj().T)/2+np.diag(1/diagonal**2)
        return np.linalg.cholesky(matrix), diagonal, response

    def evaluate(self, parameters):
        parameters = np.asarray(parameters)
        if self.last is not None and np.array_equal(parameters, self.last['parameters']):
            return self.last
        gains = self.gains(parameters)
        gradient = np.zeros(len(parameters))
        coefficients = [None]*self.channels
        total, quadratic, determinant = 0., 0., 0.
        for group in self.groups:
            channels = group['channels']
            root, diagonal, response = self.precision(group, gains[channels])
            inverse = cho_solve((root, True), np.eye(self.columns), check_finite=False)
            cross = response*(gains[channels].conj()@group['cross'])
            latent = cho_solve((root, True), cross, check_finite=False)/diagonal
            original_latent = group['right']@latent
            logdet = float(2*(np.log(root.diagonal().real).sum()+np.log(diagonal).sum()))
            q = float(np.vdot(latent, latent).real)
            for local, c in enumerate(channels):
                prediction = gains[c]*(group['directions'][local]@(group['singular']*latent))
                residual = self.data[c]-prediction
                q += float(np.vdot(residual, residual).real)+self.records[c]['outside']
                cross_residual = np.vdot(residual, prediction)
                volume = abs(gains[c])**2*np.einsum('i,ij,ji,j->', response,
                    inverse, group['grams'][local], response).real
                gradient[c] = 2*(volume-cross_residual.real)
                if c in self.other:
                    gradient[self.channels+self.other.index(c)] = 2*cross_residual.imag
                coefficients[c] = gains[c]*original_latent/self.scale
            quadratic += q
            determinant += logdet
            total += q+logdet
        self.last = {'parameters': parameters.copy(), 'value': total-self.energy,
            'negative_log_likelihood_without_pi_constant': total, 'quadratic': quadratic,
            'logdet': determinant, 'gradient': gradient, 'coefficients': coefficients,
            'physical_band_gains': gains/self.scale}
        return self.last

    def objective(self, parameters):
        result = self.evaluate(parameters)
        return result['value'], result['gradient']

    def phase_matrix(self, log_amplitudes):
        """Exact phase-dependent marginal-likelihood quadratic at fixed powers."""
        if not self.shared:
            raise ValueError('Independent responses have no relative-phase likelihood')
        gains = np.exp(log_amplitudes)
        group = self.groups[0]
        root, _, response = self.precision(group, gains)
        cross = response[:, None]*group['cross'].T*gains
        matrix = cross.conj().T@cho_solve((root, True), cross, check_finite=False)
        return (matrix+matrix.conj().T)/2

    def fit(self, initial_logs=None, initial_phases=None):
        if initial_logs is None:
            initial_logs = np.array([.5*np.log(max(np.vdot(y, y).real-len(y), 1.)/
                max(np.linalg.norm(a)**2, 1e-30)) for a, y in zip(self.designs, self.data)])
        initial_logs = np.clip(initial_logs, -12., 12.)
        starts = [np.zeros(len(self.other))]
        if self.shared:
            matrix = self.phase_matrix(initial_logs)
            _, vectors = np.linalg.eigh(matrix)
            z = vectors[:, -1]/vectors[self.reference, -1]
            starts.append(-np.angle(z)[self.other])
            if initial_phases is not None:
                phase = np.asarray(initial_phases)-initial_phases[self.reference]
                starts.append(phase[self.other])
        bounds = [(-12., 12.)]*self.channels+[(None, None)]*len(self.other)
        trials = []
        def optimize(initial):
            fit = minimize(self.objective, initial, jac=True, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 300, 'gtol': 1e-7, 'ftol': 1e-14, 'maxls': 40, 'maxcor': 20})
            parameters, refinement = fit.x, None
            if np.max(abs(self.objective(parameters)[1])) > 1e-5:
                lower = np.r_[np.full(self.channels, -12.), np.full(len(self.other), -np.inf)]
                upper = -lower
                parameters, refinement = polish(self.objective, parameters, np.eye(len(parameters)),
                    np.zeros((0, len(parameters))), np.array([]), np.array([]), lower, upper)
            value = self.evaluate(parameters)
            gradient = value['gradient'].copy()
            p = parameters[:self.channels]
            gradient[:self.channels][(p <= -12+1e-8)&(gradient[:self.channels] > 0)] = 0.
            gradient[:self.channels][(p >= 12-1e-8)&(gradient[:self.channels] < 0)] = 0.
            return {'parameters': parameters.tolist(), 'value': value['value'],
                'maximum_kkt_residual': float(abs(gradient).max()), 'iterations': fit.nit,
                'stationary': bool(abs(gradient).max() <= 1e-5), 'stationarity_polish': refinement,
                'termination': str(fit.message), 'amplitude_bound_hits': int(np.sum(abs(p) >= 12-1e-7))}
        for phase in starts:
            trials.append(optimize(np.r_[initial_logs, phase]))
        best = min(trials, key=lambda r: r['value'])
        certificate = None
        if self.shared:
            # This certifies only the phase subproblem at the final powers.
            # Refit powers if its global solution improves a local start.
            for _ in range(3):
                parameters = np.asarray(best['parameters'])
                z, certificate = solve_phases(self.phase_matrix(parameters[:self.channels]),
                                              reference=self.reference)
                candidate = parameters.copy()
                candidate[self.channels:] = -np.angle(z)[self.other]
                if self.objective(candidate)[0] >= best['value']-1e-7:
                    break
                trials.append(optimize(candidate))
                best = min(trials, key=lambda r: r['value'])
            parameters = np.asarray(best['parameters'])
            matrix = self.phase_matrix(parameters[:self.channels])
            _, certificate = solve_phases(matrix, reference=self.reference)
            z = np.ones(self.channels, complex)
            z[self.other] = np.exp(-1j*parameters[self.channels:])
            captured = float(np.vdot(z, matrix@z).real)
            gap = max(0., certificate['upper_bound']-captured)
            scale = np.linalg.norm(matrix-np.diag(matrix.diagonal()), 2)
            certificate = {**certificate, 'retained_solution_captured_energy': captured,
                'retained_solution_gap': gap,
                'retained_solution_globally_certified': bool(gap <= max(2e-8*scale, 1e-10)),
                'scope': 'Global bound only for band phases at the reported fixed powers and stipulated Gaussian causal scene.'}
        result = self.evaluate(np.asarray(best['parameters']))
        return result['coefficients'], {'shared_response': self.shared, 'reference_channel': self.reference,
            'scale': self.scale, 'trials': trials, 'selected': best, 'phase_certificate': certificate,
            'negative_log_likelihood_without_pi_constant': result['negative_log_likelihood_without_pi_constant'],
            'quadratic': result['quadratic'], 'logdet': result['logdet'],
            'band_gains_real_imag': np.c_[result['physical_band_gains'].real,
                                        result['physical_band_gains'].imag].tolist()}


def prediction(record, coefficient):
    fitted = record['design']@coefficient
    error = record['data']-fitted
    power = float(np.vdot(fitted, fitted).real)
    return {'prediction_error': float(np.vdot(error, error).real)+record['outside'],
        'prediction_energy': power,
        'cross_real_imag': [float(np.vdot(fitted, record['data']).real), float(np.vdot(fitted, record['data']).imag)],
        'data_energy': float(np.vdot(record['data'], record['data']).real)+record['outside'],
        'matched_energy': float(abs(np.vdot(fitted, record['data']))**2/power) if power else 0.}
