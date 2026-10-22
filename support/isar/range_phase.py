"""Profile shared scattering phases while refining channel envelope origins.

Motion, physical points and antenna geometry remain fixed. The range-origin
parameter changes the compressed envelope, while a separate constant phase
absorbs the carrier convention. Together they represent an affine phase
correction in range frequency. No correction is made to the measured samples.
"""
import numpy as np
from scipy.linalg import qr
from scipy.optimize import minimize

from phase_coupling import PhaseCoupling
from profile_information import profile
from validate_motion import validate, summarize


class RangePhase:
    def __init__(self, bands, origins, phase_seed, *, reference=3, scale_m=10., joint=False):
        self.bands, self.reference, self.scale = bands, reference, scale_m
        self.other = [c for c in range(len(bands)) if c != reference]
        self.points = len(bands[0].points)
        self.initial = np.asarray(origins)/scale_m
        self.seeds = np.atleast_2d(np.asarray(phase_seed)).copy()
        if self.seeds.shape[1] != self.points+len(self.other) or not np.isfinite(self.seeds).all():
            raise ValueError('Need shared point phases and nonreference band phases')
        self.joint = joint
        if joint:
            self.initial = np.r_[self.initial, self.seeds[0]]
        self.energy = sum(b.energy for b in bands)
        self.prepared = [None]*len(bands)
        self.last, self.scaling_information, self.evaluations = None, None, 0

    def prepare(self, origins):
        for c, (band, origin) in enumerate(zip(self.bands, origins)):
            if self.prepared[c] is not None and self.prepared[c]['origin_m'] == origin:
                continue
            target, derivative = band.design(origin, derivatives=True)
            design = band.nuisance.remove(target[band.selected])
            q, r = qr(design, mode='economic', check_finite=False)
            singular = np.linalg.svd(r, compute_uv=False)
            if singular[-1] <= singular[0]*1e-10:
                raise ValueError('Range shift made the scene dictionary rank deficient')
            rhs = q.conj().T@band.clean
            self.prepared[c] = {'origin_m': origin, 'target': target, 'design': design,
                'derivative': derivative[band.selected], 'r': r, 'rhs': rhs,
                'outside': float(np.linalg.norm(band.clean-q@rhs)**2),
                'condition': float(singular[0]/singular[-1])}
        return self.prepared

    def coupling(self, entries):
        return PhaseCoupling([np.vstack([e['r'], np.zeros(self.points)]) for e in entries],
            [np.r_[e['rhs'], np.sqrt(e['outside'])] for e in entries], reference=self.reference)

    def phase_solution(self, coupling):
        candidates = []
        for seed in self.seeds:
            start, _ = coupling.coordinate_refine(seed, sweeps=8)
            seeded = minimize(coupling.objective, start, jac=True, method='L-BFGS-B',
                options={'gtol': 1e-10, 'ftol': 1e-15, 'maxiter': 2000, 'maxcor': 40, 'maxls': 40})
            candidates.append(seeded.x)
        _, canonical = coupling.fit(coordinate_sweeps=8)
        candidates.append(np.asarray(canonical['parameters']))
        return min(candidates, key=lambda x: coupling.evaluate(x)['value']).copy()

    def initialize_phases(self):
        if not self.joint:
            raise ValueError('Explicit phase initialization requires the joint parameterization')
        entries = self.prepare(self.initial[:len(self.bands)]*self.scale)
        self.initial[len(self.bands):] = self.phase_solution(self.coupling(entries))
        self.last, self.scaling_information = None, None

    def evaluate(self, parameters):
        parameters = np.asarray(parameters)
        if self.last is not None and np.array_equal(self.last['parameters'], parameters):
            return self.last
        nrange = len(self.bands)
        entries = self.prepare(parameters[:nrange]*self.scale)
        coupling = self.coupling(entries)
        phase = parameters[nrange:] if self.joint else self.phase_solution(coupling)
        inner = coupling.evaluate(phase)
        if not self.joint and abs(inner['gradient']).max() > 1e-7:
            raise ValueError('Inner shared-phase fit is not stationary')
        phase_factors, band_phases = coupling.phases(phase)
        coefficients = inner['coefficients']
        gradient, value = np.zeros(len(self.bands)), 0.
        information = np.zeros((nrange+len(phase), nrange+len(phase)))
        for c, (band, entry, amplitude) in enumerate(zip(self.bands, entries, coefficients)):
            residual = band.clean-entry['design']@amplitude
            derivative = entry['derivative']@amplitude*self.scale
            gradient[c] = -2*np.real(np.vdot(residual, derivative))/self.energy
            value += np.vdot(residual, residual).real
            if self.scaling_information is None:
                fixed = np.zeros((len(residual), len(information)), complex)
                fixed[:, c] = derivative
                fixed[:, nrange:nrange+self.points] = 1j*entry['design']*amplitude
                if c in self.other:
                    fixed[:, nrange+self.points+self.other.index(c)] = 1j*entry['design']@amplitude
                fixed = band.nuisance.remove(fixed)
                real_fixed = np.r_[fixed.real, fixed.imag]
                active = abs(amplitude) > 0
                real_design = np.r_[(entry['design']*phase_factors[c]).real,
                                    (entry['design']*phase_factors[c]).imag][:, active]
                q, _ = qr(real_design, mode='economic', check_finite=False)
                tangent = real_fixed-q@(q.T@real_fixed)
                information += 2*(tangent.T@tangent)
        value /= self.energy
        if abs(value-inner['value']) > 1e-10:
            raise ValueError('Compressed and original physical likelihoods disagree')
        if self.scaling_information is None:
            reduced = information
            if not self.joint:
                reduced, _ = profile(information, np.arange(nrange), np.arange(nrange, len(information)), cutoff=1e-10)
            self.scaling_information = reduced/self.energy
        if self.joint:
            gradient = np.r_[gradient, inner['gradient']]
        self.evaluations += 1
        self.last = {'parameters': parameters.copy(), 'value': float(value), 'gradient': gradient,
            'gauss_newton': self.scaling_information, 'scene_conditions': [e['condition'] for e in entries],
            'phase_parameters': phase.copy(), 'phase_gradient': float(abs(inner['gradient']).max()),
            'band_phase_radians': band_phases.copy(), 'coefficients': coefficients.copy(),
            'independent_coefficients': coupling.independent.copy(), 'targets': [e['target'] for e in entries]}
        return self.last

    def objective(self, parameters):
        value = self.evaluate(parameters)
        return value['value'], value['gradient']

    def validation(self, parameters, interval):
        value = self.evaluate(parameters)
        result = {}
        for label, key in [('coupled', 'coefficients'), ('independent', 'independent_coefficients')]:
            checks = [validate(b.field, d, b.nuisance, b.selected_time, interval,
                coefficients=a, stationary_mode='project') for b, d, a in zip(self.bands, value['targets'], value[key])]
            result[label] = {'channels': checks, 'summary': summarize(checks)}
        return result
