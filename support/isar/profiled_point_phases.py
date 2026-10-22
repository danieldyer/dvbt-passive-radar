"""Profile the small nonlinear phase problem inside physical point refinement.

Every coordinate trial uses the same deterministic phase-start family and one
fixed pivot-coordinate seed. The inner choice uses training likelihood alone.
This changes the optimizer's parameterization, not the scattering hypothesis.
The outer envelope-theorem gradient is used only after phase stationarity.
"""
import numpy as np
from scipy.optimize import minimize

from phase_coupling import PhaseCoupling
from profile_information import profile


class ProfiledPointPhases:
    def __init__(self, model, seed, *, coordinate_sweeps=8):
        self.model, self.seed = model, np.asarray(seed).copy()
        self.initial = self.seed[:2*model.count].copy()
        self.coordinate_sweeps = coordinate_sweeps
        self.last, self.scaling_information, self.evaluations = None, None, 0

    def evaluate(self, coordinates):
        coordinates = np.asarray(coordinates)
        if self.last is not None and np.array_equal(coordinates, self.last['parameters']):
            return self.last
        model, n = self.model, self.model.count
        prepared = model.prepare(coordinates)
        designs = [np.vstack([b['r'], np.zeros(n)]) for b in prepared['bands']]
        data = [np.r_[b['rhs'], np.sqrt(b['outside'])] for b in prepared['bands']]
        coupling = PhaseCoupling(designs, data, reference=model.reference)
        start = np.r_[self.seed[2*n:3*n]+prepared['pivot'], self.seed[3*n:]]
        start, seed_coordinates = coupling.coordinate_refine(start, sweeps=self.coordinate_sweeps)
        seeded = minimize(coupling.objective, start, jac=True, method='L-BFGS-B',
            options={'gtol': 1e-10, 'ftol': 1e-15, 'maxiter': 2000, 'maxcor': 40, 'maxls': 40})
        _, canonical = coupling.fit(coordinate_sweeps=self.coordinate_sweeps)
        options = [(seeded.x, 'fixed_seed'), (np.asarray(canonical['parameters']), 'canonical_starts')]
        phase, choice = min(options, key=lambda item: coupling.evaluate(item[0])['value'])
        inner = coupling.evaluate(phase)
        inner_gradient = float(abs(inner['gradient']).max())
        if inner_gradient > 1e-7:
            raise ValueError(f'Inner phase profile not stationary: {inner_gradient:g}')
        full = np.r_[coordinates, phase[:n]-prepared['pivot'], phase[n:]]
        evaluation = model.evaluate(full, jacobian=self.scaling_information is None)
        if abs(evaluation['value']-inner['value']) > 1e-10:
            raise ValueError('Compressed phase solve changed the physical likelihood')
        if self.scaling_information is None:
            expected, _ = profile(evaluation['expected_information'], np.arange(2*n),
                np.arange(2*n, len(full)), cutoff=1e-10)
            self.scaling_information = expected/model.energy
        self.evaluations += 1
        self.last = {'parameters': coordinates.copy(), 'value': evaluation['value'],
            'gradient': evaluation['gradient'][:2*n].copy(), 'gauss_newton': self.scaling_information,
            'scene_conditions': [b['condition'] for b in evaluation['bands']],
            'full_parameters': full, 'inner_phase_gradient': inner_gradient, 'phase_start_choice': choice,
            'seed_coordinates': seed_coordinates,
            'canonical_phase_trials': [{k: t[k] for k in ('anchor', 'objective', 'maximum_gradient', 'initialization')}
                for t in canonical['trials']]}
        return self.last

    def objective(self, coordinates):
        result = self.evaluate(coordinates)
        return result['value'], result['gradient']
