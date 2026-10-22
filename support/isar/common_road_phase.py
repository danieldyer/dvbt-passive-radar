"""Test a shared interband phase reference on independent short vehicle arcs.

The point hypothesis has one scattering phase per target, one phase per band
shared by all targets, and nonnegative band amplitudes. Seven fixes the phase
gauge. This is a hypothesis to validate, not an assumed calibration fact.
The archived empirical deramp constant is restored exactly for every band.
"""
import numpy as np

from positive_phase import maximum
from road_arc_raw import RawRoadGeometry


def positive_gain(y, u, derivative):
    """Profile one nonnegative real amplitude, with its exact real Jacobian."""
    norm = np.vdot(u, u).real
    if norm <= 1e-20:
        raise ValueError('Target has no retained support')
    amplitude = max(0., np.vdot(u, y).real/norm)
    if amplitude == 0:
        zero = np.zeros_like(derivative)
        return -y, zero, 0., zero
    da = (np.real(derivative.conj().T@y)-amplitude*2*np.real(u.conj()@derivative))/norm
    residual = amplitude*u-y
    jacobian = amplitude*derivative+u[:, None]*da
    mean = amplitude*derivative
    mean -= u[:, None]*(np.real(u.conj()@mean)/norm)[None, :]
    return residual, jacobian, amplitude, mean


def phase_seeds(correlations, count=12):
    """Training-only alternating exact circular solves; keep distinct optima."""
    correlations = np.asarray(correlations)
    rng = np.random.default_rng(20261017)
    starts = [np.zeros(correlations.shape[1])]
    starts.extend(np.angle(row) for row in correlations)
    starts.extend(rng.uniform(-np.pi, np.pi, correlations.shape[1]) for _ in range(count))
    solutions = []
    for beta in starts:
        beta = beta-beta[0]
        for _ in range(100):
            phi = np.array([maximum(row*np.exp(-1j*beta))[0] for row in correlations])
            new = np.array([maximum(correlations[:, c]*np.exp(-1j*phi))[0] for c in range(len(beta))])
            phi += new[0]; new -= new[0]
            change = np.max(abs(np.angle(np.exp(1j*(new-beta)))))
            beta = new
            if change < 1e-12:
                break
        amplitude = np.maximum(0., np.real(correlations*np.exp(-1j*(phi[:, None]+beta))))
        score = float(np.sum(amplitude**2))
        parameters = np.r_[beta[1:], phi]
        if not any(np.max(abs(np.angle(np.exp(1j*(parameters-p))))) < 1e-4 for _, p in solutions):
            solutions.append((score, parameters))
    return sorted(solutions, key=lambda s: -s[0])


class CommonRoadGeometry(RawRoadGeometry):
    def __init__(self, independent, parameters, *, phase_initial=None, fixed_band_phases=None):
        super().__init__(independent.geometry, independent.likelihoods)
        self.base_count = len(self.geometry.initial)
        self.band_count = len(self.likelihoods[0])
        self.fixed_band_phases = None if fixed_band_phases is None else np.asarray(fixed_band_phases)
        self.free_band_count = self.band_count-1 if self.fixed_band_phases is None else 0
        correlations = []
        for index, bands in enumerate(self.likelihoods):
            row = []
            for ch, likelihood in enumerate(bands):
                value, _, _ = independent.band_template(parameters, index, ch)
                h = likelihood.nuisance.remove(value[likelihood.selected])
                phase = likelihood.field.k*likelihood.field.metadata['phase_reference_offset_m']
                row.append(np.vdot(h, likelihood.clean)*np.exp(1j*phase)/np.linalg.norm(h))
            correlations.append(row)
        if self.fixed_band_phases is None:
            self.initial_phase_solutions = phase_seeds(np.array(correlations))
        else:
            if self.fixed_band_phases.shape != (self.band_count,):
                raise ValueError('A transferred phase is required for each band')
            corrected = np.array(correlations)*np.exp(-1j*self.fixed_band_phases)
            phases = np.array([maximum(row)[0] for row in corrected])
            score = np.sum(np.maximum(0., np.real(corrected*np.exp(-1j*phases[:, None])))**2)
            self.initial_phase_solutions = [(float(score), phases)]
        phase_initial = self.initial_phase_solutions[0][1] if phase_initial is None else phase_initial
        self.initial = np.r_[parameters, phase_initial]
        self.lower = np.r_[self.lower, np.full(len(phase_initial), -np.inf)]
        self.upper = np.r_[self.upper, np.full(len(phase_initial), np.inf)]

    def band_template(self, parameters, index, channel):
        value, derivative, columns = super().band_template(parameters, index, channel)
        target_index = self.base_count+self.free_band_count+index
        phase = parameters[target_index]
        if self.fixed_band_phases is not None:
            phase += self.fixed_band_phases[channel]
        elif channel:
            phase += parameters[self.base_count+channel-1]
        field = self.likelihoods[index][channel].field
        phase -= field.k*field.metadata['phase_reference_offset_m']
        multiplier = np.exp(1j*phase)
        value, derivative = value*multiplier, derivative*multiplier
        indices = [target_index]+([self.base_count+channel-1] if channel and self.free_band_count else [])
        derivative = np.column_stack([derivative, *[1j*value for _ in indices]])
        return value, derivative, np.r_[columns, indices]

    profile = staticmethod(positive_gain)

    def phases(self, parameters):
        phases = (self.fixed_band_phases if self.fixed_band_phases is not None else
                  np.r_[0., parameters[self.base_count:self.base_count+self.free_band_count]])
        return {'band_phase_rad': phases.tolist(),
            'target_phase_rad': parameters[self.base_count+self.free_band_count:].tolist(),
            'band_phases_fixed_from_other_targets': self.fixed_band_phases is not None,
            'scope': __doc__}
