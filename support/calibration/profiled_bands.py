"""Exact elimination of positive band gains and one common target phase.

For fixed geometry and shared channel calibration, each band contributes one
complex sufficient statistic. On each interval between its phase sign changes,
the captured energy is a sinusoid in twice the target phase. Checking the
stationary points and interval boundaries gives the global gain/phase optimum.
No empirical reweighting or independently adjustable band phase is introduced.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from coherent_fit import OTHER


class MissingReferenceBand(ValueError):
    """The legacy amplitude-ratio format cannot represent a zero SBS band."""


def common_phase(statistics):
    z = np.asarray(statistics)
    boundaries = np.sort(np.remainder(np.r_[np.angle(z)+np.pi/2, np.angle(z)-np.pi/2], 2*np.pi))
    boundaries = np.r_[boundaries, boundaries[0]+2*np.pi]
    candidates = list(boundaries[:-1])
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        active = np.real(z*np.exp(-.5j*(lo+hi))) > 0
        quadratic = np.sum(z[active]**2)
        phase = .5*np.angle(quadratic)
        phase += np.ceil((lo-phase)/np.pi)*np.pi
        if phase <= hi:
            candidates.append(phase)
    candidates = np.asarray(candidates)
    amplitudes = np.maximum(np.real(np.exp(-1j*candidates[:, None])*z), 0)
    best = int(np.argmax(np.sum(amplitudes**2, axis=1)))
    return float(candidates[best]), amplitudes[best]


def profile_bands(data, templates, derivatives=None):
    norms = np.array([np.vdot(u, u).real for u in templates])
    if np.any(norms < 1e-20):
        raise ValueError('Band template outside the selected observation subspace')
    roots = np.sqrt(norms)
    normalized = [u/n for u, n in zip(templates, roots)]
    z = np.array([np.vdot(u, y) for u, y in zip(normalized, data)])
    phase, magnitude = common_phase(z)
    rotation = np.exp(1j*phase)
    amplitudes = rotation*magnitude/roots
    error = np.concatenate([a*u-y for a, u, y in zip(amplitudes, templates, data)])*np.sqrt(2)
    if derivatives is None:
        return error, amplitudes
    dn = np.array([2*np.real(u.conj()@du) for u, du in zip(templates, derivatives)])
    dz = np.array([du.conj().T@y/n for du, y, n in zip(derivatives, data, roots)])-z[:, None]*dn/(2*norms[:, None])
    active = magnitude > 0
    quadratic = np.sum(z[active]**2)
    # Exactly flat phase optima are non-identifiable. Use a deterministic
    # branch; the smooth-case derivative is checked against finite differences.
    dphase = (np.imag(np.sum(z[active, None]*dz[active], axis=0)/quadratic)
              if abs(quadratic) > 1e-12*np.sum(abs(z)**2) else np.zeros(dz.shape[1]))
    dm = (np.real(rotation.conjugate()*dz)+np.imag(rotation.conjugate()*z)[:, None]*dphase)*active[:, None]
    jacobian = []
    for ch, (v, du) in enumerate(zip(normalized, derivatives)):
        dv = du/roots[ch]-v[:, None]*dn[ch]/(2*norms[ch])
        jacobian.append(rotation*(magnitude[ch]*dv+v[:, None]*(dm[ch]+1j*magnitude[ch]*dphase)))
    return error, np.sqrt(2)*np.concatenate(jacobian), amplitudes


class ProfiledBandProblem:
    """Same point-response likelihood, fewer numerical nuisance parameters."""
    def __init__(self, joint):
        self.joint = joint
        self.bias_offset = joint.magnitude_offset
        self.count = self.bias_offset+(4 if joint.range_bias else 0)

    def initial(self, state):
        state = np.asarray(state)
        return np.r_[state[:self.bias_offset], state[self.joint.bias_offset:]]

    def bounds(self):
        low, high = self.joint.bounds()
        return (np.r_[low[:self.bias_offset], low[self.joint.bias_offset:]],
                np.r_[high[:self.bias_offset], high[self.joint.bias_offset:]])

    def fit_columns(self, start, active, iterations):
        from calibration_solver import solve
        def residual(x, jacobian=False):
            state=start.copy();state[active]=x
            value=self.residual(state,jacobian)
            return value[:,active] if jacobian else value
        low,high=self.bounds()
        fit=solve(residual,start[active],(low[active],high[active]),iterations)
        state=start.copy();state[active]=fit.x
        return fit,state

    def channel_model(self, p, index, channel, jacobian=False, field=None):
        joint = self.joint
        problem = joint.problems[index]
        offset = 9+index*joint.n
        state = np.r_[p[offset:offset+joint.n], p[:9]]
        beta = np.zeros(5)
        beta[OTHER] = p[joint.beta_offset:joint.magnitude_offset]
        other = None if channel == 1 else int(np.flatnonzero(OTHER == channel)[0])
        bias = p[self.bias_offset+other] if joint.range_bias and other is not None else 0.
        if field is None:
            base, local = problem.channel_model(state, channel, True, jacobian, bias, joint.range_bias)
        else:
            path, local = problem.model.paths(state,field.time,np.full(len(field.time),channel),problem.sites,True)
            base, local = field.template(path,local if jacobian else None,
                                         bias+problem.range_origins[channel],joint.range_bias)
        u = np.exp(1j*beta[channel])*base
        if not jacobian:
            return u, None
        local *= np.exp(1j*beta[channel])
        derivative = np.zeros((len(u), self.count), complex)
        derivative[:, offset:offset+joint.n] = local[:, :joint.n]
        derivative[:, :9] = local[:, joint.n:joint.n+9]
        if other is not None:
            derivative[:, joint.beta_offset+other] = 1j*u
            if joint.range_bias:
                derivative[:, self.bias_offset+other] = local[:, -1]
        return u, derivative

    def target(self, p, index, jacobian=False):
        templates, derivatives, data = [], [], []
        for ch in range(5):
            u, derivative = self.channel_model(p, index, ch, jacobian)
            templates.append(u)
            data.append(self.joint.problems[index].channel_data(ch))
            if jacobian:
                derivatives.append(derivative)
        return profile_bands(data, templates, derivatives if jacobian else None)

    def residual(self, p, jacobian=False, include_motion=True, include_site_prior=True):
        joint = self.joint
        pieces = []
        for index, problem in enumerate(joint.problems):
            result = self.target(p, index, jacobian)
            value = result[1] if jacobian else result[0]
            pieces.append(np.r_[value.real, value.imag])
            offset = 9+index*joint.n
            local = np.r_[p[offset:offset+joint.n], p[:9]]
            for v in problem.auxiliary(local, True, jacobian, with_motion=include_motion)[:-1]:
                if jacobian:
                    mapped = np.zeros((len(v), self.count))
                    mapped[:, offset:offset+joint.n] = v[:, :joint.n]
                    mapped[:, :9] = v[:, joint.n:]
                    pieces.append(mapped)
                else:
                    pieces.append(v)
        if include_site_prior:
            pieces.append(np.c_[np.diag(joint.prior_weights), np.zeros((9, self.count-9))]
                          if jacobian else p[:9]*joint.prior_weights)
        return np.concatenate(pieces)

    def expanded_state(self, p):
        """Represent the optimum in the existing saved-state convention.

        The old log-gain convention cannot express an exactly absent band.
        Report those explicitly and use the smallest positive floating value
        for serialization, without applying the old numerical gain bounds.
        """
        biases = p[self.bias_offset:self.bias_offset+4] if self.joint.range_bias else []
        state = np.r_[p[:self.bias_offset], np.zeros(4*self.joint.m), biases]
        missing = []
        for index in range(self.joint.m):
            _, amplitudes = self.target(p, index)
            if abs(amplitudes[1]) < 1e-20:
                raise MissingReferenceBand('SBS amplitude is zero; old state convention cannot represent this optimum')
            ratios = abs(amplitudes[OTHER]/amplitudes[1])
            missing.append(OTHER[ratios == 0].tolist())
            first = self.joint.magnitude_offset+4*index
            state[first:first+4] = np.log(np.maximum(ratios, np.finfo(float).tiny))
        return state, missing


def main():
    from calibration_solver import solve
    from coherent_fit import information
    from road_fit import load_trial
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('fit', 'histories', 'patches', 'templates', 'routes', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--range-bias', action='store_true',
        help='Test four shared range-envelope origins, relative to SBS; carrier phase remains a separate calibration gauge')
    args = parser.parse_args()
    saved = json.loads(args.fit.read_text())
    joint = load_trial(saved, args.routes, args.histories, args.patches, args.templates)
    original = np.array(saved['state'])
    if args.range_bias and not joint.range_bias:
        from coherent_fit import CoherentProblem
        joint = CoherentProblem(joint.problems, range_bias=True,
                                site_prior_sigma=np.array(saved['site_prior_sigma_m']))
        original = np.r_[original, np.zeros(4)]
    problem = ProfiledBandProblem(joint)
    start = problem.initial(original)
    before = problem.residual(start)
    clock = time.monotonic()
    def progress(intermediate_result):
        if intermediate_result.nfev%10 == 0:
            print('eval', intermediate_result.nfev, 'cost', round(intermediate_result.cost, 3),
                  'elapsed', round(time.monotonic()-clock, 1), flush=True)
    fit = solve(problem.residual, start, problem.bounds(), args.iterations, progress)
    result = {**saved, 'cost': float(fit.cost), 'converged': bool(fit.success),
        'evaluations': fit.nfev, 'optimality': float(fit.optimality), 'termination': str(fit.message),
        'source_fit': str(args.fit), 'reduced_state': fit.x.tolist(),
        'site_corrections_enu_m': (fit.x[:9].reshape(3, 3)*[30, 30, 75]).tolist(),
        'profiled_band_gains': True, 'range_bias': joint.range_bias,
        'range_bias_m': fit.x[problem.bias_offset:].tolist(),
        'initial_profiled_cost': float(before@before/2),
        'scope': 'Exact nonnegative band magnitudes and shared target phase profiled at each geometric trial. Zero fitted bands are reported, not counted as coherent validation. This is an offline trial, not an adopted calibration.'}
    try:
        state, absent = problem.expanded_state(fit.x)
        result.update(state=state.tolist(), zero_amplitude_channels_by_target=absent,
                      information=information(joint, state))
        check = joint.residual(state)
        result['expanded_state_cost_difference'] = float(check@check/2-fit.cost)
    except ValueError as error:
        result.pop('state', None)
        result.pop('information', None)
        result['state_conversion_error'] = str(error)
    args.output.write_text(json.dumps(result, indent=2))
    print('finished', fit.cost, fit.success, 'elapsed', time.monotonic()-clock, flush=True)


if __name__ == '__main__':
    main()
