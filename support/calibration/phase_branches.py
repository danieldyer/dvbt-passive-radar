"""Profile constant path ambiguities before refining the exact road likelihood.

This is an initializer, not a replacement measurement model. Each trial of
four shared band phases searches every target's entire delay grid. A target
cannot get stuck on a local wavelength branch during this screening step.
Actual road paths, range envelopes and positive band amplitudes determine
the final likelihood and refinement.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import differential_evolution

from calibration_solver import solve
from coherent_fit import OTHER, constant_path_direction, transport_path_shift, information
from complex_likelihood import projected_profile
from profiled_bands import ProfiledBandProblem
from road_fit import load_trial


class BranchScreen:
    def __init__(self, statistics, slopes, shifts, gradient=None, curvature=None):
        z = np.asarray(statistics)
        self.energy = np.sum(abs(z)**2, axis=1)
        self.shifts = np.asarray(shifts)
        self.coefficients = (z*abs(z))[:, :, None]*np.exp(1j*np.asarray(slopes)[:, :, None]*self.shifts[:, None, :])
        gradient = np.zeros(len(z)) if gradient is None else np.asarray(gradient)
        curvature = np.zeros(len(z)) if curvature is None else np.asarray(curvature)
        self.penalty = gradient[:, None]*self.shifts+.5*curvature[:, None]*self.shifts**2

    def scores(self, beta):
        beta = np.atleast_2d(beta)
        rotations = np.ones((len(beta), 5), complex)
        rotations[:, OTHER] = np.exp(-1j*beta)
        captured = abs(np.einsum('tbd,pb->tpd', self.coefficients, rotations, optimize=True))**2
        return captured/np.maximum(self.energy[:, None, None], 1e-30)-self.penalty[:, None, :]

    def objective(self, beta):
        beta = np.asarray(beta)
        score = np.max(self.scores(beta.T), axis=-1).sum(axis=0)
        value = self.energy.sum()-score
        return float(value[0]) if beta.ndim == 1 else value

    def path_shifts(self, beta):
        indices = np.argmax(self.scores(beta)[:, 0], axis=-1)
        return self.shifts[np.arange(len(indices)), indices]


def build_screen(joint, state, points=501, maximum_shift=25.):
    statistics, slopes, shifts, gradient, curvature = [], [], [], [], []
    biases = np.zeros(5)
    if joint.range_bias:
        biases[OTHER] = state[joint.bias_offset:]
    for index, problem in enumerate(joint.problems):
        first = 9+index*joint.n
        local = np.r_[state[first:first+joint.n], state[:9]]
        direction = constant_path_direction(problem, local)
        n = problem.model.radial_count
        low, high = problem.model.bounds()
        active = abs(direction) > 1e-10
        left = (low[:n][active]-local[:n][active])/direction[active]
        right = (high[:n][active]-local[:n][active])/direction[active]
        lo = max(-maximum_shift, np.minimum(left, right).max())+1e-6
        hi = min(maximum_shift, np.maximum(left, right).min())-1e-6
        if lo >= hi:
            raise ValueError('No supported constant-path ambiguity interval')
        z, k, g, h = [], [], 0., 0.
        for ch, field in enumerate(problem.fields):
            path, j = problem.model.paths(local, field.time, np.full(len(field.time), ch), problem.sites, True)
            derivative = j[:, :n]@direction
            u, du = field.template(path, derivative[:, None], biases[ch])
            z.append(np.vdot(u, field.y)/np.sqrt(np.vdot(u, u).real))
            k.append(field.k*np.mean(derivative))
            residual, dr, _ = projected_profile(field.y, u, du)
            g += np.vdot(residual, dr[:, 0]).real
            h += np.vdot(dr[:, 0], dr[:, 0]).real
        # Common RF phase is eliminated by the complex target amplitude.
        # Removing this carrier from the screen improves numeric precision.
        statistics.append(z);slopes.append(np.array(k)-k[1])
        shifts.append(np.linspace(lo, hi, points));gradient.append(g);curvature.append(h)
    return BranchScreen(statistics, slopes, shifts, gradient, curvature)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('fit', 'histories', 'patches', 'templates', 'routes', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--generations', type=int, default=300)
    parser.add_argument('--refine', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=400)
    parser.add_argument('--maximum-shift-m', type=float, default=25.)
    parser.add_argument('--grid-points', type=int, default=501)
    args = parser.parse_args()
    saved = json.loads(args.fit.read_text())
    joint = load_trial(saved, args.routes, args.histories, args.patches, args.templates)
    state = np.array(saved['state'])
    if args.maximum_shift_m <= 0 or args.grid_points < 3:
        parser.error('The ambiguity screen needs positive extent and at least three samples')
    screen = build_screen(joint, state, args.grid_points, args.maximum_shift_m)
    problem = ProfiledBandProblem(joint)
    clock = time.monotonic()
    def progress(intermediate_result):
        if intermediate_result.nit%20 == 0:
            print('generation', intermediate_result.nit, 'screen cost', round(intermediate_result.fun, 3),
                  'elapsed', round(time.monotonic()-clock, 1), flush=True)
    fit = differential_evolution(screen.objective, [(-np.pi, np.pi)]*4, vectorized=True,
        updating='deferred', popsize=24, maxiter=args.generations, rng=np.random.default_rng(45918),
        polish=False, tol=1e-8, atol=1e-7, callback=progress)
    r = problem.residual(problem.initial(state))
    candidates = [(float(r@r/2), problem.initial(state), None)]
    for beta in fit.population[np.argsort(fit.population_energies)[:32]]:
        trial = state.copy()
        trial[joint.beta_offset:joint.magnitude_offset] = beta
        changes = screen.path_shifts(beta)
        for index, source in enumerate(joint.problems):
            first = 9+index*joint.n
            local = np.r_[state[first:first+joint.n], state[:9]]
            moved, _ = transport_path_shift(source, local, changes[index])
            trial[first:first+joint.n] = moved[:joint.n]
        p = problem.initial(trial)
        r = problem.residual(p)
        candidates.append((float(r@r/2), p, changes.tolist()))
    candidates.sort(key=lambda item:item[0])
    distinct = []
    for candidate in candidates:
        if not distinct or candidate[0]-distinct[-1][0] > .01:
            distinct.append(candidate)
    results = []
    for cost, start, changes in distinct[:args.refine]:
        print('exact start', cost, 'path shifts', changes, flush=True)
        result = solve(problem.residual, start, problem.bounds(), args.iterations)
        results.append(result)
        print('refined', result.cost, result.success, 'elapsed', round(time.monotonic()-clock, 1), flush=True)
    best = min(results, key=lambda item:item.cost)
    result = {**saved, 'source_fit': str(args.fit), 'cost': float(best.cost),
        'converged': bool(best.success), 'evaluations': best.nfev, 'optimality': float(best.optimality),
        'termination': str(best.message), 'reduced_state': best.x.tolist(),
        'site_corrections_enu_m': (best.x[:9].reshape(3, 3)*[30, 30, 75]).tolist(),
        'profiled_band_gains': True, 'range_bias': joint.range_bias,
        'range_bias_m': best.x[problem.bias_offset:].tolist(),
        'branch_screen': {'generations': fit.nit, 'converged': bool(fit.success),
            'maximum_path_shift_m': args.maximum_shift_m, 'grid_points': args.grid_points,
            'cost': float(fit.fun), 'exact_proposal_costs': [q[0] for q in distinct],
            'scope': 'Approximate phase/envelope screen only generates seeds. Full complex range/phase likelihood evaluates and refines all retained proposals; no certificate of global optimality.'}}
    try:
        expanded, missing = problem.expanded_state(best.x)
        result.update(state=expanded.tolist(), zero_amplitude_channels_by_target=missing,
                      information=information(joint, expanded))
    except ValueError as error:
        result.pop('state', None);result.pop('information', None)
        result['state_conversion_error'] = str(error)
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
