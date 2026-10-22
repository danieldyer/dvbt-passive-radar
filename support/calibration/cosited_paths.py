"""Test a common excess-path history before imposing geographic geometry.

Four licensed main services share Ornata. This diagnostic allows an independent
constant complex gain per band, so it tests temporal phase, not absolute target
scattering phase. The small licensed mounting-height differences are explicitly
outside the coincident-centre hypothesis. Fits use alternating one-second blocks;
association and the reversible deramp seeds were chosen using the full capture.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from complex_likelihood import operator_for, projected_profile
from phase_fit import load_histories
from polar_model import PolarTrajectory


class PathHistory:
    def __init__(self, seed, knot_interval=1.5):
        self.model = PolarTrajectory(seed, knot_interval=knot_interval)
        self.seed = seed
        self.count = self.model.radial_count-1

    def basis(self, time):
        base, matrix = self.model.length_basis(time)
        # Remove the unobservable constant path. Partition of unity makes the
        # remaining n-1 spline contrasts sufficient for every path variation.
        centre = self.model.spline(0.)
        return base-self.seed['path'][0], (matrix-centre)[:, 1:]

    def path(self, p, time):
        base, matrix = self.basis(time)
        return base+matrix@p


class PathProblem:
    def __init__(self, channels, model, selected=None):
        self.channels, self.model = channels, model
        self.observations = []
        for index, d in enumerate(channels):
            mask = None if selected is None else selected[index]
            q = operator_for(d, mask)
            base, matrix = model.basis(d['time'])
            self.observations.append((q, q@d['value'],
                -d['k']*(base-d['seed_delta']), -d['k']*matrix))

    def template(self, p, index, jacobian=False):
        q, _, phase, matrix = self.observations[index]
        u = np.exp(1j*(phase+matrix@p))
        return q@u, q@(1j*u[:, None]*matrix) if jacobian else None

    def residual(self, p, jacobian=False):
        pieces = []
        for index, (_, value, _, _) in enumerate(self.observations):
            u, du = self.template(p, index, jacobian)
            r = projected_profile(value, u, du)[int(jacobian)]
            pieces.append(np.r_[r.real, r.imag])
        return np.concatenate(pieces)

    def initial(self, index):
        d = self.channels[index]
        q = self.observations[index][0]
        smooth = d['basis']@np.linalg.lstsq(q@d['basis'], q@d['value'], rcond=None)[0]
        phase = np.unwrap(np.angle(smooth))
        inferred = d['seed_delta']-phase/d['k']
        inferred -= np.interp(self.model.seed['epoch'], d['time'], inferred)
        base, matrix = self.model.basis(d['time'])
        w = abs(smooth)/np.sqrt(d['variance'])
        return np.linalg.lstsq(matrix*w[:, None], (inferred-base)*w, rcond=None)[0]

    def fit(self, starts):
        fits = [least_squares(self.residual, start,
            jac=lambda p:self.residual(p, True), max_nfev=350,
            ftol=1e-9, xtol=1e-9, gtol=1e-6) for start in starts]
        return min(fits, key=lambda f:f.cost)

    def evaluate(self, p, testing):
        result = []
        for index, (_, v, _, _) in enumerate(testing.observations):
            u, _ = self.template(p, index)
            _, amplitude = projected_profile(self.observations[index][1], u)
            u, _ = testing.template(p, index)
            energy = abs(np.vdot(u, v))**2/np.vdot(u, u).real
            error = v-amplitude*u
            result.append({'heldout_focused_energy':float(energy),
                'heldout_prediction_error':float(np.vdot(error, error).real),
                'training_amplitude':[float(amplitude.real), float(amplitude.imag)]})
        return result


def check(channels, description, knot_interval=1.5, extra_starts=()):
    model = PathHistory(channels[1]['seed'], knot_interval)
    mask = [np.floor(d['time']-description['start']).astype(int)%2==0 for d in channels]
    train = PathProblem(channels, model, mask)
    test = PathProblem(channels, model, [~m for m in mask])
    initial = [train.initial(i) for i in range(len(channels))]
    independent = []
    for i, d in enumerate(channels):
        pr = PathProblem([d], model, [mask[i]])
        fit = pr.fit([initial[i], np.zeros(model.count), *extra_starts])
        independent.append(fit)
    # Each band supplies a possible phase-unwrapping branch. All optimization
    # scores below use training samples only.
    common = train.fit([f.x for f in independent]+[np.zeros(model.count), *extra_starts])
    # The common model is nested inside each separate model. Refit from its
    # solution as well so a missed local branch cannot reverse that ordering.
    for i, d in enumerate(channels):
        pr = PathProblem([d], model, [mask[i]])
        fit = pr.fit([common.x])
        if fit.cost < independent[i].cost:
            independent[i] = fit
    common_metrics = train.evaluate(common.x, test)
    individual_metrics = []
    for i, d in enumerate(channels):
        pr = PathProblem([d], model, [mask[i]])
        held = PathProblem([d], model, [~mask[i]])
        individual_metrics.extend(pr.evaluate(independent[i].x, held))
    time = np.linspace(description['start'], description['end'], 601)
    path = model.path(common.x, time)
    separate = np.array([model.path(f.x, time) for f in independent])
    loss = [10*np.log10(i['heldout_focused_energy']/c['heldout_focused_energy'])
            for c, i in zip(common_metrics, individual_metrics)]
    return {'knot_interval_s':knot_interval, 'common_converged':bool(common.success),
        'independent_converged':[bool(f.success) for f in independent],
        'common_training_cost':float(common.cost),
        'independent_training_cost':float(sum(f.cost for f in independent)),
        'common_metrics':common_metrics, 'independent_metrics':individual_metrics,
        'heldout_common_loss_db':loss, 'time_s':time.tolist(),
        'common_path_variation_m':path.tolist(),
        'independent_minus_common_path_m':(separate-path).tolist(),
        'common_state':common.x.tolist(), 'independent_states':[f.x.tolist() for f in independent]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--histories', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--labels', nargs='+', default=['A', 'B', 'C', 'E', 'J'])
    p.add_argument('--knot-interval', type=float, default=1.5)
    p.add_argument('--seeds', type=Path, nargs='*', default=[],
                   help='Optional full-data geometric branch seeds; validation remains conditional')
    p.add_argument('--path-seeds',type=Path,help='Optional sampled excess-path initializers; no added observations')
    args = p.parse_args()
    result = {'channels':['Seven', 'SBS', 'Nine', 'Ten'], 'targets':[],
        'hypothesis':'coincident Ornata phase centres; independent constant complex band gains',
        'scope':'conditional on full-data association and seeds; no geographic trajectory or transmitter fit',
        'limitations':'Mounting-height differences, variable scattering, and receiver direction response are not included. Empirical noise covariance is not a certified thermal-noise model.',
        'geometric_branch_sources':[str(path) for path in args.seeds]}
    seed_targets = [target for path in args.seeds for target in json.loads(path.read_text())['targets']]
    path_seeds=[] if args.path_seeds is None else json.loads(args.path_seeds.read_text())['targets']
    for label in args.labels:
        channels, description = load_histories(args.histories, label, 1.)
        starts = []
        model = PathHistory(channels[1]['seed'], args.knot_interval)
        for target in path_seeds:
            if target['label']!=label:continue
            time=np.asarray(target['time_s']);variation=np.asarray(target['excess_path_m'])
            variation=variation-np.interp(model.seed['epoch'],time,variation)
            base,matrix=model.basis(time)
            starts.append(np.linalg.lstsq(matrix,variation-base,rcond=None)[0])
        for target in seed_targets:
            if target['label'] != label:
                continue
            original = PathHistory(channels[1]['seed'])
            coefficients = np.array(target['state'][:original.model.radial_count])
            time = np.linspace(description['start'], description['end'], 601)
            base, matrix = model.basis(time)
            old_base, old_matrix = original.model.length_basis(time)
            variation = old_base+old_matrix@coefficients
            variation -= channels[1]['seed']['path'][0]+original.model.spline(0.)@coefficients
            starts.append(np.linalg.lstsq(matrix, variation-base, rcond=None)[0])
        answer = check(channels[:4], description, args.knot_interval, starts)
        answer['label'] = label
        result['targets'].append(answer)
        args.output.write_text(json.dumps(result, indent=2))
        print(label, 'common path heldout loss dB',
              np.round(answer['heldout_common_loss_db'], 3),
              'converged', answer['common_converged'], answer['independent_converged'], flush=True)


if __name__ == '__main__':
    main()
