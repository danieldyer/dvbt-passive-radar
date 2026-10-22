"""Slow-target screen with explicit constant-delay nuisance responses.

Both moving and stationary templates pass through the original range/temporal
observation operator. Nuisance coefficients fit training samples and predict
withheld samples. No Doppler interval is removed from the archived profiles.
This is a conditional candidate screen, not an antenna calibration.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from complex_likelihood import projected_profile
from cosited_paths import PathHistory, PathProblem
from geographic_io import read_profiles
from phase_fit import load_histories
from phase_history import C, FS
from range_fit import Field, load_fields


class Observation:
    def __init__(self, source, path, selected, stationary_paths):
        self.field = Field(source.data, source.pulse, source.seed, selected=selected)
        self.path = path
        self.base, self.matrix = path.basis(self.field.time)
        self.base += source.seed['path'][0]
        self.stationary = np.column_stack([
            self.field.template(np.full(len(self.field.time), delay))[0]
            for delay in stationary_paths])
        u, s, vh = np.linalg.svd(self.stationary, full_matrices=False)
        keep = s > s[0]*1e-10
        self.basis = u[:, keep]
        self.inverse = vh[keep].conj().T/s[keep]
        self.y = self.remove(self.field.y)

    def remove(self, value):
        return value-self.basis@(self.basis.conj().T@value)

    def coefficients(self, value):
        return self.inverse@(self.basis.conj().T@value)

    def template(self, p, derivative=False):
        u, j = self.field.template(self.base+self.matrix@p,
                                   self.matrix if derivative else None)
        return self.remove(u), self.remove(j) if derivative else None

    def evaluate(self, p, test):
        u, _ = self.template(p)
        _, alpha = projected_profile(self.y, u)
        original, _ = self.field.template(self.base+self.matrix@p)
        beta = self.coefficients(self.field.y-alpha*original)
        beta0 = self.coefficients(self.field.y)
        other, _ = test.field.template(test.base+test.matrix@p)
        prediction = alpha*other+test.stationary@beta
        reference = test.stationary@beta0
        tested, _ = test.template(p)
        energy = abs(np.vdot(tested, test.y))**2/np.vdot(tested, tested).real
        error = float(np.sum(abs(test.field.y-prediction)**2))
        baseline = float(np.sum(abs(test.field.y-reference)**2))
        return {'heldout_prediction_error': error,
                'stationary_only_heldout_error': baseline,
                'heldout_error_reduction_fraction': 1-error/baseline,
                'heldout_projected_focused_energy': float(energy),
                'training_target_information_loss_db': float(-10*np.log10(
                    np.vdot(u, u).real/np.vdot(original, original).real)),
                'stationary_rank': self.basis.shape[1],
                'stationary_columns': self.stationary.shape[1]}


def residual(observations, p, jacobian=False):
    parts = []
    for observation in observations:
        u, j = observation.template(p, jacobian)
        r = projected_profile(observation.y, u, j)[int(jacobian)]
        parts.append(np.r_[r.real, r.imag])
    return np.concatenate(parts)


def fit(observations, starts):
    trials = [least_squares(lambda p: residual(observations, p), start,
                            jac=lambda p: residual(observations, p, True),
                            max_nfev=250, ftol=1e-9, xtol=1e-9, gtol=1e-6)
              for start in starts]
    return min(trials, key=lambda result: result.cost)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('histories', 'patches', 'templates', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--labels', nargs='+', required=True)
    args = parser.parse_args()
    _, records = read_profiles(args.templates, True)
    report = {'channels': ['Seven', 'SBS', 'Nine', 'Ten'], 'targets': [],
              'scope': 'Conditional temporal-path screen under nominal coincident Ornata centres. Actual physical trajectories and antenna coordinates are not fitted. Original complex profiles, matched range response and covariance are retained. Alternating one-second training fits target and stationary coefficients; prediction uses the other seconds. Association and seeds were discovered in the full recording. Empirical covariance is not a thermal CRLB.'}
    for label in args.labels:
        channels, description = load_histories(args.histories, label, 1.)
        fields = load_fields(args.patches, label, description, records)[:4]
        path = PathHistory(channels[1]['seed'], 3.)
        selected = [np.floor(f.time-description['start']).astype(int)%2 == 0 for f in fields]
        train, test = [], []
        for source, mask in zip(fields, selected):
            samples = source.seed_path*FS/C
            delays = np.arange(np.floor(samples.min())+source.offsets.min()-5,
                               np.ceil(samples.max())+source.offsets.max()+6)*C/FS
            train.append(Observation(source, path, mask, delays))
            test.append(Observation(source, path, ~mask, delays))
        # Independent measured-phase and unperturbed Doppler seeds provide
        # deterministic basins. Their samples are not added to the likelihood.
        masks = [np.floor(d['time']-description['start']).astype(int)%2 == 0 for d in channels[:4]]
        simple = PathProblem(channels[:4], path, masks)
        starts = [np.zeros(path.count)]+[simple.initial(i) for i in range(4)]
        independent = [fit([o], [starts[0], starts[i+1]]) for i, o in enumerate(train)]
        common = fit(train, [starts[0]]+[result.x for result in independent])
        for i in range(4):
            revised = fit([train[i]], [common.x])
            if revised.cost < independent[i].cost:
                independent[i] = revised
        times = np.linspace(description['start'], description['end'], 401)
        variation = path.path(common.x, times)
        result = {'label': label, 'time_s': times.tolist(),
                  'common_path_variation_m': variation.tolist(),
                  'median_common_path_rate_mps': float(np.median(np.gradient(variation, times))),
                  'common_converged': bool(common.success),
                  'independent_converged': [bool(f.success) for f in independent],
                  'common_training_cost': float(common.cost),
                  'independent_training_cost': float(sum(f.cost for f in independent)),
                  'common_metrics': [a.evaluate(common.x, b) for a, b in zip(train, test)],
                  'independent_metrics': [a.evaluate(f.x, b) for a, b, f in zip(train, test, independent)],
                  'common_state': common.x.tolist(),
                  'independent_states': [f.x.tolist() for f in independent]}
        report['targets'].append(result)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(label, 'rate', round(result['median_common_path_rate_mps'], 4),
              'heldout error reductions', [round(v['heldout_error_reduction_fraction'], 4) for v in result['common_metrics']],
              'projected energy', [round(v['heldout_projected_focused_energy'], 2) for v in result['common_metrics']], flush=True)


if __name__ == '__main__':
    main()
