"""Test stationary nuisance responses in the physical road-target likelihood.

The moving model, road, antenna positions and original observations are kept.
Constant-delay responses pass through the same range/time operator as moving
responses. Exact Gaussian nuisance projection is used only inside this support
likelihood; no archive sample or Doppler interval is altered. Both complete
background transfer and moving prediction after identical test-nuisance
profiling are reported, with their different scopes made explicit.
"""
import argparse
import fcntl
import json
from pathlib import Path
import time

import numpy as np

from fit_shared_target_geometry import load_experiment
from phase_history import C, FS
from scene_dictionary import fingerprint
from shared_target_geometry import RoadTarget, complex_gain
from validate_shared_geometry import fit_target


class StationaryField:
    def __init__(self, field, delays, origin, cutoff):
        self.original, self.time = field, field.time
        self.delays = np.asarray(delays)
        self.stationary = np.column_stack([field.template(np.full(len(field.time), delay),
            range_bias=origin)[0] for delay in delays])
        u, s, vh = np.linalg.svd(self.stationary, full_matrices=False)
        keep = s > s[0]*cutoff
        self.basis, self.inverse = u[:, keep], vh[keep].conj().T/s[keep]
        self.singular_values, self.cutoff = s, cutoff
        self.y = self.remove(field.y)

    def remove(self, value):
        return value-self.basis@(self.basis.conj().T@value)

    def coefficients(self, value):
        return self.inverse@(self.basis.conj().T@value)

    def template(self, path, jacobian=None, range_bias=0.):
        u, derivative = self.original.template(path, jacobian, range_bias=range_bias)
        return self.remove(u), None if derivative is None else self.remove(derivative)


class StationaryRoad(RoadTarget):
    def __init__(self, source, *, delays=None, margin=5, cutoff=1e-10):
        if delays is None:
            delays = []
            for field, origin in zip(source.training, source.origins):
                samples = field.seed_path*FS/C
                values = np.arange(np.floor(samples.min())+field.offsets.min()-margin,
                    np.ceil(samples.max())+field.offsets.max()+margin+1)*C/FS-origin
                # The existing PSF interpolator is verified only on this
                # support. Do not silently truncate a wide static dictionary.
                lag_min = field.offsets.min()-(values.max()+origin-field.seed_path.min())*FS/C
                lag_max = field.offsets.max()-(values.min()+origin-field.seed_path.max())*FS/C
                if min(lag_min, -lag_max) < -32:
                    raise ValueError('Stationary support exceeds the measured PSF interpolation support')
                delays.append(values)
        training = [StationaryField(f, d, o, cutoff) for f, d, o in zip(source.training, delays, source.origins)]
        validation = [StationaryField(f, d, o, cutoff) for f, d, o in zip(source.validation, delays, source.origins)]
        super().__init__(source.problem, training, validation, source.initial, source.transmitters, source.antennas)
        # Preserve the source's constant optimizer scale and original noise
        # units. Projection is a nuisance likelihood, not target reweighting.
        self.energy = source.energy

    def raw_template(self, motion, antenna, channel, field):
        return RoadTarget.template(self, motion, antenna, channel, field.original)

    def metrics(self, motion, antenna, channel, amplitude):
        train, test = self.training[channel], self.validation[channel]
        u, _ = self.raw_template(motion, antenna, channel, train)
        v, _ = self.raw_template(motion, antenna, channel, test)
        beta = train.coefficients(train.original.y-amplitude*u)
        beta0 = train.coefficients(train.original.y)
        prediction = amplitude*v+test.stationary@beta
        reference = test.stationary@beta0
        projected_error = test.remove(test.original.y-amplitude*v)
        information = float(np.vdot(train.remove(u), train.remove(u)).real/np.vdot(u, u).real)
        return {'prediction_error': float(np.linalg.norm(test.original.y-prediction)**2),
            'stationary_only_full_prediction_error': float(np.linalg.norm(test.original.y-reference)**2),
            'moving_prediction_error_after_test_nuisance': float(np.vdot(projected_error, projected_error).real),
            'stationary_only_projected_error': float(np.vdot(test.y, test.y).real),
            'data_energy': float(np.vdot(test.original.y, test.original.y).real),
            'training_target_information_fraction': information,
            'training_nuisance_rank': train.basis.shape[1], 'test_nuisance_rank': test.basis.shape[1],
            'nuisance_columns': len(train.delays), 'coefficient_real_imag': [float(amplitude.real), float(amplitude.imag)]}

    def predict(self, motion, antenna):
        records = []
        for c, train in enumerate(self.training):
            u, du = self.template(motion, antenna, c, train)
            _, _, amplitude, _ = complex_gain(train.y, u, du)
            records.append(self.metrics(motion, antenna, c, amplitude))
        return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('geometry-fit', 'ship-fit', 'road-fit', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--labels', nargs='+', default=['E', 'Q', 'P', 'S'])
    parser.add_argument('--cutoff', type=float, default=1e-10)
    parser.add_argument('--iterations', type=int, default=180)
    parser.add_argument('--fields', type=Path, default=Path('build/isar-20261015/full-fields'))
    parser.add_argument('--geometry', type=Path, default=Path('docs/data/calibration_20261013'))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = args.output.with_suffix('.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not 0 < args.cutoff < 1:
        raise ValueError('Use an explicit relative singular-value cutoff')
    fitted, saved = [json.loads(p.read_text()) for p in (args.geometry_fit, args.road_fit)]
    if not fitted['complete'] or not fitted['fit']['converged']:
        raise ValueError('Use a completed shared-geometry fit or matched fixed-site control')
    for key in ('sources', 'implementation_sha256'):
        for path, expected in fitted['identity'][key].items():
            if fingerprint(Path(path)) != expected:
                raise ValueError('Original geometry experiment changed: '+path)
    for path in (args.ship_fit, args.road_fit):
        if fitted['identity']['sources'].get(str(path)) != fingerprint(path):
            raise ValueError('The supplied geometry was fitted to different inputs')
    joint, _, _, labels, dropped = load_experiment(args.ship_fit, args.road_fit,
        args.fields, args.geometry, fitted['identity']['labels'], fitted['identity']['nodes'])
    if dropped or set(args.labels)-set(labels):
        raise ValueError('Unsupported or overlapping target selection')
    p = np.asarray(fitted['parameters'])
    antenna = p[joint.antenna_slice]
    identity = {'sources': {str(path): fingerprint(path) for path in (args.geometry_fit, args.ship_fit, args.road_fit)},
        'labels': args.labels, 'cutoff': args.cutoff, 'iterations': args.iterations,
        'implementation_sha256': {str(path): fingerprint(path) for path in
            [Path(__file__), Path(__file__).with_name('shared_target_geometry.py'),
             Path(__file__).with_name('validate_shared_geometry.py'), Path(__file__).with_name('fit_physical_profile.py')]}}
    result = {'identity': identity, 'scope': __doc__, 'complete': False, 'targets': [],
        'accepted_calibration': False, 'production_changed': False,
        'prediction_scope': 'Full prediction transfers all coefficients from training. The second metric profiles stationary nuisance separately on test data, identically for both moving models, while retaining training-derived moving amplitude and motion.'}
    if args.output.exists():
        result = json.loads(args.output.read_text())
        if result['identity'] != identity:
            raise ValueError('Cannot resume changed stationary-road controls')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    def save():
        temporary = args.output.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result, indent=2)+'\n'); temporary.replace(args.output)
    for label in args.labels:
        if any(t['label'] == label for t in result['targets']):
            continue
        index, old_index = labels.index(label), saved['labels'].index(label)
        original = joint.roads[index]
        model = StationaryRoad(original, cutoff=args.cutoff)
        starts = [p[joint.road_slices[index]], np.asarray(saved['selected_roads'][old_index]['state'])]
        baseline, revised = [fit_target(road, antenna, starts, args.iterations) for road in (original, model)]
        bp = np.array(baseline['starts'][baseline['selected_start']]['parameters'])
        comparison = [model.metrics(bp, antenna, c, complex(*record['coefficient_real_imag']))
                      for c, record in enumerate(baseline['prediction'])]
        before = sum(r['moving_prediction_error_after_test_nuisance'] for r in comparison)
        after = sum(r['moving_prediction_error_after_test_nuisance'] for r in revised['prediction'])
        full_before = sum(r['prediction_error'] for r in baseline['prediction'])
        full_after = sum(r['prediction_error'] for r in revised['prediction'])
        item = {'label': label, 'baseline': baseline, 'with_stationary': revised,
            'baseline_under_same_test_nuisance': comparison,
            'moving_prediction_error_reduction_percent': 100*(1-after/before),
            'full_prediction_error_reduction_percent': 100*(1-full_after/full_before),
            'selected_fits_stationary': [r['starts'][r['selected_start']]['fit']['converged'] for r in (baseline, revised)]}
        result['targets'].append(item); save()
        print(json.dumps({'label': label, **{k: item[k] for k in
            ('moving_prediction_error_reduction_percent', 'full_prediction_error_reduction_percent', 'selected_fits_stationary')},
            'seconds': time.monotonic()-started}), flush=True)
    result.update(complete=True, seconds=time.monotonic()-started); save()


if __name__ == '__main__':
    main()
