"""Full-bin reference for stationary-background vehicle calibration.

Every selected saved 20-ms bin and range sample is retained. Only invertible
whitening by the declared empirical covariance precedes Gaussian profiling
of the stationary dictionary. This removes the offline Doppler-gate loss;
it does not certify the prior bin averaging, covariance or scattering model.
"""
import argparse
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from fit_shared_target_geometry import load_experiment
from integrated_stationary_road import IntegratedField, IntegratedRoad, RecordedBins
from scene_dictionary import fingerprint
from shared_target_geometry import RoadTarget
from stationary_road_control import StationaryRoad
from stationary_sufficient_road import UnionField, UnionRoad, coefficients
from validate_shared_geometry import fit_target


class FullField(UnionField):
    def __init__(self, original, binned):
        self.original, self.time, self.selected = original, original.time, original.selected
        self.sigma = np.sqrt(original.data['variance'][self.selected, len(original.offsets)//2])
        if np.any(self.sigma <= 0):
            raise ValueError('Positive declared bin variances are required')
        self.white_stationary = self.white(binned)
        self.y = self.apply(original.data['value'])

    def apply(self, values):
        result = self.white(values)
        return result.reshape((-1,)+result.shape[2:])


class FullRoad(StationaryRoad):
    def __init__(self, previous, binned):
        def convert(field, values):
            original = FullField(field.original, values)
            return IntegratedField(SimpleNamespace(original=original, time=field.time,
                delays=field.delays, cutoff=field.cutoff), values)
        training = [convert(f, b) for f, b in zip(previous.training, binned)]
        validation = [convert(f, b) for f, b in zip(previous.validation, binned)]
        RoadTarget.__init__(self, previous.problem, training, validation, previous.initial,
            previous.transmitters, previous.antennas)
        self.energy = previous.energy


def predict_at(road, motion, antenna, c, amplitude, beta):
    field = road.validation[c]
    u, _ = road.raw_template(motion, antenna, c, field)
    residual = field.original.y-amplitude*u-field.stationary@beta
    moving_residual = field.remove(field.original.y-amplitude*u)
    return {'prediction_error': float(np.vdot(residual, residual).real),
        'data_energy': float(np.vdot(field.original.y, field.original.y).real),
        'moving_prediction_error_with_identical_test_nuisance': float(np.vdot(moving_residual, moving_residual).real),
        'stationary_only_projected_error': float(np.vdot(field.y, field.y).real),
        'complex_observations': len(residual)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('geometry-fit', 'ship-fit', 'road-fit', 'control', 'integrated', 'sufficient', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--fields', type=Path, default=Path('build/isar-20261015/full-fields'))
    parser.add_argument('--geometry', type=Path, default=Path('docs/data/calibration_20261013'))
    parser.add_argument('--iterations', type=int, default=180)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = args.output.with_suffix('.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sufficient, integrated, control, fitted, saved = [json.loads(p.read_text()) for p in
        (args.sufficient, args.integrated, args.control, args.geometry_fit, args.road_fit)]
    if not sufficient['complete'] or not integrated['complete']:
        raise ValueError('Finish both compressed-data controls first')
    for source in (sufficient, integrated):
        for group in ('sources', 'implementation_sha256'):
            for path, digest in source['identity'][group].items():
                if fingerprint(Path(path)) != digest:
                    raise ValueError('Reference control changed: '+path)
    for path in (args.geometry_fit, args.ship_fit, args.road_fit, args.control, args.integrated):
        if sufficient['identity']['sources'].get(str(path)) != fingerprint(path):
            raise ValueError('Use identical source geometry and observations')
    joint, _, _, labels, _ = load_experiment(args.ship_fit, args.road_fit, args.fields,
        args.geometry, fitted['identity']['labels'], fitted['identity']['nodes'])
    antenna = np.asarray(fitted['parameters'])[joint.antenna_slice]
    sources = [args.geometry_fit, args.ship_fit, args.road_fit, args.control, args.integrated, args.sufficient]
    implementation = [Path(__file__), Path(__file__).with_name('stationary_sufficient_road.py'),
        Path(__file__).with_name('integrated_stationary_road.py'), Path(__file__).with_name('stationary_road_control.py')]
    identity = {'sources': {str(p): fingerprint(p) for p in sources}, 'iterations': args.iterations,
        'implementation_sha256': {str(p): fingerprint(p) for p in implementation}}
    result = {'identity': identity, 'scope': __doc__, 'complete': False, 'targets': [],
        'accepted_calibration': False, 'production_changed': False,
        'prediction_scope': 'All compared models transfer training-derived motion and coefficients. Each test observation operator is identical across estimators. The separately labelled moving-error metric profiles the same stationary nuisance on test samples.'}
    if args.output.exists():
        result = json.loads(args.output.read_text())
        if result['identity'] != identity:
            raise ValueError('Cannot resume a changed full-bin experiment')
    def save():
        temporary = args.output.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result, indent=2)+'\n')
        temporary.replace(args.output)
    started = time.monotonic()
    for record in sufficient['targets']:
        label = record['label']
        if any(t['label'] == label for t in result['targets']):
            continue
        previous = StationaryRoad(joint.roads[labels.index(label)], cutoff=control['identity']['cutoff'])
        binned = []
        for c, field in enumerate(previous.training):
            with np.load(Path(saved['input_patches'])/f'{label}-{c}.npz') as raw:
                bins = RecordedBins(field.original, raw['row_time'], raw['row_information'])
                binned.append(bins.stationary(field.original, field.delays, previous.origins[c]))
        gated, union, full = IntegratedRoad(previous, binned), UnionRoad(previous, binned), FullRoad(previous, binned)
        fits = [next(t for t in integrated['targets'] if t['label'] == label)['fit'], record['fit']]
        states = [np.asarray(f['starts'][f['selected_start']]['parameters']) for f in fits]
        revised = fit_target(full, antenna, states, args.iterations)
        motion = np.asarray(revised['starts'][revised['selected_start']]['parameters'])
        channels = []
        for c in range(5):
            case = {'channel': c}
            for name, source, state in [('original_gate', gated, states[0]), ('stationary_union', union, states[1]),
                                        ('full_bins', full, motion)]:
                amplitude, beta = coefficients(source, state, antenna, c)
                case[name] = {space: predict_at(destination, state, antenna, c, amplitude, beta)
                    for space, destination in [('original_gate', gated), ('full_bins', full)]}
            channels.append(case)
        summary = {space: {name: {metric: sum(c[name][space][metric] for c in channels)
            for metric in ('prediction_error', 'moving_prediction_error_with_identical_test_nuisance')}
            for name in ('original_gate', 'stationary_union', 'full_bins')} for space in ('original_gate', 'full_bins')}
        item = {'label': label, 'fit': revised, 'channels': channels, 'summary': summary,
            'selected_fit_stationary': revised['starts'][revised['selected_start']]['fit']['converged']}
        result['targets'].append(item); save()
        print(json.dumps({'label': label, 'stationary': item['selected_fit_stationary'],
            'summary': summary, 'seconds': time.monotonic()-started}), flush=True)
    result.update(complete=True, seconds=time.monotonic()-started); save()


if __name__ == '__main__':
    main()
