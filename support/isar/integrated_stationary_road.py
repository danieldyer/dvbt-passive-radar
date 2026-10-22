"""Average stationary templates through the actual recorded row/bin operator.

Moving-target deramping leaves stationary returns at nonzero phase rate. A
mean timestamp does not describe that average. This control uses the saved
row epochs/information weights, verifies reconstruction of the archived bins,
and changes only the stationary forward dictionary. The moving trajectory,
range PSF, observation covariance and held-out split remain explicit controls.
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
from shared_target_geometry import RoadTarget
from stationary_road_control import StationaryField, StationaryRoad
from validate_shared_geometry import fit_target


class RecordedBins:
    def __init__(self, field, row_time, row_information, *, step=.02):
        start = field.seed['epoch']-field.seed['duration_s']/2
        rows = np.floor((np.asarray(row_time)-start)/step).astype(int)
        bins = np.floor((field.time-start)/step).astype(int)
        if len(set(bins)) != len(bins) or rows.min() < 0:
            raise ValueError('Inconsistent retained bin epochs')
        lookup = np.full(max(rows.max(), bins.max())+1, -1)
        lookup[bins] = np.arange(len(bins))
        self.keep = lookup[rows] >= 0
        self.index = lookup[rows[self.keep]]
        self.time = np.asarray(row_time)[self.keep]
        weights = np.asarray(row_information)[self.keep]
        if np.any(weights <= 0):
            raise ValueError('Recorded rows require positive information')
        self.count = len(bins)
        total = np.bincount(self.index, weights=weights, minlength=self.count)
        if np.any(total == 0):
            raise ValueError('A retained bin has no original rows')
        self.weights = weights/total[self.index]
        np.testing.assert_allclose(self.mean(self.time), field.time, atol=2e-10, rtol=0.)

    def mean(self, values):
        values = np.asarray(values)
        if len(values) != len(self.index):
            raise ValueError('Average values at the retained original row epochs')
        flat = values.reshape(len(values), -1)
        result = np.column_stack([np.bincount(self.index, weights=self.weights*z.real,
            minlength=self.count)+(1j*np.bincount(self.index, weights=self.weights*z.imag,
            minlength=self.count) if np.iscomplexobj(values) else 0.) for z in flat.T])
        return result.reshape((self.count,)+values.shape[1:])

    def stationary(self, field, delays, origin=0.):
        seed_path = np.polynomial.polynomial.polyval(
            (self.time-field.seed['epoch'])/field.seed.get('scale', 15.), field.seed['path'])
        columns = []
        for delay in delays:
            lag = field.offsets[None, :]-(delay+origin-seed_path[:, None])*FS/C
            phase = np.exp(-1j*field.k*(delay-(seed_path-field.seed['path'][0])))
            columns.append(self.mean(phase[:, None]*field.pulse(lag)))
        return np.stack(columns, axis=-1)


class IntegratedField(StationaryField):
    def __init__(self, previous, binned):
        self.original, self.time, self.delays = previous.original, previous.time, previous.delays
        self.stationary = np.column_stack([self.original.apply(column).ravel()
            for column in np.moveaxis(binned, -1, 0)])
        u, s, vh = np.linalg.svd(self.stationary, full_matrices=False)
        keep = s > s[0]*previous.cutoff
        self.basis, self.inverse = u[:, keep], vh[keep].conj().T/s[keep]
        self.singular_values, self.cutoff = s, previous.cutoff
        self.y = self.remove(self.original.y)


class IntegratedRoad(StationaryRoad):
    def __init__(self, previous, binned):
        training = [IntegratedField(field, values) for field, values in zip(previous.training, binned)]
        validation = [IntegratedField(field, values) for field, values in zip(previous.validation, binned)]
        RoadTarget.__init__(self, previous.problem, training, validation, previous.initial,
                            previous.transmitters, previous.antennas)
        self.energy = previous.energy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('geometry-fit', 'ship-fit', 'road-fit', 'control', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--fields', type=Path, default=Path('build/isar-20261015/full-fields'))
    parser.add_argument('--geometry', type=Path, default=Path('docs/data/calibration_20261013'))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = args.output.with_suffix('.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    control, fitted, saved = [json.loads(p.read_text()) for p in (args.control, args.geometry_fit, args.road_fit)]
    if not control['complete']:
        raise ValueError('Finish the midpoint control first')
    for group in ('sources', 'implementation_sha256'):
        for source, digest in control['identity'][group].items():
            if fingerprint(Path(source)) != digest:
                raise ValueError('Midpoint control changed: '+source)
    for path in (args.geometry_fit, args.ship_fit, args.road_fit):
        if control['identity']['sources'].get(str(path)) != fingerprint(path):
            raise ValueError('The midpoint control uses different physical geometry or observations')
    joint, _, _, labels, _ = load_experiment(args.ship_fit, args.road_fit, args.fields,
        args.geometry, fitted['identity']['labels'], fitted['identity']['nodes'])
    antenna = np.asarray(fitted['parameters'])[joint.antenna_slice]
    sources = [args.control, args.geometry_fit, args.ship_fit, args.road_fit,
        *[Path(saved['input_patches'])/f"{t['label']}-{c}.npz" for t in control['targets'] for c in range(5)]]
    identity = {'sources': {str(p): fingerprint(p) for p in sources},
        'implementation_sha256': {str(p): fingerprint(p) for p in [Path(__file__),
            Path(__file__).with_name('stationary_road_control.py')]}}
    result = {'identity': identity, 'scope': __doc__, 'complete': False, 'targets': [],
        'accepted_calibration': False, 'production_changed': False,
        'remaining_approximations': 'Moving response is still evaluated at retained-bin mean time; stationary response uses original rows. Both use the same interval-averaged range PSF and declared empirical covariance.'}
    if args.output.exists():
        result = json.loads(args.output.read_text())
        if result['identity'] != identity:
            raise ValueError('Cannot resume changed integrated stationary controls')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    def save():
        temporary = args.output.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result, indent=2)+'\n')
        temporary.replace(args.output)
    for target in control['targets']:
        label = target['label']
        if any(t['label'] == label for t in result['targets']):
            continue
        previous = StationaryRoad(joint.roads[labels.index(label)], cutoff=control['identity']['cutoff'])
        binned, checks = [], []
        for channel, field in enumerate(previous.training):
            path = Path(saved['input_patches'])/f'{label}-{channel}.npz'
            with np.load(path) as raw:
                bins = RecordedBins(field.original, raw['row_time'], raw['row_information'])
                reconstructed = bins.mean(raw['row_value'][bins.keep])
                discrepancy = np.linalg.norm(reconstructed-field.original.data['value'])/np.linalg.norm(reconstructed)
                if discrepancy > 1e-10:
                    raise ValueError('Saved row means do not reproduce the fitted observations')
            values = bins.stationary(field.original, field.delays, previous.origins[channel])
            binned.append(values)
            current = IntegratedField(field, values)
            checks.append({'channel': channel, 'recorded_bin_relative_error': float(discrepancy),
                'stationary_dictionary_relative_midpoint_error': float(np.linalg.norm(current.stationary-field.stationary)/np.linalg.norm(current.stationary))})
        model = IntegratedRoad(previous, binned)
        old = target['with_stationary']
        initial = np.asarray(old['starts'][old['selected_start']]['parameters'])
        baseline = target['baseline']
        second = np.asarray(baseline['starts'][baseline['selected_start']]['parameters'])
        frozen = model.predict(initial, antenna)
        revised = fit_target(model, antenna, [initial, second], control['identity']['iterations'])
        def totals(records):
            return {key: float(sum(r[key] for r in records)) for key in
                ('prediction_error', 'moving_prediction_error_after_test_nuisance', 'stationary_only_projected_error')}
        item = {'label': label, 'operator_checks': checks, 'midpoint': totals(old['prediction']),
            'integrated_frozen_motion': totals(frozen), 'integrated_refit': totals(revised['prediction']),
            'fit': revised, 'selected_fit_stationary': revised['starts'][revised['selected_start']]['fit']['converged'],
            'comparison_scope': 'Full prediction errors use identical original held-out rows. Projected moving errors use each dictionary\'s declared stationary nuisance and are not the same observation subspace across midpoint/integrated cases.'}
        result['targets'].append(item)
        save()
        print(json.dumps({'label': label, 'stationary': item['selected_fit_stationary'],
            'midpoint_error': item['midpoint']['prediction_error'], 'integrated_error': item['integrated_refit']['prediction_error'],
            'seconds': time.monotonic()-started}), flush=True)
    result.update(complete=True, seconds=time.monotonic()-started)
    save()


if __name__ == '__main__':
    main()
