"""Transfer shared geometry to a completely excluded vehicle interval.

Geometry is frozen before fitting the excluded target's motion and independent
complex band responses on even one-second blocks. Odd blocks test prediction.
The same target, initial motion branches, priors and noise units are used at
nominal and transferred antenna positions. Road discovery used the recording:
this is conditional transfer, not independent confirmation of road identity.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from fit_physical_profile import solve
from fit_shared_target_geometry import load_experiment
from road_validation import related_intervals
from scene_dictionary import fingerprint


class RoadMotion:
    """Fit only a road target's motion; physical antenna coordinates stay fixed."""
    def __init__(self, road, antenna, initial):
        self.road = road
        self.antenna = np.array(antenna, copy=True)
        self.initial = np.array(initial, copy=True)
        self.last, self.evaluations = None, 0

    def evaluate(self, parameters):
        if self.last is not None and np.array_equal(parameters, self.last['parameters']):
            return self.last
        result = self.road.evaluate(parameters, self.antenna)
        n = self.road.nmotion
        self.evaluations += 1
        self.last = {'parameters': np.array(parameters, copy=True),
            'value': result['value']/self.road.energy,
            'gradient': result['gradient'][:n]/self.road.energy,
            'gauss_newton': result['gauss_newton'][:n, :n]/self.road.energy,
            'scene_conditions': [], 'evaluations': self.evaluations}
        return self.last

    def objective(self, parameters):
        result = self.evaluate(parameters)
        return result['value'], result['gradient']


def fit_target(road, antenna, starts, iterations):
    lower, upper = road.problem.model.bounds()
    candidates = []
    for initial in starts:
        problem = RoadMotion(road, antenna, np.clip(initial, lower, upper))
        parameters, fit = solve(problem, np.zeros((0, road.nmotion)), np.array([]),
            np.array([]), lower, upper, iterations=iterations)
        candidates.append({'parameters': parameters.tolist(), 'fit': fit})
    selected = min(range(len(candidates)), key=lambda k: candidates[k]['fit']['objective'])
    parameters = np.asarray(candidates[selected]['parameters'])
    return {'selected_start': selected, 'starts': candidates,
            'prediction': road.predict(parameters, antenna)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('calibration', 'ship-fit', 'road-fit', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--labels', nargs='+', required=True)
    parser.add_argument('--fields', type=Path, default=Path('build/isar-20261015/full-fields'))
    parser.add_argument('--geometry', type=Path, default=Path('docs/data/calibration_20261013'))
    parser.add_argument('--iterations', type=int, default=180)
    args = parser.parse_args()
    started = time.monotonic()
    calibration, saved = [json.loads(p.read_text()) for p in (args.calibration, args.road_fit)]
    if not calibration['complete'] or not calibration['fit']['converged']:
        raise ValueError('Finish the excluded-target geometry fit before testing its transfer')
    if (calibration['original_row_maximum_kkt_residual'] > 1e-7
            or calibration['original_row_constraint_violation'] > 1e-7):
        raise ValueError('Calibration has not passed original-row stationarity and feasibility checks')
    identity = calibration['identity']
    for group in ('sources', 'implementation_sha256'):
        for path, expected in identity[group].items():
            if fingerprint(Path(path)) != expected:
                raise ValueError(f'Calibration input changed: {path}')
    actual_sources = [args.ship_fit, args.road_fit,
        *[args.fields/f'channel-{c}.npz' for c in range(5)],
        *[args.geometry/name for name in ('geometry-prior.json', 'sites-main.json', 'acma-transmitters.json')]]
    for path in actual_sources:
        if identity['sources'].get(str(path)) != fingerprint(path):
            raise ValueError('The calibration and transfer test use different source fits')
    labels = list(dict.fromkeys([*identity['labels'], *args.labels]))
    joint, _, _, selected, dropped = load_experiment(args.ship_fit, args.road_fit,
        args.fields, args.geometry, labels, identity['nodes'])
    if dropped or selected != labels:
        raise ValueError('Resolve overlapping target groups before a whole-target transfer test')
    if joint.ship.antennas.channel_sites.tolist() != identity['site_assignment']:
        raise ValueError('The calibration and transfer test use different licensed assignments')
    roads = dict(zip(selected, joint.roads))
    for label in args.labels:
        if label in identity['labels'] or any(related_intervals(roads[label].problem, roads[old].problem)
                                              for old in identity['labels']):
            raise ValueError(f'{label} or a possibly overlapping interval was used in calibration')
    antenna = np.asarray(calibration['parameters'])[joint.antenna_slice]
    result = {'scope': __doc__, 'complete': False, 'accepted_calibration': False,
        'calibration': str(args.calibration), 'calibration_sha256': fingerprint(args.calibration),
        'calibration_labels': identity['labels'], 'withheld_labels': args.labels,
        'source_sha256': {str(p): fingerprint(p) for p in [args.ship_fit, args.road_fit,
            *[Path(saved['input_histories'])/(label+'.json') for label in args.labels],
            *[Path(saved['input_patches'])/f'{label}-{c}.npz' for label in args.labels for c in range(5)]]},
        'implementation_sha256': {str(Path(__file__)): fingerprint(Path(__file__))},
        'targets': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for label in args.labels:
        road = roads[label]
        index = saved['labels'].index(label)
        # Both controls start in the same two previously discovered local road
        # branches. Neither the transferred geometry nor validation score
        # chooses an initial branch or the winning training solution.
        starts = [np.asarray(saved['selected_roads'][index]['state']), road.initial]
        nominal = fit_target(road, np.zeros_like(antenna), starts, args.iterations)
        transferred = fit_target(road, antenna, starts, args.iterations)
        a, b = nominal['prediction'], transferred['prediction']
        error_a, error_b = [sum(c['prediction_error'] for c in case) for case in (a, b)]
        energy_a, energy_b = [sum(c['matched_energy'] for c in case) for case in (a, b)]
        summary = {'prediction_error_reduction_percent': float(100*(1-error_b/error_a)),
            'matched_energy_change_db': float(10*np.log10(energy_b/energy_a)),
            'channel_error_reduction_percent': [float(100*(1-y['prediction_error']/x['prediction_error']))
                for x, y in zip(a, b)],
            'selected_fits_stationary': [case['starts'][case['selected_start']]['fit']['converged']
                                        for case in (nominal, transferred)]}
        result['targets'].append({'label': label, 'nominal': nominal,
                                  'transferred': transferred, 'summary': summary})
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({'label': label, **summary}), flush=True)
    result.update(complete=True, seconds=time.monotonic()-started)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
