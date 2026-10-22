"""Matched vehicle-motion prediction before/after covariance-aware gating.

Sites, origins, roads, motion priors and Fourier bandwidth stay fixed. Each
operator fits training blocks only. Both predictions are scored on both
held-out operators, so a changed observation space cannot masquerade as gain.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from covariance_gate import whiten_first_field
from fit_shared_target_geometry import load_experiment
from scene_dictionary import fingerprint
from shared_origin_geometry import OriginRoad
from shared_origin_transfer import FrozenOriginRoad
from shared_target_geometry import RoadTarget
from validate_shared_geometry import fit_target


def transfer(fitted, scoring, motion, antenna, origins):
    result = []
    for c, field in enumerate(fitted.training):
        u = fitted.template(motion, antenna, origins, c, field)[0]
        amplitude = np.vdot(u, field.y)/np.vdot(u, u).real
        channel = {}
        for name, model in scoring.items():
            field = model.validation[c]
            v = model.template(motion, antenna, origins, c, field)[0]
            predicted = amplitude*v
            error = field.y-predicted
            norm = np.vdot(predicted, predicted).real
            channel[name] = {'prediction_error': float(np.vdot(error, error).real),
                'matched_energy': float(abs(np.vdot(predicted, field.y))**2/norm) if norm else 0.,
                'data_energy': float(np.vdot(field.y, field.y).real)}
        result.append(channel)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'ship-fit', 'road-fit', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--fields', type=Path, default=Path('build/isar-20261015/full-fields'))
    p.add_argument('--geometry', type=Path, default=Path('docs/data/calibration_20261013'))
    p.add_argument('--labels', nargs='+', default=['A', 'O', 'K', 'Q'])
    p.add_argument('--iterations', type=int, default=180)
    args = p.parse_args(); started = time.monotonic()
    if args.output.exists():
        raise ValueError('Use a new result file')
    source = json.loads(args.source.read_text()); identity = source['identity']
    if not source['complete'] or not source['fit']['converged'] or source['original_row_maximum_kkt_residual'] > 1e-7:
        raise ValueError('Require an original-row stationary source')
    inherited = identity['legacy_identity']
    for path in (args.ship_fit, args.road_fit):
        if fingerprint(path) != inherited['sources'][str(path)]:
            raise ValueError('Source fits differ')
    joint, _, _, labels, dropped = load_experiment(args.ship_fit, args.road_fit,
        args.fields, args.geometry, identity['labels'], inherited['nodes'])
    if dropped or any(label not in labels for label in args.labels):
        raise ValueError('Use known disjoint road hypotheses')
    parameters = np.asarray(source['parameters']); antenna = parameters[joint.antenna_slice]
    origins = np.asarray(source['range_origins_m'])
    paths = [args.source, args.ship_fit, args.road_fit, Path(__file__),
        *[Path(__file__).with_name(name) for name in ('covariance_gate.py', 'shared_origin_geometry.py',
          'shared_origin_transfer.py', 'validate_shared_geometry.py', 'fit_physical_profile.py')]]
    report = {'complete': False, 'scope': __doc__, 'targets': [],
        'source_sha256': {str(path): fingerprint(path) for path in paths},
        'accepted_calibration': False, 'production_changed': False,
        'limitation': 'Conditional on the same empirical covariance, road hypotheses, origins and interval-averaged pulse. Time-bin mean response is retained; its separately audited error was negligible at the initial fits. This is not antenna calibration or independent ground-truth motion validation.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for label in args.labels:
        index = labels.index(label); road = joint.roads[index]
        alternatives = {'legacy': road, 'whiten_first': RoadTarget(road.problem,
            [whiten_first_field(f) for f in road.training], [whiten_first_field(f) for f in road.validation],
            road.initial, road.transmitters, road.antennas)}
        models = {name: OriginRoad(value) for name, value in alternatives.items()}
        starts = [parameters[joint.road_slices[index]], road.initial]
        trials = []
        for name, candidate in alternatives.items():
            fit = fit_target(FrozenOriginRoad(candidate, origins), antenna, starts, args.iterations)
            selected = fit['starts'][fit['selected_start']]
            motion = np.asarray(selected['parameters'])
            prediction = transfer(models[name], models, motion, antenna, origins)
            trials.append({'operator': name, 'fit': fit, 'prediction': prediction,
                'selected_fit_stationary': selected['fit']['converged']})
            print(json.dumps({'label': label, 'operator': name,
                'stationary': selected['fit']['converged'], 'kkt': selected['fit']['maximum_kkt_residual'],
                'seconds': time.monotonic()-started}), flush=True)
        comparisons = {}
        for metric in models:
            errors = [sum(c[metric]['prediction_error'] for c in t['prediction']) for t in trials]
            energy = [sum(c[metric]['matched_energy'] for c in t['prediction']) for t in trials]
            comparisons[metric] = {'prediction_error_change_percent': 100*(errors[1]/errors[0]-1),
                'matched_energy_change_db': float(10*np.log10(energy[1]/energy[0]))}
        report['targets'].append({'label': label, 'trials': trials, 'comparison': comparisons})
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    report.update(complete=True, seconds=time.monotonic()-started,
        all_selected_fits_stationary=all(t['selected_fit_stationary'] for r in report['targets'] for t in r['trials']))
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
