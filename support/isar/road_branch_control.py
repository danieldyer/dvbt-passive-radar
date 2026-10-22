"""Reconsider road associations under the current independent-band likelihood.

The source road search supplies alternatives and starting points only. Every
candidate is refitted to the same common-source observations and noise weights.
Its five complex scattering coefficients are independent, as in the shared
road/ship geometry experiment. Antenna coordinates are frozen for this control.
Road discovery used this recording; prediction remains conditional validation.
"""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import time

import numpy as np

from fit_shared_target_geometry import load_experiment
from road_geometry import RoadCurve
from road_model import RoadTrajectory
from scene_dictionary import fingerprint
from shared_target_geometry import RoadTarget
from validate_shared_geometry import fit_target


def branch(source, choice, routes):
    """Replace only a target's connected road and its trajectory coordinates."""
    problem = copy.copy(source.problem)
    old = problem.model
    problem.model = RoadTrajectory(old.seed, RoadCurve(routes[choice['route']]),
        choice['coarse_state'], old.channel_sites, old.channel_offsets, controls=old.radial_count)
    problem.model.receiver_enu_m = old.receiver_enu_m.copy()
    problem.model.receiver_channel_offsets_enu_m = old.receiver_channel_offsets_enu_m.copy()
    return RoadTarget(problem, source.training, source.validation, choice['state'],
                      source.transmitters, source.antennas)


def rebase_motion(parameters, previous_coarse, model):
    """Change a cubic along-road origin without changing the physical curve."""
    t = np.linspace(-1., 1., max(41, 3*model.radial_count))
    difference = (np.polynomial.polynomial.polyval(t, previous_coarse)
                  -np.polynomial.polynomial.polyval(t, model.coarse))
    basis = model.spline(t)
    correction = np.linalg.lstsq(basis, difference, rcond=None)[0]
    if np.max(abs(basis@correction-difference)) > 1e-8:
        raise ValueError('The new along-road basis cannot preserve the old physical curve')
    result = np.array(parameters, copy=True)
    result[:model.radial_count] += correction
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('geometry-fit', 'ship-fit', 'road-fit', 'trials', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--labels', nargs='+')
    parser.add_argument('--iterations', type=int, default=180)
    parser.add_argument('--fields', type=Path, default=Path('build/isar-20261015/full-fields'))
    parser.add_argument('--geometry', type=Path, default=Path('docs/data/calibration_20261013'))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = args.output.with_suffix('.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fitted, saved = [json.loads(p.read_text()) for p in (args.geometry_fit, args.road_fit)]
    if not fitted['complete'] or not fitted['fit']['converged']:
        raise ValueError('Use a completed shared-geometry fit or matched fixed-site control')
    for key in ('sources', 'implementation_sha256'):
        for path, expected in fitted['identity'][key].items():
            if fingerprint(Path(path)) != expected:
                raise ValueError('Source or implementation changed: '+path)
    for path in (args.ship_fit, args.road_fit):
        if fitted['identity']['sources'].get(str(path)) != fingerprint(path):
            raise ValueError('The supplied initial geometry belongs to different targets')
    labels = args.labels or fitted['identity']['labels']
    if set(labels)-set(fitted['identity']['labels']):
        raise ValueError('A requested road was not in the geometry experiment')
    joint, _, _, selected, dropped = load_experiment(args.ship_fit, args.road_fit,
        args.fields, args.geometry, fitted['identity']['labels'], fitted['identity']['nodes'])
    if selected != fitted['identity']['labels'] or dropped:
        raise ValueError('Source target grouping changed')
    antenna = np.asarray(fitted['parameters'])[joint.antenna_slice]
    parameters = np.asarray(fitted['parameters'])
    routes = json.loads(Path(saved['input_routes']).read_text())['routes']
    # Historical trial codes use Ornata/Eyre/Ridge indices, unlike site IDs.
    code = ''.join(map(str, saved['assignment']))
    files = [args.geometry_fit, args.ship_fit, args.road_fit, Path(saved['input_routes']),
             *[args.trials/f'{code}-{label}.json' for label in labels]]
    implementation = [Path(__file__), Path(__file__).with_name('validate_shared_geometry.py'),
        Path(__file__).with_name('shared_target_geometry.py'), Path(__file__).with_name('fit_physical_profile.py'),
        Path(__file__).resolve().parents[1]/'calibration'/'road_model.py',
        Path(__file__).resolve().parents[1]/'calibration'/'road_geometry.py']
    identity = {'sources': {str(p): fingerprint(p) for p in files}, 'labels': labels,
        'iterations': args.iterations, 'frozen_antenna_parameters': antenna.tolist(),
        'implementation_sha256': {str(p): fingerprint(p) for p in implementation}}
    result = {'identity': identity, 'scope': __doc__, 'complete': False, 'targets': [],
        'accepted_calibration': False, 'selection': 'Lowest training objective, including declared motion/road priors. Withheld prediction never selects the road or starting point.',
        'limitations': 'Finite previously discovered alternatives and local starts, not a global road search. Antenna geometry is held fixed; road identities are not independently observed.'}
    if args.output.exists():
        result = json.loads(args.output.read_text())
        if result['identity'] != identity:
            raise ValueError('Cannot resume a changed road-association control')
    def save():
        temporary = args.output.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result, indent=2)+'\n')
        temporary.replace(args.output)
    started = time.monotonic()
    for label in labels:
        records = next((t for t in result['targets'] if t['label'] == label), None)
        if records is not None and records['complete']:
            continue
        source_index, old_index = selected.index(label), saved['labels'].index(label)
        source = joint.roads[source_index]
        previous = saved['selected_roads'][old_index]
        choices = {c['route']: c for c in json.loads((args.trials/f'{code}-{label}.json').read_text())['trials']}
        # Keep the current branch's exact parameterization and warm start.
        choices[previous['route']] = previous
        if records is None:
            records = {'label': label, 'previous_route': previous['route'], 'previous_name': previous['name'],
                       'cases': [], 'complete': False}
            result['targets'].append(records)
        for choice in choices.values():
            if any(c['route'] == choice['route'] for c in records['cases']):
                continue
            target = branch(source, choice, routes)
            starts = [np.asarray(choice['state'])]
            if choice['route'] == previous['route']:
                starts.append(rebase_motion(parameters[joint.road_slices[source_index]],
                                           source.problem.model.coarse, target.problem.model))
            else:
                # The existing independent road start and a zero residual
                # curve are two declared local basins; neither uses test scores.
                starts.append(np.zeros(target.nmotion))
            trial = fit_target(target, antenna, starts, args.iterations)
            best = trial['starts'][trial['selected_start']]
            p = np.asarray(best['parameters'])
            prediction = trial['prediction']
            item = {'route': choice['route'], 'name': choice['name'],
                'coarse_state': choice['coarse_state'], 'state': p.tolist(), 'fit': trial,
                'training_objective': best['fit']['objective'], 'stationary': best['fit']['converged'],
                'prediction_error': sum(c['prediction_error'] for c in prediction),
                'matched_energy': sum(c['matched_energy'] for c in prediction),
                'midpoint_enu_m': target.problem.model.positions(p,
                    np.array([target.problem.model.seed['epoch']]), target.problem.sites)[0][0].tolist()}
            records['cases'].append(item)
            save()
            print(json.dumps({'label': label, 'road': choice['name'], 'route': choice['route'],
                'stationary': item['stationary'], 'objective': item['training_objective'],
                'seconds': time.monotonic()-started}), flush=True)
        winner = min(records['cases'], key=lambda c: c['training_objective'])
        control = next(c for c in records['cases'] if c['route'] == previous['route'])
        records.update(complete=True, selected_route=winner['route'], selected_name=winner['name'],
            selected_stationary=winner['stationary'], road_changed=winner['route'] != previous['route'],
            prediction_error_reduction_percent=100*(1-winner['prediction_error']/control['prediction_error']),
            matched_energy_change_db=float(10*np.log10(winner['matched_energy']/control['matched_energy'])))
        save()
        print(json.dumps({'target_complete': label, 'selected_road': winner['name'],
            'changed': records['road_changed'], 'prediction_error_reduction_percent': records['prediction_error_reduction_percent']}), flush=True)
    result.update(complete=True, seconds=time.monotonic()-started)
    save()


if __name__ == '__main__':
    main()
