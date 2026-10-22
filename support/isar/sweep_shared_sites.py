"""Run matched fixed/free vehicle-and-ship fits for every licensed assignment.

Each assignment uses identical vehicle hypotheses and its existing ship seed.
Ranking uses training objective only; withheld predictions remain diagnostics.
The fit is conditional on road associations and fixed reflector support.
Numerical convergence is not evidence of surveyed antenna locations.
"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from scene_dictionary import fingerprint


def verified(path, *, fixed):
    value = json.loads(path.read_text())
    if not value['complete'] or value['identity']['fixed_antennas'] != fixed:
        raise ValueError(f'Incomplete or wrong fixed/free result: {path}')
    for key in ('sources', 'implementation_sha256'):
        for source, digest in value['identity'][key].items():
            if fingerprint(Path(source)) != digest:
                raise ValueError(f'Changed input or implementation in {path}: {source}')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ship-fits', type=Path, required=True)
    parser.add_argument('--road-fit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse', nargs=3, action='append', default=[], metavar=('INDEX', 'FIXED', 'JOINT'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # A live process owns this experiment. A delayed tool observation must not
    # launch a duplicate sweep or overwrite its checkpoints.
    lock = (args.output/'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fits = sorted(args.ship_fits.glob('assignment-??.json'))
    if len(fits) != 48:
        raise ValueError('The declared licensed experiment contains exactly 48 assignments')
    assignment = {int(p.stem.rsplit('-', 1)[1]): p for p in fits}
    sites = {i: [c['site_id'] for c in json.loads(p.read_text())['site_assignment']]
             for i, p in assignment.items()}
    order = sorted(assignment, key=lambda i: (len(set(sites[i])) != 3, i))
    fitter = Path(__file__).with_name('fit_shared_target_geometry.py')
    identity = {'ship_sources': {str(p): fingerprint(p) for p in fits},
        'road_fit': str(args.road_fit), 'road_sha256': fingerprint(args.road_fit),
        'implementation': {str(p): fingerprint(p) for p in [Path(__file__), fitter]},
        'order': order, 'reuse': args.reuse}
    manifest = args.output/'experiment.json'
    result = {'identity': identity, 'scope': __doc__, 'cases': [], 'complete': False,
        'accepted_calibration': False, 'yaw_fitted': False,
        'selection_scope': 'Training penalized profile likelihood, not marginal model evidence or posterior site probabilities.',
        'road_scope': 'The same ten provisional road branches for every assignment; their discovery used the recording.',
        'order_scope': 'Three-tower configurations first, then all remaining configurations. No assignment is excluded.'}
    if manifest.exists():
        result = json.loads(manifest.read_text())
        if result['identity'] != identity:
            raise ValueError('Cannot resume a sweep with changed inputs, implementation or fit order')
    reused = {int(i): [Path(a), Path(b)] for i, a, b in args.reuse}
    started = time.monotonic()
    def save():
        result['complete'] = len(result['cases']) == len(order)
        eligible = [c for c in result['cases'] if c.get('both_stationary')]
        result['stationary_pairs'] = len(eligible)
        result['training_selected_among_completed'] = min(eligible, key=lambda c: c['joint_training_objective'])['index'] if eligible else None
        temporary = manifest.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result, indent=2)+'\n')
        temporary.replace(manifest)
    # Reused pairs are verified before running new jobs, and retain their exact
    # source paths and scientific identities rather than silently re-fitting.
    order = list(reused)+[i for i in order if i not in reused]
    save()
    for index in order:
        if any(c['index'] == index for c in result['cases']):
            continue
        paths = reused.get(index, [args.output/f'assignment-{index:02d}-{mode}.json' for mode in ('fixed', 'joint')])
        records, failed = [], None
        for fixed, path in zip((True, False), paths):
            if not path.exists():
                command = [sys.executable, '-u', str(fitter), '--ship-fit', str(assignment[index]),
                    '--road-fit', str(args.road_fit), '--output', str(path)]
                checkpoint = path.with_suffix('.checkpoint.json')
                if fixed:
                    command.append('--fixed-antennas')
                if checkpoint.exists():
                    command.extend(['--initial', str(checkpoint)])
                elif not fixed:
                    command.extend(['--initial', str(paths[0])])
                print(json.dumps({'stage': 'start', 'index': index, 'fixed': fixed,
                    'sites': sites[index], 'output': str(path)}), flush=True)
                with path.with_suffix('.log').open('a') as log:
                    returncode = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
                if returncode:
                    failed = {'stage': 'fixed' if fixed else 'joint', 'returncode': returncode}
                    break
            record = verified(path, fixed=fixed)
            if str(assignment[index]) not in record['identity']['sources'] or str(args.road_fit) not in record['identity']['sources']:
                raise ValueError('A reused result has different source motion or road hypotheses')
            records.append(record)
        entry = {'index': index, 'sites': sites[index], 'files': [str(p) for p in paths]}
        if failed:
            entry['failure'] = failed
        else:
            first, second = records
            a, b = [dict(r['identity']) for r in records]
            a.pop('fixed_antennas'); b.pop('fixed_antennas')
            if a != b:
                raise ValueError('Matched controls differ in more than antenna-coordinate fitting')
            entry.update(source_sha256={str(p): fingerprint(p) for p in paths},
                both_stationary=all(r['fit']['converged'] and r['original_row_maximum_kkt_residual'] < 1e-8 for r in records),
                original_row_kkt=[r['original_row_maximum_kkt_residual'] for r in records],
                fixed_training_objective=first['fit']['objective'], joint_training_objective=second['fit']['objective'],
                antenna_bound_hits=second['antenna_bound_hits'],
                ship_validation=[r['validation']['ship_summary'] for r in records],
                vehicle_validation_error=[sum(c['prediction_error'] for values in r['validation']['vehicles'].values() for c in values) for r in records])
        result['cases'].append(entry)
        save()
        print(json.dumps({'stage': 'pair_complete', 'index': index, 'completed': len(result['cases']),
            'stationary_pairs': result['stationary_pairs'], 'failure': failed,
            'elapsed_seconds': time.monotonic()-started}), flush=True)


if __name__ == '__main__':
    main()
