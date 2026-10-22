"""Leave one entire vehicle out of antenna, range-origin and band-phase fitting.

The held-out vehicle's position/motion and target amplitude/phase are fitted on
its declared training rows. Antennas, envelope origins and (where applicable)
interband phases are transferred unchanged from the other vehicles. Predictions
use untouched held-out rows, including every band and weak/failed hypotheses.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from common_road_phase import CommonRoadGeometry
from fit_road_arc_raw import describe, load_problem, solve


def common_fit(independent, fit, phases=None):
    model = CommonRoadGeometry(independent, fit.x, fixed_band_phases=phases)
    trials = [solve(model, np.r_[fit.x, p], iterations=180)
              for _, p in model.initial_phase_solutions[:4]]
    fitted = min(trials, key=lambda trial: trial.cost)
    result = describe(model, fitted); result.update(model.phases(fitted.x))
    result['phase_start_costs'] = [float(t.cost) for t in trials]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=Path('docs/data/isar_20261017/road-arc-geometry/fits.json'))
    parser.add_argument('--assignment', default='02002')
    parser.add_argument('--labels', nargs='+', default=['A', 'B', 'C', 'E', 'K', 'O', 'Q', 'R', 'S'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-independent', type=Path, help='Reuse completed independent controls; repeat the common-phase search')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'validation.json').exists():
        raise ValueError('Use a fresh output for leave-one-target-out validation')
    started = time.monotonic(); assignment = list(map(int, args.assignment))
    result = {'scope': __doc__, 'complete': False, 'accepted_calibration': False,
        'assignment': assignment, 'labels': args.labels, 'trials': []}
    previous = None
    if args.reuse_independent:
        previous = json.loads(args.reuse_independent.read_text())
        if not previous['complete'] or previous['assignment'] != assignment or previous['labels'] != args.labels:
            raise ValueError('Reused independent controls must be complete and use the same hypotheses')
        previous = {row['held_target']: row for row in previous['trials']}
        result['reused_independent_controls'] = str(args.reuse_independent)
    sources = set([Path(__file__), Path(__file__).with_name('fit_road_arc_raw.py'),
        Path(__file__).with_name('common_road_phase.py'), Path(__file__).with_name('road_arc_raw.py')])
    if args.reuse_independent:
        sources.add(args.reuse_independent)
    for held in args.labels:
        labels = [label for label in args.labels if label != held]
        model, associations, paths = load_problem(args.summary, assignment, labels, free_antennas=True, free_origins=True)
        sources.update(paths)
        if previous is None:
            fitted = solve(model, search_motion=True); independent = describe(model, fitted)
        else:
            independent = previous[held]['independent_calibration']
            fitted = SimpleNamespace(x=np.array(independent['parameters']))
        independent.update(assignment=assignment, labels=labels)
        common = common_fit(model, fitted); common.update(assignment=assignment, labels=labels)
        row = {'held_target': held, 'independent_calibration': independent, 'common_calibration': common}
        for name, calibration in [('nominal', None), ('independent', independent), ('common', common)]:
            if previous is not None and name != 'common':
                row[name+'_held_target'] = previous[held][name+'_held_target']
                continue
            test, association, paths = load_problem(args.summary, assignment, [held], fixed_calibration=calibration)
            sources.update(paths)
            target_fit = solve(test, search_motion=True)
            prediction = (common_fit(test, target_fit, common['band_phase_rad']) if name == 'common'
                          else describe(test, target_fit))
            row[name+'_held_target'] = prediction
        result['trials'].append(row); result['seconds'] = time.monotonic()-started
        result['source_sha256'] = {str(p): hashlib.file_digest(p.open('rb'), 'sha256').hexdigest() for p in sorted(sources)}
        (args.output/'validation.json').write_text(json.dumps(result, indent=2, allow_nan=False))
        errors = {name: sum(c['error'] for c in row[name+'_held_target']['prediction'][0]['channels'])
                  for name in ('nominal', 'independent', 'common')}
        print('held', held, 'errors', {k: round(v, 2) for k, v in errors.items()},
              'seconds', round(result['seconds'], 2), flush=True)
    result['complete'] = True
    (args.output/'validation.json').write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
