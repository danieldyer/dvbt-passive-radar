"""Fit and validate a shared band-phase hypothesis on archived vehicle arcs."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from common_road_phase import CommonRoadGeometry
from fit_road_arc_raw import describe, load_problem, solve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--independent', type=Path, required=True)
    parser.add_argument('--summary', type=Path, default=Path('docs/data/isar_20261017/road-arc-geometry/fits.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--starts', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=180)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'fits.json').exists():
        raise ValueError('Use a fresh output for controlled phase-hypothesis trials')
    start = time.monotonic(); old = json.loads(args.independent.read_text())
    independent, associations, sources = load_problem(args.summary, old['assignment'], old['labels'],
        free_antennas=old['free_antennas'], free_origins=old['free_origins'], arc_index=old.get('arc_index', 0),
        span_arcs=old.get('span_arcs', False))
    refined = solve(independent, np.array(old['parameters']), search_motion=True)
    base = refined.x; model = CommonRoadGeometry(independent, base)
    # Audit full geometry + stationary/amplitude profiling + shared phases.
    point = model.initial; residual, jacobian = model.evaluate(point)
    rng = np.random.default_rng(45); checks = []
    for _ in range(3):
        direction = rng.normal(size=len(point)); direction /= np.linalg.norm(direction)
        delta = direction*1e-5
        numeric = (model.evaluate(point+delta)[0]-model.evaluate(point-delta)[0])/2e-5
        checks.append(float(np.linalg.norm(numeric-jacobian@direction)/np.linalg.norm(numeric)))
    if max(checks) > 2e-5:
        raise ArithmeticError(f'Full common-phase Jacobian audit failed: {checks}')
    result = {'scope': __doc__, 'complete': False, 'accepted_calibration': False,
        'assignment': old['assignment'], 'labels': old['labels'], 'associations': associations,
        'independent': str(args.independent), 'jacobian_relative_errors': checks,
        'independent_refined': describe(independent, refined),
        'phase_model': model.phases(model.initial)['scope'], 'trials': []}
    sources.extend([args.independent, Path(__file__), Path(__file__).with_name('common_road_phase.py'),
        Path(__file__).with_name('road_arc_raw.py'), Path(__file__).with_name('fit_road_arc_raw.py')])
    result['source_sha256'] = {str(p): hashlib.file_digest(p.open('rb'), 'sha256').hexdigest() for p in sources}
    def save():
        (args.output/'fits.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print('phase seeds', len(model.initial_phase_solutions), 'Jacobian errors', checks, flush=True)
    for index, (_, phases) in enumerate(model.initial_phase_solutions[:args.starts]):
        initial = np.r_[base, phases]
        fitted = solve(model, initial, iterations=args.iterations)
        row = describe(model, fitted); row.update(model.phases(fitted.x))
        row['start'] = index; result['trials'].append(row)
        result['seconds'] = time.monotonic()-start; save()
        print('common phase', index, 'cost', round(fitted.cost, 4), 'optimality', round(fitted.optimality, 5),
            'evaluations', fitted.nfev, 'seconds', round(result['seconds'], 2), flush=True)
    result['best_training_trial'] = int(np.argmin([r['cost'] for r in result['trials']]))
    result['complete'] = True; save()


if __name__ == '__main__':
    main()
