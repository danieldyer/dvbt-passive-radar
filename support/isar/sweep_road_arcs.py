"""Compare all 32 permitted primary/backup hypotheses on raw vehicle arcs.

Every hypothesis uses the same immutable samples, training/validation split,
covariance, physical priors and multi-start motion search. Road associations
remain explicit conditional hypotheses. No hypothesis is accepted by this
screen alone; geometric conditioning and target transfer are separate tests.
"""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import time

from fit_road_arc_raw import describe, load_problem, solve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=Path('docs/data/isar_20261017/road-arc-geometry/fits.json'))
    parser.add_argument('--labels', nargs='+', default=['A', 'B', 'C', 'E', 'K', 'O', 'Q', 'R', 'S'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'fits.json').exists():
        raise ValueError('Use a fresh output for the controlled 32-hypothesis screen')
    started = time.monotonic(); sources = {Path(__file__)}
    result = {'scope': __doc__, 'complete': False, 'accepted_calibration': False,
        'labels': args.labels, 'site_order': ['Ornata', 'Eyre', 'Ridge'], 'trials': []}
    for assignment in product((0, 1), (2, 0), (0, 1), (0, 1), (2, 0)):
        model, associations, paths = load_problem(args.summary, assignment, args.labels,
            free_antennas=True, free_origins=True)
        sources.update(paths)
        fitted = solve(model, search_motion=True)
        row = describe(model, fitted)
        row.update(assignment=list(assignment), associations=associations)
        result['trials'].append(row); result['seconds'] = time.monotonic()-started
        (args.output/'fits.json').write_text(json.dumps(result, indent=2, allow_nan=False))
        error = sum(c['error'] for t in row['prediction'] for c in t['channels'])
        print(''.join(map(str, assignment)), 'train', round(fitted.cost, 3), 'test', round(error, 3),
              'bounds', row['bound_indices'], 'seconds', round(result['seconds'], 2), flush=True)
    sources.update(Path(__file__).with_name(n) for n in
                   ['fit_road_arc_raw.py', 'road_arc_raw.py', 'road_arc_geometry.py', 'coherent_arc.py', 'road_arc.py'])
    result.update(complete=True, source_sha256={str(p): hashlib.file_digest(p.open('rb'), 'sha256').hexdigest() for p in sorted(sources)})
    (args.output/'fits.json').write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
