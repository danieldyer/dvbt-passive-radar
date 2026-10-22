"""Select shared physical points on training data; report blocked predictions.

The requested support counts are declared before fitting. Held-out samples
never choose locations or coefficients. This is a conditional scene diagnostic
at the existing motion, not a calibrated physical hull or channel phase fit.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from joint_support import select
from linear_observation import LinearObservation
from scene_dictionary import fingerprint


def load(directory, channel, split):
    prefix = directory/f'channel-{channel}-{split}'
    with np.load(str(prefix)+'-data.npz') as data:
        return LinearObservation(np.load(str(prefix)+'-design.npy', mmap_mode='r'),
            data['data'], float(data['orthogonal_energy']), int(data['observations']))


def assess(train, test, support):
    channels = []
    for a, b in zip(train, test):
        design = a.design[:, support]
        coefficient, _, rank, singular = np.linalg.lstsq(design, a.data, rcond=1e-10)
        prediction_error = float(np.linalg.norm(b.data-b.design[:, support]@coefficient)**2+b.orthogonal_energy)
        channels.append({'rank': int(rank), 'condition': float(singular[0]/singular[-1]),
            'coefficient_real_imag': np.c_[coefficient.real, coefficient.imag].tolist(),
            'training_error': float(np.linalg.norm(a.data-design@coefficient)**2+a.orthogonal_energy),
            'prediction_error': prediction_error, 'baseline_error': b.energy,
            'explained_fraction': 1-prediction_error/b.energy})
    return {'count': len(support), 'support_indices': support, 'channels': channels,
        'heldout_explained_fraction': 1-sum(c['prediction_error'] for c in channels)/sum(c['baseline_error'] for c in channels)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--statistics', type=Path, required=True)
    parser.add_argument('--counts', type=int, nargs='+', default=[8, 16, 32, 64, 128])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source_path = args.statistics/'fit.json'
    source = json.loads(source_path.read_text())
    if not source['complete']:
        raise ValueError('Finish the dense scene reference before testing sparse support')
    counts = sorted(set(args.counts))
    if not counts or counts[0] <= 0:
        raise ValueError('Need positive requested support counts')
    train = [load(args.statistics, c, 'train') for c in range(5)]
    test = [load(args.statistics, c, 'test') for c in range(5)]
    result = {'source_statistics': str(args.statistics), 'source_sha256': fingerprint(source_path),
        'source_motion': source['source_motion'], 'requested_counts': counts,
        'holdout_seconds': source['holdout_seconds'], 'scope': __doc__, 'complete': False,
        'points_body_m': source['points_body_m'], 'trials': [], 'steps': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for support, step in select([o.design for o in train], [o.data for o in train], counts[-1]):
        result['steps'].append(step)
        if len(support) not in counts:
            continue
        trial = assess(train, test, support)
        trial['points_body_m'] = np.array(source['points_body_m'])[support].tolist()
        result['trials'].append(trial)
        result['seconds'] = time.perf_counter()-started
        result['complete'] = len(support) == counts[-1]
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({'count': len(support), 'heldout_explained_fraction': trial['heldout_explained_fraction'],
            'seconds': result['seconds']}), flush=True)
    if not result['complete']:
        raise ValueError('Dictionary exhausted before the declared maximum support')


if __name__ == '__main__':
    main()
