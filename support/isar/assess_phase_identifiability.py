"""Check channel phase sensitivity to the declared distributed-scene prior.

Uses saved QR statistics. No data are renormalized, and no parameter is chosen
by this test's held-out scores. All trials keep the physical motion fixed.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from linear_observation import LinearObservation
from phase_registration import PhaseRegistration


def load_observation(path):
    with np.load(path) as f:
        return LinearObservation(f['design'], f['data'], float(f['orthogonal_energy']), int(f['observations']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--multipliers', nargs='+', type=float, default=[.01, .1, 1., 10., 100.])
    args = parser.parse_args()
    scene = json.loads(args.scene.read_text())
    if not scene.get('statistics_directory'):
        raise ValueError('Scene result must contain its saved statistics path')
    directory = Path(scene['statistics_directory'])
    train = [load_observation(directory/f'channel-{c}-train.npz') for c in range(5)]
    test = [load_observation(directory/f'channel-{c}-test.npz') for c in range(5)]
    strength = scene['trials'][0]['channels'][0]['regularization']['regularization_strength']
    started = time.perf_counter()
    model = PhaseRegistration([o.design for o in train], [o.data for o in train], strength,
        orthogonal_energy=sum(o.orthogonal_energy for o in train), retain_basis=True)
    baseline = sum(o.energy for o in test)
    result = {'source_scene': str(args.scene), 'scope': __doc__, 'training_gcv_strength': strength,
              'trials': [], 'heldout_stationary_only_energy': baseline}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({'stage': 'Exact SVD ready', 'seconds': time.perf_counter()-started}), flush=True)
    for multiplier in args.multipliers:
        model.regularize(strength*multiplier)
        phases, coefficient, detail = model.fit()
        error = sum(o.error(coefficient, p) for o, p in zip(test, phases))
        no_phase, original, _ = model.project(np.zeros(4))
        original_error = sum(o.error(original, p) for o, p in zip(test, no_phase))
        detail.update(strength_multiplier=multiplier, heldout_prediction_error=error,
            heldout_explained_fraction=1-error/baseline,
            fixed_phase_heldout_explained_fraction=1-original_error/baseline,
            coefficient_norm=float(np.linalg.norm(coefficient)))
        if multiplier == 1.:
            detail['data_information'] = model.phase_information(coefficient)
        result['trials'].append(detail)
        result['seconds'] = time.perf_counter()-started
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k:detail[k] for k in ('strength_multiplier', 'heldout_explained_fraction',
            'relative_phase_degrees', 'maximum_gradient')}|{'seconds':result['seconds']}), flush=True)


if __name__ == '__main__':
    main()
