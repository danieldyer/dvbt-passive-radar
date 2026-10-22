"""Check whether pairwise ship phase estimates form one transferable calibration.

Fit every pair independently; do not force the five-band model onto a pair.
The same fixed support, motion and site dictionaries are used throughout.
Constant transmitter phase offsets must have zero wrapped phase around every
triangle and must transfer between aperture halves. These are necessary, not
sufficient, conditions for treating a ship-derived phase as calibration.
"""
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from phase_coupling import PhaseCoupling
from scene_dictionary import fingerprint
from stationary import Stationary


def closures(phases):
    """phases[i,j] is phase of band j minus band i, in radians."""
    phases = np.asarray(phases)
    return [{'channels': list(triangle), 'phase_radians': float(np.angle(np.exp(1j*(
        phases[i, j]+phases[j, k]-phases[i, k]))))}
        for triangle in combinations(range(len(phases)), 3) for i, j, k in [triangle]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dictionary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--coordinate-sweeps', type=int, default=0)
    args = parser.parse_args()
    path = args.dictionary/'fit.json'
    source = json.loads(path.read_text())
    if not source['complete']:
        raise ValueError('Finish the exact licensed-site dictionary')
    sites = ['12013', '12013', '12013', '12013', '12014']
    identity = {'dictionary': str(args.dictionary), 'source_sha256': fingerprint(path), 'sites': sites}
    if args.coordinate_sweeps:
        identity['coordinate_sweeps'] = args.coordinate_sweeps
    result = {'identity': identity, 'scope': __doc__, 'complete': False,
        'correction_adopted': False, 'trials': [], 'closures': {}}
    if args.output.exists():
        result = json.loads(args.output.read_text())
        if result['identity'] != identity:
            raise ValueError('Cannot resume pair controls with changed input')
    for label, interval in [('full', [142., 172.]), ('first', [142., 157.]), ('second', [157., 172.])]:
        training, testing, observations, testing_data = [], [], [], []
        for c, channel in enumerate(source['channels']):
            with np.load(args.dictionary/f'channel-{c}-observation.npz') as loaded:
                data, time, background = loaded['data'], loaded['time'], loaded['background']
            keep = (time >= interval[0])&(time < interval[1])
            rows = np.repeat(keep, data.shape[1])
            data, time, background = data[keep], time[keep], background[rows]
            block = np.floor(time-interval[0]).astype(int)
            selected = np.repeat((block%2 == 0)|(block == block.max()), data.shape[1])
            option = next(o for o in channel['options'] if o['site_id'] == sites[c])
            target = np.load(args.dictionary/option['design'], mmap_mode='r')[rows]
            train, test = Stationary(background, selected), Stationary(background, ~selected)
            # Keep orthogonal training residuals in PhaseCoupling's exact QR
            # likelihood; they do not affect the fitted phase but set its scale.
            training.append(train.remove(target[selected]))
            observations.append(train.remove(data.ravel()[selected]))
            testing.append(test.remove(target[~selected]))
            testing_data.append(test.remove(data.ravel()[~selected]))
        for i, j in combinations(range(5), 2):
            if any(t['interval'] == label and t['channels'] == [i, j] for t in result['trials']):
                continue
            model = PhaseCoupling([training[i], training[j]], [observations[i], observations[j]], reference=0)
            coefficients, fit = model.fit(coordinate_sweeps=args.coordinate_sweeps)
            variants = {}
            for key, amplitudes in [('independent', model.independent), ('coupled', coefficients)]:
                errors = [float(np.linalg.norm(testing_data[c]-testing[c]@a)**2)
                    for c, a in zip((i, j), amplitudes)]
                variants[key] = {'channels': errors, 'total': sum(errors)}
            trial = {'interval': label, 'interval_s': interval, 'channels': [i, j], 'fit': fit,
                'prediction_error': variants,
                'prediction_error_penalty_percent': 100*(variants['coupled']['total']/variants['independent']['total']-1)}
            result['trials'].append(trial)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps({'interval': label, 'channels': [i, j], 'converged': fit['converged'],
                'phase_degrees': float(np.degrees(fit['channel_phase_radians'][1])),
                'penalty_percent': trial['prediction_error_penalty_percent']}), flush=True)
        phases = np.zeros((5, 5))
        for trial in result['trials']:
            if trial['interval'] == label:
                i, j = trial['channels']
                phases[i, j] = trial['fit']['channel_phase_radians'][1]
                phases[j, i] = -phases[i, j]
        result['closures'][label] = closures(phases)
    result['complete'] = True
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
