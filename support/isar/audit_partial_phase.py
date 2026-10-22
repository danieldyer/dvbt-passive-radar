"""Test common reflector phases without imposing them on all ship scattering.

Rank anchor candidates on full-aperture training data only, by their weakest-
band conditional information. Declared anchor counts are evaluated on untouched
time blocks. First/second halves use the same selected anchors and test whether
calibration is stable; those halves remain conditional on anchor discovery.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from partial_phase import common_visibility, fit
from scene_dictionary import fingerprint
from stationary import Stationary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dictionary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--counts', type=int, nargs='+', default=[2, 4, 8, 16, 32])
    args = parser.parse_args()
    source_path = args.dictionary/'fit.json'
    source = json.loads(source_path.read_text())
    if not source['complete']:
        raise ValueError('Finish the exact licensed-site dictionary')
    selected_sites = ['12013', '12013', '12013', '12013', '12014']
    result = {'scope': __doc__, 'dictionary': str(args.dictionary), 'source_sha256': fingerprint(source_path),
        'sites': selected_sites, 'complete': False, 'correction_adopted': False, 'trials': []}
    ranking = None
    for label, interval in [('full', [142., 172.]), ('first', [142., 157.]), ('second', [157., 172.])]:
        training, testing, observations, testing_data, outside = [], [], [], [], []
        for c, channel in enumerate(source['channels']):
            with np.load(args.dictionary/f'channel-{c}-observation.npz') as loaded:
                data, time, background = loaded['data'], loaded['time'], loaded['background']
            keep = (time >= interval[0])&(time < interval[1])
            row_keep = np.repeat(keep, data.shape[1])
            data, time, background = data[keep], time[keep], background[row_keep]
            block = np.floor(time-interval[0]).astype(int)
            selected = np.repeat((block%2 == 0)|(block == block.max()), data.shape[1])
            option = next(o for o in channel['options'] if o['site_id'] == selected_sites[c])
            target = np.load(args.dictionary/option['design'], mmap_mode='r')[row_keep]
            train_nuisance, test_nuisance = Stationary(background, selected), Stationary(background, ~selected)
            design, value = train_nuisance.remove(target[selected]), train_nuisance.remove(data.ravel()[selected])
            q, r = np.linalg.qr(design, mode='reduced')
            rhs = q.conj().T@value
            training.append(r); observations.append(rhs)
            outside.append(float(np.linalg.norm(value-q@rhs)**2))
            testing.append(test_nuisance.remove(target[~selected]))
            testing_data.append(test_nuisance.remove(data.ravel()[~selected]))
        if ranking is None:
            ranking, visibility = common_visibility(training, observations)
            result.update(anchor_ranking=ranking.tolist(), conditional_information_by_channel=visibility.tolist())
        independent = [np.linalg.solve(a, y) for a, y in zip(training, observations)]
        independent_error = sum(np.linalg.norm(y-a@coefficient)**2 for a, y, coefficient in zip(testing, testing_data, independent))
        for count in args.counts:
            coefficients, details = fit(training, observations, ranking[:count], reference=3)
            error = float(sum(np.linalg.norm(y-a@coefficient)**2 for a, y, coefficient in zip(testing, testing_data, coefficients)))
            train_error = float(sum(np.linalg.norm(y-a@coefficient)**2+orthogonal for a, y, coefficient, orthogonal in
                zip(training, observations, coefficients, outside)))
            trial = {'interval': label, 'interval_s': interval, 'count': count, 'fit': details,
                'training_error': train_error, 'prediction_error': error,
                'independent_prediction_error': float(independent_error),
                'prediction_error_penalty_percent': 100*(error/independent_error-1)}
            result['trials'].append(trial)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps({'interval': label, 'count': count, 'converged': details['converged'],
                'penalty_percent': trial['prediction_error_penalty_percent'],
                'phase_degrees': np.degrees(details['channel_phase_radians']).tolist()}), flush=True)
    result['complete'] = True
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
