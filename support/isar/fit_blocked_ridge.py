"""Nested temporal validation of the fixed-motion distributed ship scene.

Outer odd one-second blocks are excluded from both fitting and penalty choice.
Inner training uses blocks 0,4,8,...; validation uses 2,6,10,... . A single
absolute ridge penalty per band is then used to refit all even blocks. Physical
motion and original support decisions remain conditional on this aperture.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from blocked_ridge import RidgePath, tune
from fit_joint_support import load
from fit_gpga import binned_metrics
from linear_observation import LinearObservation
from matched_phase import history
from scene_dictionary import fingerprint
from stationary import Stationary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--statistics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source_path = args.statistics/'fit.json'
    source = json.loads(source_path.read_text())
    if not source['complete'] or source['holdout_seconds'] != 1.:
        raise ValueError('Need complete one-second-block statistics')
    dictionary = Path(source['dictionary'])
    manifest = json.loads((dictionary/'dictionary.json').read_text())
    points = np.array(source['identity']['point_indices'])
    args.output.mkdir(parents=True, exist_ok=True)
    result = {'source_statistics': str(args.statistics), 'source_sha256': fingerprint(source_path),
        'scope': __doc__, 'source_motion': source['source_motion'], 'channels': [], 'complete': False,
        'limitation': 'New nested validation only selects the scene penalty. It does not independently validate previously fitted motion, absolute heading or transmitter locations.'}
    started = time.perf_counter()
    for c in range(5):
        with np.load(dictionary/f'channel-{c}-observation.npz') as f:
            observation = dict(f)
        matrix = np.load(dictionary/f'channel-{c}-design.npy', mmap_mode='r')
        if fingerprint(dictionary/f'channel-{c}-design.npy') != manifest['channels'][c]['design_sha256']:
            raise ValueError('Source response dictionary has changed')
        block = np.floor(observation['time']-manifest['interval_s'][0]).astype(int)
        inner = []
        for phase in (0, 2):
            rows = np.repeat(block%4 == phase, observation['data'].shape[1])
            nuisance = Stationary(observation['background'], rows)
            a = nuisance.remove(matrix[np.ix_(rows, points)])
            y = nuisance.remove(observation['data'].ravel()[rows])
            inner.append(LinearObservation.compress(a, y, nuisance.rank))
            del a, y, nuisance
        strength, tuning = tune(*inner, np.logspace(-12, 2, 57))
        del inner
        print(json.dumps({'channel': c, 'stage': 'Inner penalty selected', 'relative_penalty': tuning['relative_strengths'][tuning['chosen_index']],
            'seconds': time.perf_counter()-started}), flush=True)
        training, testing = [load(args.statistics, c, split) for split in ('train', 'test')]
        path = RidgePath(training.design, training.data)
        coefficient = path.coefficients([strength])[:, 0]
        prediction_error = testing.error(coefficient)
        full = np.zeros(matrix.shape[1], complex)
        full[points] = coefficient
        prediction = (matrix@full).reshape(observation['data'].shape)
        selected = block%2 == 0
        rows = np.repeat(~selected, observation['data'].shape[1])
        nuisance = Stationary(observation['background'], rows)
        y = nuisance.remove(observation['data'].ravel()[rows]).reshape((~selected).sum(), -1)
        p = nuisance.remove(prediction.ravel()[rows]).reshape(y.shape)
        relative_error = abs(float(np.linalg.norm(y-p)**2)/prediction_error-1)
        if relative_error > 2e-8:
            raise ValueError('Original-row and QR prediction scores disagree')
        z, weight = history(y, p)
        metrics = binned_metrics(observation['time'][~selected], z, weight, np.ones(len(z), bool), np.ones(1, complex), 1.)
        result['channels'].append({'channel': c, 'tuning': tuning, 'absolute_penalty': strength,
            'final_relative_penalty': float(strength/path.singular[0]**2),
            'training_error': training.error(coefficient), 'prediction_error': prediction_error,
            'baseline_error': testing.energy, 'heldout_explained_fraction': 1-prediction_error/testing.energy,
            'one_second_phase_metrics': metrics, 'regenerated_prediction_error_relative_check': relative_error,
            'coefficient_real_imag': np.c_[coefficient.real, coefficient.imag].tolist()})
        np.savez(args.output/f'channel-{c}-prediction.npz', time=observation['time'], prediction=prediction, selected=selected)
        result['complete'] = c == 4
        result['seconds'] = time.perf_counter()-started
        result['heldout_explained_fraction'] = 1-sum(v['prediction_error'] for v in result['channels'])/sum(v['baseline_error'] for v in result['channels'])
        (args.output/'fit.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({'channel': c, 'heldout_explained_fraction': result['channels'][-1]['heldout_explained_fraction'],
            'phase_rms_degrees': metrics['phase_rms_degrees_by_point'][0], 'seconds': result['seconds']}), flush=True)
        del matrix, observation, path, training, testing


if __name__ == '__main__':
    main()
