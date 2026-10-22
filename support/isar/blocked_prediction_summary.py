"""Separate interior interpolation, terminal extrapolation and clutter overlap."""
import argparse
import json
from pathlib import Path

import numpy as np

from fit_gpga import binned_metrics
from matched_phase import history
from stationary import Stationary


def assess(observation, prediction, selected):
    rows = np.repeat(selected, observation['data'].shape[1])
    background = Stationary(observation['background'], rows)
    y = background.remove(observation['data'].ravel()[rows]).reshape(selected.sum(), -1)
    p = background.remove(prediction.ravel()[rows]).reshape(y.shape)
    error, baseline = float(np.linalg.norm(y-p)**2), float(np.linalg.norm(y)**2)
    z, information = history(y, p)
    metrics = binned_metrics(observation['time'][selected], z, information,
        np.ones(selected.sum(), bool), np.ones(1, complex), 1.)
    product, energy = np.vdot(p, y), float(np.linalg.norm(p)**2)
    return {'prediction_error': error, 'baseline_error': baseline, 'explained_fraction': 1-error/baseline,
        'matched_energy': float(abs(product)**2/energy), 'phase_rms_degrees': metrics['phase_rms_degrees_by_point'][0],
        'template_energy_fraction_outside_stationary_subspace': float(energy/np.linalg.norm(prediction[selected])**2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.work/'dictionary/dictionary.json').read_text())
    stats = json.loads((args.work/'core-blocks-1s/fit.json').read_text())
    sparse = json.loads((args.work/'core-support-blocks-1s.json').read_text())
    indices = np.array(stats['identity']['point_indices'])
    result = {'scope': 'All original odd one-second blocks, interior odd blocks, and the terminal odd block are scored separately, projecting the same stationary basis out of data and prediction in each region. Models are not refitted or selected using these diagnostics.',
        'regions': ['all', 'interior', 'terminal'], 'channels': []}
    for c in range(5):
        with np.load(args.work/'dictionary'/f'channel-{c}-observation.npz') as f:
            observation = dict(f)
        time = observation['time']
        block = np.floor(time-manifest['interval_s'][0]).astype(int)
        odd = block%2 == 1
        regions = {'all': odd, 'interior': odd&(block < block.max()), 'terminal': odd&(block == block.max())}
        predictions = {}
        for name, path in [('gcv', args.work/'core-blocks-1s'/f'independent-channel-{c}-prediction.npz'),
                           ('nested_ridge', args.work/'core-nested-blocks-1s'/f'channel-{c}-prediction.npz')]:
            with np.load(path) as f:
                predictions[name] = f['prediction']
        matrix = np.load(args.work/'dictionary'/f'channel-{c}-design.npy', mmap_mode='r')
        for count in (64, 128):
            trial = next(t for t in sparse['trials'] if t['count'] == count)
            positions = indices[trial['support_indices']]
            a = matrix[:, positions]
            coefficient = np.array(trial['channels'][c]['coefficient_real_imag'])
            predictions[f'sparse_{count}'] = (a@(coefficient[:, 0]+1j*coefficient[:, 1])).reshape(observation['data'].shape)
            phase_file = args.work/f'core-support-{count}-phase/fit.json'
            phase = json.loads(phase_file.read_text())
            channel = next(x for x in phase['channels'] if x['channel'] == c)
            fit = channel['trials'][0]
            coefficient = np.array(fit['coefficient_real_imag'])
            predictions[f'sparse_{count}_phase'] = np.exp(1j*np.array(fit['phase_radians']))[:, None]*(a@(coefficient[:, 0]+1j*coefficient[:, 1])).reshape(observation['data'].shape)
        checks = {name: {region: assess(observation, prediction, selected) for region, selected in regions.items()}
                  for name, prediction in predictions.items()}
        result['channels'].append({'channel': c, 'models': checks})
        del matrix
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
