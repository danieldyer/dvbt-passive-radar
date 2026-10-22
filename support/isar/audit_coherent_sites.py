"""Compare all licensed site combinations under one fixed coherent scene model.

Use the exact dictionaries cached by audit_ship_sites. Every combination gets
the same shared point phases, independent positive band strengths and free
constant band phases. Training likelihood selects the candidate; withheld
interior time blocks evaluate it. Motion and point support remain conditional.
"""
import argparse
from itertools import product
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]/'calibration'))
from transmitter_config import candidate_acma_ids
import time

import numpy as np

from phase_coupling import PhaseCoupling
from scene_dictionary import fingerprint
from stationary import Stationary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dictionary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, help='Stop after this many combinations; resume with the same output')
    parser.add_argument('--truth-sites', nargs=5,
        help='Known-truth control: synthesize these five sites, without using measured ship values')
    parser.add_argument('--sites', nargs=5, help='Restrict the control to this declared combination')
    parser.add_argument('--spatial-phase', action='store_true',
        help='Diagnostic only: add a static affine phase across the ship for each non-reference band')
    parser.add_argument('--interval', type=float, nargs=2, help='Restrict the cached rows to this time interval')
    args = parser.parse_args()
    source_path = args.dictionary/'fit.json'
    source = json.loads(source_path.read_text())
    allowed = candidate_acma_ids()
    if not source['complete'] or len(source['channels']) != 5:
        raise ValueError('Finish the five-channel licensed-site dictionary first')
    identity = {'source_sha256': fingerprint(source_path), 'source_directory': str(args.dictionary),
                'candidate_site_ids':[list(sites) for sites in allowed]}
    if args.truth_sites:
        identity['synthetic_sites'] = args.truth_sites
    for key, value in [('sites', args.sites), ('spatial_phase', args.spatial_phase), ('interval', args.interval)]:
        if value:
            identity[key] = value
    result = {'scope': __doc__, 'identity': identity, 'complete': False,
        'correction_adopted': False, 'trials': []}
    if args.output.exists():
        result = json.loads(args.output.read_text())
        if result['identity'] != identity:
            raise ValueError('Cannot resume with a changed site dictionary')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    truth = None
    if args.truth_sites:
        rng = np.random.default_rng(9431)
        phase = rng.uniform(-np.pi, np.pi, source['count'])
        channel_phase = np.r_[rng.uniform(-np.pi, np.pi, 3), 0., rng.uniform(-np.pi, np.pi)]
        truth = rng.uniform(.5, 1.5, (5, source['count']))*np.exp(1j*(phase+channel_phase[:, None]))
        result['known_truth'] = {'seed': 9431, 'sites': args.truth_sites,
            'channel_phase_radians': channel_phase.tolist(),
            'coefficients_real_imag': np.stack([truth.real, truth.imag], axis=-1).tolist(),
            'measured_ship_values_used': False, 'noise_added': False}
    options, train_data, test_data = [], [], []
    for channel in source['channels']:
        c = channel['channel']
        with np.load(args.dictionary/f'channel-{c}-observation.npz') as loaded:
            data, selected, background, epochs = loaded['data'], loaded['selected'], loaded['background'], loaded['time']
        if truth is not None:
            correct = next(o for o in channel['options'] if o['site_id'] == args.truth_sites[c])
            data = (np.load(args.dictionary/correct['design'], mmap_mode='r')@truth[c]).reshape(data.shape)
        retained = np.ones(len(epochs), bool)
        if args.interval:
            retained = (epochs >= args.interval[0])&(epochs < args.interval[1])
            epochs, data = epochs[retained], data[retained]
            block = np.floor(epochs-args.interval[0]).astype(int)
            selected = (block%2 == 0)|(block == block.max())
            background = background[np.repeat(retained, data.shape[1])]
        mask = np.repeat(selected, data.shape[1])
        training, testing = Stationary(background, mask), Stationary(background, ~mask)
        train_data.append(training.remove(data.ravel()[mask]))
        test_data.append(testing.remove(data.ravel()[~mask]))
        candidates = []
        for option in channel['options']:
            if option['site_id'] not in allowed[c]:
                continue
            if args.sites and option['site_id'] != args.sites[c]:
                continue
            target = np.load(args.dictionary/option['design'], mmap_mode='r')
            if args.interval:
                target = target[np.repeat(retained, data.shape[1])]
            candidates.append({'site_id': option['site_id'],
                'train': training.remove(target[mask]), 'test': testing.remove(target[~mask])})
        options.append(candidates)
    elapsed = time.perf_counter()
    done = {tuple(t['sites']) for t in result['trials']}
    combinations = list(product(*options))
    if not combinations:
        raise ValueError('Requested sites are not present in the licensed dictionary')
    coordinates = None
    if args.spatial_phase:
        coordinates = np.array(json.loads(Path(source['source_support']).read_text())['points_body_m'])[:source['count'], :2]
    for combination in combinations:
        sites = [o['site_id'] for o in combination]
        if tuple(sites) in done:
            continue
        if args.limit is not None and len(result['trials']) >= args.limit:
            break
        coupling = PhaseCoupling([o['train'] for o in combination], train_data, reference=3,
            point_coordinates_m=coordinates)
        coefficients, fit = coupling.fit()
        variants = {}
        for name, responses in [('coupled', coefficients), ('independent', coupling.independent)]:
            channels = []
            for option, observation, amplitude in zip(combination, test_data, responses):
                prediction = option['test']@amplitude
                energy = float(np.vdot(prediction, prediction).real)
                cross = np.vdot(prediction, observation)
                channels.append({'prediction_error': float(np.sum(abs(observation-prediction)**2)),
                    'matched_energy': float(abs(cross)**2/energy), 'prediction_energy': energy,
                    'prediction_cross_real_imag': [float(cross.real), float(cross.imag)]})
            variants[name] = {'channels': channels,
                'prediction_error': sum(c['prediction_error'] for c in channels),
                'matched_energy': sum(c['matched_energy'] for c in channels)}
        trial = {'sites': sites, 'fit': fit, 'validation': variants}
        result['trials'].append(trial)
        result['complete'] = len(result['trials']) == len(combinations)
        result['training_selected_index'] = min(range(len(result['trials'])),
            key=lambda i: result['trials'][i]['fit']['objective'])
        result['this_run_seconds'] = time.perf_counter()-elapsed
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({'trial': len(result['trials']), 'sites': sites, 'training': fit['objective'],
            'converged': fit['converged'], 'gradient': fit['maximum_gradient'],
            'test_errors': {k: v['prediction_error'] for k, v in variants.items()},
            'seconds': result['this_run_seconds']}), flush=True)


if __name__ == '__main__':
    main()
