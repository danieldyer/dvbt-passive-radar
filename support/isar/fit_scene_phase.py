"""Blocked prediction test of phase autofocus with a refitted sparse scene."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from audit_residual_phase import score
from scene_dictionary import fingerprint
from scene_phase import ScenePhase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--support', type=Path)
    source.add_argument('--dictionary', type=Path, help='Use all points in a declared physical-response cache')
    parser.add_argument('--count', type=int, default=64, help='Requested selection size when using --support')
    parser.add_argument('--knot-seconds', type=float, default=3.)
    parser.add_argument('--channels', type=int, nargs='+', default=list(range(5)))
    parser.add_argument('--range-folds', action='store_true')
    parser.add_argument('--observe-aperture-edges', action='store_true',
        help='Keep the last one-second block in training to test interpolation only')
    parser.add_argument('--maximum-phase-radians', type=float, default=.8,
        help='Declared search bound; reported separately from fit convergence')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.support:
        support = json.loads(args.support.read_text())
        chosen = next(t for t in support['trials'] if t['count'] == args.count)
        stats = json.loads((Path(support['source_statistics'])/'fit.json').read_text())
        dictionary = Path(stats['dictionary'])
        indices = np.array(stats['identity']['point_indices'])[chosen['support_indices']]
    else:
        dictionary = args.dictionary
    manifest = json.loads((dictionary/'dictionary.json').read_text())
    if not manifest['complete']:
        raise ValueError('Complete the physical response cache first')
    if not args.support:
        indices = np.arange(len(manifest['points_body_m']))
    args.output.mkdir(parents=True, exist_ok=True)
    result = {'source_support': str(args.support) if args.support else manifest.get('source_support'),
        'source_dictionary': str(dictionary), 'source_dictionary_sha256': fingerprint(dictionary/'dictionary.json'),
        'support_count': len(indices), 'knot_seconds': args.knot_seconds,
        'maximum_phase_radians': args.maximum_phase_radians, 'channels': [], 'complete': False,
        'phase_gauge': 'Constant and linear phase projected out on training times; no channel offset or Doppler-centroid adjustment.',
        'scope': ('Conditional phase diagnostic: physical motion/sites remain fixed; scene amplitudes and '
            'stationary nuisance re-solved at every phase trial. '+('Odd interior one-second blocks test predictions.'
            if args.observe_aperture_edges else 'All odd one-second blocks test predictions.')),
        'original_row_correction_adopted': False}
    result['observe_aperture_edges'] = args.observe_aperture_edges
    if args.support:
        result['source_support_sha256'] = fingerprint(args.support)
    result['source_code_sha256'] = {name: fingerprint(Path(__file__).with_name(name)) for name in
        ('fit_scene_phase.py', 'scene_phase.py', 'audit_residual_phase.py', 'stationarity.py')}
    started = time.perf_counter()
    for c in args.channels:
        with np.load(dictionary/f'channel-{c}-observation.npz') as f:
            observation = dict(f)
        matrix_path = dictionary/f'channel-{c}-design.npy'
        if fingerprint(matrix_path) != manifest['channels'][c]['design_sha256']:
            raise ValueError('Physical response cache has changed')
        design = np.load(matrix_path, mmap_mode='r')[:, indices]
        field = SimpleNamespace(y=observation['data'], time=observation['time'])
        block = np.floor(field.time-manifest['interval_s'][0]).astype(int)
        # Observe both aperture edges: this is an interpolation experiment,
        # not an undocumented extrapolation at the final validation block.
        selected = block%2 == 0
        if args.observe_aperture_edges:
            selected |= block == block.max()
        ranges = np.arange(field.y.shape[1])
        folds = [('all', ranges, ranges)]
        if args.range_folds:
            folds += [('even to odd', ranges[::2], ranges[1::2]), ('odd to even', ranges[1::2], ranges[::2])]
        channel = {'channel': c, 'time_s': field.time.tolist(), 'trials': []}
        for label, train_range, test_range in folds:
            model = ScenePhase(design, field.y, observation['background'], field.time, selected,
                train_range, manifest['interval_s'], args.knot_seconds)
            original = model.project(np.zeros(model.basis.shape[1]))[0]
            phase, coefficient, detail = model.fit(maximum_phase_radians=args.maximum_phase_radians)
            before = score(field, observation['background'], (design@original).reshape(field.y.shape),
                np.zeros(len(field.time)), selected, test_range)
            after = score(field, observation['background'], (design@coefficient).reshape(field.y.shape),
                phase, selected, test_range)
            trial = {'ranges': label, 'refinement': detail, 'before': before, 'after': after,
                'matched_gain_db': float(10*np.log10(after['matched_energy']/before['matched_energy'])),
                'prediction_error_change_percent': 100*(after['prediction_error']/before['prediction_error']-1),
                'phase_radians': phase.tolist(), 'coefficient_real_imag': np.c_[coefficient.real, coefficient.imag].tolist()}
            channel['trials'].append(trial)
            print(json.dumps({'channel': c, 'ranges': label, 'converged': detail['converged'],
                'matched_gain_db': trial['matched_gain_db'], 'prediction_error_change_percent': trial['prediction_error_change_percent'],
                'seconds': time.perf_counter()-started}), flush=True)
        result['channels'].append(channel)
        result['complete'] = len(result['channels']) == len(args.channels)
        result['seconds'] = time.perf_counter()-started
        (args.output/'fit.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
