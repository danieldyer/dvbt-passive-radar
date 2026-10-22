"""Globally bounded channel-phase likelihood on the exact retained operator.

Only constant channel phases are fitted. Roll/pitch, motion, range origins,
points and band magnitudes remain fixed. Each channel's stationary nuisance is
profiled. Temporal tests refit neither phase nor moving coefficients on holdout.
This tests a shared complex scene; it does not assume that model is correct.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from linear_observation import LinearObservation
from phase_quadratic import phase_matrix, solve
from scene_dictionary import fingerprint
from stationary import Stationary


def fit(training, testing):
    designs, data = [o.design for o in training], [o.data for o in training]
    matrix, r, projected = phase_matrix(designs, data)
    phase, certificate = solve(matrix, reference=3)
    coefficient = np.linalg.solve(r, projected@phase)
    baseline = np.linalg.solve(r, projected@np.ones(len(phase)))
    original_energy = sum(o.energy for o in training)
    actual = sum(o.error(coefficient, z.conjugate()) for o, z in zip(training, phase))
    expected = original_energy-certificate['captured_energy']
    if abs(actual-expected) > 1e-10*original_energy:
        raise ValueError('Quadratic reduction differs from original likelihood')
    independent = [np.linalg.lstsq(o.design, o.data, rcond=None)[0] for o in training]
    all_design = np.concatenate([g.conjugate()*a for a, g in zip(designs, phase)])
    q = np.linalg.qr(all_design, mode='reduced')[0]
    derivative = np.zeros((len(q), len(phase)-1), complex)
    first, j = 0, 0
    for c, a in enumerate(designs):
        if c != 3:
            derivative[first:first+len(a), j] = 1j*phase[c].conjugate()*(a@coefficient)
            j += 1
        first += len(a)
    derivative -= q@(q.conj().T@derivative)
    fisher = 2*np.real(derivative.conj().T@derivative)
    channels = []
    for c, (train, test) in enumerate(zip(training, testing)):
        channels.append({'channel': c, 'training_error': train.error(coefficient, phase[c].conjugate()),
            'test_error': test.error(coefficient, phase[c].conjugate()),
            'original_phase_test_error': test.error(baseline),
            'independent_scene_test_error': test.error(independent[c]),
            'test_energy': test.energy})
    return {'certificate': certificate, 'phase_increment_radians': (-np.angle(phase)).tolist(),
        'training_error': actual, 'channels': channels,
        'phase_fisher_information': fisher.tolist(),
        'conditional_phase_crlb_degrees': np.degrees(np.sqrt(np.diag(np.linalg.inv(fisher)))).tolist(),
        'fisher_scope': 'Known unit conditional noise, fixed geometry/support/magnitudes, common complex coefficients eliminated. Not a real-data error bar.',
        'coefficient_real_imag': np.c_[coefficient.real, coefficient.imag].tolist(),
        'heldout_error_change_percent': float(100*(sum(c['test_error'] for c in channels)/
            sum(c['original_phase_test_error'] for c in channels)-1)),
        'heldout_penalty_vs_independent_percent': float(100*(sum(c['test_error'] for c in channels)/
            sum(c['independent_scene_test_error'] for c in channels)-1))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dictionary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    manifest_path = args.dictionary/'dictionary.json'
    manifest = json.loads(manifest_path.read_text())
    if not manifest['complete']:
        raise ValueError('Need complete exact forward responses')
    result = {'scope': __doc__, 'dictionary': str(args.dictionary),
        'dictionary_manifest_sha256': fingerprint(manifest_path), 'trials': [], 'complete': False,
        'implementation_sha256': {name: fingerprint(Path(__file__).with_name(name)) for name in
            ('audit_phase_quadratic.py', 'phase_quadratic.py')},
        'conditional_support': 'Existing motion and continuous support were chosen on this interval; temporal scores validate this phase adjustment only.',
        'correction_adopted': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cached = [[] for _ in range(3)]
    for entry in manifest['channels']:
        c = entry['channel']
        path = args.dictionary/f'channel-{c}-design.npy'
        if fingerprint(path) != entry['design_sha256']:
            raise ValueError('Exact dictionary has changed')
        design = np.load(path, mmap_mode='r')
        with np.load(args.dictionary/f'channel-{c}-observation.npz') as f:
            time_axis, y, background = f['time'], f['data'], f['background']
        midpoint = np.mean(manifest['interval_s'])
        masks = [np.floor(time_axis-manifest['interval_s'][0]).astype(int)%2 == 0,
                 time_axis < midpoint, time_axis >= midpoint]
        for i, selected in enumerate(masks):
            pair = []
            for mask in (selected, ~selected):
                rows = np.repeat(mask, y.shape[1])
                nuisance = Stationary(background, rows)
                pair.append(LinearObservation.compress(nuisance.remove(design[rows]),
                    nuisance.remove(y.ravel()[rows]), nuisance.rank))
            cached[i].append(pair)
        print(json.dumps({'channel': c, 'stage': 'Exact projected QR statistics',
            'seconds': time.perf_counter()-started}), flush=True)
    for label, pairs in zip(('alternate_one_second', 'first_half', 'second_half'), cached):
        fitted = fit([v[0] for v in pairs], [v[1] for v in pairs])
        fitted['training_split'] = label
        result['trials'].append(fitted)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({'split': label, 'global_gap': fitted['certificate']['absolute_gap'],
            'globally_certified': fitted['certificate']['globally_certified'],
            'heldout_error_change_percent': fitted['heldout_error_change_percent'],
            'penalty_vs_independent_percent': fitted['heldout_penalty_vs_independent_percent']}), flush=True)
    phase = np.array([v['phase_increment_radians'] for v in result['trials']])
    difference = np.angle(np.exp(1j*(phase[1]-phase[2])))
    result.update(complete=True, seconds=time.perf_counter()-started,
        half_to_half_phase_difference_degrees=np.degrees(difference).tolist())
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
