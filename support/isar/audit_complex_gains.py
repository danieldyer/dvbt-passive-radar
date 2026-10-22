"""Test constant complex gain calibration on fixed exact QR observations.

Motion and spatial support are fixed. Scene regularization is selected by GCV
on training observations at the original gains, then held fixed for both gain
tests. This is a conditional comparison, not a global joint-calibration proof.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from assess_dense_scene import fit
from dense_scatterers import gcv
from gain_registration import GainRegistration
from linear_observation import LinearObservation
from phase_registration import PhaseRegistration
from scene_dictionary import fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--statistics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.statistics/'fit.json'
    identity = json.loads(source.read_text())
    if not identity.get('statistics_complete'):
        raise ValueError('Need completed exact train/test QR statistics')
    train, test = [], []
    for c in range(5):
        for label, target in [('train', train), ('test', test)]:
            with np.load(args.statistics/f'channel-{c}-{label}-data.npz') as f:
                target.append(LinearObservation(np.load(args.statistics/f'channel-{c}-{label}-design.npy'),
                    f['data'], float(f['orthogonal_energy']), int(f['observations'])))
    started = time.perf_counter()
    a, y = [v.design for v in train], [v.data for v in train]
    orthogonal = sum(v.orthogonal_energy for v in train)
    baseline, regularization = gcv(np.concatenate(a), np.concatenate(y), 0,
        orthogonal_energy=orthogonal, observations=sum(v.observations for v in train))
    strength = regularization['regularization_strength']
    phase, phase_scene, phase_detail = PhaseRegistration(a, y, strength,
        orthogonal_energy=orthogonal).fit()
    gain, gain_scene, gain_detail = GainRegistration(a, y, strength,
        orthogonal_energy=orthogonal).fit(phase_start=phase)
    result = {'scope': __doc__, 'statistics': str(args.statistics),
        'statistics_identity_sha256': fingerprint(source),
        'regularization': regularization, 'phase_fit': phase_detail, 'complex_gain_fit': gain_detail,
        'independent': fit(train, test, independent=True), 'trials': [],
        'implementation_sha256': {name: fingerprint(Path(__file__).with_name(name)) for name in
            ('audit_complex_gains.py', 'gain_registration.py', 'linear_observation.py')}}
    for label, coefficient, factors in [('original', baseline, np.ones(5)),
            ('phase', phase_scene, phase), ('complex_gain', gain_scene, gain)]:
        channels = [{'channel': c, 'training_error': tr.error(coefficient, factor),
            'test_error': te.error(coefficient, factor), 'test_energy': te.energy}
            for c, (tr, te, factor) in enumerate(zip(train, test, factors))]
        trial = {'variant': label, 'channels': channels,
            'training_error': sum(v['training_error'] for v in channels),
            'test_error': sum(v['test_error'] for v in channels),
            'coefficient_real_imag': np.c_[coefficient.real, coefficient.imag].tolist(),
            'gain_increment_real_imag': np.c_[factors.real, factors.imag].tolist()}
        result['trials'].append(trial)
        print(json.dumps({k: trial[k] for k in ('variant', 'training_error', 'test_error')}), flush=True)
    result.update(complete=True, seconds=time.perf_counter()-started)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
