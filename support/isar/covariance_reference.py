"""Finite-sample reference for the field covariance diagnostic.

The Hann taper correlates neighboring Fourier cells. Simulating independent
frequency cells would falsely flag harmless sample covariance as structure.
This reference propagates the exact DFT/taper covariance before comparison.
It checks the diagnostic high-Doppler band only, not the ship's signal band.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def statistics(covariance):
    diagonal = covariance.diagonal(axis1=-2, axis2=-1).real
    mean = diagonal.mean(axis=-1)
    correlation = covariance/np.sqrt(diagonal[..., :, None]*diagonal[..., None, :])
    eigen = np.linalg.eigvalsh(covariance/mean[..., None, None])
    return {'mean_variance': mean, 'maximum_range_correlation': np.max(abs(correlation-np.eye(diagonal.shape[-1])), axis=(-2, -1)),
        'normalized_eigenvalue_spread': eigen[..., -1]-eigen[..., 0]}


def reference(record, *, trials=2000, seed=9114):
    frequency = np.array(record['frequency_hz'])
    length = record['samples_per_block']
    step = 1/(length*(frequency[1]-frequency[0]))
    low, high = record['doppler_band_hz']
    selected = (abs(frequency) >= low)&(abs(frequency) <= high)
    window = np.hanning(length); window /= np.linalg.norm(window)
    transform = np.exp(-2j*np.pi*frequency[selected, None]*np.arange(length)*step)*window
    weights = np.maximum(0., np.linalg.eigvalsh(transform@transform.conj().T))
    weights = np.tile(weights, record['time_blocks'])/(selected.sum()*record['time_blocks'])
    cells = len(record['range_variance'])
    rng = np.random.default_rng(seed)
    values = []
    for first in range(0, trials, 25):
        count = min(25, trials-first)
        z = (rng.normal(size=(count, len(weights), cells))+1j*rng.normal(size=(count, len(weights), cells)))/np.sqrt(2)
        covariance = np.einsum('tnr,n,tns->trs', z, weights, z.conj(), optimize=True)
        values.append(statistics(covariance))
    return {key: np.concatenate([v[key] for v in values]) for key in values[0]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--trials', type=int, default=4000)
    args = p.parse_args()
    audit = json.loads(args.audit.read_text())
    if not audit['complete']:
        raise ValueError('Complete the covariance measurements first')
    null = reference(audit['channels'][0]['crops'][0], trials=args.trials)
    result = {'scope': __doc__, 'trials': args.trials,
        'finite_sample_null_95_intervals': {k: np.quantile(v, [.025, .975]).tolist() for k, v in null.items()},
        'channels': [], 'complete': True,
        'limitation': 'Uniform epochs use the measured cadence. Existing audit reports interval deviations. Crops are correlated and these individual tail probabilities are diagnostic, not independent replications or target-free guarantees.'}
    for channel in audit['channels']:
        output = {'channel': channel['channel'], 'crops': []}
        for crop in channel['crops']:
            if crop['time_blocks'] != audit['channels'][0]['crops'][0]['time_blocks']:
                raise ValueError('Recompute the reference for a different number of time blocks')
            covariance = np.array(crop['covariance_real'])+1j*np.array(crop['covariance_imag'])
            measured = statistics(covariance[None])
            output['crops'].append({'range_shift_samples': crop['range_shift_samples'],
                'metrics': {k: {'value': float(v[0]), 'upper_tail_probability': float((1+(null[k] >= v[0]).sum())/(args.trials+1))}
                    for k, v in measured.items()}})
        result['channels'].append(output)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
