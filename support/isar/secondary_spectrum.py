"""Multitaper covariance diagnostics on primary and secondary range crops.

The expected white-noise covariance propagates the exact saved stationary
projection through each taper and Fourier bin. Samples are not rescaled to
make spectra flat. Disjoint time tiles remain separate for transfer checks.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal.windows import dpss

from scene_dictionary import fingerprint


def tile_statistics(original, residual, q, tapers, bands):
    """Return spectra and proper-complex covariances in declared bands."""
    n, ranges = original.shape
    a = np.fft.fft(tapers[:, :, None]*original[None], axis=1)
    b = np.fft.fft(tapers[:, :, None]*residual[None], axis=1)
    z = np.fft.fft(tapers[:, :, None, None]*q[None], axis=1)
    covariance, nominal = [], []
    for keep in bands:
        value = b[:, keep].reshape(-1, ranges)
        covariance.append(value.T@value.conj()/len(value))
        selected = z[:, keep]
        expected = np.eye(ranges)-np.einsum('kfrp,kfsp->rs', selected, selected.conj())/(len(tapers)*keep.sum())
        nominal.append((expected+expected.conj().T)/2)
    return {'original_spectrum': np.mean(abs(a)**2, axis=(0, 2)),
        'residual_spectrum': np.mean(abs(b)**2, axis=(0, 2)),
        'nominal_spectrum': 1-np.mean(np.sum(abs(z)**2, axis=3), axis=(0, 2)),
        'covariance': np.array(covariance), 'nominal_covariance': np.array(nominal)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--secondary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rows', type=int, default=512)
    args = p.parse_args()
    source = json.loads((args.secondary/'secondary.json').read_text())
    if not source['complete']:
        raise ValueError('Complete secondary extraction first')
    args.output.mkdir(parents=True, exist_ok=True)
    band_limits = [[0., .5], [.5, 2.], [2., 5.], [15., 25.]]
    tapers = dpss(args.rows, 2., Kmax=3)
    np.testing.assert_allclose(tapers@tapers.T, np.eye(3), atol=1e-12)
    output = {'scope': __doc__, 'bands_hz': band_limits, 'tile_rows': args.rows,
        'dpss_time_half_bandwidth': 2., 'tapers': 3,
        'source': str(args.secondary/'secondary.json'), 'source_sha256': fingerprint(args.secondary/'secondary.json'),
        'channels': [], 'complete': False}
    for c in range(5):
        with np.load(args.secondary/f'channel-{c}.npz') as d:
            time, original, residual, shifts = d['time'], d['original'], d['residual'], d['shifts']
            step = float(np.median(np.diff(time)))
            frequency = np.fft.fftfreq(args.rows, step)
            bands = [(abs(frequency) >= a)&(abs(frequency) < b) for a, b in band_limits]
            starts = np.arange(0, len(time)-args.rows+1, args.rows)
            centres = np.array([time[s:s+args.rows].mean() for s in starts])
            records = []
            maximum_epoch_error = 0.
            for group, shift in enumerate(shifts):
                q = d[f'nuisance_q_{group}']
                tiles = []
                for start in starts:
                    rows = slice(start, start+args.rows)
                    maximum_epoch_error = max(maximum_epoch_error,
                        float(np.max(abs(time[rows]-time[start]-step*np.arange(args.rows)))))
                    tiles.append(tile_statistics(original[group, rows], residual[group, rows], q[rows], tapers, bands))
                records.append({k: np.array([t[k] for t in tiles]) for k in tiles[0]})
                variance = [float(np.trace(records[-1]['covariance'][:, b].mean(axis=0)).real/original.shape[-1])
                    for b in range(len(bands))]
                print(json.dumps({'channel': c, 'shift': int(shift), 'band_mean_variance': variance}), flush=True)
        path = args.output/f'channel-{c}.npz'
        np.savez(path, frequency_hz=frequency, shifts=shifts, tile_centres_s=centres,
            **{key: np.array([r[key] for r in records]) for key in records[0]})
        output['channels'].append({'channel': c, 'path': str(path), 'sha256': fingerprint(path),
            'maximum_fft_epoch_error_s': maximum_epoch_error,
            'maximum_nyquist_phase_error_degrees': 180*maximum_epoch_error/step})
        (args.output/'spectra.json').write_text(json.dumps(output, indent=2)+'\n')
    output['complete'] = True
    (args.output/'spectra.json').write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
