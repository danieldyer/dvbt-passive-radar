"""Known-matrix covariance efficiency with the actual 30-second echo operator."""
import argparse
import json
from pathlib import Path

import numpy as np

from matrix_filter_benchmark import frequency_labels, signal_block
from matrix_spectral_gls import matched_filter, shrink
from scene_dictionary import fingerprint
from secondary_filter_benchmark import templates
from spectral_gls import orthogonal_nuisance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('benchmark', 'secondary', 'fields', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--trials', type=int, default=8000)
    args = p.parse_args()
    experiment = json.loads(args.benchmark.read_text())
    channel = next(c for c in experiment['channels'] if c['channel'] == 1)
    model = json.loads(Path(experiment['motion']).read_text())
    covariance = np.load(channel['data'])['covariance'][0]
    with np.load(args.secondary/'channel-1.npz') as data:
        time = data['time']
        step = float(np.median(np.diff(time)))
        n = int(round(channel['long_aperture_seconds']/step))
        start = int(np.searchsorted(time, channel['long_test_interval_s'][0]))
        selected = slice(start, start+n)
        time = time[selected]
        primary = int(np.flatnonzero(data['shifts'] == 0)[0])
        q = orthogonal_nuisance(data[f'nuisance_q_{primary}'][selected])
        q = np.fft.fft(q, axis=0, norm='ortho')
    point_time, values = templates(args.fields/'channel-1.npz', model, 1, np.array([[0., 0., 20.]]))
    a = signal_block(point_time, values, time)[:, :, 0]
    labels = frequency_labels(n, step)
    h, information = matched_filter(a, q, labels, covariance)
    h0, _ = matched_filter(a, q, labels, shrink(covariance, 0.))
    weights = np.stack([h0, h], axis=-1)
    # Applying L^H to filters is exactly equivalent to generating n=Lz and
    # filtering n. The dense covariance tests independently check this algebra.
    root = np.linalg.cholesky(covariance)
    transformed = np.einsum('frs,frk->fsk', root[labels].conj(), weights)
    variance = np.sum(abs(transformed)**2, axis=(0, 1))
    rng = np.random.default_rng(385108)
    errors = []
    for first in range(0, args.trials, 128):
        count = min(128, args.trials-first)
        z = (rng.normal(size=(count, *a.shape))+1j*rng.normal(size=(count, *a.shape)))/np.sqrt(2)
        errors.append(z.reshape(count, -1)@transformed.reshape(a.size, 2).conj())
    errors = np.concatenate(errors)
    alpha = np.sqrt(1000/information)*np.exp(.71j)
    phase = np.angle((alpha+errors[:, 1])*alpha.conjugate())
    result = {'scope': __doc__, 'source': str(args.benchmark), 'source_sha256': fingerprint(args.benchmark),
        'implementation_sha256': {name: fingerprint(Path(__file__).with_name(name)) for name in
            ('audit_matrix_spectral_gls.py', 'matrix_spectral_gls.py', 'matrix_filter_benchmark.py')},
        'channel': 1, 'rows': n, 'samples': int(a.size), 'nuisance_rank': q.shape[-1],
        'aperture_seconds': n*step, 'trials': args.trials,
        'truth': 'The fitted covariance is deliberately made known true Gaussian Fourier-range covariance. This validates the conditional numerical estimator, not transfer of the recording covariance or physical ship motion.',
        'unit_complex_response_error': float(abs(np.vdot(h, a)-1)),
        'nuisance_response_error': float(np.max(abs(q.reshape(a.size, -1).conj().T@h.ravel()))),
        'conditional_amplitude_crlb': 1/information, 'calculated_variance': variance.tolist(),
        'calculated_mse_over_crlb': float(variance[1]*information),
        'measured_mse': np.mean(abs(errors)**2, axis=0).tolist(),
        'measured_mse_over_variance': (np.mean(abs(errors)**2, axis=0)/variance).tolist(),
        'conditional_oracle_gain_db': float(10*np.log10(variance[0]/variance[1])),
        'integrated_snr_db': 30., 'phase_crlb_rms_degrees': float(np.rad2deg(1/np.sqrt(2000))),
        'phase_rms_degrees': float(np.rad2deg(np.sqrt(np.mean(phase**2)))),
        'phase_mse_over_crlb': float(np.mean(phase**2)*2000)}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
