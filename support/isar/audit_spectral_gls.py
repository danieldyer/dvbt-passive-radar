"""Known-covariance amplitude/phase efficiency on an actual fixed-N/E template."""
import argparse
import json
from pathlib import Path

import numpy as np

from scene_dictionary import fingerprint
from secondary_filter_benchmark import templates
from spectral_gls import matched_filter, orthogonal_nuisance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('secondary', 'fields', 'benchmark', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--trials', type=int, default=8000)
    args = p.parse_args()
    experiment = json.loads(args.benchmark.read_text())
    model = json.loads(Path(experiment['motion']).read_text())
    c = 1
    record = experiment['channels'][c]
    logvar = np.log(record['outer_fits'][0]['fit']['variance'])
    n = experiment['tile_rows']
    tile = record['test_tiles'][0]
    with np.load(args.secondary/f'channel-{c}.npz') as d:
        rows = slice(tile*n, (tile+1)*n)
        time = d['time'][rows]
        primary = int(np.flatnonzero(d['shifts'] == 0)[0])
        q = orthogonal_nuisance(d[f'nuisance_q_{primary}'][rows])
    epochs, signal = templates(args.fields/f'channel-{c}.npz', model, c, np.array([[0., 0., 20.]]))
    first = int(np.searchsorted(epochs, time[0]))
    np.testing.assert_array_equal(epochs[first:first+n], time)
    a = np.fft.fft(signal[first:first+n, :, 0], axis=0, norm='ortho')
    a /= np.linalg.norm(a)
    q = np.fft.fft(q, axis=0, norm='ortho')
    frequency = np.fft.fftfreq(n, float(np.median(np.diff(time))))
    values = abs(frequency) if experiment.get('spectral_model', 'absolute_bands') == 'absolute_bands' else frequency
    labels = np.digitize(values, record.get('spectral_partition_edges_hz', [.5, 2., 5.]))
    h, information = matched_filter(a, q, labels, logvar)
    h0, _ = matched_filter(a, q, labels, logvar*0.)
    variance = np.broadcast_to(np.exp(logvar[labels])[:, None], a.shape)
    expected = np.array([np.sum(abs(h0)**2*variance), np.sum(abs(h)**2*variance)])
    rng = np.random.default_rng(45249)
    errors = []
    for first in range(0, args.trials, 128):
        count = min(128, args.trials-first)
        noise = ((rng.normal(size=(count, *a.shape))+1j*rng.normal(size=(count, *a.shape)))/
            np.sqrt(2)*np.sqrt(variance))
        errors.append(np.stack([np.einsum('nfr,fr->n', noise, w.conj()) for w in (h0, h)], axis=-1))
    errors = np.concatenate(errors)
    amplitude = np.sqrt(1000/information)*np.exp(.71j)
    phase = np.angle((amplitude+errors[:, 1])*amplitude.conjugate())
    result = {'scope': __doc__, 'source': str(args.benchmark), 'source_sha256': fingerprint(args.benchmark),
        'implementation_sha256': {name: fingerprint(Path(__file__).with_name(name)) for name in
            ('audit_spectral_gls.py', 'spectral_gls.py', 'secondary_filter_benchmark.py')},
        'channel': c, 'tile': tile, 'samples': int(a.size), 'nuisance_rank': q.shape[-1],
        'trials': args.trials, 'truth': 'Independent proper-complex Gaussian Fourier bins, with the fitted secondary spectrum deliberately made the known true covariance. This is a conditional synthetic efficiency check, not the real recording noise.',
        'unit_response_error': float(abs(np.vdot(h, a)-1)),
        'stationary_response_error': float(np.max(abs(q.reshape(a.size, -1).conj().T@h.ravel()))),
        'conditional_complex_amplitude_crlb': 1/information,
        'calculated_variance': expected.tolist(), 'measured_mse': np.mean(abs(errors)**2, axis=0).tolist(),
        'mse_over_known_variance': (np.mean(abs(errors)**2, axis=0)/expected).tolist(),
        'conditional_optimal_gain_db': float(10*np.log10(expected[0]/expected[1])),
        'injected_integrated_snr_db': 30.,
        'phase_crlb_rms_degrees': float(np.rad2deg(1/np.sqrt(2000))),
        'measured_phase_rms_degrees': float(np.rad2deg(np.sqrt(np.mean(phase**2)))),
        'phase_mse_over_crlb': float(np.mean(phase**2)*2000)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
