"""Compare a shared causal scattering prior with independent band responses.

Motion, physical sites, envelope origins, spatial support and observations are
fixed. Inner folds select a delay-support hypothesis before outer prediction.
The common model estimates band phases and powers by marginal likelihood;
the independent control has identical per-band marginal covariance freedom.
Neither a likelihood improvement nor a phase-subproblem bound alone accepts
the physical scattering model or the resulting cross-band calibration.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from prepare_causal_scene import read_records
from scene_dictionary import fingerprint
from spectral_scene import SpectralScene, prediction


def summarize(channels):
    power = sum(c['prediction_energy'] for c in channels)
    cross = sum(complex(*c['cross_real_imag']) for c in channels)
    return {'prediction_error': sum(c['prediction_error'] for c in channels),
        'data_energy': sum(c['data_energy'] for c in channels),
        'prediction_energy': power, 'matched_energy': float(abs(cross)**2/power) if power else 0.,
        'scope': 'Correlation of the concatenated five-band prediction with withheld observations, not a calibrated physical image-SNR measurement.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--fold', choices=('inner', 'outer'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initial', type=Path, help='Completed inner result for the same causal support')
    args = parser.parse_args()
    started = time.monotonic()
    cache_path = args.cache/'cache.json'
    cache = json.loads(cache_path.read_text())
    if not cache['complete']:
        raise ValueError('Finish both original-row covariance/quadrature checks before fitting')
    for path, expected in cache['identity']['implementation_sha256'].items():
        if fingerprint(Path(path)) != expected:
            raise ValueError(f'Template-cache implementation changed: {path}')
    if fingerprint(Path(cache['identity']['source'])) != cache['identity']['source_sha256']:
        raise ValueError('Physical source model changed')
    override = cache['identity'].get('motion_override')
    if override and fingerprint(Path(override['source'])) != override['sha256']:
        raise ValueError('The overridden physical motion changed')
    for channel in cache['channels']:
        if fingerprint(Path(channel['file'])) != channel['sha256']:
            raise ValueError('Original-row sufficient statistics changed')
    source = json.loads(Path(cache['identity']['source']).read_text())
    gains = np.asarray(source['channel_gains_real_imag'])@np.array([1., 1j])
    training = [read_records(args.cache, c, args.fold+'_train') for c in range(5)]
    validation = [read_records(args.cache, c, args.fold+'_test') for c in range(5)]
    identity = {'cache': str(args.cache), 'cache_sha256': fingerprint(cache_path), 'fold': args.fold,
        'implementation_sha256': {str(p): fingerprint(p) for p in [Path(__file__),
            Path(__file__).with_name('spectral_scene.py'), Path(__file__).with_name('phase_quadratic.py'),
            Path(__file__).with_name('stationarity.py')]}}
    initial = None
    if args.initial:
        initial = json.loads(args.initial.read_text())
        if (not initial['complete'] or initial['identity']['cache_sha256'] != identity['cache_sha256']
                or initial['identity']['fold'] != 'inner' or args.fold != 'outer'):
            raise ValueError('An outer fit may start only from its matching completed inner result')
        identity['initial_sha256'] = fingerprint(args.initial)
    result = {'identity': identity, 'scope': __doc__, 'complete': False,
        'delay_width_m': cache['identity']['delay_width_m'], 'delay_nodes': len(cache['delays_m']),
        'points': len(cache['points_enu_m']), 'motion_fitted': False, 'yaw_fitted': False,
        'accepted_calibration': False, 'cases': [], 'source_baseline': cache['source_baseline']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        if previous['identity'] != identity:
            raise ValueError('Do not mix changed causal-model experiments in one result')
        result = previous
    for shared in (False, True):
        if any(c['shared_response'] == shared for c in result['cases']):
            continue
        model = SpectralScene(training, shared=shared, reference=3)
        logs, phase = None, np.angle(gains)
        if initial:
            previous = next(c for c in initial['cases'] if c['shared_response'] == shared)['fit']
            p = np.asarray(previous['selected']['parameters'])
            logs = p[:5]+np.log(model.scale/previous['scale'])
            if shared:
                phase = np.zeros(5); phase[model.other] = p[5:]
        elif shared:
            logs = np.asarray(result['cases'][0]['fit']['selected']['parameters'])
        coefficients, fit = model.fit(initial_logs=logs, initial_phases=phase)
        channels = [prediction(r, a) for r, a in zip(validation, coefficients)]
        summary = summarize(channels)
        if args.fold == 'outer':
            np.testing.assert_allclose(summary['data_energy'], cache['source_baseline']['stationary_only_energy'], rtol=1e-12)
        result['cases'].append({'shared_response': shared, 'fit': fit, 'channels': channels,
            'summary': summary,
            'coefficients_real_imag': np.stack([np.array(coefficients).real, np.array(coefficients).imag], axis=-1).tolist()})
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({'shared': shared, 'width': result['delay_width_m'],
            'stationary': fit['selected']['stationary'], 'kkt': fit['selected']['maximum_kkt_residual'],
            **summary, 'seconds': time.monotonic()-started}), flush=True)
    independent, shared = [c['summary'] for c in result['cases']]
    result['comparison'] = {'shared_prediction_error_reduction_percent': 100*(1-shared['prediction_error']/independent['prediction_error']),
        'shared_matched_energy_change_db': float(10*np.log10(shared['matched_energy']/independent['matched_energy']))}
    result.update(complete=True, seconds=time.monotonic()-started)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
