"""Test evidence-learned point variances on untouched physical observations.

This uses the zero-delay original-row cache. The conditional inner prediction
test precedes any outer test or motion/antenna changes. Independent band
responses are an explicit control with separate point-variance estimates.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from fit_causal_scene import summarize
from prepare_causal_scene import read_records
from relevance_scene import RelevanceScene
from scene_dictionary import fingerprint
from spectral_scene import prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--fold', choices=('inner', 'outer'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initial', type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    cache_path = args.cache/'cache.json'
    cache = json.loads(cache_path.read_text())
    if not cache['complete'] or cache['identity']['delay_width_m'] != 0:
        raise ValueError('Use the checked instantaneous original-row scene cache')
    for path, expected in cache['identity']['implementation_sha256'].items():
        if fingerprint(Path(path)) != expected:
            raise ValueError(f'Template implementation changed: {path}')
    for channel in cache['channels']:
        if fingerprint(Path(channel['file'])) != channel['sha256']:
            raise ValueError('Original-row sufficient statistics changed')
    source_path = Path(cache['identity']['source'])
    if fingerprint(source_path) != cache['identity']['source_sha256']:
        raise ValueError('The fixed physical model changed')
    source = json.loads(source_path.read_text())
    gains = np.asarray(source['channel_gains_real_imag'])@np.array([1., 1j])
    training = [read_records(args.cache, c, args.fold+'_train') for c in range(5)]
    testing = [read_records(args.cache, c, args.fold+'_test') for c in range(5)]
    names = ['fit_relevance_scene.py', 'relevance_scene.py', 'fit_causal_scene.py', 'spectral_scene.py',
             'scene_statistics.py', 'phase_quadratic.py', 'stationarity.py']
    identity = {'cache': str(args.cache), 'cache_sha256': fingerprint(cache_path), 'fold': args.fold,
        'implementation_sha256': {str(Path(__file__).with_name(n)): fingerprint(Path(__file__).with_name(n)) for n in names}}
    initial = None
    if args.initial:
        initial = json.loads(args.initial.read_text())
        if not initial['complete'] or initial['identity']['cache_sha256'] != identity['cache_sha256'] or initial['identity']['fold'] != 'inner' or args.fold != 'outer':
            raise ValueError('Outer initialization requires its matching completed inner experiment')
        identity['initial_sha256'] = fingerprint(args.initial)
    result = {'identity': identity, 'scope': __doc__, 'complete': False, 'cases': [],
        'points_enu_m': cache['points_enu_m'], 'motion_fitted': False, 'yaw_fitted': False,
        'antenna_positions_fitted': False, 'accepted_calibration': False,
        'limitations': 'Motion and original spatial support used this recording. Type-II evidence optima are local; no posterior variance is claimed to include estimated hyperparameter or physical model uncertainty.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        saved = json.loads(args.output.read_text())
        if saved['identity'] != identity:
            raise ValueError('Cannot resume changed relevance experiments')
        result = saved
    for shared in (False, True):
        if any(case['shared_response'] == shared for case in result['cases']):
            continue
        model = RelevanceScene(training, shared=shared, reference=3)
        parameters = None
        if initial:
            previous = next(c['fit'] for c in initial['cases'] if c['shared_response'] == shared)
            parameters = np.asarray(previous['selected']['parameters'])
            parameters[:model.variance_count] += np.log(model.scale/previous['scale'])
        coefficients, fit = model.fit(initial_phases=np.angle(gains), initial_parameters=parameters)
        channels = [prediction(record, coefficient) for record, coefficient in zip(testing, coefficients)]
        summary = summarize(channels)
        if args.fold == 'outer':
            np.testing.assert_allclose(summary['data_energy'], cache['source_baseline']['stationary_only_energy'], rtol=1e-12)
        result['cases'].append({'shared_response': shared, 'fit': fit, 'channels': channels,
            'summary': summary, 'coefficients_real_imag': np.stack([np.array(coefficients).real, np.array(coefficients).imag], axis=-1).tolist()})
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({'shared': shared, 'stationary': fit['selected']['stationary'],
            'kkt': fit['selected']['maximum_kkt_residual'], **summary,
            'seconds': time.monotonic()-started}), flush=True)
    independent, shared = [c['summary'] for c in result['cases']]
    result['comparison'] = {'shared_prediction_error_reduction_percent': 100*(1-shared['prediction_error']/independent['prediction_error']),
        'shared_matched_energy_change_db': float(10*np.log10(shared['matched_energy']/independent['matched_energy']))}
    result.update(complete=True, seconds=time.monotonic()-started)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
