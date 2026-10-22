"""Compare local point arithmetic on the identical recorded-data likelihood.

Preserve the ongoing fit's snapshot, bounds, observed samples and coefficients'
least-squares treatment. Only the order of position-offset arithmetic changes.
Directional finite differences diagnose derivatives; no new fit is accepted.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from bounded_point_profile import BoundedPointProfile
from field import Field
from fit_shared_ne_points import bands_for
from scene_dictionary import fingerprint
from stable_point_field import StablePointField


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('experiment', 'fields', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--starts', type=int, nargs='+', default=[3, 0])
    args = p.parse_args(); started = time.monotonic()
    if args.output.exists():
        raise ValueError('Preserve completed diagnostics')
    raw = (args.experiment/'experiment.json').read_bytes(); experiment = json.loads(raw)
    for name, expected in experiment['source_sha256'].items():
        if fingerprint(Path(name)) != expected:
            raise ValueError('Fitted source changed: '+name)
    snapshot = args.output.with_suffix('.source.json')
    if snapshot.exists():
        raise ValueError('Preserve the existing input snapshot')
    snapshot.write_bytes(raw)
    model = json.loads(Path(experiment['source_motion']).read_text())
    fields = []
    for c in range(5):
        path = args.fields/f'channel-{c}.npz'
        if fingerprint(path) != experiment['source_sha256'][str(path)]:
            raise ValueError('Use the identical observations')
        with np.load(path) as d:
            fields.append(Field(dict(d), d['pulse_samples']))
    old_bands = bands_for(model, fields); local_bands = []
    for band in old_bands:
        other = copy.copy(band); other.field = StablePointField(band.field); local_bands.append(other)
    sources = [snapshot, Path(__file__), *[Path(__file__).with_name(n) for n in
        ('stable_point_field.py', 'bounded_point_profile.py', 'point_profile.py', 'field.py', 'fit_shared_ne_points.py')]]
    result = {'complete': False, 'scope': __doc__, 'source_sha256': {
        str(path): fingerprint(path) for path in sources}, 'cases': [], 'accepted_calibration': False}
    for source_band in args.starts:
        trial = next(t for t in experiment['trials'] if t['source_band'] == source_band)
        parameters = np.asarray(trial['parameters']); points = np.asarray(trial['initial_points_body_m'])
        old = BoundedPointProfile(old_bands, points, model['range_bias_m'])
        local = BoundedPointProfile(local_bands, points, model['range_bias_m'])
        first, second = old.evaluate(parameters), local.evaluate(parameters)
        np.testing.assert_allclose(first['value'], trial['fit']['objective'], atol=1e-12, rtol=0)
        direction = second['gradient']/np.linalg.norm(second['gradient'])
        response_errors = [float(np.linalg.norm(a['target']-b['target'])/np.linalg.norm(b['target']))
                           for a, b in zip(first['bands'], second['bands'])]
        record = {'source_band': source_band, 'original_stationary': trial['fit']['converged'],
            'original_objective': first['value'], 'local_objective': second['value'],
            'template_relative_errors_by_band': response_errors,
            'maximum_gradient_change': float(abs(first['gradient']-second['gradient']).max()),
            'maximum_scene_condition': max(first['scene_conditions']), 'checks': []}
        for step in (.01, .001, .0001, .00001):
            item = {'step_m': step}
            for name, problem, value in (('original', old, first), ('local', local, second)):
                plus, minus = [problem.objective(parameters+sign*step*direction)[0] for sign in (1, -1)]
                numeric = (plus-minus)/(2*step)
                analytic = value['gradient']@direction
                item[name] = {'finite_difference': float(numeric), 'analytic': float(analytic),
                    'absolute_error': float(abs(numeric-analytic))}
            record['checks'].append(item)
            print(json.dumps({'source_band': source_band, **item, 'seconds': time.monotonic()-started}), flush=True)
        result['cases'].append(record)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
    result.update(complete=True, seconds=time.monotonic()-started)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
