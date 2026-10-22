"""Predictive check of a common-location marginal scene on recorded data.

The physical forward model, original rows, noise metric, point count, point
boxes and half-second holdouts match the unregularized shared-support control.
Only the scene likelihood changes. Fixed motion/origins/height and inherited
support remain conditional; no physical calibration is accepted by this fit.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from field import Field
from fit_shared_ne_points import bands_for
from refine_shared_ne_points import refine
from scene_dictionary import fingerprint
from shared_marginal_points import SharedMarginalPoints
from validate_motion import summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'fields', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--starts', type=int, nargs='+', default=[3, 0])
    args = p.parse_args(); started = time.monotonic()
    if (args.output.exists() or len(set(args.starts)) != len(args.starts)
            or any(c not in range(5) for c in args.starts)):
        raise ValueError('Use a fresh output and distinct source bands from zero to four')
    source = json.loads(args.source.read_text())
    for path, expected in source['source_sha256'].items():
        if fingerprint(Path(path)) != expected:
            raise ValueError('Original experiment input changed: '+path)
    model = json.loads(Path(source['source_motion']).read_text()); fields = []
    for c in range(5):
        path = args.fields/f'channel-{c}.npz'
        if fingerprint(path) != source['source_sha256'][str(path)]:
            raise ValueError('Use the identical observed fields')
        with np.load(path) as d:
            fields.append(Field(dict(d), d['pulse_samples']))
    bands = bands_for(model, fields)
    paths = [args.source, Path(__file__), *[Path(__file__).with_name(name) for name in
        ('shared_marginal_points.py', 'marginal_points.py', 'gaussian_scene.py',
         'fit_shared_ne_points.py', 'refine_shared_ne_points.py', 'field.py', 'full_motion.py')]]
    result = {'complete': False, 'scope': __doc__, 'source_sha256': {
        str(path): fingerprint(path) for path in paths}, 'source_experiment': str(args.source),
        'initial_support_bands': args.starts, 'maximum_shift_m': source['maximum_shift_m'],
        'independent_band_support_prediction': source['independent_band_support_prediction'],
        'trials': [], 'accepted_calibration': False, 'yaw_fitted': False,
        'statistical_model': 'Independent proper-complex Gaussian point coefficients per band; scene variance fitted on training rows with full marginal likelihood including log determinant. Thermal metric unchanged.'}
    args.output.mkdir(parents=True)
    def save():
        (args.output/'experiment.json').write_text(json.dumps(result, indent=2)+'\n')
    save()
    print(json.dumps({'stage': 'loaded', 'initial_support_bands': args.starts,
        'seconds': time.monotonic()-started}), flush=True)
    for c in args.starts:
        points = np.asarray(model['selected_points_body_m'][c])
        problem = SharedMarginalPoints(bands, model['parameters'], points)
        before = problem.validation(problem.initial)
        def checkpoint(parameters, trace):
            (args.output/f'band-{c}.checkpoint.json').write_text(json.dumps({
                'parameters': parameters.tolist(), 'trace': trace}, indent=2)+'\n')
            print(json.dumps({'source_band': c, **trace[-1]}), flush=True)
        count = len(problem.initial); limit = source['maximum_shift_m']
        parameters, fit = refine(problem, np.full(count, -limit), np.full(count, limit),
            evaluations=500, checkpoint=checkpoint)
        final = problem.evaluate(parameters); after = problem.validation(parameters)
        trial = {'source_band': c, 'fit': fit, 'parameters': parameters.tolist(),
            'initial_points_body_m': points.tolist(), 'points_body_m': problem.coordinates(parameters).tolist(),
            'before': {'channels': before, 'summary': summarize(before)},
            'after': {'channels': after, 'summary': summarize(after)},
            'scene_conditions': final['scene_conditions'], 'scene_variances': final['scene_variances'],
            'effective_scene_ranks': final['effective_scene_ranks'],
            'bound_hits': int(np.isclose(abs(parameters), limit, atol=1e-6, rtol=0).sum())}
        result['trials'].append(trial)
        save()
        print(json.dumps({'stage': 'completed_start', 'source_band': c, 'converged': fit['converged'],
            'kkt': fit['maximum_kkt_residual'], 'prediction': trial['after']['summary'],
            'seconds': time.monotonic()-started}), flush=True)
    eligible = [trial for trial in result['trials'] if trial['fit']['converged']]
    selected = min(eligible, key=lambda trial: trial['fit']['objective']) if eligible else None
    result.update(complete=True, selected_source_band=selected['source_band'] if selected else None,
        all_starts_stationary=all(trial['fit']['converged'] for trial in result['trials']),
        selection='Minimum training marginal likelihood among stationary fits; withheld samples do not select the start.',
        seconds=time.monotonic()-started)
    save()


if __name__ == '__main__':
    main()
