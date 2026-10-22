"""Continue one terminal shared-point fit with bounded quasi-Newton steps.

The exact original-row likelihood, initial support, physical box and acceptance
gate are unchanged. The source is an immutable snapshot of a completed start;
other starts may still be running in their original experiment.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize

from bounded_point_profile import BoundedPointProfile
from field import Field
from fit_shared_ne_points import bands_for, prediction
from scene_dictionary import fingerprint
from stationarity import check


def refine(problem, lower, upper, *, evaluations=600, checkpoint=None):
    empty, rhs = np.empty((0, len(problem.initial))), np.empty(0)
    first = problem.evaluate(problem.initial); started = time.monotonic(); trace = []
    def objective(parameters):
        if np.any(parameters < lower) or np.any(parameters > upper):
            raise ValueError('Native bounded refinement left the physical box')
        return problem.objective(parameters)
    def callback(intermediate_result):
        evaluation = problem.evaluate(intermediate_result.x)
        kkt, violation = check(intermediate_result.x, evaluation['gradient'], empty, rhs, rhs, lower, upper)
        trace.append({'iteration': len(trace)+1, 'objective': evaluation['value'],
            'kkt_residual': kkt, 'constraint_violation': violation,
            'evaluations': problem.evaluations, 'seconds': time.monotonic()-started})
        if checkpoint:
            checkpoint(intermediate_result.x, trace)
        if kkt <= 1e-7 and violation <= 1e-7 and evaluation['value'] <= first['value']:
            raise StopIteration
    fit = minimize(objective, problem.initial, method='L-BFGS-B', jac=True,
        bounds=list(zip(lower, upper)), callback=callback,
        options={'maxfun': evaluations, 'maxiter': evaluations, 'maxcor': 20,
                 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 30})
    last = problem.evaluate(fit.x)
    kkt, violation = check(fit.x, last['gradient'], empty, rhs, rhs, lower, upper)
    return fit.x, {'converged': bool(kkt <= 1e-7 and violation <= 1e-7),
        'method': 'L-BFGS-B with native physical bounds and exact profiled gradient',
        'termination': str(fit.message), 'initial_objective': first['value'], 'objective': last['value'],
        'maximum_kkt_residual': kkt, 'maximum_constraint_violation': violation,
        'iterations': len(trace), 'evaluations': problem.evaluations, 'trace': trace,
        'scene_conditions': last['scene_conditions'], 'seconds': time.monotonic()-started}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'fields', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--source-band', type=int, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Preserve the existing continuation')
    source = json.loads(args.source.read_text())
    for path, expected in source['source_sha256'].items():
        if fingerprint(Path(path)) != expected:
            raise ValueError('Shared support source changed: '+path)
    original = next(t for t in source['trials'] if t['source_band'] == args.source_band)
    if original['fit']['converged']:
        raise ValueError('This continuation is for a terminal nonstationary start')
    model = json.loads(Path(source['source_motion']).read_text())
    fields = []
    for c in range(5):
        path = args.fields/f'channel-{c}.npz'
        if fingerprint(path) != source['source_sha256'][str(path)]:
            raise ValueError('Use the identical observed fields')
        with np.load(path) as d:
            fields.append(Field(dict(d), d['pulse_samples']))
    bands = bands_for(model, fields)
    joint = BoundedPointProfile(bands, np.asarray(original['initial_points_body_m']), model['range_bias_m'])
    joint.initial = np.asarray(original['parameters'])
    np.testing.assert_allclose(joint.evaluate(joint.initial)['value'], original['fit']['objective'], atol=1e-12, rtol=0)
    def checkpoint(parameters, trace):
        args.output.with_suffix('.checkpoint.json').write_text(json.dumps({
            'parameters': parameters.tolist(), 'trace': trace}, indent=2)+'\n')
        print(json.dumps(trace[-1]), flush=True)
    n = len(joint.initial); limit = source['maximum_shift_m']
    parameters, fit = refine(joint, np.full(n, -limit), np.full(n, limit), checkpoint=checkpoint)
    trial = {**original, 'parameters': parameters.tolist(), 'fit': fit,
        'points_body_m': joint.coordinates(parameters).tolist(),
        'after': prediction(bands, joint.evaluate(parameters)['bands'], model['interval_s']),
        'bound_hits': int(np.isclose(abs(parameters), limit, atol=1e-6, rtol=0).sum())}
    result = {'complete': True, 'scope': __doc__, 'source': str(args.source),
        'source_sha256': {str(args.source): fingerprint(args.source), str(Path(__file__)): fingerprint(Path(__file__))},
        'source_trial': original, 'trial': trial, 'accepted_calibration': False}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'complete': True, 'converged': fit['converged'],
        'kkt': fit['maximum_kkt_residual'], 'prediction': trial['after']['summary']}), flush=True)


if __name__ == '__main__':
    main()
