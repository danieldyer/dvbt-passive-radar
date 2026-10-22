"""Add the diffuse covariance volume to the existing yaw-free motion fit.

Only the statistical objective changes. Original observations, scene support,
training-selected penalties, thermal weighting and physical constraints are
identical. Final stationarity and validation use all original sample epochs.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from fit_distributed_motion import aggregate
from fit_mixed_motion import changes
from fit_physical_profile import solve
from marginal_scene import MarginalMotion, profile
from mixed_experiment import load
from refine_full_capture_motion import constraints
from scene_dictionary import fingerprint
from spline_motion import SplineMotion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=80)
    parser.add_argument('--resume-checkpoint', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve completed comparisons')
    experiment = load(args.source, accelerated=True)
    model = experiment.model
    extras = (model['parameters'], experiment.weights, model['core_counts'], model['fixed_penalties'])
    exact, fast = [MarginalMotion(bands, *extras) for bands in (experiment.bands, experiment.fast)]
    curve = SplineMotion(model)
    _, matrix, lo, hi, lower, upper = constraints(model, common_path=True)
    frozen = (exact.active >= curve.sections[1].start)&(exact.active < curve.sections[1].stop)
    lower[frozen] = upper[frozen] = 0.
    start = time.monotonic()
    first = exact.evaluate(exact.initial)
    before = exact.validation(exact.initial)
    print(json.dumps({'stage': 'before', 'channels': first['channels']}), flush=True)
    identity = {'source_motion': str(args.source), 'source_motion_sha256': fingerprint(args.source),
        'scope': __doc__, 'statistical_model': 'Proper complex Gaussian stationary contrasts; unknown deterministic core and integrated diffuse coefficients with variance 1/lambda; declared unit thermal noise.',
        'noise_variances': [1.]*5, 'yaw_fitted': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume_checkpoint:
        saved = json.loads(args.output.with_suffix('.checkpoint.json').read_text())
        if saved['source_motion_sha256'] != identity['source_motion_sha256']:
            raise ValueError('Checkpoint source changed')
        fast.initial = np.asarray(saved['parameters'])[fast.active]
    def checkpoint(stage):
        def save(parameters, trace):
            print(json.dumps({'stage': stage, **trace[-1]}), flush=True)
            args.output.with_suffix('.checkpoint.json').write_text(json.dumps({**identity,
                'parameters': exact.expand(parameters).tolist(), 'trace': trace}, indent=2)+'\n')
        return save
    parameters, search = solve(fast, matrix, lo, hi, lower, upper,
        iterations=args.iterations, checkpoint=checkpoint('search'))
    exact.initial = parameters.copy()
    parameters, fit = solve(exact, matrix, lo, hi, lower, upper, iterations=0,
        curvature_gradient=lambda p: fast.objective(p)[1], checkpoint=checkpoint('original'))
    final = exact.evaluate(parameters)
    after = exact.validation(parameters)
    final_parameters = exact.expand(parameters)
    gradient_checks = []
    c = next(i for i, v in enumerate(model['fixed_penalties']) if v is not None)
    band, scale = experiment.bands[c], np.sqrt(experiment.weights[c])
    core, strength = model['core_counts'][c], model['fixed_penalties'][c]
    fitted = profile(band, final_parameters, scale, core, strength, derivatives=True)
    direction = np.random.default_rng(874918).normal(size=len(final_parameters))
    direction[curve.sections[1]] = 0.
    direction /= np.linalg.norm(direction)
    for h in (.01, .001, .0001):
        plus, minus = [profile(band, final_parameters+sign*h*direction, scale, core, strength)
            for sign in (1, -1)]
        gradient_checks.append({'step_m': h,
            'normalized_likelihood_gradient_error': abs((plus['value']-minus['value'])/(2*h)-fitted['gradient']@direction)/band.energy,
            'logdet_gradient_error': abs((plus['logdet']-minus['logdet'])/(2*h)-fitted['volume_gradient']@direction)})
    if min(c['normalized_likelihood_gradient_error'] for c in gradient_checks)>1e-7 or min(c['logdet_gradient_error'] for c in gradient_checks)>1e-5:
        raise ValueError('Original-row marginal likelihood gradient failed')
    output = {**model, **identity, 'parameters': final_parameters.tolist(), 'fit': fit,
        'search_fit': search, 'marginal_before': first['channels'], 'marginal_after': final['channels'],
        'marginal_before_objective': first['value'], 'marginal_after_objective': final['value'],
        'before': before, 'after': after, 'motion_change': changes(aggregate(before), aggregate(after)),
        'gradient_checks': gradient_checks, 'gradient_channel': c,
        'maximum_parameter_change_m': float(np.max(abs(final_parameters-np.asarray(model['parameters'])))),
        'complete': True, 'seconds': time.monotonic()-start,
        'implementation_sha256': {name: fingerprint(Path(__file__).with_name(name)) for name in
            ('fit_marginal_motion.py', 'marginal_scene.py', 'mixed_experiment.py', 'mixed_scene.py',
             'field.py', 'full_motion.py', 'ridge_motion.py', 'spline_motion.py', 'fit_physical_profile.py')}}
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({'stage': 'complete', 'converged': fit['converged'],
        'kkt': fit['maximum_kkt_residual'], 'change': output['motion_change'],
        'max_parameter_change_m': output['maximum_parameter_change_m']}), flush=True)


if __name__ == '__main__':
    main()
