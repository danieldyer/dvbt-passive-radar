"""Same recovery trial with a physical trust-region metric.

Fisher coordinates measure statistical error; they need not define a useful
nonlinear search ball far from a minimum. Here all free coordinates are metres,
and a 0.1 m initial trust radius limits physical steps, not fitted parameters.
No box, acceleration, scene or motion prior is added. The likelihood decides
step acceptance and radius adaptation. Final stationarity uses original rows
and the unchanged expected Fisher scaling, independently of search units.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from audit_marginal_efficiency import draw_trial, quadrature_bands, synthetic_components
from marginal_refinement import CurvatureMotion, fit
from marginal_score import ScoreMotion
from mixed_experiment import load
from scene_dictionary import fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--information', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trial', type=int, default=0)
    parser.add_argument('--seed', type=int, default=881924)
    parser.add_argument('--nodes', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=50)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve existing metric comparisons')
    started = time.monotonic()
    bound = json.loads(args.information.read_text())
    if not bound['complete'] or bound['source_sha256'] != fingerprint(args.source):
        raise ValueError('Use the complete information audit for this source')
    e = load(args.source, accelerated=True)
    e.fast = quadrature_bands(e, args.nodes)
    m = e.model
    draw_trial(e, synthetic_components(e, bound), args.seed, args.trial)
    extras = (m['parameters'], e.weights, m['core_counts'], m['fixed_penalties'])
    fast, exact = CurvatureMotion(e.fast, *extras), ScoreMotion(e.bands, *extras)
    active = np.array(bound['active_physical_indices'])
    np.testing.assert_array_equal(active, fast.active)
    truth = np.array(m['parameters'])[active]
    initial = truth+.02*np.sin(np.linspace(0., 2*np.pi, len(truth)))
    fisher = np.array(bound['unit_noise_information']['total'])
    eigen, vectors = np.linalg.eigh(fisher)
    metric = vectors/np.sqrt(eigen)
    search_scale = .1*np.eye(len(truth))
    result = {'source': str(args.source), 'source_sha256': fingerprint(args.source),
        'information': str(args.information), 'information_sha256': fingerprint(args.information),
        'seed': args.seed, 'trial': args.trial, 'scope': __doc__,
        'truth': truth.tolist(), 'fixed_initial': initial.tolist(), 'nodes': args.nodes,
        'initial_trust_radius_m': .1, 'trace': [], 'complete': False,
        'implementation_sha256': {name: fingerprint(Path(__file__).with_name(name)) for name in
            ('audit_marginal_metric.py','audit_marginal_efficiency.py','marginal_refinement.py',
             'model_quadrature.py','marginal_score.py','marginal_scene.py','mixed_scene.py',
             'mixed_experiment.py','field.py','full_motion.py','ridge_motion.py','spline_motion.py')}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def progress(record):
        evaluation = fast.last
        row = {**record, 'search_gradient_norm': record['score_norm'],
            'score_norm': float(np.linalg.norm(metric.T@evaluation['gradient'])),
            'maximum_coefficient_error_m': float(np.max(abs(evaluation['parameters']-truth)))}
        result['trace'].append(row)
        result['parameters'] = evaluation['parameters'].tolist()
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(row), flush=True)
    parameters, search = fit(fast, initial, search_scale, iterations=args.iterations,
        method='trust-exact', progress=progress)
    final = exact.evaluate(parameters)
    score = float(np.linalg.norm(metric.T@final['gradient']))
    error = parameters-truth
    result.update(complete=True, search=search, parameters=parameters.tolist(), error=error.tolist(),
        original_objective=final['value'], original_information_normalized_score_norm=score,
        stationary=bool(score <= .01), mahalanobis_error_per_parameter=float(error@fisher@error/len(error)),
        seconds=time.monotonic()-started)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('complete','stationary',
        'original_information_normalized_score_norm','mahalanobis_error_per_parameter','seconds')}), flush=True)


if __name__ == '__main__':
    main()
