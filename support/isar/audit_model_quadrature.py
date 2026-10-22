"""Check forward-model quadrature in statistical, not signal-energy, units."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from audit_marginal_efficiency import draw_trial, quadrature_bands, synthetic_components
from marginal_score import ScoreMotion
from mixed_experiment import load
from scene_dictionary import fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--information', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--nodes', type=int, nargs='+', default=[2, 4])
    parser.add_argument('--seed', type=int, default=881924)
    parser.add_argument('--trial', type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve existing quadrature audits')
    bound = json.loads(args.information.read_text())
    if not bound['complete'] or bound['source_sha256'] != fingerprint(args.source):
        raise ValueError('Use the complete information calculation for this source')
    experiment = load(args.source, accelerated=True)
    model = experiment.model
    active = np.asarray(bound['active_physical_indices'])
    truth = np.asarray(model['parameters'])[active]
    eigen, vectors = np.linalg.eigh(bound['unit_noise_information']['total'])
    transform = vectors/np.sqrt(eigen)
    extras = (model['parameters'], experiment.weights, model['core_counts'], model['fixed_penalties'])
    components = synthetic_components(experiment, bound)
    draw_trial(experiment, components, args.seed, args.trial)
    result = {'source': str(args.source), 'source_sha256': fingerprint(args.source),
        'information': str(args.information), 'information_sha256': fingerprint(args.information),
        'seed': args.seed, 'trial': args.trial, 'scope': __doc__,
        'gate': 'Norm of exact-minus-approximate score after expected Fisher scaling <= 0.01 at the declared truth.',
        'records': [], 'complete': False}
    for name, bands in [('original', experiment.bands), ('midpoint', experiment.fast),
            *((f'gauss-{n}', quadrature_bands(experiment, n)) for n in args.nodes)]:
        joint = ScoreMotion(bands, *extras)
        started = time.monotonic()
        fitted = joint.evaluate(truth)
        score = transform.T@fitted['gradient']
        if not result['records']:
            exact_score, exact_value = score, fitted['value']
        record = {'model': name, 'seconds': time.monotonic()-started,
            'objective': fitted['value'], 'objective_difference': fitted['value']-exact_value,
            'score': score.tolist(), 'score_norm': float(np.linalg.norm(score)),
            'score_discrepancy': float(np.linalg.norm(score-exact_score)),
            'observations_identical': all(np.array_equal(a.field.y,b.field.y) for a,b in zip(experiment.bands,bands))}
        record['passes_statistical_score_gate'] = record['score_discrepancy'] <= .01
        result['records'].append(record)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(record), flush=True)
    result.update(complete=True, implementation_sha256={name:fingerprint(Path(__file__).with_name(name))
        for name in ('audit_model_quadrature.py','model_quadrature.py','audit_marginal_efficiency.py',
            'marginal_score.py','marginal_scene.py','mixed_experiment.py','full_motion.py','field.py','spline_motion.py')})
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
