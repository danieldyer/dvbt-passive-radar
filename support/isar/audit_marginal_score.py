"""Compare independent marginal likelihood implementations on original data."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from marginal_scene import profile as reference
from marginal_score import profile
from mixed_experiment import load
from scene_dictionary import fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--information', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve completed checks')
    experiment = load(args.source)
    model = experiment.model
    bound = json.loads(args.information.read_text())
    if not bound['complete'] or bound['source_sha256'] != fingerprint(args.source):
        raise ValueError('Use the bound for this exact model')
    active = np.array(bound['active_physical_indices'])
    covariance = np.array(bound['crlb_covariance'])
    delta = np.zeros(len(active))
    result = {'source': str(args.source), 'source_sha256': fingerprint(args.source),
        'information': str(args.information), 'information_sha256': fingerprint(args.information),
        'channels': [], 'complete': False}
    for c, (band, weight, core, strength) in enumerate(zip(experiment.bands, experiment.weights,
            model['core_counts'], model['fixed_penalties'])):
        start = time.monotonic()
        expected = reference(band, model['parameters'], np.sqrt(weight), core, strength, derivatives=True)
        reference_seconds = time.monotonic()-start
        start = time.monotonic()
        actual = profile(band, model['parameters'], np.sqrt(weight), core, strength)
        score_seconds = time.monotonic()-start
        difference = (actual['gradient']-expected['gradient'])[active]
        delta += difference
        prediction_difference = np.linalg.norm(actual['target']@(actual['physical_coefficient']-expected['physical_coefficient']))
        prediction_error = prediction_difference/np.linalg.norm(expected['target']@expected['physical_coefficient'])
        result['channels'].append({'channel': c, 'absolute_objective_error': abs(actual['value']-expected['value']),
            'relative_objective_error': abs(actual['value']/expected['value']-1),
            'maximum_normalized_gradient_error': float(abs(difference).max()/band.energy),
            'prediction_relative_complex_error': float(prediction_error),
            'prediction_difference_in_unit_noise_norm': float(prediction_difference),
            'augmented_reference_seconds': reference_seconds, 'single_svd_score_seconds': score_seconds})
        print(json.dumps(result['channels'][-1]), flush=True)
    result.update(complete=True, information_normalized_score_discrepancy=float(np.sqrt(max(0., delta@covariance@delta))),
        implementation_sha256={name: fingerprint(Path(__file__).with_name(name)) for name in
            ('audit_marginal_score.py', 'marginal_score.py', 'marginal_scene.py', 'mixed_experiment.py', 'mixed_scene.py', 'field.py', 'full_motion.py', 'ridge_motion.py')})
    # A fixed relative waveform tolerance can reject harmless disagreement in
    # an ill-conditioned nuisance coefficient solve. Bound the actual complex
    # prediction error in unit thermal-noise norm, and the resulting first-order
    # motion change in CRLB units. Each must be below one percent of noise.
    result['numerically_adequate_for_efficiency_trials'] = bool(
        result['information_normalized_score_discrepancy'] <= .01 and
        max(c['prediction_difference_in_unit_noise_norm'] for c in result['channels']) <= .01)
    result['precision_gate'] = 'Prediction difference norm and Fisher-normalized score difference each <= 0.01 under the declared unit-noise model. Not a model-validity or achieved estimator-efficiency claim.'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    if not result['numerically_adequate_for_efficiency_trials']:
        raise ValueError('Independent score precision does not meet the declared statistical gate')
    print(json.dumps({'complete': True, 'normalized_score_error': result['information_normalized_score_discrepancy']}), flush=True)


if __name__ == '__main__':
    main()
