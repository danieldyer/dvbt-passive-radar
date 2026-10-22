"""Refine yaw-free motion with independently acquired delayed-path nuisance."""
import argparse
import json
from pathlib import Path

import numpy as np

from companion_motion import add_companion
from field import Field
from fit_marginal_motion import make_bands
from fit_physical_profile import solve
from marginal_motion import MarginalMotion
from refine_full_capture_motion import constraints
from scene_dictionary import fingerprint
from spline_motion import SplineMotion
from validate_motion import summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('fields', 'motion', 'companion', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--iterations', type=int, default=160)
    args = p.parse_args()
    model = json.loads(args.motion.read_text())
    acquired = json.loads(args.companion.read_text())
    if args.output.exists() or not acquired['complete'] or acquired['source_sha256'] != fingerprint(args.motion):
        raise ValueError('Need a completed acquisition at this source motion and a fresh output file')
    if not all(t['fit']['converged'] for t in acquired['trials']):
        raise ValueError('Finish numerical convergence of the companion hypotheses first')
    curve = SplineMotion(model)
    if not model.get('fixed_ne_frame') or np.any(curve.yaw_coefficients):
        raise ValueError('Keep yaw fixed at zero')
    fields = []
    for c in range(5):
        with np.load(args.fields/f'channel-{c}.npz') as d:
            fields.append(Field(dict(d), d['pulse_samples']))
    bands = make_bands(fields, model)
    diagnostic = [add_companion(b, model['parameters'], [t for t in acquired['trials'] if t['channel'] == c])
        for c, b in enumerate(bands)]
    joint = MarginalMotion(bands, model['parameters'])
    before = joint.validation(joint.initial)
    _, matrix, lo, hi, lower, upper = constraints(model, common_path=True)
    frozen = (joint.active >= curve.sections[1].start)&(joint.active < curve.sections[1].stop)
    lower[frozen] = upper[frozen] = 0.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    identity = {'source_motion': str(args.motion), 'source_motion_sha256': fingerprint(args.motion),
        'companion_acquisition': str(args.companion), 'companion_sha256': fingerprint(args.companion),
        'companion_nuisance': diagnostic, 'nuisance_scope': __doc__,
        'implementation_sha256': {name: fingerprint(Path(__file__).with_name(name)) for name in
            ('fit_companion_motion.py', 'companion_motion.py', 'delayed_path.py', 'gaussian_scene.py',
             'marginal_motion.py', 'field.py', 'full_motion.py', 'fit_physical_profile.py', 'stationarity.py')}}
    def checkpoint(parameters, trace):
        if len(trace)%5 == 0:
            print(json.dumps(trace[-1]), flush=True)
            args.output.with_suffix('.checkpoint.json').write_text(json.dumps({**identity,
                'parameters': joint.expand(parameters).tolist(), 'trace': trace}, indent=2)+'\n')
    parameters, fit = solve(joint, matrix, lo, hi, lower, upper,
        iterations=args.iterations, checkpoint=checkpoint)
    validation = joint.validation(parameters)
    final = joint.evaluate(parameters)
    result = {**model, **identity, 'parameters': joint.expand(parameters).tolist(), 'fit': fit,
        'before_same_nuisance': before, 'before_same_nuisance_summary': summarize(before),
        'original_row_validation_after': validation, 'after_summary': summarize(validation),
        'scene_variances': final['scene_variances'], 'effective_scene_ranks': final['effective_scene_ranks'],
        'phase_centres_validated': False, 'fitted_orientation': False,
        'rendering_data': 'Original archived profiles; the fitted nuisance is used only in autofocus, not removed from the rendered signal.',
        'validation_scope': 'Same fixed companion/stationary projection on both sides of the motion comparison. Companion path hypotheses use training samples only. Heldout nuisance amplitudes are eliminated; no heldout main-target coefficient or phase is fitted.',
        'complete': True}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'converged': fit['converged'], 'kkt': fit['maximum_kkt_residual'],
        'heldout_error_change_percent': 100*(result['after_summary']['prediction_error']/
            result['before_same_nuisance_summary']['prediction_error']-1),
        'heldout_matched_energy_change_db': 10*np.log10(result['after_summary']['matched_energy']/
            result['before_same_nuisance_summary']['matched_energy'])}), flush=True)


if __name__ == '__main__':
    main()
