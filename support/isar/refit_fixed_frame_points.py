"""Refit spatial support under the yaw-free original-row observation operator.

The translation curve, antenna positions, range origins and point heights are
frozen. Only the horizontal locations of the existing eight reflectors per
band change, within declared bounds, on the original training half-seconds.
These are scattering-model controls, not measured physical point positions.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from field import Field
from fit_marginal_motion import make_bands
from fit_physical_profile import solve
from marginal_points import MarginalPoints
from scene_dictionary import fingerprint
from spline_motion import SplineMotion
from validate_motion import summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('fields', 'motion', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--maximum-shift-m', type=float, default=60.)
    p.add_argument('--iterations', type=int, default=120)
    args = p.parse_args()
    model = json.loads(args.motion.read_text())
    if not model.get('fixed_ne_frame') or np.any(SplineMotion(model).yaw_coefficients):
        raise ValueError('Yaw must be fixed at zero')
    if args.maximum_shift_m <= 0 or args.output.exists():
        raise ValueError('Use positive coordinate bounds and a fresh result path')
    fields = []
    for c in range(5):
        with np.load(args.fields/f'channel-{c}.npz') as d:
            fields.append(Field(dict(d), d['pulse_samples']))
    bands = make_bands(fields, model)
    result = {**model, 'source_motion': str(args.motion), 'source_motion_sha256': fingerprint(args.motion),
        'motion_fit_before_point_refinement': model['fit'],
        'point_refinement_scope': __doc__, 'initial_points_body_m': model['selected_points_body_m'],
        'maximum_coordinate_shift_m': args.maximum_shift_m, 'point_refinement': [], 'complete': False,
        'implementation_sha256': {name: fingerprint(Path(__file__).with_name(name)) for name in
            ('refit_fixed_frame_points.py', 'marginal_points.py', 'gaussian_scene.py',
             'field.py', 'full_motion.py', 'fit_physical_profile.py', 'stationarity.py')},
        'phase_centres_validated': False}
    result.pop('fit')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    before, after, locations = [], [], []
    for c, band in enumerate(bands):
        joint = MarginalPoints(band, model['parameters'])
        initial = joint.validation(joint.initial)
        n = len(joint.initial)
        def checkpoint(parameters, trace):
            if len(trace)%10 == 0:
                print(json.dumps({'channel': c, **trace[-1]}), flush=True)
        parameters, details = solve(joint, np.empty((0, n)), np.empty(0), np.empty(0),
            np.full(n, -args.maximum_shift_m), np.full(n, args.maximum_shift_m),
            iterations=args.iterations, checkpoint=checkpoint)
        final = joint.validation(parameters)
        before.append(initial); after.append(final)
        points = joint.coordinates(parameters)
        locations.append(points.tolist())
        result['point_refinement'].append({'channel': c, 'fit': details,
            'before': initial, 'after': final, 'points_body_m': points.tolist(),
            'coordinate_shift_m': parameters.reshape(-1, 2).tolist(),
            'bound_hits': int(np.sum(abs(parameters) >= args.maximum_shift_m-1e-6)),
            'scene_variance': joint.evaluate(parameters)['scene_variance'],
            'effective_scene_rank': joint.evaluate(parameters)['effective_scene_rank']})
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({'channel': c, 'converged': details['converged'],
            'kkt': details['maximum_kkt_residual'], 'heldout_error_change_percent':
            100*(final['prediction_error']/initial['prediction_error']-1)}), flush=True)
    result.update(complete=True, selected_points_body_m=locations, point_refinement_before=summarize(before),
        point_refinement_after=summarize(after), original_row_validation_after=after, after_summary=summarize(after),
        point_refinement_all_converged=all(c['fit']['converged'] for c in result['point_refinement']))
    total = sum(b.energy for b in bands)
    result['fit'] = {'converged': result['point_refinement_all_converged'],
        'objective': sum(b.energy*c['fit']['objective'] for b, c in zip(bands, result['point_refinement']))/total,
        'maximum_kkt_residual': max(b.energy*c['fit']['maximum_kkt_residual']/total
            for b, c in zip(bands, result['point_refinement'])),
        'maximum_constraint_violation': max(c['fit']['maximum_constraint_violation'] for c in result['point_refinement']),
        'scope': 'Conditional horizontal-coordinate fit at fixed motion. Joint motion/scene stationarity has not been asserted.'}
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
