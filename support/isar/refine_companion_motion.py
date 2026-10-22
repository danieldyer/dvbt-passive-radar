"""Accelerate search for yaw-free motion, with exact final acceptance.

The midpoint model proposes an initial motion and supplies Newton curvature.
Original-row values, gradients, physical feasibility and KKT stationarity
govern every final correction. Both models use the exact same observations
and fixed nuisance projector. No approximation enters final predictions or BP.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ais import utc_seconds
from companion_motion import add_companion
from field import Field
from fit_marginal_motion import make_bands
from fit_physical_profile import solve
from full_motion import Band
from imaging import motion_state
from marginal_motion import MarginalMotion
from midpoint_model import representative, discrepancy
from refine_full_capture_motion import constraints
from scene_dictionary import fingerprint
from spline_motion import SplineMotion
from validate_motion import summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('fields', 'motion', 'companion', 'checkpoint', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Preserve earlier fits')
    model = json.loads(args.motion.read_text())
    acquisition = json.loads(args.companion.read_text())
    blob = args.checkpoint.read_bytes()
    checkpoint = json.loads(blob)
    if (not acquisition['complete'] or acquisition['source_sha256'] != fingerprint(args.motion)
            or checkpoint['source_motion_sha256'] != fingerprint(args.motion)
            or checkpoint['companion_sha256'] != fingerprint(args.companion)
            or not all(t['fit']['converged'] for t in acquisition['trials'])):
        raise ValueError('The saved search and nuisance acquisition must have identical inputs')
    curve = SplineMotion({**model, 'parameters': checkpoint['parameters']})
    if not model.get('fixed_ne_frame') or np.any(curve.yaw_coefficients):
        raise ValueError('Keep yaw fixed at zero')
    fields = []
    for c in range(5):
        with np.load(args.fields/f'channel-{c}.npz') as d:
            fields.append(Field(dict(d), d['pulse_samples']))
    bands = make_bands(fields, model)
    cost = [add_companion(b, model['parameters'], [t for t in acquisition['trials'] if t['channel'] == c])
        for c, b in enumerate(bands)]
    fast = []
    for b in bands:
        f = representative(b.field)
        prior = motion_state(None, utc_seconds(model['nominal_start_utc'])+f.data['row_time'], model)
        candidate = Band(f, prior, b.reference_transmitter, b.receiver, b.points, model,
            b.selected_time, range_bias_m=b.range_bias)
        candidate.nuisance, candidate.clean, candidate.energy = b.nuisance, b.clean, b.energy
        fast.append(candidate)
    exact = MarginalMotion(bands, checkpoint['parameters'])
    approximate = MarginalMotion(fast, checkpoint['parameters'])
    np.testing.assert_array_equal(exact.active, approximate.active)
    _, matrix, lo, hi, lower, upper = constraints(model, common_path=True)
    frozen = (exact.active >= curve.sections[1].start)&(exact.active < curve.sections[1].stop)
    lower[frozen] = upper[frozen] = 0.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    before = exact.validation(exact.initial)
    parameters, search = solve(approximate, matrix, lo, hi, lower, upper, iterations=120,
        checkpoint=lambda p, trace: print(json.dumps({'stage': 'search', **trace[-1]}), flush=True)
        if len(trace)%10 == 0 else None)
    print(json.dumps({'stage': 'search_done', 'converged': search['converged'],
        'kkt': search['maximum_kkt_residual']}), flush=True)
    exact.initial = parameters.copy()
    parameters, fit = solve(exact, matrix, lo, hi, lower, upper, iterations=0,
        curvature_gradient=lambda p: approximate.objective(p)[1])
    validation = exact.validation(parameters)
    final = exact.evaluate(parameters)
    errors = []
    for full, short in zip(bands, fast):
        a, b = [band.design(exact.expand(parameters))[0] for band in (full, short)]
        errors.append([discrepancy(a[:, j], b[:, j]) for j in range(a.shape[1])])
    result = {**model, 'parameters': exact.expand(parameters).tolist(), 'fit': fit,
        'original_row_validation_after': validation, 'after_summary': summarize(validation),
        'before_same_nuisance': before, 'before_same_nuisance_summary': summarize(before),
        'source_motion': str(args.motion), 'source_motion_sha256': fingerprint(args.motion),
        'companion_acquisition': str(args.companion), 'companion_sha256': fingerprint(args.companion),
        'starting_checkpoint': str(args.checkpoint), 'starting_checkpoint_sha256': hashlib.sha256(blob).hexdigest(),
        'starting_parameters': checkpoint['parameters'], 'search_fit': search,
        'companion_nuisance': cost, 'scene_variances': final['scene_variances'],
        'effective_scene_ranks': final['effective_scene_ranks'], 'search_model_discrepancy': errors,
        'scope': __doc__, 'fitted_orientation': False, 'phase_centres_validated': False,
        'calibration_validated': False, 'rendering_data': 'Unmodified archived profiles; no companion subtraction.',
        'implementation_sha256': {name: fingerprint(Path(__file__).with_name(name)) for name in
            ('refine_companion_motion.py', 'companion_motion.py', 'midpoint_model.py', 'stationarity.py',
             'fit_physical_profile.py', 'marginal_motion.py', 'gaussian_scene.py', 'field.py')}, 'complete': True}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'stage': 'exact_final', 'converged': fit['converged'],
        'kkt': fit['maximum_kkt_residual'], 'objective': fit['objective'], 'heldout': result['after_summary']}), flush=True)


if __name__ == '__main__':
    main()
