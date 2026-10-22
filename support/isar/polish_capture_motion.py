"""Finish saved single-anchor fits with a checked local Newton correction.

The observations, scene support, physical constraints and objective are fixed.
The correction is accepted only with positive tangent curvature, feasibility,
improved stationarity and no material likelihood increase. Original files are
preserved and original-row withheld predictions are recomputed.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ais import utc_seconds
from field import Field
from fit_joint_motion import information_scaling
from full_motion import Band
from imaging import motion_state, sites
from midpoint_model import discrepancy, representative
from physical_profile import PhysicalProfile
from refine_full_capture_motion import constraints
from scene_dictionary import fingerprint
from stationarity import check, polish
from validate_motion import summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('fields', 'seed', 'source', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--windows', type=int, nargs='+', required=True)
    args = p.parse_args()
    if args.source.resolve()==args.output.resolve():
        raise ValueError('Preserve source fits in a separate directory')
    seed = json.loads(args.seed.read_text())
    tx, rx, _ = sites(Path(seed['geometry_directory']))
    fields = []
    for c in range(5):
        with np.load(args.fields/f'channel-{c}.npz') as f:
            fields.append(Field(dict(f), f['pulse_samples']))
    args.output.mkdir(parents=True, exist_ok=True)
    for window in args.windows:
        path = args.source/f'window-{window:03d}.json'
        source = json.loads(path.read_text())
        if not (source['hardware_range_offsets_fixed'] and source.get('linear_reference')):
            raise ValueError('Use a fixed-hardware single-anchor physical motion fit')
        model = {**seed, **source}
        model.pop('path_reference_time_s', None)
        _, matrix, lo, hi, lower, upper = constraints(model, common_path=True)
        exact, fast = [], []
        for c, full in enumerate(fields):
            field = full.windowed(model['interval_s'][0]+.03, model['interval_s'][1]-.03)
            selected = np.floor((field.time-model['interval_s'][0])/.5).astype(int)%2==0
            for item, destination in ((field, exact), (representative(field), fast)):
                prior = motion_state(None, utc_seconds(model['nominal_start_utc'])+item.data['row_time'], model)
                destination.append(Band(item, prior, tx[3], rx, model['selected_points_body_m'][c], model,
                    selected, range_bias_m=model['range_bias_m'][c]))
        joint = PhysicalProfile(fast, model['parameters'])
        initial = joint.evaluate(joint.initial)
        transform, _ = information_scaling(initial['gauss_newton'], 1., lower<upper)
        fitted, details = polish(joint.objective, joint.initial, transform, matrix, lo, hi, lower, upper)
        final = joint.evaluate(fitted)
        kkt, violation = check(fitted, final['gradient'], matrix, lo, hi, lower, upper)
        parameters = joint.expand(fitted)
        original = PhysicalProfile(exact, parameters)
        checks = original.validation(original.initial)
        approximation = []
        for a, b in zip(exact, fast):
            da, _ = a.design(parameters)
            db, _ = b.design(parameters)
            approximation.append([discrepancy(da[:, j], db[:, j]) for j in range(da.shape[1])])
        result = {**source, 'parameters': parameters.tolist(), 'after_summary': summarize(checks),
            'original_row_validation_after': checks, 'fit_before_refinement': source['fit'],
            'refinement_source': str(path), 'refinement_source_sha256': fingerprint(path),
            'refinement': details, 'refinement_scope': __doc__,
            'midpoint_approximation_at_fitted_motion': approximation,
            'fit': {**source['fit'], 'objective': final['value'], 'maximum_kkt_residual': kkt,
                'maximum_constraint_violation': violation, 'stationarity_polish': details,
                'converged': bool(kkt<=1e-7 and violation<=1e-7)}}
        (args.output/path.name).write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({'window': window, 'accepted': details['accepted'], 'converged': result['fit']['converged'],
            'kkt': kkt, 'objective_before': initial['value'], 'objective_after': final['value'],
            'heldout_error': result['after_summary']['prediction_error']}), flush=True)


if __name__ == '__main__':
    main()
