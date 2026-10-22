"""Compare free yaw with translation-only fits on identical 30-second data.

Both fits use the same physical point support at the aperture midpoint,
training/heldout samples, stationary nuisance model and translation bounds.
The fixed-frame model keeps east/north/up axes throughout the aperture. No AIS
trajectory, heading penalty, phase-only correction or display normalization is
added to the likelihood. Complex point responses are profiled independently in
each band. Antenna positions and the calibration used for rendering stay fixed.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ais import utc_seconds
from field import Field
from fit_physical_profile import solve
from fixed_frame_motion import geographic_frame, point_centred_start, restrict_interval
from full_capture_motion import constrained_curve
from full_motion import Band
from imaging import motion_state, sites
from midpoint_model import discrepancy, representative
from physical_profile import PhysicalProfile
from refine_full_capture_motion import constraints
from scene_dictionary import fingerprint
from spline_motion import SplineMotion
from validate_motion import summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('fields', 'seed', 'source', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--windows', type=int, nargs='+', required=True)
    p.add_argument('--duration', type=float, default=30.)
    p.add_argument('--iterations', type=int, default=140)
    p.add_argument('--offset-starts', action='store_true', help='Also seed translation at the four support corners')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
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
        centre = np.mean(source['interval_s'])
        model = {**seed, **source}
        model.pop('path_reference_time_s', None)
        model = restrict_interval(model, [centre-args.duration/2, centre+args.duration/2])
        fixed = geographic_frame(model)
        variants = [('free', model), ('fixed-ne', fixed)]
        if args.offset_starts:
            points = np.concatenate(model['selected_points_body_m'])
            corners = [[x, y, 20.] for x in (points[:, 0].min(), points[:, 0].max())
                                  for y in (points[:, 1].min(), points[:, 1].max())]
            variants += [(f'fixed-ne-start-{i}', point_centred_start(model, fixed, point, tx[3], rx))
                         for i, point in enumerate(corners)]
        for variant, description in variants:
            output = args.output/f'window-{window:03d}-{variant}.json'
            if output.exists():
                previous = json.loads(output.read_text())
                if (not args.resume or previous['source_fit_sha256'] != fingerprint(path) or
                        previous['interval_s'] != description['interval_s'] or previous['variant'] != variant):
                    raise ValueError('Use a fresh output directory for this paired comparison')
                continue
            exact, fast = [], []
            for c, full in enumerate(fields):
                field = full.windowed(description['interval_s'][0]+.03, description['interval_s'][1]-.03)
                selected = np.floor((field.time-description['interval_s'][0])/.5).astype(int)%2 == 0
                for item, target in ((field, exact), (representative(field), fast)):
                    prior = motion_state(None, utc_seconds(description['nominal_start_utc'])+item.data['row_time'], description)
                    target.append(Band(item, prior, tx[3], rx, description['selected_points_body_m'][c], description,
                        selected, range_bias_m=description['range_bias_m'][c]))
            joint = PhysicalProfile(fast, description['parameters'])
            _, matrix, lo, hi, lower, upper = constraints(description, common_path=True)
            if variant != 'free':
                yaw = SplineMotion(description).sections[1]
                frozen = (joint.active >= yaw.start)&(joint.active < yaw.stop)
                lower[frozen] = upper[frozen] = 0.
            # Re-basing spline knots can slightly overshoot an acceleration
            # limit. Project the starting coefficients onto the same declared
            # feasible set; this adds no penalty to the radar likelihood.
            free = lower < upper
            variable = np.any(matrix[:, free] != 0., axis=1)
            initial = joint.initial.copy()
            joint.initial[free] = constrained_curve(np.eye(free.sum()), initial[free],
                np.ones(free.sum()), np.r_[matrix[variable][:, free], np.eye(free.sum())],
                np.r_[hi[variable], upper[free]])
            feasible_initial = joint.expand(joint.initial)
            def checkpoint(parameters, trace):
                if len(trace)%10 == 0:
                    print(json.dumps({'window': window, 'variant': variant, **trace[-1]}), flush=True)
            parameters, fit = solve(joint, matrix, lo, hi, lower, upper,
                iterations=args.iterations, checkpoint=checkpoint)
            full = joint.expand(parameters)
            original = PhysicalProfile(exact, full)
            validation = original.validation(original.initial)
            approximation = []
            for a, b in zip(exact, fast):
                da, _ = a.design(full); db, _ = b.design(full)
                approximation.append([discrepancy(da[:, j], db[:, j]) for j in range(da.shape[1])])
            result = {**description, 'parameters': full.tolist(), 'fit': fit,
                'original_row_validation_after': validation, 'after_summary': summarize(validation),
                'midpoint_approximation_at_fitted_motion': approximation,
                'source_fit': str(path), 'source_fit_sha256': fingerprint(path), 'variant': variant,
                'paired_control_scope': __doc__, 'scope': __doc__, 'complete': True,
                'fitted_real_parameters': int(np.sum(lower < upper)),
                'feasible_initial_parameters': feasible_initial.tolist(),
                'fitted_orientation': variant == 'free'}
            output.write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps({'window': window, 'variant': variant, 'converged': fit['converged'],
                'training': fit['objective'], 'heldout': result['after_summary']['prediction_error'],
                'conditional_loss_db': result['after_summary']['conditional_coherent_loss_db']}), flush=True)


if __name__ == '__main__':
    main()
