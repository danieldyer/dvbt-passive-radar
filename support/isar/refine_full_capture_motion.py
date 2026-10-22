"""Refine whole-capture radar motion on complex range/phase observations.

Each local fit profiles independent complex reflector responses per band and
stationary nuisance amplitudes. Delay, carrier phase and symbol Doppler move
together. The midpoint forward approximation is checked against original-row
predictions; final held-out scores use the exact original-row model.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from ais import read_csv, utc_seconds
from field import Field
from fit_physical_profile import solve
from full_capture_motion import constrained_curve, join_curves
from full_motion import Band
from imaging import physical_state, motion_state, sites
from midpoint_model import discrepancy, representative
from physical_profile import PhysicalProfile
from spline_motion import SplineMotion, basis, canonical_model
from validate_motion import summarize
from motion import enu_basis
from radar_rate_initialization import rate_seed


def constraints(model, *, common_path=False):
    interval = model['interval_s']
    time = np.linspace(*interval, int(np.ceil(np.diff(interval)[0]*8))+1)
    a = basis(time, interval, model['path_spacing_seconds'], fixed_midpoint=True)
    b = basis(time, interval, model['heading_spacing_seconds'])
    n, m = a.shape[1], b.shape[1]
    active_path = n if common_path else n-1
    dp = basis(time, interval, model['path_spacing_seconds'], derivative=2)[:, :active_path]
    dy = np.concatenate([basis(time, interval, model['heading_spacing_seconds'], derivative=i) for i in (1, 2)])
    p = np.array(model['parameters'])
    path_design = basis(time, interval, model['path_spacing_seconds']) if common_path else a[:, :-1]
    source_path = basis(time, interval, model['path_spacing_seconds'],
        fixed_midpoint=model.get('anchor_path', True))@p[:n]
    path = constrained_curve(path_design, source_path, np.ones(len(time)),
        np.r_[dp, np.eye(active_path)], np.r_[np.full(len(dp), .15), np.full(active_path, 200.)])
    yaw_seed = b@p[n:n+m]+model.get('initial_yaw_rate_deg_ps', 0.)*(time-np.mean(interval))
    yaw = constrained_curve(b, yaw_seed, np.ones(len(time)),
        np.r_[dy, np.eye(m)], np.r_[np.full(len(time), .5), np.full(len(time), .02), np.full(m, 25.)])
    p = np.r_[path, yaw] if common_path else np.r_[path, 0., yaw]
    matrix = np.zeros((len(time)*3, active_path+m+(0 if common_path else 5)))
    matrix[:len(time), :active_path] = dp
    matrix[len(time):, active_path:active_path+m] = dy
    limit = np.r_[np.full(len(time), .15), np.full(len(time), .5), np.full(len(time), .02)]
    lower = -np.r_[np.full(active_path, 200.), np.full(m, 25.), np.full(0 if common_path else 5, 100.)]
    if model.get('transverse_spacing_seconds') is not None:
        if not common_path:
            raise ValueError('Two-dimensional motion requires fixed hardware range offsets')
        spacing = model['transverse_spacing_seconds']
        cross = basis(time, interval, spacing)
        rate = basis(time, interval, spacing, derivative=1)
        acceleration = basis(time, interval, spacing, derivative=2)
        count = cross.shape[1]
        active = count-int(model.get('anchor_transverse', False))
        extended = np.zeros((len(matrix)+2*len(time), matrix.shape[1]+active))
        extended[:len(matrix), :matrix.shape[1]] = matrix
        extended[len(matrix):, matrix.shape[1]:] = np.r_[rate[:, :active], acceleration[:, :active]]
        matrix = extended
        limit = np.r_[limit, np.full(len(time), 5.), np.full(len(time), .1)]
        lower = np.r_[lower, np.full(active, -1000.)]
        p = np.r_[p, np.full(count, model.get('initial_transverse_m', 0.))]
    return p, matrix, -limit, limit, lower, -lower


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fields', 'phase-seed', 'ais', 'geometry-directory', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--windows', type=int, nargs='+')
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--rescore', action='store_true',
        help='Recompute matching before/after original-row validation for existing completed fits')
    parser.add_argument('--initial-fits', type=Path,
        help='Initialize from completed local full-response fits instead of the phase-only initializer')
    parser.add_argument('--transverse-spacing', type=float,
        help='Also fit physical horizontal transverse displacement, in metres; spline knot interval in seconds')
    parser.add_argument('--initial-transverse-m', type=float, default=0.)
    parser.add_argument('--initial-yaw-rate-deg-ps', type=float, default=0.,
        help='Initial rotation-rate perturbation only; no penalty or constraint toward this value during fitting')
    parser.add_argument('--radar-rate-start', action='store_true',
        help='Initialize physical path rate from training-only, stationary-projected phase products')
    parser.add_argument('--zero-motion-start', action='store_true',
        help='Start the full-response likelihood at the coarse AIS frame instead of the phase-only correction')
    parser.add_argument('--fixed-heading-reference', action='store_true',
        help='Use only the coarse midpoint heading as reference; infer all heading evolution from radar')
    parser.add_argument('--linear-reference', action='store_true',
        help='Use one coarse midpoint pose and velocity; fit motion from radar without inheriting an AIS trajectory')
    parser.add_argument('--anchor-transverse', action='store_true',
        help='Keep the coarse transverse position at the aperture midpoint; fit its radar-derived rate and curvature')
    parser.add_argument('--common-path', action='store_true',
        help='Fit the physical path origin from radar; keep hardware range offsets fixed across the capture')
    args = parser.parse_args()
    if args.radar_rate_start and not (args.zero_motion_start and args.common_path and args.linear_reference):
        raise ValueError('Radar rate acquisition requires a zero-path, single-anchor, fixed-hardware control')
    seed = json.loads(args.phase_seed.read_text())
    track = read_csv(args.ais, reference_height_ellipsoid_m=seed['reference_height_ellipsoid_m'], mmsi=563052200)
    tx, rx, _ = sites(args.geometry_directory)
    epoch = utc_seconds(seed['nominal_start_utc'])
    fields = []
    for c in range(5):
        with np.load(args.fields/f'channel-{c}.npz') as f:
            fields.append(Field(dict(f), f['pulse_samples']))
    args.output.mkdir(parents=True, exist_ok=True)
    locals = []
    for i, source in enumerate(sorted(Path(seed['local_fits']).glob('window-*.json'))):
        if args.windows is not None and i not in args.windows:
            continue
        path = args.output/source.name
        existing = json.loads(path.read_text()) if path.exists() else None
        if existing is not None:
            identity = (existing['hardware_range_offsets_fixed'], existing.get('transverse_spacing_seconds'),
                existing.get('anchor_transverse', False), existing.get('zero_motion_start', False),
                'fixed_reference_heading_deg' in existing, 'linear_reference' in existing,
                existing.get('initial_yaw_rate_deg_ps', 0.), existing.get('radar_rate_start', False))
            requested = (args.common_path, args.transverse_spacing, args.anchor_transverse, args.zero_motion_start,
                args.fixed_heading_reference, args.linear_reference, args.initial_yaw_rate_deg_ps, args.radar_rate_start)
            if identity != requested:
                raise ValueError(f'Existing local fit uses another reference or parameterization: {path}')
            if not args.rescore:
                locals.append(existing); continue
        if args.rescore and existing is None:
            raise ValueError(f'No completed fit to rescore: {path}')
        model = {**seed, **json.loads(source.read_text())}
        if args.initial_fits is not None:
            initial = json.loads((args.initial_fits/source.name).read_text())
            if initial['interval_s'] != model['interval_s'] or not initial['complete']:
                raise ValueError('Initial local fit has different support or is incomplete')
            model.update({key: initial[key] for key in ('parameters', 'anchor_path')})
        model.pop('path_reference_time_s', None)
        model.pop('rock_spacing_seconds', None)
        model = canonical_model(model)
        if args.zero_motion_start:
            model['parameters'] = np.zeros_like(model['parameters'])
        if args.fixed_heading_reference:
            anchor = track.sample([epoch+np.mean(model['interval_s'])], max_gap_s=400.)
            velocity = np.einsum('nji,nj->ni', enu_basis(anchor['latitude'], anchor['longitude']), anchor['velocity_ecef_mps'])
            model['fixed_reference_heading_deg'] = float(np.degrees(np.arctan2(velocity[0, 0], velocity[0, 1])))
        if args.linear_reference:
            if args.transverse_spacing is None:
                raise ValueError('A free radar trajectory needs both horizontal translation coordinates')
            anchor_time = epoch+np.mean(model['interval_s'])
            anchor = physical_state(track, [anchor_time], seed['attitude_prior'], max_gap_s=400.)
            model['linear_reference'] = {'utc_seconds': anchor_time,
                **{key: anchor[key][0].tolist() for key in ('position_ecef_m', 'velocity_ecef_mps', 'body_to_ecef')},
                'scope': 'Single coarse pose/velocity initialization; no AIS trajectory enters the likelihood. Local tangent-plane baseline; all fitted motion is physical.'}
        if args.transverse_spacing is not None:
            model.update(transverse_spacing_seconds=args.transverse_spacing, horizontal_translation=True,
                         initial_transverse_m=args.initial_transverse_m, anchor_transverse=args.anchor_transverse)
        model['initial_yaw_rate_deg_ps'] = args.initial_yaw_rate_deg_ps
        model['parameters'], matrix, lo, hi, lower, upper = constraints(model, common_path=args.common_path)
        model['anchor_path'] = not args.common_path
        exact_bands, fast_bands = [], []
        for c, full in enumerate(fields):
            exact = full.windowed(model['interval_s'][0]+.03, model['interval_s'][1]-.03)
            selected = np.floor((exact.time-model['interval_s'][0])/.5).astype(int)%2 == 0
            points = np.array(model['selected_points_body_m'][c])
            for item, destination in ((exact, exact_bands), (representative(exact), fast_bands)):
                prior = motion_state(track, epoch+item.data['row_time'], {**model, 'max_ais_gap_s': 400.})
                destination.append(Band(item, prior, tx[3], rx, points, model, selected,
                    range_bias_m=seed['range_bias_m'][c]))
        acquisition = None
        if args.radar_rate_start:
            model['parameters'], acquisition = rate_seed(exact_bands, model['parameters'])
        joint = PhysicalProfile(fast_bands, model['parameters'], fit_ranges=not args.common_path)
        initial_exact = PhysicalProfile(exact_bands, model['parameters'], fit_ranges=not args.common_path)
        before = initial_exact.validation(initial_exact.initial)
        started = time.perf_counter()
        def checkpoint(parameters, trace):
            if len(trace)%10 == 0:
                print(json.dumps({'window': i, **trace[-1]}), flush=True)
                (args.output/f'window-{i:03d}.checkpoint.json').write_text(json.dumps({
                    'parameters': joint.expand(parameters).tolist(), 'ranges_m': parameters[joint.nmotion:].tolist(),
                    'trace': trace}, indent=2)+'\n')
        if existing is None:
            fitted, diagnostics = solve(joint, matrix, lo, hi, lower, upper, iterations=args.iterations,
                checkpoint=checkpoint, polish_stationarity=False)
            motion = joint.expand(fitted)
            origins = np.asarray(seed['range_bias_m']) if args.common_path else fitted[joint.nmotion:]
        else:
            diagnostics = existing['fit']
            motion = SplineMotion({**model, 'parameters': existing['parameters']}).parameters
            origins = np.asarray(existing['range_bias_m'])
        exact = PhysicalProfile(exact_bands, motion, fit_ranges=not args.common_path)
        for band, origin in zip(exact.bands, origins):
            band.range_bias = origin
        x = motion[exact.active] if args.common_path else np.r_[motion[exact.active], origins]
        after = exact.validation(x)
        for initial_score, fitted_score in zip(before, after):
            if (initial_score['stationary_validation_mode'] != fitted_score['stationary_validation_mode'] or
                initial_score['stationary_only_error'] != fitted_score['stationary_only_error']):
                raise ValueError('Before/after validation must use exactly the same observations and nuisance projection')
        approximation = []
        for a, b, origin in zip(exact_bands, fast_bands, origins):
            da, _ = a.design(motion, range_bias_m=origin)
            db, _ = b.design(motion, range_bias_m=origin)
            approximation.append([discrepancy(da[:, j], db[:, j]) for j in range(da.shape[1])])
        result = {'interval_s': model['interval_s'], 'parameters': motion.tolist(),
            'anchor_path': model['anchor_path'], 'hardware_range_offsets_fixed': args.common_path,
            'path_spacing_seconds': model['path_spacing_seconds'], 'heading_spacing_seconds': model['heading_spacing_seconds'],
            'range_bias_m': origins.tolist(), 'selected_points_body_m': model['selected_points_body_m'],
            'fit': diagnostics, 'original_row_validation_before': before, 'original_row_validation_after': after,
            'before_summary': summarize(before), 'after_summary': summarize(after),
            'initial_parameters': model['parameters'].tolist(), 'validation_version': 2,
            'zero_motion_start': args.zero_motion_start,
            'initial_yaw_rate_deg_ps': args.initial_yaw_rate_deg_ps,
            'radar_rate_start': args.radar_rate_start, 'radar_rate_acquisition': acquisition,
            'validation_operator': 'Identical independent stationary-nuisance projection on heldout data and templates before and after; motion and complex target coefficients frozen from training.',
            'midpoint_approximation_at_fitted_motion': approximation, 'seconds': time.perf_counter()-started,
            'scope': __doc__, 'complete': True}
        if existing is not None:
            result['fit_seconds'] = existing.get('fit_seconds', existing['seconds'])
            result['reconstructed_initial_objective_change'] = joint.evaluate(joint.initial)['value']-diagnostics['initial_objective']
            if existing.get('validation_version', 1) == 1:
                result['superseded_validation_version_1'] = {
                    'before': existing['original_row_validation_before'], 'before_summary': existing['before_summary'],
                    'reason': 'Before used stationary prediction; after used projection. Those raw error differences were not comparable.'}
            elif 'superseded_validation_version_1' in existing:
                result['superseded_validation_version_1'] = existing['superseded_validation_version_1']
        if args.transverse_spacing is not None:
            result.update(transverse_spacing_seconds=args.transverse_spacing, horizontal_translation=True,
                          anchor_transverse=args.anchor_transverse,
                          initial_transverse_m=args.initial_transverse_m,
                          transverse_motion_scope='Physical horizontal correction to the AIS initialization; fixed per-channel hardware offsets and independent band scattering amplitudes. No AIS residual term in the likelihood.')
        if args.fixed_heading_reference:
            result['fixed_reference_heading_deg'] = model['fixed_reference_heading_deg']
        if args.linear_reference:
            result['linear_reference'] = model['linear_reference']
        path.write_text(json.dumps(result, indent=2)+'\n'); locals.append(result)
        print(json.dumps({'window': i, 'converged': diagnostics['converged'],
            'exact_heldout_error_change_percent': 100*(result['after_summary']['prediction_error']/result['before_summary']['prediction_error']-1),
            'seconds': result['seconds']}), flush=True)
    # Do not silently form a capture-wide curve from an incomplete subset.
    expected = len(list(Path(seed['local_fits']).glob('window-*.json')))
    if len(locals) == expected and not args.common_path:
        parameters, continuity = join_curves(locals, seed['interval_s'])
        model = {**seed, 'parameters': parameters.tolist(), 'local_fits': str(args.output),
            'scope': __doc__, 'continuity': continuity,
            'limitation': 'Joined local full-response fits; global joint stationarity and physical channel calibration remain separate checks.'}
        (args.output/'motion.json').write_text(json.dumps(model, indent=2)+'\n')


if __name__ == '__main__':
    main()
