"""Fit bounded physical support and channel phases jointly on training blocks.

The same original two-metre cells bound the point coordinates as in the earlier
independent-response refinement. Motion, heights, antenna geometry and envelope
origins stay fixed. Withheld observations never choose a fitted parameter.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares, minimize

from ais import read_csv, utc_seconds
from channel_alignment import AlignmentBand
from coherent_points import CoherentPoints
from field import Field
from fit_physical_profile import solve as constrained_solve
from imaging import physical_state, sites
from phase_coupling import PhaseCoupling
from profiled_point_phases import ProfiledPointPhases
from scene_dictionary import fingerprint
from spline_motion import SplineMotion
from validate_motion import validate, summarize


def assess(model, parameters, interval):
    result = model.evaluate(parameters)
    checks = [validate(b.field, out['target'], b.nuisance, b.selected_time, interval,
        coefficients=out['coefficient'], stationary_mode='project')
        for b, out in zip(model.bands, result['bands'])]
    return {'channels': checks, 'summary': summarize(checks)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('motion', 'support', 'alignment', 'ais', 'geometry-directory', 'phase-control', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--iterations', type=int, default=300)
    p.add_argument('--start', type=int, default=0, help='Training-objective rank of the existing phase start')
    p.add_argument('--solver', choices=['trf', 'lbfgs', 'profiled'], default='trf')
    p.add_argument('--resume', type=Path, help='Continue a saved parameter checkpoint in the same physical coordinates')
    args = p.parse_args()
    motion, support, alignment, phase_control = [json.loads(path.read_text())
        for path in (args.motion, args.support, args.alignment, args.phase_control)]
    if motion.get('antenna_fit') or motion.get('rock_spacing_seconds') or np.any(motion.get('channel_path_rate_bias_mps', 0.)):
        raise ValueError('Use fixed roll/pitch and nominal antennas without channel-rate corrections')
    if not support['complete'] or not support['fit']['converged']:
        raise ValueError('Complete the independent coordinate refinement first')
    points = np.asarray(support['initial_points_body_m'])
    initial_points = np.asarray(support['points_body_m'])
    if not np.array_equal(initial_points[:, 2], points[:, 2]):
        raise ValueError('This control holds heights fixed')
    curve = SplineMotion(motion)
    tx, rx, _ = sites(args.geometry_directory)
    track = read_csv(args.ais, reference_height_ellipsoid_m=motion['reference_height_ellipsoid_m'], mmsi=563052200)
    bands = []
    for c in range(5):
        with np.load(Path(alignment['input_fields'])/f'channel-{c}.npz') as f:
            field = Field(dict(f), f['pulse_samples']).windowed(curve.interval[0]+.02, curve.interval[1]-.02)
        state = physical_state(track, utc_seconds(motion['nominal_start_utc'])+field.data['row_time'], motion['attitude_prior'])
        state = curve.apply(field.data['row_time'], state, tx[3], rx)
        block = np.floor(field.time-curve.interval[0]).astype(int)
        selected = (block%2 == 0)|(block == block.max())
        bands.append(AlignmentBand(field, state, points, selected))
    model = CoherentPoints(bands, points, motion['range_bias_m'], reference=3)
    previous = next(t for t in phase_control['trials'] if t['interval_s'] == motion['interval_s'])['fit']
    trial = sorted(previous['trials'], key=lambda t: t['objective'])[args.start]
    phase = np.asarray(trial['parameters'])
    if len(phase) != model.count+4:
        raise ValueError('Expected one phase per point and four band phases')
    band_phase = np.zeros(5); band_phase[model.other] = phase[model.count:]
    initial = model.initial(initial_points, phase[:model.count], band_phase)
    if args.resume:
        initial = np.asarray(json.loads(args.resume.read_text())['parameters'])
        if initial.shape != (3*model.count+4,):
            raise ValueError('Checkpoint uses a different physical parameterization')
    initial[:2*model.count] = np.clip(initial[:2*model.count], -1., 1.)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_paths = [args.motion, args.support, args.alignment, args.ais, args.phase_control,
        Path(__file__), *[Path(__file__).with_name(name+'.py') for name in
            ('coherent_points', 'profiled_point_phases', 'phase_coupling', 'fit_physical_profile', 'stationarity')]]
    if args.resume:
        source_paths.append(args.resume)
    result = {'scope': __doc__, 'source_sha256': {str(path): fingerprint(path) for path in source_paths},
        'source_motion': str(args.motion), 'source_support': str(args.support),
        'initial_training_rank': args.start, 'initial_phase_start': trial,
        'initial_points_body_m': initial_points.tolist(), 'cell_centres_body_m': points.tolist(),
        'position_bound_m': 1., 'correction_adopted': False, 'complete': False}
    result.update(solver=args.solver, resumed_from=str(args.resume) if args.resume else None)
    if args.resume:
        result['previous_trace'] = json.loads(args.resume.read_text()).get('trace', [])
    started = time.perf_counter()
    first = model.evaluate(initial)
    if not args.resume and abs(first['value']-trial['objective']) > 1e-9:
        raise ValueError('Joint fit changed the fixed-coordinate phase likelihood')
    result['before_training_objective'] = first['value']
    result['before'] = assess(model, initial, motion['interval_s'])
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'initial_objective': first['value'], 'initial_prediction_error':
        result['before']['summary']['prediction_error'], 'seconds': time.perf_counter()-started}), flush=True)
    bounds = [(-1., 1.)]*(2*model.count)+[(None, None)]*(model.count+4)
    trace = []
    def kkt(parameters, gradient):
        pg = gradient.copy()
        coord = pg[:2*model.count]
        coord[(parameters[:2*model.count] <= -1.+1e-8)&(coord > 0)] = 0.
        coord[(parameters[:2*model.count] >= 1.-1e-8)&(coord < 0)] = 0.
        return float(abs(pg).max())
    def callback(parameters):
        evaluation = model.evaluate(parameters)
        item = {'iteration': len(trace)+1, 'evaluations': model.evaluations, 'objective': evaluation['value'],
            'maximum_kkt_residual': kkt(parameters, evaluation['gradient']), 'seconds': time.perf_counter()-started}
        trace.append(item)
        args.output.with_suffix('.checkpoint.json').write_text(json.dumps({'parameters': parameters.tolist(),
            'points_body_m': model.coordinates(parameters).tolist(), 'trace': trace}, indent=2)+'\n')
        print(json.dumps(item), flush=True)
    profiled_details = None
    if args.solver == 'profiled':
        reduced = ProfiledPointPhases(model, initial)
        ncoords = len(reduced.initial)
        def checkpoint(coordinates, history):
            point = reduced.evaluate(coordinates)
            args.output.with_suffix('.checkpoint.json').write_text(json.dumps({
                'parameters': point['full_parameters'].tolist(), 'coordinates': coordinates.tolist(),
                'points_body_m': model.coordinates(point['full_parameters']).tolist(), 'trace': history}, indent=2)+'\n')
            print(json.dumps({**history[-1], 'inner_phase_gradient': point['inner_phase_gradient'],
                'phase_start_choice': point['phase_start_choice']}), flush=True)
        coordinates, profiled_details = constrained_solve(reduced, np.empty((0, ncoords)), np.empty(0), np.empty(0),
            np.full(ncoords, -1.), np.full(ncoords, 1.), iterations=args.iterations, checkpoint=checkpoint,
            polish_stationarity=False)
        from scipy.optimize import OptimizeResult
        fit = OptimizeResult(x=reduced.evaluate(coordinates)['full_parameters'],
            message=profiled_details['termination'])
        trace = profiled_details['trace']
    elif args.solver == 'lbfgs':
        fit = minimize(model.objective, initial, jac=True, method='L-BFGS-B', bounds=bounds, callback=callback,
            options={'gtol': 1e-9, 'ftol': 1e-14, 'maxiter': args.iterations, 'maxcor': 50, 'maxls': 40})
    else:
        lower = np.r_[np.full(2*model.count, -1.), np.full(model.count+4, -np.inf)]
        upper = np.r_[np.full(2*model.count, 1.), np.full(model.count+4, np.inf)]
        fit = least_squares(lambda x: model.evaluate(x, jacobian=True)['real_residual'], initial,
            jac=lambda x: model.evaluate(x, jacobian=True)['residual_jacobian'],
            bounds=(lower, upper), x_scale='jac', gtol=5e-10, ftol=1e-14, xtol=1e-12,
            max_nfev=args.iterations, callback=callback)
    final = model.evaluate(fit.x)
    maximum_kkt = kkt(fit.x, final['gradient'])
    result.update(fit={'converged': maximum_kkt < 1e-7, 'maximum_kkt_residual': maximum_kkt,
        'objective': final['value'], 'termination': str(fit.message), 'iterations': len(trace),
        'evaluations': model.evaluations, 'trace': trace}, parameters=fit.x.tolist(),
        points_body_m=model.coordinates(fit.x).tolist(),
        band_phase_radians=final['band_phase_radians'], point_phase_radians=final['point_phase_radians'],
        coordinate_bound_hits=int(np.sum(abs(fit.x[:2*model.count]) >= 1.-1e-6)),
        after=assess(model, fit.x, motion['interval_s']), seconds=time.perf_counter()-started, complete=True)
    if profiled_details is not None:
        profiled_details['convergence_scope'] = ('First-order constrained stationarity of the physical point '
            'coordinates with profiled nonnegative band strengths and shared point phases. '
            'Motion, sites, heights and envelope origins remain fixed. Not a global or calibration certificate.')
        result['profiled_phase_solver'] = profiled_details
    # At the selected physical coordinates, retain the independent-band
    # control without refitting any position or phase on validation data.
    coupling = PhaseCoupling([b.nuisance.remove(o['target'][b.selected]) for b, o in zip(bands, final['bands'])],
        [b.clean for b in bands], reference=3)
    independent = [validate(b.field, o['target'], b.nuisance, b.selected_time, motion['interval_s'],
        coefficients=a, stationary_mode='project') for b, o, a in zip(bands, final['bands'], coupling.independent)]
    result['independent_at_selected_points'] = {'channels': independent, 'summary': summarize(independent)}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'converged': result['fit']['converged'], 'kkt': maximum_kkt,
        'prediction_change_percent': 100*(result['after']['summary']['prediction_error']/
            result['before']['summary']['prediction_error']-1), 'seconds': result['seconds']}), flush=True)


if __name__ == '__main__':
    main()
