"""Joint channel range-origin and shared-phase control at fixed physical motion.

This uses the converged physical point support, jointly optimizes range origins
and phases, profiles nonnegative band strengths exactly, and tests withheld
one-second blocks.
The half-aperture controls retain the same support and motion; they are
conditional calibration-transfer checks, not new target discovery experiments.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from ais import read_csv, utc_seconds
from channel_alignment import AlignmentBand
from field import Field
from fit_physical_profile import solve
from imaging import physical_state, sites
from range_phase import RangePhase
from scene_dictionary import fingerprint
from spline_motion import SplineMotion


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('support', 'alignment', 'ais', 'geometry-directory', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--split-halves', action='store_true')
    p.add_argument('--phase-control', type=Path,
        help='Add the completed full/half fixed-origin phase solutions to the deterministic initializations')
    p.add_argument('--warm-start', type=Path,
        help='Reuse documented range/phase initializations from this same physical control; final convergence is rechecked')
    p.add_argument('--iterations', type=int, default=80)
    args = p.parse_args()
    support, alignment = [json.loads(path.read_text()) for path in (args.support, args.alignment)]
    if not support['complete'] or not support['fit']['converged']:
        raise ValueError('Use a completed stationary physical point fit')
    motion_path = Path(support['source_motion'])
    motion = json.loads(motion_path.read_text())
    if motion.get('antenna_fit') or motion.get('rock_spacing_seconds') or np.any(motion.get('channel_path_rate_bias_mps', 0.)):
        raise ValueError('Hold roll/pitch, sites and channel path rates fixed')
    curve = SplineMotion(motion)
    tx, rx, _ = sites(args.geometry_directory)
    track = read_csv(args.ais, reference_height_ellipsoid_m=motion['reference_height_ellipsoid_m'], mmsi=563052200)
    points = np.asarray(support['points_body_m'])
    seed = np.r_[support['point_phase_radians'], np.asarray(support['band_phase_radians'])[[0, 1, 2, 4]]]
    seeds = [seed]
    if args.phase_control:
        phase_control = json.loads(args.phase_control.read_text())
        if (not phase_control['complete'] or Path(phase_control['source_support']).resolve() != args.support.resolve()
                or not all(t['fit']['converged'] for t in phase_control['trials'])):
            raise ValueError('Use completed phase controls at the same physical support')
        seeds += [np.asarray(t['fit']['parameters']) for t in phase_control['trials']]
    intervals = [motion['interval_s']]
    if args.split_halves:
        centre = float(np.mean(curve.interval))
        intervals += [[float(curve.interval[0]), centre], [centre, float(curve.interval[1])]]
    sources = [args.support, args.alignment, args.ais, motion_path, Path(__file__),
        *[Path(__file__).with_name(name+'.py') for name in
            ('range_phase', 'phase_coupling', 'fit_physical_profile', 'stationarity')]]
    if args.phase_control:
        sources.append(args.phase_control)
    warm = None
    if args.warm_start:
        warm = json.loads(args.warm_start.read_text())
        if Path(warm['source_support']).resolve() != args.support.resolve():
            raise ValueError('Warm start uses a different physical support')
        sources.append(args.warm_start)
    result = {'scope': __doc__, 'source_support': str(args.support), 'source_motion': str(motion_path),
        'source_sha256': {str(path): fingerprint(path) for path in sources},
        'range_origin_bounds_m': [-75., 75.], 'reference_channel': 3,
        'roll_pitch_fitted': False, 'correction_adopted': False, 'complete': False, 'trials': []}
    result['phase_seeds_radians'] = [s.tolist() for s in seeds]
    result['solver'] = 'Joint range and phase variables; only real nonnegative amplitudes are profiled.'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for index, interval in enumerate(intervals):
        bands = []
        for c in range(5):
            with np.load(Path(alignment['input_fields'])/f'channel-{c}.npz') as loaded:
                field = Field(dict(loaded), loaded['pulse_samples']).windowed(interval[0]+.02, interval[1]-.02)
            prior = physical_state(track, utc_seconds(motion['nominal_start_utc'])+field.data['row_time'], motion['attitude_prior'])
            state = curve.apply(field.data['row_time'], prior, tx[3], rx)
            block = np.floor(field.time-interval[0]).astype(int)
            selected = (block%2 == 0)|(block == block.max())
            bands.append(AlignmentBand(field, state, points, selected))
        previous = (next((t for t in warm['trials'] if t['interval_s'] == interval), None)
                    if warm else None)
        starts = list(seeds)
        if previous:
            starts.append(np.r_[previous['point_phase_radians'], np.asarray(previous['band_phase_radians'])[[0, 1, 2, 4]]])
        model = RangePhase(bands, motion['range_bias_m'], starts, reference=3, joint=True)
        model.initialize_phases()
        before = model.validation(model.initial, interval)
        if index == 0 and model.last['value'] > support['fit']['objective']+1e-9:
            raise ValueError('Initial phase search lost the existing physical support solution')
        if previous:
            model.initial[:5] = np.asarray(previous['range_origins_m'])/model.scale
            model.initialize_phases()
        count = len(model.initial)
        def checkpoint(parameters, trace):
            print(json.dumps({'interval_s': interval, **trace[-1],
                'range_origins_m': (parameters[:5]*model.scale).tolist(),
                'phase_gradient': model.last['phase_gradient']}), flush=True)
            args.output.with_suffix(f'.interval-{index}.checkpoint.json').write_text(json.dumps({
                'interval_s': interval, 'parameters': parameters.tolist(), 'trace': trace}, indent=2)+'\n')
        parameters, details = solve(model, np.empty((0, count)), np.empty(0), np.empty(0),
            np.r_[np.full(5, -75./model.scale), np.full(count-5, -np.inf)],
            np.r_[np.full(5, 75./model.scale), np.full(count-5, np.inf)], iterations=args.iterations,
            checkpoint=checkpoint, polish_stationarity=False)
        final = model.evaluate(parameters)
        details['converged'] = bool(details['converged'] and final['phase_gradient'] < 1e-7)
        details['convergence_scope'] = ('First-order constrained stationarity of range origins with '
            'range origins and shared phases, with nonnegative band strengths profiled. Motion and point coordinates fixed; '
            'not an antenna-location estimate or global optimality certificate.')
        after = model.validation(parameters, interval)
        trial = {'interval_s': interval, 'fit': details, 'range_origins_m': (parameters[:5]*model.scale).tolist(),
            'band_phase_radians': final['band_phase_radians'].tolist(),
            'point_phase_radians': final['phase_parameters'][:len(points)].tolist(),
            'maximum_phase_gradient': final['phase_gradient'], 'before': before, 'after': after,
            'coefficients_real_imag': np.stack([final['coefficients'].real, final['coefficients'].imag], axis=-1).tolist()}
        result['trials'].append(trial)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({'interval_s': interval, 'converged': details['converged'],
            'kkt': details['maximum_kkt_residual'], 'range_origins_m': trial['range_origins_m'],
            'prediction_error_change_percent': 100*(after['coupled']['summary']['prediction_error']/
                before['coupled']['summary']['prediction_error']-1), 'seconds': time.perf_counter()-started}), flush=True)
    result.update(complete=True, seconds=time.perf_counter()-started)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
