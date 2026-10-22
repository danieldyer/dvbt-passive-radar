"""Matched fixed-site motion fits with average versus known block responses.

Recorded observations, whitening, physical support, constraints and time split
are identical. Only the known forward pulse changes, including the stationary
nuisance templates. Antennas and yaw stay fixed. All derivatives and likelihood
evaluations use original symbol rows; no midpoint surrogate drops the kernels.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from ais import utc_seconds
from field import Field
from fit_physical_profile import solve
from full_motion import Band
from imaging import motion_state, sites
from physical_profile import PhysicalProfile
from refine_full_capture_motion import constraints
from scene_dictionary import fingerprint
from spline_motion import SplineMotion
from stationarity import check
from validate_motion import summarize
from waveform_kernels import paired_fields, read_kernels


def derivative_check(problem, initial, lower, upper):
    measured = problem.evaluate(initial)
    free = lower < upper
    rng = np.random.default_rng(62517)
    checks = []
    for n in range(2):
        direction = rng.normal(size=len(initial))*free
        direction /= np.linalg.norm(direction)
        expected = float(measured['gradient']@direction)
        for h in (1e-3, 1e-4, 1e-5, 1e-6):
            upper_value = problem.evaluate(initial+h*direction)['value']
            lower_value = problem.evaluate(initial-h*direction)['value']
            numerical = (upper_value-lower_value)/(2*h)
            checks.append({'direction': n, 'analytic': expected, 'finite_difference': numerical,
                'absolute_error': abs(expected-numerical), 'step': h})
            print(json.dumps({'stage': 'derivative_check', **checks[-1]}), flush=True)
    if any(sum(c['absolute_error'] <= 2e-7 for c in checks if c['direction'] == n) < 2 for n in range(2)):
        raise ValueError('Recorded known-waveform gradient does not agree with finite differences')
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('motion', 'kernels', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--models', choices=('mean', 'known'), nargs='+', default=['mean', 'known'])
    parser.add_argument('--iterations', type=int, default=180)
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--resume', type=Path, help='Continue one completed same-model result without changing its data or bounds')
    parser.add_argument('--resume-driver-archive', type=Path, help='Original driver source if only the driver changed since the result')
    parser.add_argument('--polish-stationarity', action='store_true')
    args = parser.parse_args(); started = time.monotonic()
    if args.output.exists():
        raise ValueError('Use a new output directory')
    model = json.loads(args.motion.read_text()); curve = SplineMotion(model)
    if not model['fixed_ne_frame'] or np.any(curve.yaw_coefficients) or model.get('rock_spacing_seconds'):
        raise ValueError('Keep fixed north/east axes with no fitted attitude')
    if (np.max(abs(np.asarray(model['antenna_fit']['transmitter_offsets_enu_m']))) > 1e-6 or
            np.max(abs(np.asarray(model['antenna_fit']['receiver_offset_enu_m']))) > 1e-6):
        raise ValueError('This matched control keeps nominal sites')
    header, records, pulses = read_kernels(args.kernels/'response.kernels')
    tx, rx, _ = sites(Path(model['geometry_directory']))
    bands = {key: [] for key in args.models}; availability = []
    args.output.mkdir(parents=True)
    source_files = [args.motion, args.kernels/'response.kernels', args.kernels/'request.bin',
        args.kernels/'rows.npz', Path(__file__), *[Path(__file__).with_name(n) for n in
        ('block_pulse.py', 'waveform_kernels.py', 'field.py', 'full_motion.py', 'physical_profile.py',
         'fit_physical_profile.py', 'stationarity.py', 'spline_motion.py')]]
    for c in range(5):
        path = Path(header['channels'][c]['field']); source_files.append(path)
        with np.load(path) as data:
            base = Field(dict(data), data['pulse_samples']).windowed(curve.interval[0]+.03, curve.interval[1]-.03)
        base, corrected, valid = paired_fields(base, args.kernels, header, records[c], pulses[c], c)
        availability.append({'channel': c, 'blocks': len(valid), 'known_blocks': int(valid.sum()),
            'data_changed_between_models': False, 'range_metric_changed_between_models': False})
        prior = motion_state(None, utc_seconds(model['nominal_start_utc'])+base.data['row_time'], model)
        selected = np.floor((base.time-curve.interval[0])/.5).astype(int)%2 == 0
        for key, field in [('mean', base), ('known', corrected)]:
            if key in bands:
                bands[key].append(Band(field, prior, tx[model['reference_channel']], rx,
                    model['selected_points_body_m'][c], model, selected, range_bias_m=model['range_bias_m'][c]))
        print(json.dumps({'stage': 'loaded_channel', **availability[-1], 'seconds': time.monotonic()-started}), flush=True)
    _, matrix, lo, hi, lower, upper = constraints(model, common_path=True)
    source_fit = json.loads(Path(model['source_fit']).read_text())
    origin_lower, origin_upper = source_fit['range_origin_bounds_m']
    lower, upper = np.r_[lower, origin_lower], np.r_[upper, origin_upper]
    matrix = np.pad(matrix, ((0, 0), (0, 5)))
    initial = PhysicalProfile(next(iter(bands.values())), curve.parameters, fit_ranges=True).initial
    active = curve.gauge_fixed_parameters()[1]
    yaw = (active >= curve.sections[1].start)&(active < curve.sections[1].stop)
    frozen = np.flatnonzero(yaw); lower[frozen] = upper[frozen] = 0.
    if len(initial) != len(lower):
        raise ValueError('Physical motion gauge/constraint dimension changed')
    if args.resume:
        previous = json.loads(args.resume.read_text())
        if not previous['complete'] or args.models != [previous['name']]:
            raise ValueError('Resume exactly one completed matching model')
        for source, expected in previous['source_sha256'].items():
            if fingerprint(Path(source)) != expected:
                if (Path(source).resolve() != Path(__file__).resolve() or not args.resume_driver_archive or
                        fingerprint(args.resume_driver_archive) != expected):
                    raise ValueError('Resume source changed: '+source)
        if previous['source_sha256'].get(str(args.motion)) != fingerprint(args.motion):
            raise ValueError('Resume changed the input physical motion model')
        initial = np.asarray(previous['parameters'])
        source_files.append(args.resume)
        if args.resume_driver_archive: source_files.append(args.resume_driver_archive)
    sources = {str(p): fingerprint(p) for p in source_files+[Path(model['source_fit'])]}
    manifest = {'scope': __doc__, 'complete': False, 'source_sha256': sources,
        'availability': availability, 'antennas_fixed': True, 'yaw_fixed_zero': True,
        'holdout_seconds': .5, 'models': [], 'scope_limit': 'Same empirical mean range covariance in both models; this test does not certify the unknown recorded interference metric.'}
    for name, family in bands.items():
        problem = PhysicalProfile(family, curve.parameters, fit_ranges=True)
        problem.initial = initial.copy()
        initial_value = problem.evaluate(initial)
        before = problem.validation(initial)
        audit = derivative_check(problem, initial, lower, upper) if name == 'known' else None
        record = {'name': name, 'source_sha256': sources, 'complete': False,
            'initial_objective': initial_value['value'], 'before': before,
            'before_summary': summarize(before), 'derivative_audit': audit,
            'nuisance_rank_by_channel': [b.nuisance.rank for b in family]}
        path = args.output/(name+'.json')
        path.write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps({'stage': 'initial', 'model': name, 'objective': initial_value['value'],
            'summary': record['before_summary'], 'gradient_check': audit}), flush=True)
        def checkpoint(parameters, trace):
            temporary = path.with_suffix('.checkpoint.tmp')
            temporary.write_text(json.dumps({'parameters': parameters.tolist(), 'trace': trace}, indent=2)+'\n')
            temporary.replace(path.with_suffix('.checkpoint.json'))
            print(json.dumps({'stage': 'fit', 'model': name, **trace[-1]}), flush=True)
        if args.evaluate_only:
            parameters, fit = initial, None
        else:
            parameters, fit = solve(problem, matrix, lo, hi, lower, upper,
                iterations=args.iterations, checkpoint=checkpoint, polish_stationarity=args.polish_stationarity)
        final = problem.evaluate(parameters)
        kkt, violation = check(parameters, final['gradient'], matrix, lo, hi, lower, upper)
        after = problem.validation(parameters)
        record.update(complete=True, parameters=parameters.tolist(), motion_parameters=problem.expand(parameters).tolist(),
            range_bias_m=parameters[problem.nmotion:].tolist(), fit=fit,
            original_row_kkt=kkt, constraint_violation=violation,
            after=after, after_summary=summarize(after), seconds=time.monotonic()-started)
        path.write_text(json.dumps(record, indent=2)+'\n')
        manifest['models'].append({'name': name, 'path': str(path), 'sha256': fingerprint(path),
            'kkt': kkt, 'fit_converged': bool(fit and fit['converged']), 'summary': record['after_summary']})
        (args.output/'experiment.json').write_text(json.dumps(manifest, indent=2)+'\n')
    manifest.update(complete=True, seconds=time.monotonic()-started)
    (args.output/'experiment.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
