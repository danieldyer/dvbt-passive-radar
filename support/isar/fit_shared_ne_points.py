"""Test common physical reflector locations on the current fixed-N/E aperture.

Motion, nominal antennas, five shared envelope origins and point heights remain
fixed. Each band profiles its own complex coefficients. Five inherited band
supports supply explicitly recorded initializations; training objective alone
selects among them. Original rows and the existing half-second holdouts are
retained. This is conditional scene prediction, not antenna calibration.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from ais import utc_seconds
from field import Field
from bounded_point_profile import BoundedPointProfile, solve_points
from full_motion import Band
from imaging import motion_state, sites
from physical_profile import profile
from point_profile import profile_points
from scene_dictionary import fingerprint
from spline_motion import SplineMotion
from validate_motion import summarize, validate


def bands_for(model, fields):
    curve = SplineMotion(model)
    tx, rx, _ = sites(Path(model['geometry_directory']))
    bands = []
    for c, full in enumerate(fields):
        field = full.windowed(curve.interval[0]+.03, curve.interval[1]-.03)
        prior = motion_state(None, utc_seconds(model['nominal_start_utc'])+field.data['row_time'], model)
        selected = np.floor((field.time-curve.interval[0])/.5).astype(int) % 2 == 0
        band = Band(field, prior, tx[model['reference_channel']], rx,
            model['selected_points_body_m'][c], model, selected, range_bias_m=model['range_bias_m'][c])
        band.state = curve.apply(band.time, prior, band.reference_transmitter, band.receiver)
        bands.append(band)
    return bands


def prediction(bands, results, interval):
    channels = [validate(band.field, result['target'], band.nuisance, band.selected_time,
        interval, coefficients=result['coefficient'], stationary_mode='project')
        for band, result in zip(bands, results)]
    return {'channels': channels, 'summary': summarize(channels)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('motion', 'fields', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--starts', type=int, nargs='+', default=[3, 0, 1, 2, 4])
    p.add_argument('--maximum-shift-m', type=float, default=24.)
    p.add_argument('--iterations', type=int, default=180)
    args = p.parse_args(); started = time.monotonic()
    if args.output.exists() or args.maximum_shift_m <= 0:
        raise ValueError('Use positive bounds and a fresh output directory')
    if len(set(args.starts)) != len(args.starts) or any(c not in range(5) for c in args.starts):
        raise ValueError('Select distinct source-band starts from zero to four')
    model = json.loads(args.motion.read_text()); curve = SplineMotion(model)
    if (not model['complete'] or not model['fit']['converged']
            or model['original_row_kkt'] > 1e-7 or not model['fixed_ne_frame']
            or np.any(curve.yaw_coefficients)):
        raise ValueError('Use a completed stationary fixed-N/E motion control')
    for value in ('transmitter_offsets_enu_m', 'receiver_offset_enu_m'):
        if np.max(abs(np.asarray(model.get('antenna_fit', {}).get(value, [0.])))) > 1e-6:
            raise ValueError('Keep nominal physical antenna locations')
    paths = [args.motion, *[args.fields/f'channel-{c}.npz' for c in range(5)], Path(__file__),
        *[Path(__file__).with_name(name) for name in ('point_profile.py', 'physical_profile.py',
            'full_motion.py', 'field.py', 'spline_motion.py', 'bounded_point_profile.py', 'stationarity.py',
            'validate_motion.py')]]
    fields = []
    for path in paths[1:6]:
        with np.load(path) as d:
            fields.append(Field(dict(d), d['pulse_samples']))
    bands = bands_for(model, fields)
    original = [profile(band, curve.parameters, range_origin=band.range_bias) for band in bands]
    equivalent = [profile_points(band, band.points, band.range_bias) for band in bands]
    # Independently assembled point and motion operators must describe the same
    # original-row observations before changing the scene correspondence.
    equivalence = []
    for a, b in zip(original, equivalent):
        error = np.linalg.norm(a['target']-b['target'])/np.linalg.norm(a['target'])
        if error > 1e-7:
            raise ValueError('Physical point and motion operators disagree')
        equivalence.append(float(error))
    independent = prediction(bands, original, model['interval_s'])
    result = {'complete': False, 'scope': __doc__, 'source_sha256': {
        str(path): fingerprint(path) for path in paths}, 'source_motion': str(args.motion),
        'interval_s': model['interval_s'], 'maximum_shift_m': args.maximum_shift_m,
        'initial_support_bands': args.starts, 'operator_relative_errors': equivalence,
        'independent_band_support_prediction': independent, 'trials': [],
        'accepted_calibration': False, 'yaw_fitted': False,
        'limitation': 'Support and motion were inferred from this recording. Holding a common point height fixed is a scene assumption, not measured elevation. This only tests a shared horizontal support with independent band phases; no coherent cross-band gain is asserted.'}
    args.output.mkdir(parents=True)
    def save():
        (args.output/'experiment.json').write_text(json.dumps(result, indent=2)+'\n')
    save()
    print(json.dumps({'stage': 'loaded', 'operator_relative_errors': equivalence,
        'independent_prediction': independent['summary'], 'seconds': time.monotonic()-started}), flush=True)
    for c in args.starts:
        points = np.asarray(model['selected_points_body_m'][c])
        joint = BoundedPointProfile(bands, points, model['range_bias_m'])
        before = prediction(bands, joint.evaluate(joint.initial)['bands'], model['interval_s'])
        n = len(joint.initial)
        def checkpoint(parameters, trace):
            state = {'source_band': c, 'parameters': parameters.tolist(), 'trace': trace}
            (args.output/f'band-{c}.checkpoint.json').write_text(json.dumps(state, indent=2)+'\n')
            print(json.dumps({'source_band': c, **trace[-1]}), flush=True)
        parameters, fit = solve_points(joint,
            np.full(n, -args.maximum_shift_m), np.full(n, args.maximum_shift_m),
            evaluations=args.iterations, checkpoint=checkpoint)
        final = joint.evaluate(parameters)
        after = prediction(bands, final['bands'], model['interval_s'])
        trial = {'source_band': c, 'fit': fit, 'parameters': parameters.tolist(),
            'initial_points_body_m': points.tolist(), 'points_body_m': joint.coordinates(parameters).tolist(),
            'before': before, 'after': after, 'scene_conditions': final['scene_conditions'],
            'bound_hits': int(np.isclose(abs(parameters), args.maximum_shift_m, rtol=0, atol=1e-6).sum())}
        result['trials'].append(trial); save()
        print(json.dumps({'stage': 'completed_start', 'source_band': c,
            'converged': fit['converged'], 'kkt': fit['maximum_kkt_residual'],
            'prediction': after['summary'], 'seconds': time.monotonic()-started}), flush=True)
    eligible = [trial for trial in result['trials'] if trial['fit']['converged']]
    selected = min(eligible, key=lambda trial: trial['fit']['objective']) if eligible else None
    result.update(complete=True, selected_source_band=selected['source_band'] if selected else None,
        selected_all_starts_stationary=all(trial['fit']['converged'] for trial in result['trials']),
        selection='Smallest training objective among stationary fits; every requested start is retained.',
        seconds=time.monotonic()-started)
    save()
    print(json.dumps({'stage': 'complete', 'selected_source_band': result['selected_source_band'],
        'all_starts_stationary': result['selected_all_starts_stationary']}), flush=True)


if __name__ == '__main__':
    main()
