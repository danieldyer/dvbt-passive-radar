"""Test whether a delayed return predicts contamination of the ship crop.

A point-path delay/rate and complex amplitude are measured in the companion
crop using training half-second blocks only. The exact same complex amplitude
is then transferred into the primary crop; no new companion phase/gain is fit
there. The main point strengths are refit, and frozen combined predictions
are checked on withheld samples. Geometry and yaw remain unchanged.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ais import utc_seconds
from delayed_path import DelayedPath
from field import Field
from fit_physical_profile import solve
from full_motion import Band
from gaussian_scene import marginal
from imaging import motion_state, sites
from scene_dictionary import fingerprint
from spline_motion import SplineMotion
from stationary import Stationary


def prediction_score(y, prediction):
    energy = float(np.vdot(y, y).real)
    model = float(np.vdot(prediction, prediction).real)
    cross = np.vdot(prediction, y)
    error = float(np.sum(abs(y-prediction)**2))
    return {'data_energy': energy, 'prediction_energy': model, 'prediction_error': error,
        'explained_fraction': 1-error/energy, 'cross_phase_degrees': float(np.degrees(np.angle(cross))),
        'matched_energy': float(abs(cross)**2/model) if model else 0.}


def matched_change(before, after):
    """No dB ratio exists when the fitted scene contains zero signal."""
    a, b = before['matched_energy'], after['matched_energy']
    return float(10*np.log10(b/a)) if min(a, b) > 0 else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('primary', 'companion', 'motion', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--centres', type=float, nargs='+', default=[150., 160., 170., 180.])
    p.add_argument('--duration', type=float, default=3.)
    p.add_argument('--full-interval', action='store_true', help='Cover the entire fitted interval with nonoverlapping blocks')
    p.add_argument('--continue-from', type=Path, help='Reuse a preserved partial acquisition with identical scientific inputs and kernels')
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Preserve earlier experiments; choose a new output path')
    model = json.loads(args.motion.read_text())
    if not model.get('fixed_ne_frame') or np.any(SplineMotion(model).yaw_coefficients):
        raise ValueError('Keep yaw fixed at zero')
    if args.full_interval:
        span = model['interval_s'][1]-model['interval_s'][0]
        if abs(span/args.duration-round(span/args.duration)) > 1e-8:
            raise ValueError('The declared block duration must divide the fitted interval')
        args.centres = (model['interval_s'][0]+(np.arange(round(span/args.duration))+.5)*args.duration).tolist()
    tx, rx, _ = sites(Path(model['geometry_directory']))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {'scope': __doc__, 'source_motion': str(args.motion), 'source_sha256': fingerprint(args.motion),
        'companion_fields': str(args.companion/'fields.json'), 'centres_s': args.centres,
        'duration_s': args.duration, 'delay_bounds_m': [350., 750.], 'relative_rate_bounds_mps': [-3., 3.],
        'implementation_sha256': {name: fingerprint(Path(__file__).with_name(name)) for name in
            ('companion_transfer.py', 'delayed_path.py', 'field.py', 'gaussian_scene.py', 'full_motion.py')},
        'trials': [], 'complete': False, 'correction_adopted': False,
        'scope_limits': 'Conditional fixed-motion, single-extra-path test. Stationary/main nuisance is explicit. Other delayed scatterers and nonconstant extra path rate are not assumed absent; failed transfer must not be relabelled as calibrated cancellation.'}
    if args.continue_from:
        previous = json.loads(args.continue_from.read_text())
        keys = ('source_sha256', 'companion_fields', 'centres_s', 'duration_s',
            'delay_bounds_m', 'relative_rate_bounds_mps')
        kernels = {k: v for k, v in result['implementation_sha256'].items() if k != 'companion_transfer.py'}
        if (any(previous[k] != result[k] for k in keys) or
                any(previous['implementation_sha256'][k] != v for k, v in kernels.items())):
            raise ValueError('The saved acquisition has different scientific inputs or kernels')
        result['trials'] = previous['trials']
        result['reused_results'] = {'source': str(args.continue_from),
            'sha256': fingerprint(args.continue_from), 'trials': len(previous['trials']),
            'implementation_sha256': previous['implementation_sha256']}
    completed = {(t['channel'], t['centre_s']) for t in result['trials']}
    for c in range(5):
        fields = []
        for path in (args.primary, args.companion):
            with np.load(path/f'channel-{c}.npz') as d:
                fields.append(Field(dict(d), d['pulse_samples']))
        for centre in args.centres:
            if (c, centre) in completed:
                continue
            bands = []
            interval = [max(centre-args.duration/2, model['interval_s'][0]+.03),
                min(centre+args.duration/2, model['interval_s'][1]-.03)]
            for full in fields:
                f = full.windowed(*interval)
                prior = motion_state(None, utc_seconds(model['nominal_start_utc'])+f.data['row_time'], model)
                selected = np.floor((f.time-model['interval_s'][0])/.5).astype(int)%2 == 0
                bands.append(Band(f, prior, tx[model['reference_channel']], rx,
                    model['selected_points_body_m'][c], model, selected, range_bias_m=model['range_bias_m'][c]))
            primary, companion = bands
            np.testing.assert_array_equal(primary.field.time, companion.field.time)
            joint = DelayedPath(companion, model['parameters'])
            delays, rates = np.arange(350., 751., 10.), np.arange(-3., 3.001, .1)
            initial, grid = joint.acquire(delays, rates)
            starts = []
            available = np.ones(grid.shape, bool)
            for _ in range(3):
                di, vi = np.unravel_index(np.argmax(np.where(available, grid, -np.inf)), grid.shape)
                starts.append(np.array([delays[di], rates[vi]]))
                available &= ((abs(delays[:, None]-delays[di]) >= 40.)|
                    (abs(rates[None, :]-rates[vi]) >= .4))
            fitted = []
            for start in starts:
                joint.initial = start
                parameters, details = solve(joint, np.empty((0, 2)), np.empty(0), np.empty(0),
                    np.array([350., -3.]), np.array([750., 3.]), iterations=70)
                fitted.append((parameters, details))
            parameters, details = min(fitted, key=lambda item:item[1]['objective'])
            solution = joint.evaluate(parameters)
            amplitude = solution['amplitude']
            test = ~companion.selected
            validation_nuisance = Stationary(joint.nuisance.design, test)
            companion_score = prediction_score(validation_nuisance.remove(companion.field.y.ravel()[test]),
                validation_nuisance.remove(solution['target'][test]*amplitude))
            # Identical phase origin and exact per-crop linear response.
            transferred = DelayedPath(primary, model['parameters']).response(parameters)*amplitude
            train_y = primary.clean
            main = primary.design(model['parameters'])[0]
            train_a = primary.nuisance.remove(main[primary.selected])
            baseline = marginal(train_a, train_y)
            updated = marginal(train_a, train_y-primary.nuisance.remove(transferred[primary.selected]))
            test = ~primary.selected
            nuisance = Stationary(primary.nuisance.design, test)
            y = nuisance.remove(primary.field.y.ravel()[test])
            before = prediction_score(y, nuisance.remove(main[test]@baseline['coefficient']))
            after = prediction_score(y, nuisance.remove(main[test]@updated['coefficient']+transferred[test]))
            leakage = nuisance.remove(transferred[test])
            result['trials'].append({'channel': c, 'centre_s': centre, 'parameters': parameters.tolist(),
                'interval_s': interval, 'phase_epoch_s': joint.epoch,
                'amplitude_real_imag': [float(amplitude.real), float(amplitude.imag)],
                'fit': details, 'starts': [{'initial': s.tolist(), 'parameters': p.tolist(), 'fit': f}
                    for s, (p, f) in zip(starts, fitted)], 'companion_validation': companion_score,
                'primary_before': before, 'primary_after': after,
                'transferred_primary_energy': float(np.vdot(leakage, leakage).real),
                'primary_error_change_percent': 100*(after['prediction_error']/before['prediction_error']-1),
                'primary_matched_energy_change_db': matched_change(before, after)})
            args.output.write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps({'channel': c, 'centre_s': centre, 'delay_rate': parameters.tolist(),
                'converged': details['converged'], 'companion_explained': companion_score['explained_fraction'],
                'primary_error_change_percent': result['trials'][-1]['primary_error_change_percent']}), flush=True)
    result['complete'] = True
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
