"""Original-row grid refinement and physical checks after translation fitting.

The fitted motion and per-area scene penalties stay fixed. This checks whether
the coarser spatial quadrature approximates the same model after autofocus;
it does not retune on test data or establish cross-band calibration.
"""
import argparse
import json
from pathlib import Path
import time as wall_time

import numpy as np

from ais import utc_seconds
from distributed_fixed_frame import prediction_metrics, quadrature
from field import Field
from full_motion import Band
from imaging import motion_state, sites
from midpoint_model import discrepancy
from ridge_motion import profile
from scene_dictionary import fingerprint
from spline_motion import SplineMotion
from stationary import Stationary


def physical_check(model, tx, rx):
    curve = SplineMotion(model)
    # Keep both finite-difference probes inside the fitted aperture. Spline
    # extrapolation is forbidden. Component bounds below also include endpoints.
    t = np.linspace(curve.interval[0]+.005, curve.interval[1]-.005, 6001)
    epoch = utc_seconds(model['nominal_start_utc'])
    def state(time):
        prior = motion_state(None, epoch+time, model)
        return curve.apply(time, prior, tx[model['reference_channel']], rx)
    current = state(t)
    checks = []
    for h in (.004, .002):
        left, right = state(t-h), state(t+h)
        acceleration = (right['velocity_ecef_mps']-left['velocity_ecef_mps'])/(2*h)
        path_acceleration = []
        for antenna in tx:
            value = np.zeros(len(t))
            for site in (antenna, rx):
                delta = current['position_ecef_m']-site
                distance = np.linalg.norm(delta, axis=1)
                unit = delta/distance[:, None]
                v = current['velocity_ecef_mps']
                radial = np.sum(unit*v, axis=1)
                value += np.sum(unit*acceleration, axis=1)+(np.sum(v*v, axis=1)-radial**2)/distance
            path_acceleration.append(float(abs(value).max()))
        checks.append({'derivative_step_s': h,
            'maximum_physical_acceleration_mps2': float(np.linalg.norm(acceleration, axis=1).max()),
            'maximum_absolute_bistatic_path_acceleration_mps2_by_channel': path_acceleration})
    full = np.linspace(*curve.interval, 6001)
    return {'yaw_is_exactly_zero': bool(np.all(curve.yaw_coefficients == 0)),
        'sample_times_s': [float(t[0]), float(t[-1])], 'samples': len(t),
        'maximum_absolute_component_rates': abs(curve.components(full, 1)).max(axis=0).tolist(),
        'maximum_absolute_component_accelerations': abs(curve.components(full, 2)).max(axis=0).tolist(),
        'physical_acceleration_step_checks': checks,
        'scope': 'The optimization bounds spline correction coordinates. These separately checked total physical accelerations include geometry and coarse linear velocity. They are diagnostics, not newly imposed constraints.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--spacing', type=float, default=10.)
    args = p.parse_args()
    model = json.loads(args.motion.read_text())
    source = json.loads(Path(model['training']).read_text())
    if not model['complete'] or not model['fixed_ne_frame'] or np.any(SplineMotion(model).yaw_coefficients):
        raise ValueError('Complete the fixed-N/E physical fit first')
    if args.output.exists():
        raise ValueError('Preserve previous audit results')
    tx, rx, _ = sites(Path(model['geometry_directory']))
    curve = SplineMotion(model)
    started = wall_time.monotonic()
    result = {'scope': __doc__, 'motion': str(args.motion), 'motion_sha256': fingerprint(args.motion),
        'original_spacing_m': model['spacing_m'], 'refined_spacing_m': args.spacing,
        'physical_checks': physical_check(model, tx, rx), 'channels': [], 'complete': False,
        'implementation_sha256': {name: fingerprint(Path(__file__).with_name(name)) for name in
            ('audit_distributed_motion.py', 'ridge_motion.py', 'distributed_fixed_frame.py',
             'spline_motion.py', 'field.py', 'full_motion.py')}}
    for c, strength in enumerate(model['fixed_penalties']):
        if strength is None:
            result['channels'].append({'channel': c, 'zero_prediction_selected': True})
            continue
        with np.load(Path(source['fields'])/f'channel-{c}.npz') as d:
            field = Field(dict(d), d['pulse_samples']).windowed(curve.interval[0]+.03, curve.interval[1]-.03)
        prior = motion_state(None, utc_seconds(model['nominal_start_utc'])+field.data['row_time'], model)
        selected = np.floor((field.time-curve.interval[0])/source['holdout_seconds']).astype(int)%2 == 0
        fits, predictions = [], []
        for spacing in (model['spacing_m'], args.spacing):
            points, weights = quadrature(source['bounds_enu_m'], spacing)
            band = Band(field, prior, tx[model['reference_channel']], rx, points, model, selected,
                range_bias_m=model['range_bias_m'][c])
            local = profile(band, model['parameters'], np.sqrt(weights), strength)
            prediction = local['target']@local['physical_coefficient']
            nuisance = Stationary(band.nuisance.design, ~band.selected)
            y = nuisance.remove(field.y.ravel()[~band.selected]).reshape((~selected).sum(), -1)
            value = nuisance.remove(prediction[~band.selected]).reshape(y.shape)
            fits.append({'spacing_m': spacing, 'points': len(points), 'strength': strength,
                'training_objective': local['value']/band.energy,
                'rank': local['effective_rank'], 'validation': prediction_metrics(y, value, field.time[~selected])})
            predictions.append(value)
            del local
        old, new = [fit['validation'] for fit in fits]
        channel = {'channel': c, 'trials': fits,
            'prediction_error_change_percent': 100*(new['prediction_error']/old['prediction_error']-1),
            'matched_energy_change_db': float(10*np.log10(new['heldout_matched_energy']/old['heldout_matched_energy'])),
            'prediction_discrepancy': discrepancy(predictions[0], predictions[1]),
            'original_score_reproduction_relative_error': abs(old['prediction_error']/model['after'][c]['prediction_error']-1)}
        result['channels'].append(channel)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({'channel': c, 'gain_change_db': channel['matched_energy_change_db'],
            'error_change_percent': channel['prediction_error_change_percent'],
            'seconds': wall_time.monotonic()-started}), flush=True)
    result.update(complete=True, seconds=wall_time.monotonic()-started)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
