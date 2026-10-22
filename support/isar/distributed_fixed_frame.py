"""A stationary distributed scene in the fixed geographic frame.

Cell-centred quadrature represents a spatially white proper-complex Gaussian
reflectivity field over a declared rectangle. The prior is per unit area,
not per grid point, so refining the grid approximates the same scene model.
Every band has its own reflectivity. Only one variance per band is estimated,
using training samples and the full marginal-likelihood determinant. Geometry,
motion, yaw and the original observations stay fixed in this comparison.
"""
import argparse
import json
from pathlib import Path
import time as wall_time

import numpy as np

from ais import utc_seconds
from field import Field
from full_motion import Band
from gaussian_scene import marginal
from imaging import motion_state, sites
from scene_dictionary import fingerprint
from spline_motion import SplineMotion
from stationary import Stationary


def quadrature(bounds, spacing, height=20.):
    east0, east1, north0, north1 = bounds
    counts = np.array([(east1-east0)/spacing, (north1-north0)/spacing])
    if spacing <= 0 or np.any(counts < 1) or not np.allclose(counts, np.round(counts), atol=1e-12, rtol=0):
        raise ValueError('Positive spacing must exactly divide both rectangle dimensions')
    east = east0+(np.arange(int(round(counts[0])))+.5)*spacing
    north = north0+(np.arange(int(round(counts[1])))+.5)*spacing
    e, n = np.meshgrid(east, north)
    return np.c_[e.ravel(), n.ravel(), np.full(e.size, height)], np.full(e.size, spacing**2)


def prediction_metrics(y, prediction, time):
    energy = float(np.vdot(prediction, prediction).real)
    cross = np.vdot(prediction, y)
    total = float(np.vdot(y, y).real)
    phase = []
    for first in np.arange(np.floor(time[0]), np.ceil(time[-1]), 1.):
        select = (time >= first)&(time < first+1.)
        if not select.any():
            continue
        z = np.vdot(prediction[select], y[select])
        p = float(np.vdot(prediction[select], prediction[select]).real)
        phase.append({'time_s': float(first+.5), 'phase_degrees': float(np.degrees(np.angle(z))),
            'prediction_energy': p})
    return {'prediction_error': float(np.sum(abs(y-prediction)**2)),
        'stationary_only_energy': total, 'prediction_energy': energy,
        'cross_real_imag': [float(cross.real), float(cross.imag)],
        'heldout_matched_energy': float(abs(cross)**2/energy) if energy else 0.,
        'heldout_explained_fraction': float((2*cross.real-energy)/total),
        'phase_rms_degrees': float(np.sqrt(np.average([p['phase_degrees']**2 for p in phase],
            weights=[p['prediction_energy'] for p in phase]))) if energy else None,
        'phase_by_one_second': phase}


def fit_scene(band, target, weights=None):
    scale = np.ones(target.shape[1]) if weights is None else np.sqrt(weights)
    design = band.nuisance.remove(target[band.selected])*scale
    if weights is None:
        # Matched control: the existing independently profiled eight-point fit.
        coefficient = np.linalg.lstsq(design, band.clean, rcond=None)[0]
        detail = {'scene_prior': 'Unregularized existing sparse support', 'points': len(coefficient)}
    else:
        posterior = marginal(design, band.clean)
        coefficient = posterior['coefficient']*scale
        detail = {'scene_prior': 'CN(0, variance times cell area), independent spatial increments',
            'variance_density': posterior['variance'], 'variance_fit': posterior['variance_fit'],
            'effective_scene_rank': posterior['effective_scene_rank'], 'condition': posterior['condition'],
            'negative_log_marginal_likelihood': posterior['value'],
            'singular_values': posterior['singular_values'].tolist(), 'points': len(coefficient)}
        del posterior
    predicted = (target@coefficient).reshape(band.field.y.shape)
    test = ~band.selected
    nuisance = Stationary(band.nuisance.design, test)
    y = nuisance.remove(band.field.y.ravel()[test]).reshape((~band.selected_time).sum(), -1)
    value = nuisance.remove(predicted.ravel()[test]).reshape(y.shape)
    detail['validation'] = prediction_metrics(y, value, band.field.time[~band.selected_time])
    detail['coefficient_real_imag'] = np.c_[coefficient.real, coefficient.imag].tolist()
    return detail, predicted


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('fields', 'motion', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--spacing', type=float, nargs='+', default=[20., 10.])
    p.add_argument('--bounds', type=float, nargs=4, default=[-120., 120., -180., 200.])
    p.add_argument('--channels', type=int, nargs='+', default=list(range(5)))
    p.add_argument('--holdout-seconds', type=float, default=.5)
    args = p.parse_args()
    model = json.loads(args.motion.read_text())
    curve = SplineMotion(model)
    if not model.get('fixed_ne_frame') or np.any(curve.yaw_coefficients):
        raise ValueError('Yaw must stay zero in a fixed ENU frame')
    tx, rx, _ = sites(Path(model['geometry_directory']))
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'experiment.json').exists():
        raise ValueError('Use a fresh output directory')
    result = {'scope': __doc__, 'source_motion': str(args.motion), 'source_sha256': fingerprint(args.motion),
        'fields': str(args.fields), 'bounds_enu_m': args.bounds, 'spacing_m': args.spacing,
        'holdout_seconds': args.holdout_seconds, 'yaw_fitted': False, 'motion_fitted': False,
        'point_coordinate_frame': 'east_north_up', 'height_m': 20., 'channels': [], 'complete': False,
        'limitation': 'Previously fitted motion and selected sparse controls are conditional on this aperture. Spatial bounds cover their support; the grid is not selected from test residuals. Gaussian reflectivity is a model to validate, not established hull physics.',
        'implementation_sha256': {name: fingerprint(Path(__file__).with_name(name)) for name in
            ('distributed_fixed_frame.py', 'gaussian_scene.py', 'full_motion.py', 'field.py',
             'spline_motion.py', 'stationary.py')}}
    started = wall_time.monotonic()
    for c in args.channels:
        with np.load(args.fields/f'channel-{c}.npz') as d:
            field = Field(dict(d), d['pulse_samples']).windowed(curve.interval[0]+.03, curve.interval[1]-.03)
        selected = (np.arange(len(field.time))%2 == 0 if args.holdout_seconds == 0 else
            np.floor((field.time-curve.interval[0])/args.holdout_seconds).astype(int)%2 == 0)
        prior = motion_state(None, utc_seconds(model['nominal_start_utc'])+field.data['row_time'], model)
        band = Band(field, prior, tx[model['reference_channel']], rx, model['selected_points_body_m'][c],
            model, selected, range_bias_m=model['range_bias_m'][c])
        target, _ = band.design(model['parameters'])
        control, prediction = fit_scene(band, target)
        np.savez(args.output/f'channel-{c}-control.npz', time=field.time, selected=selected, prediction=prediction)
        trials = []
        for spacing in args.spacing:
            points, weights = quadrature(args.bounds, spacing)
            band.points = points
            target, _ = band.design(model['parameters'])
            trial, prediction = fit_scene(band, target, weights)
            v, old = trial['validation'], control['validation']
            trial.update(spacing_m=spacing, points_enu_m=points.tolist(),
                heldout_matched_energy_gain_db=float(10*np.log10(v['heldout_matched_energy']/old['heldout_matched_energy'])),
                heldout_prediction_error_change_percent=100*(v['prediction_error']/old['prediction_error']-1),
                seconds=wall_time.monotonic()-started)
            np.savez(args.output/f'channel-{c}-{spacing:g}m.npz', time=field.time, selected=selected,
                prediction=prediction, points_enu_m=points, quadrature_weights=weights,
                coefficients=np.array(trial['coefficient_real_imag']))
            trials.append(trial)
            print(json.dumps({'channel': c, 'spacing_m': spacing, 'points': len(points),
                'rank': trial['effective_scene_rank'], 'explained': v['heldout_explained_fraction'],
                'gain_db': trial['heldout_matched_energy_gain_db'], 'seconds': trial['seconds']}), flush=True)
            del target
        result['channels'].append({'channel': c, 'control': control, 'trials': trials,
            'field_sha256': fingerprint(args.fields/f'channel-{c}.npz')})
        (args.output/'experiment.json').write_text(json.dumps(result, indent=2)+'\n')
    result.update(complete=True, seconds=wall_time.monotonic()-started)
    (args.output/'experiment.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
