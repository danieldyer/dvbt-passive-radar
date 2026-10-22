"""Check whether a sparse point model underfits the extended ship response.

The dense dictionary uses exact point responses. Tikhonov strength is selected
only by training-set generalized cross validation. Withheld complex samples
test both the target coefficients and stationary nuisance prediction.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ais import read_csv, utc_seconds
from field import Field, OFFSETS
from imaging import attitude_basis, physical_state, sites
from spline_motion import SplineMotion
from stationary import Stationary, make_design


def points(image, count, separation):
    x, y = np.meshgrid(image['along_m'], image['port_m'])
    allowed = (abs(x) < 220)&(abs(y) < 65)
    power = abs(image['complex_image'])**2
    selected = []
    for _ in range(count):
        if not np.any(allowed):
            break
        index = np.unravel_index(np.argmax(np.where(allowed, power, -1)), power.shape)
        a, b = float(x[index]), float(y[index])
        selected.append([a, b, 20.])
        allowed &= (x-a)**2+(y-b)**2 >= separation**2
    return np.array(selected)


def gcv(design, data, nuisance_rank, *, orthogonal_energy=0., observations=None):
    u, s, vh = np.linalg.svd(design, full_matrices=False)
    projected = u.conj().T@data
    observations = len(data)-nuisance_rank if observations is None else observations
    if orthogonal_energy < 0 or observations <= 0:
        raise ValueError('Invalid orthogonal residual energy or observation count')
    strengths = s[0]**2*np.logspace(-12, 2, 141)
    shrinkage = s[:, None]**2/(s[:, None]**2+strengths)
    outside = data-u@projected
    residual = (np.vdot(outside,outside).real+orthogonal_energy+
        np.sum((1-shrinkage)**2*abs(projected[:,None])**2,axis=0))
    degrees = shrinkage.sum(axis=0)
    criterion = residual/(observations-degrees)**2
    chosen = int(np.argmin(criterion))
    amplitude = vh.conj().T@(s*projected/(s**2+strengths[chosen]))
    return amplitude, {'regularization_strength': float(strengths[chosen]),
                       'regularization_relative_to_max_gram': float(strengths[chosen]/s[0]**2),
                       'effective_degrees_of_freedom': float(degrees[chosen]),
                       'effective_observation_count': int(observations),
                       'training_residual_energy':float(residual[chosen]),
                       'strength_grid_edge': chosen in (0, len(strengths)-1),
                       'singular_values': s.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--field', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--interval', type=float, nargs=2, required=True)
    parser.add_argument('--attitude', choices=['course-prior', 'reported-course-prior'], required=True)
    parser.add_argument('--ais', type=Path, required=True)
    parser.add_argument('--nominal-start-utc', required=True)
    parser.add_argument('--motion', type=Path, help='Evaluate an already fitted physical motion/location model')
    parser.add_argument('--geometry-directory', type=Path)
    parser.add_argument('--count', type=int, default=64)
    parser.add_argument('--separation-m', type=float, default=12)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.field) as data:
        interval = np.asarray(args.interval)+([.02, -.02] if args.motion else [0., 0.])
        field = Field(dict(data), data['pulse_samples']).windowed(*interval)
    track = read_csv(args.ais, reference_height_ellipsoid_m=5, mmsi=563052200)
    state = track.sample(utc_seconds(args.nominal_start_utc)+field.data['row_time'], max_gap_s=180)
    field.data['body_to_ecef'] = attitude_basis(track, state, args.attitude)
    options = {}
    if args.motion:
        if args.geometry_directory is None:
            raise ValueError('A fitted physical motion requires its geometry directory')
        model = json.loads(args.motion.read_text())
        if model['attitude_prior'] != args.attitude or model['nominal_start_utc'] != args.nominal_start_utc:
            raise ValueError('Motion prior/epoch and dense model disagree')
        tx, rx, _ = sites(args.geometry_directory)
        physical = physical_state(track, utc_seconds(args.nominal_start_utc)+field.data['row_time'], args.attitude)
        physical = SplineMotion(model).apply(field.data['row_time'], physical, tx[3], rx)
        options['motion'] = physical
        if model.get('antenna_fit'):
            a = model['antenna_fit']; rotation = np.array(a['ecef_to_enu'])
            channel = field.metadata['channel']
            options['transmitter_offset_ecef_m'] = np.asarray(a['transmitter_offsets_enu_m'])[channel]@rotation
            options['receiver_offset_ecef_m'] = np.asarray(a['receiver_offset_enu_m'])@rotation
    with np.load(args.image) as image:
        locations = points(image, args.count, args.separation_m)
    target = np.column_stack([field.point(point, **options).ravel() for point in locations])
    design, paths = make_design(field)
    train_time = ((field.time-args.interval[0])//2)%2 == 0
    train = np.repeat(train_time, len(OFFSETS))
    nuisance = Stationary(design, train)
    data = field.y.ravel()
    projected = nuisance.remove(target[train])
    clean = nuisance.remove(data[train])
    amplitude, result = gcv(projected, clean, nuisance.rank)
    beta = nuisance.solve(data[train]-target[train]@amplitude)
    validation = data[~train]-design[~train]@beta
    model = target[~train]@amplitude
    independent = np.linalg.lstsq(target[~train], validation, rcond=None)[0]
    upper = float(np.sum(abs(target[~train]@independent)**2))
    captured = float(abs(np.vdot(model, validation))**2/np.vdot(model, model).real)
    stationary_only = nuisance.solve(data[train])
    result.update(channel=field.metadata['channel'], interval_s=args.interval, attitude=args.attitude,
                  motion=None if args.motion is None else str(args.motion),
                  points_body_m=locations.tolist(), training_captured_fraction=float(np.sum(abs(projected@amplitude)**2)/np.sum(abs(clean)**2)),
                  held_out_matched_energy=captured, held_out_independent_upper_energy=upper,
                  held_out_efficiency_loss_db=float(-10*np.log10(captured/upper)),
                  held_out_prediction_error=float(np.sum(abs(validation-model)**2)),
                  held_out_stationary_only_error=float(np.sum(abs(data[~train]-design[~train]@stationary_only)**2)),
                  scope='Dense exact complex response, independent channel amplitudes, fixed motion and phase centres. GCV on training only; 2-second alternating holdout. Dictionary peak selection used the prior full-aperture image, so feature selection is not independently validated.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(result, indent=2))
    np.savez(args.output.with_suffix('.npz'), points_body_m=locations, amplitudes=amplitude, stationary_coefficients=beta, stationary_paths_m=paths)
    np.save(args.output.with_name(args.output.stem+'-design.npy'), target)
    print(json.dumps({k: v for k, v in result.items() if k not in ('points_body_m', 'singular_values')}, indent=2))


if __name__ == '__main__':
    main()
