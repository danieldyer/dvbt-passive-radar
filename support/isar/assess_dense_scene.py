"""Test a time-invariant distributed scene before adding more motion freedom.

Fine, unresolved fixed reflectors can interfere as aspect changes. A coarse
point model may represent that as changing strengths or phases. This compares
declared spatial dictionaries on exactly the same held-out complex data.
Only training samples select Tikhonov regularization by generalized CV.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from ais import read_csv, utc_seconds
from dense_scatterers import gcv
from field import Field
from full_motion import Band
from imaging import physical_state, sites
from linear_observation import LinearObservation
from phase_registration import PhaseRegistration
from stationary import Stationary


def local_grid(points, radius, spacing):
    if radius < 0 or spacing <= 0:
        raise ValueError('Need nonnegative radius and positive spacing')
    offsets = np.arange(-radius, radius+spacing*.1, spacing)
    x, y = np.meshgrid(offsets, offsets)
    return np.unique(np.round((np.asarray(points)[:, None, :]+np.c_[x.ravel(), y.ravel(), x.ravel()*0]).reshape(-1, 3), 7), axis=0)


def rectangular_grid(bounds, spacing, height):
    if (not np.isfinite([*bounds, spacing, height]).all() or spacing <= 0 or
            not bounds[0] < bounds[1] or not bounds[2] < bounds[3]):
        raise ValueError('Need finite ordered rectangular bounds and positive spacing')
    x, y = np.meshgrid(np.arange(bounds[0], bounds[1]+spacing*.1, spacing),
                       np.arange(bounds[2], bounds[3]+spacing*.1, spacing))
    return np.c_[x.ravel(), y.ravel(), np.full(x.size, height)]


def fit(train, test, *, independent, register_phases=False, shared_solution=None):
    def solve(observations):
        return gcv(np.concatenate([o.design for o in observations]),
            np.concatenate([o.data for o in observations]), 0,
            orthogonal_energy=sum(o.orthogonal_energy for o in observations),
            observations=sum(o.observations for o in observations))
    if independent:
        solutions = [solve([a]) for a in train]
    else:
        solution = solve(train) if shared_solution is None else shared_solution
        solutions = [solution]*len(train)
    registration = None
    phases = np.ones(len(train), complex)
    if register_phases:
        if independent:
            raise ValueError('Per-channel target coefficients would make channel phase unidentifiable')
        model = PhaseRegistration([a.design for a in train], [a.data for a in train],
            solution[1]['regularization_strength'], orthogonal_energy=sum(a.orthogonal_energy for a in train))
        phases, alpha, registration = model.fit()
        solutions = [(alpha, solution[1])]*len(train)
    checks, total_error, baseline = [], 0., 0.
    for observation, phase, (coefficient, detail) in zip(test, phases, solutions):
        error, energy = observation.error(coefficient, phase), observation.energy
        total_error += error
        baseline += energy
        checks.append({'prediction_error': error, 'stationary_only_error': energy,
            'explained_fraction': 1-error/energy, 'regularization': detail,
            'coefficient_real_imag': np.c_[coefficient.real, coefficient.imag].tolist()})
    return {'independent_complex_band_responses': independent, 'prediction_error': total_error,
        'stationary_only_error': baseline, 'heldout_explained_fraction': 1-total_error/baseline,
        'phase_registration': registration, 'channels': checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('alignment', 'motion', 'ais', 'geometry-directory', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--radius-m', type=float, default=10.)
    parser.add_argument('--spacing-m', type=float, default=5.)
    parser.add_argument('--bounds-m', type=float, nargs=4, metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'),
        help='Use a complete rectangular scene support instead of patches around selected points')
    parser.add_argument('--height-m', type=float, default=20.)
    parser.add_argument('--holdout-seconds', type=float, default=0., help='Alternate fit/test blocks of this duration; zero uses alternate retained bins')
    parser.add_argument('--compress', action='store_true', help='Use exact per-band QR sufficient statistics to bound memory')
    parser.add_argument('--save-statistics', type=Path, help='Save projected observations or QR statistics for further conditional phase tests')
    args = parser.parse_args()
    source, model = json.loads(args.alignment.read_text()), json.loads(args.motion.read_text())
    if model.get('antenna_fit') or np.any(model.get('channel_path_rate_bias_mps', 0.)):
        raise ValueError('This experiment uses the nominal-site, zero-channel-rate control')
    if args.holdout_seconds < 0:
        raise ValueError('Holdout duration cannot be negative')
    points = (rectangular_grid(args.bounds_m, args.spacing_m, args.height_m) if args.bounds_m else
              local_grid(model['points_body_m'], args.radius_m, args.spacing_m))
    if len(points) > (6000 if args.compress else 1000):
        raise ValueError('Use a smaller declared dictionary before allocating the dense reference solve')
    tx, rx, _ = sites(args.geometry_directory)
    track = read_csv(args.ais, reference_height_ellipsoid_m=model['reference_height_ellipsoid_m'], mmsi=563052200)
    epoch, train, test = utc_seconds(model['nominal_start_utc']), [], []
    gains = np.array([complex(*g) for g in model['channel_gains_real_imag']])
    if args.save_statistics:
        args.save_statistics.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for c in range(5):
        with np.load(Path(source['input_fields'])/f'channel-{c}.npz') as f:
            field = Field(dict(f), f['pulse_samples']).windowed(model['interval_s'][0]+.02, model['interval_s'][1]-.02)
        prior = physical_state(track, epoch+field.data['row_time'], model['attitude_prior'])
        selected = (np.arange(len(field.time))%2 == 0 if args.holdout_seconds == 0 else
            np.floor((field.time-model['interval_s'][0])/args.holdout_seconds).astype(int)%2 == 0)
        band = Band(field, prior, tx[3], rx, points, model, selected,
            range_bias_m=model['range_bias_m'][c])
        design = band.design(model['parameters'])[0]*gains[c]
        print(json.dumps({'channel': c, 'points': len(points), 'stage': 'Exact dictionary ready', 'seconds': time.perf_counter()-started}), flush=True)
        for label, keep, container in [('train', band.selected, train), ('test', ~band.selected, test)]:
            background = band.nuisance if label == 'train' else Stationary(band.nuisance.design, keep)
            a, y = background.remove(design[keep]), background.remove(field.y.ravel()[keep])
            observation = (LinearObservation.compress(a, y, background.rank) if args.compress else
                LinearObservation(a, y, 0., len(y)-background.rank))
            container.append(observation)
            if args.save_statistics:
                np.savez(args.save_statistics/f'channel-{c}-{label}.npz',
                    design=observation.design, data=observation.data,
                    orthogonal_energy=observation.orthogonal_energy, observations=observation.observations)
            del a, y
        del design, band, field
        print(json.dumps({'channel': c, 'stage': 'Projected observations ready', 'seconds': time.perf_counter()-started}), flush=True)
    result = {'source_motion': str(args.motion), 'scope': __doc__, 'points_body_m': points.tolist(),
        'radius_m': args.radius_m, 'spacing_m': args.spacing_m,
        'support_bounds_m': args.bounds_m, 'height_m': args.height_m,
        'holdout_seconds': args.holdout_seconds,
        'qr_compressed': args.compress,
        'statistics_directory': str(args.save_statistics) if args.save_statistics else None,
        'limitation': 'Motion, registration and original sparse support are conditional on this aperture. Additional point coordinates are deterministic offsets, not selected on held-out data.', 'trials': []}
    shared_solution = None
    for independent, register in ((False, False), (True, False), (False, True)):
        trial = fit(train, test, independent=independent, register_phases=register,
                    shared_solution=shared_solution)
        if not independent and not register:
            coefficient = np.array(trial['channels'][0]['coefficient_real_imag'])
            shared_solution = (coefficient[:, 0]+1j*coefficient[:, 1], trial['channels'][0]['regularization'])
        result['trials'].append(trial)
        result['seconds'] = time.perf_counter()-started
        args.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k:trial[k] for k in ('independent_complex_band_responses', 'heldout_explained_fraction')}|
            {'seconds': result['seconds'], 'effective_dof': [c['regularization']['effective_degrees_of_freedom'] for c in trial['channels']]}), flush=True)


if __name__ == '__main__':
    main()
