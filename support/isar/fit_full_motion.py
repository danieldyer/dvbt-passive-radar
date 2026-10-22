"""Refine a physical ship-motion seed on the unchanged complex observations."""
import argparse
import json
from pathlib import Path
import time as clock

import numpy as np
from scipy.optimize import minimize

from ais import read_csv, utc_seconds
from dense_scatterers import points
from field import Field
from full_motion import Band
from imaging import physical_state, sites


def fit(bands, initial, *, iterations=150):
    initial = np.asarray(initial).copy()
    npath = bands[0].npath
    # Adding the same constant to every path coefficient is a null direction
    # because the curve is anchored at the midpoint. Remove it explicitly.
    initial[:npath] -= initial[npath-1]
    active = np.r_[np.arange(npath-1), np.arange(npath, len(initial))]
    energy = sum(b.energy for b in bands)
    evaluations, started, trace = 0, clock.perf_counter(), []
    def expand(x):
        full = np.zeros_like(initial)
        full[active] = x
        return full
    def objective(x):
        nonlocal evaluations
        evaluated = [b.evaluate(expand(x)) for b in bands]
        value = sum(v for v, g in evaluated)/energy
        gradient = sum(g for v, g in evaluated)[active]/energy
        evaluations += 1
        if evaluations%5 == 0:
            print(json.dumps({'evaluations': evaluations, 'objective': value,
                'maximum_gradient': float(abs(gradient).max()), 'seconds': clock.perf_counter()-started}), flush=True)
        return value, gradient
    def callback(x):
        # The objective is evaluated again deliberately: this records the
        # gradient at accepted iterates rather than at line-search trials.
        value, gradient = objective(x)
        trace.append({'objective': value, 'maximum_gradient': float(abs(gradient).max())})
    before = [b.validation(initial) for b in bands]
    bounds = [(-30., 30.)]*(npath-1)+[(-20., 20.)]*(len(initial)-npath)
    result = minimize(objective, initial[active], jac=True, bounds=bounds, method='L-BFGS-B', callback=callback,
        options={'maxiter': iterations, 'gtol': 1e-8, 'ftol': 1e-14, 'maxcor': 30, 'maxls': 30})
    projected = result.jac.copy()
    lower, upper = np.array(bounds).T
    projected[((result.x <= lower+1e-7)&(projected > 0))|
              ((result.x >= upper-1e-7)&(projected < 0))] = 0
    parameters = expand(result.x)
    return parameters, {'converged': bool(result.success and abs(projected).max() <= 1e-7),
        'termination': str(result.message), 'iterations': result.nit, 'evaluations': evaluations,
        'maximum_gradient': float(abs(result.jac).max()), 'maximum_projected_gradient': float(abs(projected).max()),
        'objective': float(result.fun), 'trace': trace,
        'bound_hit': bool(np.any(result.x <= lower+1e-5) or np.any(result.x >= upper-1e-5)),
        'gauge': 'Last path coefficient fixed to zero; translation curve vanishes at aperture midpoint. Body point coordinates and antenna coordinates held fixed.',
        'before': before, 'after': [b.validation(parameters) for b in bands],
        'seconds': clock.perf_counter()-started,
        'scope': 'Full range/carrier phase/symbol moment likelihood with independent per-band scatterer amplitudes and joint stationary nuisance. Numerical convergence is conditional on the fixed point support; it does not establish absolute heading, hull centroid or phase-centre calibration.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('fields', 'images', 'ais', 'geometry-directory', 'motion', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--points', type=int, default=8)
    p.add_argument('--holdout-seconds', type=float, default=2.)
    p.add_argument('--iterations', type=int, default=150)
    args = p.parse_args()
    model = json.loads(args.motion.read_text())
    epoch = utc_seconds(model['nominal_start_utc'])
    track = read_csv(args.ais, reference_height_ellipsoid_m=model['reference_height_ellipsoid_m'], mmsi=563052200)
    tx, rx, _ = sites(args.geometry_directory)
    bands = []
    for channel in range(5):
        with np.load(args.fields/f'channel-{channel}.npz') as f:
            field = Field(dict(f), f['pulse_samples']).windowed(model['interval_s'][0]+.02, model['interval_s'][1]-.02)
        with np.load(args.images/f'channel-{channel}-image.npz') as image:
            locations = points(image, args.points, 30.)
        selected = (np.arange(len(field.time))%2 == 0 if args.holdout_seconds == 0 else
            np.floor((field.time-model['interval_s'][0])/args.holdout_seconds).astype(int)%2 == 0)
        prior = physical_state(track, epoch+field.data['row_time'], model['attitude_prior'])
        bands.append(Band(field, prior, tx[3], rx, locations, model, selected))
    parameters, result = fit(bands, model['parameters'], iterations=args.iterations)
    metadata = {k: model[k] for k in ('interval_s', 'path_spacing_seconds', 'heading_spacing_seconds',
        'nominal_start_utc', 'attitude_prior', 'reference_height_ellipsoid_m', 'reference_channel',
        'yaw_convention', 'geometry_directory')}
    metadata.update(parameters=parameters.tolist(), full_likelihood=result, initial_motion=str(args.motion),
                    point_source=str(args.images), points_body_m=[b.points.tolist() for b in bands],
                    holdout_seconds=args.holdout_seconds, absolute_heading_validated=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('converged', 'iterations', 'seconds', 'maximum_projected_gradient')}), flush=True)


if __name__ == '__main__':
    main()
