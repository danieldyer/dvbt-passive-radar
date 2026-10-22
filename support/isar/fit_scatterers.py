"""Continuous point-position check with exact complex amplitude projection.

Independent channel reflectivities are permitted. AIS motion and antenna
coordinates remain fixed. Temporal holdout tests position/amplitude fitting;
initial peak selection from the prior image is explicitly not independent.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize

from ais import read_csv, utc_seconds
from field import Field
from fit_autofocus import anchors
from imaging import attitude_basis
from scatterers import normal, prediction


def projected(design, derivatives, data):
    """Concentrated Gaussian likelihood and derivative for separate columns."""
    amplitude = np.linalg.lstsq(design, data, rcond=None)[0]
    model = design@amplitude
    residual = data-model
    energy = np.vdot(data, data).real
    gradient = -2*np.real(np.einsum('n,npj,p->pj', residual.conj(), derivatives, amplitude))/energy
    return -float(np.vdot(model, model).real)/energy, gradient, amplitude


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fields', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--channel', type=int, required=True)
    parser.add_argument('--points', type=int, default=4)
    parser.add_argument('--position-radius-m', type=float, default=5.)
    parser.add_argument('--interval', nargs=2, type=float, default=[90, 210])
    parser.add_argument('--attitude', choices=['course-prior', 'reported-course-prior'])
    parser.add_argument('--ais', type=Path)
    parser.add_argument('--nominal-start-utc')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.fields/f'channel-{args.channel}.npz') as loaded:
        field = Field(dict(loaded), loaded['pulse_samples'])
    field = field.windowed(*args.interval)
    if args.attitude:
        if not args.ais or not args.nominal_start_utc:
            raise ValueError('An alternate attitude requires explicit AIS and nominal UTC')
        track = read_csv(args.ais, reference_height_ellipsoid_m=5, mmsi=563052200)
        state = track.sample(utc_seconds(args.nominal_start_utc)+field.data['row_time'], max_gap_s=180)
        data = dict(field.data)
        data['body_to_ecef'] = attitude_basis(track, state, args.attitude)
        data['metadata_json'] = json.dumps({**field.metadata, 'attitude_hypothesis': args.attitude})
        field = Field(data, field.pulse.samples)
    with np.load(args.images/f'channel-{args.channel}-image.npz') as image:
        initial = anchors(image['complex_image'], image['along_m'], image['port_m'], args.points)[:, :3]
    train = ((field.time-args.interval[0])//2)%2 == 0
    y = field.y[train].ravel()
    evaluations = 0
    clock = time.perf_counter()

    def evaluate(parameters):
        nonlocal evaluations
        points = initial.copy();points[:, :2] = parameters.reshape(-1, 2)
        evaluated = [field.point(point, derivatives=True) for point in points]
        design = np.column_stack([value[train].ravel() for value, _ in evaluated])
        derivative = np.stack([jacobian[train, :, :2].reshape(-1, 2) for _, jacobian in evaluated], axis=1)
        value, gradient, _ = projected(design, derivative, y)
        evaluations += 1
        if evaluations%20 == 0:
            print(json.dumps({'evaluations': evaluations, 'captured_fraction': -value,
                              'gradient': float(abs(gradient).max()), 'seconds': time.perf_counter()-clock}), flush=True)
        return value, gradient.ravel()

    def validation(points):
        design = np.stack([field.point(point) for point in points], axis=-1)
        fit = normal(design[train], field.y[train])
        holdout = normal(design[~train], field.y[~train])
        amplitudes = np.linalg.solve(fit['gram'], fit['score'])
        return {'training_captured_energy': float(np.vdot(fit['score'], amplitudes).real),
                'training_gram_condition': float(np.linalg.cond(fit['gram'])),
                'held_out': prediction([holdout], [amplitudes])}

    before = validation(initial)
    radius = args.position_radius_m
    seed = initial[:, :2].ravel()
    fit = minimize(evaluate, seed, jac=True, method='L-BFGS-B', bounds=list(zip(seed-radius, seed+radius)),
                   options={'maxiter': 150, 'ftol': 1e-11, 'gtol': 1e-8})
    positions = initial.copy();positions[:, :2] = fit.x.reshape(-1, 2)
    result = {'channel': args.channel, 'interval_s': args.interval, 'attitude': field.metadata['attitude_hypothesis'],
              'initial_points_body_m': initial.tolist(), 'fitted_points_body_m': positions.tolist(),
              'converged': bool(fit.success), 'message': str(fit.message), 'evaluations': fit.nfev,
              'seconds': time.perf_counter()-clock, 'position_radius_m': radius,
              'position_bound_hit': bool(np.any(abs(fit.x-seed) > .999*radius)),
              'initial': before, 'fitted': validation(positions),
              'scope': 'Independent complex channel response, continuous fixed-point positions, unchanged AIS motion and antenna coordinates. Alternating 2-second holdout; peak initialisation used the full prior image. No accepted calibration or correction.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
