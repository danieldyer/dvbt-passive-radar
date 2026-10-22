"""Common-phase diagnostic with full range responses and static nuisance.

The parameter fit is a concentrated complex Gaussian likelihood. It is a
phase-only seed evaluated on the retained 16-row bins, not an accepted physical
motion model: exact range/within-bin effects must be checked before formation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from ais import read_csv, utc_seconds
from field import Field, OFFSETS
from imaging import attitude_basis
from stationary import Stationary, make_design


def fit(time, target, stationary, data, train_time, degree=6):
    cells = data.shape[1]
    u = 2*(time-time[0])/(time[-1]-time[0])-1
    basis = np.polynomial.legendre.legvander(u, degree)[:, 1:]
    train = np.repeat(train_time, cells)
    nuisance = Stationary(stationary, train)
    y = data.ravel()
    clean = nuisance.remove(y[train])
    energy = np.vdot(clean, clean).real
    training_basis = basis[train_time]
    initial = target[train]

    def evaluate(p, full=False):
        phase = np.exp(-1j*np.repeat(training_basis@p, cells))
        a = initial*phase[:, None]
        projected = nuisance.remove(a)
        amplitude = np.linalg.lstsq(projected, clean, rcond=None)[0]
        model = a@amplitude
        beta = nuisance.solve(y[train]-model)
        residual = clean-projected@amplitude
        value = -np.vdot(projected@amplitude, projected@amplitude).real/energy
        cross = (residual.conj()*model).reshape(-1, cells).sum(axis=1)
        gradient = -2*(cross.imag@training_basis)/energy
        return (float(value), gradient, amplitude, beta) if full else (float(value), gradient)

    starts = [np.r_[linear, np.zeros(degree-1)] for linear in (-6., 0., 6.)]
    trials = [minimize(evaluate, start, jac=True, method='BFGS', options={'gtol': 1e-8, 'maxiter': 200})
              for start in starts]
    chosen = min(trials, key=lambda trial: trial.fun)

    def held_out(p):
        _, _, amplitude, beta = evaluate(p, True)
        a = target[~train]*np.exp(-1j*np.repeat(basis[~train_time]@p, cells))[:, None]
        residual_data = y[~train]-stationary[~train]@beta
        model = a@amplitude
        bound_amplitude = np.linalg.lstsq(a, residual_data, rcond=None)[0]
        bound = np.sum(abs(a@bound_amplitude)**2)
        captured = abs(np.vdot(model, residual_data))**2/np.vdot(model, model).real
        return {'prediction_error': float(np.sum(abs(residual_data-model)**2)),
                'matched_energy': float(captured), 'independent_amplitude_upper_energy': float(bound),
                'efficiency_loss_db': float(-10*np.log10(captured/bound))}

    return {'degree': degree, 'interval_s': [float(time[0]), float(time[-1])],
            'coefficients_radians': chosen.x.tolist(), 'converged': bool(chosen.success),
            'evaluations': int(sum(trial.nfev for trial in trials)),
            'training_captured_fraction': float(-chosen.fun),
            'initial': held_out(np.zeros(degree)), 'fitted': held_out(chosen.x),
            'gauge': 'Constant phase absorbed into complex amplitudes; linear phase remains free, so image location is not calibrated.',
            'scope': 'Phase-only seed with full complex range PSFs and constant-delay nuisance. Training-only concentrated likelihood; no range or within-bin motion correction applied.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--field', type=Path, required=True)
    parser.add_argument('--positions', type=Path, required=True)
    parser.add_argument('--ais', type=Path, required=True)
    parser.add_argument('--nominal-start-utc', required=True)
    parser.add_argument('--degree', type=int, default=6)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    positions = json.loads(args.positions.read_text())
    with np.load(args.field) as loaded:
        field = Field(dict(loaded), loaded['pulse_samples']).windowed(*positions['interval_s'])
    track = read_csv(args.ais, reference_height_ellipsoid_m=5, mmsi=563052200)
    state = track.sample(utc_seconds(args.nominal_start_utc)+field.data['row_time'], max_gap_s=180)
    field.data['body_to_ecef'] = attitude_basis(track, state, positions['attitude'])
    points = np.array(positions['initial_points_body_m'])
    target = np.column_stack([field.point(point).ravel() for point in points])
    stationary, _ = make_design(field)
    train = ((field.time-positions['interval_s'][0])//2)%2 == 0
    result = fit(field.time, target, stationary, field.y, train, degree=args.degree)
    result.update(channel=field.metadata['channel'], points_body_m=points.tolist(),
                  point_selection='Prior full-aperture image; holdout tests the subsequent fit only')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
