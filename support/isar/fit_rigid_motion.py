"""Fit physical ship motion with exact row geometry and temporal holdout.

Independent complex point amplitudes in each channel avoid assuming a calibrated
multiband scattering phase. Linear target and stationary nuisance amplitudes are
profiled on training samples only. Nothing is written to the production archive.
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
from rigid_motion import RigidCorrection
from stationary import Stationary, make_design


class Band:
    def __init__(self, model, points, train_time):
        self.model = model
        self.points = np.array(points)
        self.field = model.field
        self.selected = np.repeat(train_time, self.field.y.shape[1])
        self.stationary, _ = make_design(self.field)
        self.nuisance = Stationary(self.stationary, self.selected)
        self.y = self.field.y.ravel()
        self.clean = self.nuisance.remove(self.y[self.selected])
        self.energy = float(np.vdot(self.clean, self.clean).real)

    def design(self, parameters, *, derivatives=False):
        state = self.model.state(parameters)
        if not derivatives:
            return np.column_stack([self.model.point(state, point, derivatives=False).ravel()
                                    for point in self.points])
        values, jacobians = [], []
        for point in self.points:
            value, derivative = self.model.point(state, point)
            values.append(value.ravel())
            jacobians.append(derivative.reshape(-1, len(parameters)))
        return np.column_stack(values), np.stack(jacobians, axis=1)

    def evaluate(self, parameters):
        design, jacobian = self.design(parameters, derivatives=True)
        projected = self.nuisance.remove(design[self.selected])
        amplitude = np.linalg.lstsq(projected, self.clean, rcond=None)[0]
        residual = self.clean-projected@amplitude
        # The projected residual is orthogonal to the stationary and target
        # columns, so derivatives of their optimum coefficients cancel.
        derivative = np.einsum('npj,p->nj', jacobian[self.selected], amplitude)
        gradient = -2*np.real(residual.conj()@derivative)
        return float(np.vdot(residual, residual).real), gradient

    def validation(self, parameters):
        design = self.design(parameters)
        projected = self.nuisance.remove(design[self.selected])
        amplitude = np.linalg.lstsq(projected, self.clean, rcond=None)[0]
        beta = self.nuisance.solve(self.y[self.selected]-design[self.selected]@amplitude)
        reference_beta = self.nuisance.solve(self.y[self.selected])
        held_data = self.y[~self.selected]-self.stationary[~self.selected]@beta
        held_design = design[~self.selected]
        prediction = held_design@amplitude
        prediction_energy = float(np.vdot(prediction, prediction).real)
        separate = np.linalg.lstsq(held_design, held_data, rcond=None)[0]
        upper = float(np.sum(abs(held_design@separate)**2))
        captured = float(abs(np.vdot(prediction, held_data))**2/prediction_energy) if prediction_energy else 0.
        error = float(np.sum(abs(held_data-prediction)**2))
        stationary_error = float(np.sum(abs(self.y[~self.selected]-self.stationary[~self.selected]@reference_beta)**2))
        retention = np.sum(abs(projected)**2, axis=0)/np.sum(abs(design[self.selected])**2, axis=0)
        return {'channel': self.field.metadata['channel'], 'prediction_error': error,
                'stationary_only_prediction_error': stationary_error,
                'heldout_error_reduction_fraction': 1-error/stationary_error,
                'matched_validation_energy': captured, 'independently_refitted_amplitude_energy': upper,
                'conditional_efficiency_loss_db': float(-10*np.log10(captured/upper)) if captured > 0 and upper > 0 else None,
                'training_captured_fraction': float(np.sum(abs(projected@amplitude)**2)/self.energy),
                'training_gram_condition': float(np.linalg.cond(projected.conj().T@projected)),
                'stationary_nuisance_point_information_loss_db': (-10*np.log10(retention)).tolist(),
                'amplitude_real_imag': np.c_[amplitude.real, amplitude.imag].tolist(),
                'points_body_m': self.points.tolist()}


def fit(bands, *, iterations=150, initial=None):
    count = 6*bands[0].model.degree
    energy = sum(band.energy for band in bands)
    parameters = np.zeros(count) if initial is None else np.asarray(initial)
    evaluations = 0
    clock = time.perf_counter()

    def objective(p):
        nonlocal evaluations
        # Unit complex-noise negative log likelihood plus independent real
        # Gaussian coefficient priors (one half of the squared prior norm).
        value, gradient = float(p@p)/2, p.copy()
        for band in bands:
            error, derivative = band.evaluate(p)
            value += error
            gradient += derivative
        evaluations += 1
        if evaluations%10 == 0:
            print(json.dumps({'evaluations': evaluations, 'objective': value/energy,
                              'maximum_gradient': float(abs(gradient).max()/energy),
                              'seconds': time.perf_counter()-clock}), flush=True)
        return value/energy, gradient/energy

    before = [band.validation(np.zeros(count)) for band in bands]
    result = minimize(objective, parameters, jac=True, method='L-BFGS-B', bounds=[(-3., 3.)]*count,
                      options={'maxiter': iterations, 'ftol': 1e-11, 'gtol': 1e-7, 'maxcor': 20})
    projected_gradient = result.jac.copy()
    projected_gradient[((result.x <= -2.999999)&(projected_gradient > 0))|
                       ((result.x >= 2.999999)&(projected_gradient < 0))] = 0
    return {'parameters_standardised': result.x.tolist(),
            'physical_coefficients': (result.x.reshape(6, -1)*bands[0].model.scale[:, None]).tolist(),
            'converged': bool(result.success), 'termination': str(result.message),
            'evaluations': result.nfev, 'iterations': result.nit,
            'objective': float(result.fun), 'maximum_gradient': float(abs(result.jac).max()),
            'maximum_projected_gradient': float(abs(projected_gradient).max()),
            'objective_definition': '(whitened complex residual sum of squares + 0.5 * squared standardized coefficient norm) / training residual energy',
            'parameter_bound_hit': bool(np.any(abs(result.x) > 2.999)),
            'seconds': time.perf_counter()-clock, 'model': bands[0].model.description(),
            'initial': before, 'fitted': [band.validation(result.x) for band in bands],
            'accepted_calibration': False,
            'scope': 'Exact physical translation and attitude correction before original-row range/phase/moment evaluation. Independent complex scatterer responses per band and constant-delay nuisance fitted on alternating two-second blocks; other blocks test prediction. Initial image point selection used all samples. Thermal covariance and AIS/course attitude are conditional assumptions. No phase-centre update or coherent multiband image is accepted by this diagnostic.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fields', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--channels', type=int, nargs='+', default=list(range(5)))
    parser.add_argument('--interval', type=float, nargs=2, default=[142, 172])
    parser.add_argument('--points', type=int, default=4)
    parser.add_argument('--degree', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=150)
    parser.add_argument('--ais', type=Path, required=True)
    parser.add_argument('--nominal-start-utc', required=True)
    parser.add_argument('--attitude', choices=['course-prior', 'reported-course-prior'], required=True)
    parser.add_argument('--initial', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    track = read_csv(args.ais, reference_height_ellipsoid_m=5, mmsi=563052200)
    bands = []
    midpoint = track.sample([utc_seconds(args.nominal_start_utc)+np.mean(args.interval)], max_gap_s=180)
    reference = attitude_basis(track, midpoint, args.attitude)[0]
    initial_result = None if args.initial is None else json.loads(args.initial.read_text())
    if initial_result is not None:
        if initial_result['model']['degree'] != args.degree or initial_result['model']['interval_s'] != args.interval:
            raise ValueError('Initial motion has a different degree or aperture')
        reference = np.array(initial_result['model']['reference_frame_ecef'])
    for channel in args.channels:
        with np.load(args.fields/f'channel-{channel}.npz') as loaded:
            field = Field(dict(loaded), loaded['pulse_samples']).windowed(*args.interval)
        state = track.sample(utc_seconds(args.nominal_start_utc)+field.data['row_time'], max_gap_s=180)
        field.data['body_to_ecef'] = attitude_basis(track, state, args.attitude)
        model = RigidCorrection(field, args.interval, degree=args.degree, reference_frame=reference)
        with np.load(args.images/f'channel-{channel}-image.npz') as image:
            points = anchors(image['complex_image'], image['along_m'], image['port_m'], args.points)[:, :3]
        train = ((field.time-args.interval[0])//2)%2 == 0
        bands.append(Band(model, points, train))
        print('Prepared', channel, 'bins', len(field.time), 'original rows', len(field.data['row_time']), flush=True)
    initial = None if initial_result is None else initial_result['parameters_standardised']
    result = fit(bands, iterations=args.iterations, initial=initial)
    result.update(fields=str(args.fields), images=str(args.images), attitude=args.attitude)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ('converged', 'evaluations', 'seconds', 'initial', 'fitted')}), flush=True)


if __name__ == '__main__':
    main()
