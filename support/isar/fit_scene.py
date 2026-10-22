"""Refine physical scatterer coordinates at a fixed motion/location hypothesis.

Complex amplitudes and stationary nuisance are profiled exactly. Only training
rows drive the fit. The untouched temporal blocks test prediction; sharpness
and optimizer convergence alone are not an acceptance criterion.
"""
import argparse
import json
from pathlib import Path
import time as clock

import numpy as np
from scipy.optimize import minimize

from ais import read_csv, utc_seconds
from field import Field
from imaging import physical_state, sites
from spline_motion import SplineMotion
from stationary import Stationary, make_design
from validate_motion import validate


class Scene:
    def __init__(self, field, state, points, selected_time, offsets, *, scale_m=10.):
        self.field, self.state, self.initial = field, state, np.asarray(points)
        self.selected_time = selected_time
        self.selected = np.repeat(selected_time, field.y.shape[1])
        self.options = {'motion': state, 'transmitter_offset_ecef_m': offsets[0],
                        'receiver_offset_ecef_m': offsets[1]}
        self.scale = scale_m
        stationary, _ = make_design(field)
        self.nuisance = Stationary(stationary, self.selected)
        self.clean = self.nuisance.remove(field.y.ravel()[self.selected])
        self.energy = float(np.vdot(self.clean, self.clean).real)
        if self.energy <= 0 or scale_m <= 0:
            raise ValueError('Need positive information and position scale')

    def points(self, parameters):
        points = self.initial.copy()
        points[:, :2] += np.asarray(parameters).reshape(-1, 2)*self.scale
        return points

    def design(self, parameters, derivatives=False):
        values, jacobians = [], []
        for point in self.points(parameters):
            response = self.field.point(point, derivatives=derivatives, **self.options)
            if derivatives:
                value, derivative = response
                jacobians.append(derivative.reshape(-1, 3)[self.selected, :2]*self.scale)
            else:
                value = response
            values.append(value.ravel())
        return np.column_stack(values), jacobians

    def objective(self, parameters):
        target, derivatives = self.design(parameters, derivatives=True)
        design = self.nuisance.remove(target[self.selected])
        amplitudes = np.linalg.lstsq(design, self.clean, rcond=None)[0]
        residual = self.clean-design@amplitudes
        gradient = np.array([-2*np.real(amplitude*(residual.conj()@derivative))
                             for amplitude, derivative in zip(amplitudes, derivatives)])
        return float(np.vdot(residual, residual).real)/self.energy, gradient.ravel()/self.energy

    def validation(self, parameters, interval):
        target, _ = self.design(parameters)
        return validate(self.field, target, self.nuisance, self.selected_time, interval)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('fields', 'ais', 'geometry-directory', 'motion', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--channels', nargs='+', type=int, default=list(range(5)))
    p.add_argument('--radius-m', type=float, default=20.)
    p.add_argument('--iterations', type=int, default=150)
    args = p.parse_args()
    model = json.loads(args.motion.read_text())
    motion = SplineMotion(model)
    track = read_csv(args.ais, reference_height_ellipsoid_m=model['reference_height_ellipsoid_m'], mmsi=563052200)
    epoch = utc_seconds(model['nominal_start_utc'])
    tx, rx, _ = sites(args.geometry_directory)
    antenna = model.get('antenna_fit')
    rotation = np.eye(3) if antenna is None else np.array(antenna['ecef_to_enu'])
    args.output.mkdir(parents=True, exist_ok=True)
    for channel in args.channels:
        with np.load(args.fields/f'channel-{channel}.npz') as f:
            field = Field(dict(f), f['pulse_samples']).windowed(motion.interval[0]+.02, motion.interval[1]-.02)
        prior = physical_state(track, epoch+field.data['row_time'], model['attitude_prior'])
        state = motion.apply(field.data['row_time'], prior, tx[3], rx)
        offsets = (np.zeros((2, 3)) if antenna is None else
                   np.array([antenna['transmitter_offsets_enu_m'][channel], antenna['receiver_offset_enu_m']])@rotation)
        block = model['holdout_seconds']
        selected = (np.arange(len(field.time))%2 == 0 if block == 0 else
                    np.floor((field.time-motion.interval[0])/block).astype(int)%2 == 0)
        scene = Scene(field, state, model['points_body_m'][channel], selected, offsets)
        initial = np.zeros(2*len(scene.initial))
        radius = args.radius_m/scene.scale
        before = scene.validation(initial, motion.interval)
        trace, start = [], clock.perf_counter()
        def callback(parameters):
            if len(trace)%10 == 0:
                value, gradient = scene.objective(parameters)
                print(json.dumps({'channel': channel, 'iteration': len(trace)+1, 'objective': value,
                    'gradient': float(abs(gradient).max()), 'seconds': clock.perf_counter()-start}), flush=True)
            trace.append(parameters.tolist())
        result = minimize(scene.objective, initial, jac=True, method='L-BFGS-B',
            bounds=[(-radius, radius)]*len(initial), callback=callback,
            options={'maxiter': args.iterations, 'gtol': 1e-7, 'ftol': 1e-14, 'maxcor': 30})
        gradient = result.jac.copy()
        gradient[((result.x <= -radius+1e-7)&(gradient > 0))|((result.x >= radius-1e-7)&(gradient < 0))] = 0
        after = scene.validation(result.x, motion.interval)
        record = {'channel': channel, 'input_motion': str(args.motion), 'radius_m': args.radius_m,
            'parameters': result.x.tolist(), 'points_body_m': scene.points(result.x).tolist(),
            'initial_points_body_m': scene.initial.tolist(), 'before': before, 'after': after,
            'converged': bool(result.success and abs(gradient).max() <= 1e-7),
            'maximum_projected_gradient': float(abs(gradient).max()), 'iterations': result.nit,
            'seconds': clock.perf_counter()-start, 'bound_hit': bool(np.any(abs(result.x) >= radius-1e-5)),
            'scope': __doc__, 'geometry_accepted': False}
        (args.output/f'channel-{channel}.json').write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps({'channel': channel, 'converged': record['converged'],
            'heldout_loss_db': after['conditional_efficiency_loss_db'],
            'heldout_error_change_percent': 100*(after['prediction_error']/before['prediction_error']-1)}), flush=True)


if __name__ == '__main__':
    main()
