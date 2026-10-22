"""Joint path/heading seed from the phase of fixed BP histories.

Heading is independent of velocity. A shared physical translation along the
reference bistatic bisector and a body yaw curve generate exact point paths in
all five channels. This is a phase seed: changing the range interpolation and
within-symbol moment response requires subsequent full backprojection.
"""
import argparse
import json
from pathlib import Path
import time as clock

import numpy as np
from scipy.optimize import minimize

from ais import read_csv, utc_seconds
from gpga import phase_metrics
from imaging import attitude_basis, sites
from motion import C, excess_path, path_difference
from spline_motion import basis


def spline(time, interval, spacing, fixed_midpoint=False):
    return basis(time, interval, spacing, fixed_midpoint=fixed_midpoint)


class Band:
    def __init__(self, time, value, information, pixels, centre, frame, transmitter, receiver,
                 direction, rf, interval, *, path_spacing=2., heading_spacing=5., selected=None,
                 point_selection=None):
        self.time, self.value = np.asarray(time), np.asarray(value, complex)
        self.weight = np.asarray(information)*8192
        self.pixels, self.centre, self.frame = np.asarray(pixels), np.asarray(centre), np.asarray(frame)
        self.tx, self.rx, self.direction = transmitter, receiver, np.asarray(direction)
        self.k = 2*np.pi*rf/C
        self.path_basis = spline(time, interval, path_spacing, True)
        self.heading_basis = spline(time, interval, heading_spacing)
        self.npath = self.path_basis.shape[1]
        self.nheading = self.heading_basis.shape[1]
        self.count = self.npath+self.nheading
        self.base_offset = np.einsum('nij,pj->npi', self.frame, self.pixels)
        self.base = self.centre[:, None, :]+self.base_offset
        self.selected = np.arange(len(time))%2 == 0 if selected is None else selected
        self.point_selection = np.arange(value.shape[1]) if point_selection is None else np.asarray(point_selection)

    def phase(self, parameters, derivatives=False):
        path = self.path_basis@parameters[:self.npath]
        angle = np.radians(self.heading_basis@parameters[self.npath:])
        c, s = np.cos(angle)[:, None], np.sin(angle)[:, None]
        x, y, z = self.pixels.T
        rotated = np.stack([c*x-s*y, s*x+c*y, np.broadcast_to(z, c.shape[:-1]+z.shape)], axis=-1)
        offset = np.einsum('nij,npj->npi', self.frame, rotated)
        shift = path[:, None, None]*self.direction[:, None, :]+offset-self.base_offset
        delta = path_difference(self.base, shift, self.tx, self.rx)
        phase = -self.k*delta
        if not derivatives:
            return phase
        _, gradient = excess_path(self.base+shift, self.tx, self.rx)
        derivative_angle = np.stack([-s*x-c*y, c*x-s*y, np.zeros_like(c*x)], axis=-1)
        derivative_angle = np.einsum('nij,npj->npi', self.frame, derivative_angle)
        along = -self.k*np.sum(gradient*self.direction[:, None, :], axis=-1)
        yaw = -self.k*np.sum(gradient*derivative_angle, axis=-1)*np.pi/180
        return phase, along, yaw

    def evaluate(self, parameters):
        phase, along, yaw = self.phase(parameters, True)
        corrected = self.value*np.exp(-1j*phase)
        weights = self.weight*self.selected
        amplitude = (weights[:, None]*corrected).sum(axis=0)/weights.sum()
        kept = self.point_selection
        value = -weights.sum()*np.sum(abs(amplitude[kept])**2)
        derivative = -2*weights[:, None]*np.imag(corrected[:, kept]*amplitude[kept].conj())
        gradient = np.r_[self.path_basis.T@(derivative*along[:, kept]).sum(axis=1),
                         self.heading_basis.T@(derivative*yaw[:, kept]).sum(axis=1)]
        return float(value), gradient

    def validation(self, parameters):
        phase = self.phase(parameters)
        corrected = self.value*np.exp(-1j*phase)
        w, train = self.weight, self.selected
        before_ref = np.sum(w[train, None]*self.value[train], axis=0)/w[train].sum()
        after_ref = np.sum(w[train, None]*corrected[train], axis=0)/w[train].sum()
        before = phase_metrics(self.time, self.value, w, ~train, before_ref)
        after = phase_metrics(self.time, corrected, w, ~train, after_ref)
        unseen = np.setdiff1d(np.arange(self.value.shape[1]), self.point_selection)
        result = {'before': before, 'after': after,
            'heldout_power_gain_db': float(10*np.log10(sum(after['coherent_energy_by_point'])/sum(before['coherent_energy_by_point']))),
            'maximum_point_range_change_m': float(np.max(abs(phase))/self.k)}
        if len(unseen):
            result['unfitted_point_power_gain_db'] = float(10*np.log10(
                np.sum(np.array(after['coherent_energy_by_point'])[unseen])/
                np.sum(np.array(before['coherent_energy_by_point'])[unseen])))
        return result


def fit(bands, iterations=500):
    count = bands[0].count
    energy = sum(np.sum(b.weight[b.selected, None]*abs(b.value[b.selected][:, b.point_selection])**2) for b in bands)
    def objective(p):
        results = [b.evaluate(p) for b in bands]
        return sum(r[0] for r in results)/energy, sum(r[1] for r in results)/energy
    bounds = [(-5., 5.)]*bands[0].npath+[(-20., 20.)]*bands[0].nheading
    trials = []
    for heading in (-5., 0., 5.):
        initial = np.r_[np.zeros(bands[0].npath), np.full(bands[0].nheading, heading)]
        trace = []
        def callback(p):
            value, gradient = objective(p)
            trace.append({'objective': value, 'maximum_gradient': float(abs(gradient).max())})
        result = minimize(objective, initial, jac=True, bounds=bounds, method='L-BFGS-B', callback=callback,
                          options={'maxiter': iterations, 'gtol': 1e-8, 'ftol': 1e-14, 'maxcor': 30})
        trials.append((result, trace))
        print('heading start', heading, result.fun, result.success, result.nit, float(abs(result.jac).max()), flush=True)
    result, trace = min(trials, key=lambda trial: trial[0].fun)
    return result.x, {'parameters': result.x.tolist(), 'objective': float(result.fun),
        'converged': bool(result.success and abs(result.jac).max() < 1e-7),
        'termination': str(result.message), 'maximum_gradient': float(abs(result.jac).max()),
        'iterations': result.nit, 'evaluations': result.nfev, 'trace': trace,
        'bound_hit': bool(np.any(abs(result.x[:bands[0].npath]) >= 4.999) or np.any(abs(result.x[bands[0].npath:]) >= 19.999)),
        'starting_trials': [{'objective': float(r.fun), 'iterations': r.nit,
            'maximum_gradient': float(abs(r.jac).max()), 'success': bool(r.success)} for r, _ in trials],
        'bands': [b.validation(result.x) for b in bands],
        'scope': 'Exact path phase of a shared physical bisector translation and independently fitted yaw. Fixed image-selected histories; no updated range interpolation. This is a seed, requiring full-response reimaging and independent spatial validation before adoption.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('histories', 'ais', 'geometry-directory', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--nominal-start-utc', required=True)
    p.add_argument('--interval', type=float, nargs=2, required=True)
    p.add_argument('--path-spacing', type=float, default=2.)
    p.add_argument('--heading-spacing', type=float, default=5.)
    p.add_argument('--iterations', type=int, default=500)
    p.add_argument('--fit-points', type=int, nargs='+')
    args = p.parse_args()
    tx, rx, _ = sites(args.geometry_directory)
    track = read_csv(args.ais, reference_height_ellipsoid_m=5, mmsi=563052200)
    epoch = utc_seconds(args.nominal_start_utc)
    bands = []
    for channel in range(5):
        with np.load(args.histories/f'channel-{channel}-histories.npz') as f:
            keep = (f['time'] >= args.interval[0]) & (f['time'] < args.interval[1])
            time, value, information = (f[key][keep] for key in ('time', 'value', 'information'))
            pixels, rf = f['pixels'][:, :3], f['phase_scale']*C/(2*np.pi)
        state = track.sample(epoch+time, max_gap_s=180)
        frame = attitude_basis(track, state, 'course-prior')
        _, gradient = excess_path(state['position_ecef_m'], tx[3], rx)
        direction = gradient/np.sum(gradient**2, axis=1)[:, None]
        bands.append(Band(time, value, information, pixels, state['position_ecef_m'], frame, tx[channel], rx,
            direction, rf, args.interval, path_spacing=args.path_spacing, heading_spacing=args.heading_spacing,
            point_selection=args.fit_points))
    started = clock.perf_counter()
    parameters, result = fit(bands, args.iterations)
    args.output.mkdir(parents=True, exist_ok=True)
    for channel, band in enumerate(bands):
        np.savez(args.output/f'channel-{channel}.npz', time=band.time, phase_radians=band.phase(parameters),
            path_correction_m=band.path_basis@parameters[:band.npath],
            yaw_correction_degrees=band.heading_basis@parameters[band.npath:])
    result.update(interval_s=args.interval, path_spacing_seconds=args.path_spacing,
        heading_spacing_seconds=args.heading_spacing, fit_points=args.fit_points,
        input_histories=str(args.histories), seconds=clock.perf_counter()-started,
        nominal_start_utc=args.nominal_start_utc, attitude_prior='course-prior',
        reference_height_ellipsoid_m=5., reference_channel=3,
        yaw_convention='Body yaw positive towards port; nautical heading correction is its negative',
        geometry_directory=str(args.geometry_directory))
    (args.output/'motion-seed.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
