"""Conditional geometric test of independently measured short vehicle arcs.

All five band summaries and their full range/rate/curvature covariances enter
the objective. Road association is an explicit hypothesis, not ground truth.
The free-origin control tests five shared envelope-reference offsets separately
from antenna coordinates; it does not introduce per-target or per-band rates.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares

from antenna_motion import Antennas
from flash_geometry import observables
from road_geometry import RoadCurve
from site_hypotheses import licensed_geometry
from terrain import RasterTerrain


def road_state(road, state):
    """Position, velocity and acceleration for one smooth road-centreline arc."""
    distance, speed, acceleration, lateral, height = state
    s = np.array([distance])
    normal, dnormal = road.frame(s); normal, dnormal = normal[0], dnormal[0]
    tangent, second, third = [road.centre(s, order)[0] for order in (1, 2, 3)]
    q = np.linalg.norm(tangent[:2]); dot = tangent[:2]@second[:2]
    turn = np.array([-second[1], second[0], 0.])
    d2normal = np.array([-third[1], third[0], 0.])/q-2*turn*dot/q**3
    d2normal -= normal*(second[:2]@second[:2]+tangent[:2]@third[:2])/q**2
    d2normal += 3*normal*dot**2/q**4
    position = road.centre(s)[0]+lateral*normal+np.array([0., 0., height])
    direction = tangent+lateral*dnormal
    return position, direction*speed, direction*acceleration+(second+lateral*d2normal)*speed**2


class RoadGeometry:
    def __init__(self, arcs, routes, transmitters, sites, rotation, *, free_origins=False, free_antennas=True,
                 columns=(0, 1, 2), receiver=None, range_origins=None):
        self.arcs, self.roads, self.tx = arcs, [RoadCurve(r) for r in routes], np.asarray(transmitters)
        self.antennas = Antennas(rotation, sites, receiver_sigma=(0., 0., 7.5))
        self.free_antennas, self.free_origins = free_antennas, free_origins
        self.receiver = np.zeros(3) if receiver is None else np.asarray(receiver)
        self.range_origins = np.zeros(5) if range_origins is None else np.asarray(range_origins)
        self.antenna_count = 18 if free_antennas else 0
        self.origin_count = 5 if free_origins else 0
        self.start = self.antenna_count+self.origin_count
        self.scale = np.array([100., 10., 1., 5., 5.])
        self.columns = np.array(columns)
        self.whiten = []
        for arc in arcs:
            covariance = np.array([r['absolute_observable_covariance'] for r in arc['channels']])
            self.whiten.append([solve_triangular(np.linalg.cholesky(v[np.ix_(columns, columns)]),
                np.eye(len(columns)), lower=True) for v in covariance])
        self.initial = np.r_[np.zeros(self.start), *[np.asarray(a['initial_road_state'])/self.scale for a in arcs]]
        lower, upper = [], []
        if free_antennas:
            lower.extend(np.full(18, -3.)); upper.extend(np.full(18, 3.))
            lower[17], upper[17] = -10/7.5, 10/7.5
        if free_origins:
            # Metres, shared across all target arcs. Bounds are sensitivity
            # limits, not evidence of an instrumental-delay calibration.
            lower.extend(np.full(5, -3.)); upper.extend(np.full(5, 3.))
        for road in self.roads:
            directions = road.description['directions']
            vmax = road.description['posted_speed_mps']
            vmax = 40. if vmax is None else max(vmax*1.2, vmax+3.)
            lower.extend(np.array([1e-4, -vmax if -1 in directions else 0., -4., -15., 0.])/self.scale)
            upper.extend(np.array([road.length-1e-4, vmax if 1 in directions else 0., 4., 15., 8.])/self.scale)
        self.lower, self.upper = np.array(lower), np.array(upper)

    def split(self, p):
        physical = self.antennas.factor@p[:18] if self.free_antennas else np.zeros(18)
        origins = self.range_origins+(30*p[self.antenna_count:self.start] if self.free_origins else 0.)
        states = p[self.start:].reshape(-1, 5)*self.scale
        return self.tx+physical[:15].reshape(5, 3), self.receiver+physical[15:], origins, states

    def predictions(self, p):
        tx, rx, origins, states = self.split(p)
        result = np.array([observables(*road_state(road, state), tx, rx) for road, state in zip(self.roads, states)])
        result[:, :, 0] += origins
        return result

    def residual(self, p, priors=True):
        predicted = self.predictions(p)
        residual = []
        for arc, value, whiten in zip(self.arcs, predicted, self.whiten):
            measured = np.array([r['absolute_observables'] for r in arc['channels']])
            residual.extend(w@e[self.columns] for w, e in zip(whiten, value-measured))
        if priors:
            residual.append(p[:self.antenna_count])
            # Road centreline is not a lane survey; height is above the DEM.
            states = self.split(p)[3]
            residual.extend([states[:, 3]/5., (states[:, 4]-2.)/5., states[:, 2]/4.])
        return np.concatenate(residual)

    def fit(self):
        return least_squares(self.residual, np.clip(self.initial, self.lower+1e-7, self.upper-1e-7),
            jac='3-point', bounds=(self.lower, self.upper), max_nfev=250,
            ftol=1e-10, xtol=1e-10, gtol=1e-7, x_scale='jac')

    def describe(self, fit):
        data = self.residual(fit.x, False)
        tx, rx, origins, states = self.split(fit.x)
        residual = self.predictions(fit.x)-np.array([[r['absolute_observables'] for r in a['channels']] for a in self.arcs])
        result = {'parameters': fit.x.tolist(), 'converged': bool(fit.success), 'termination': fit.message,
            'maximum_scaled_gradient': float(fit.optimality), 'evaluations': int(fit.nfev),
            'data_squared_residual': float(data@data), 'prior_squared_residual': float(fit.fun@fit.fun-data@data),
            'bound_indices': np.flatnonzero((fit.x < self.lower+1e-4)|(fit.x > self.upper-1e-4)).tolist(),
            'range_origins_m': origins.tolist(), 'road_states': states.tolist(),
            'predicted_observables': self.predictions(fit.x).tolist(),
            'residual_rms_range_rate_acceleration': np.sqrt(np.mean(residual**2, axis=(0, 1))).tolist(),
            'range_residuals_m': residual[:, :, 0].tolist(), 'rate_residuals_mps': residual[:, :, 1].tolist(),
            'antenna_fit': self.antennas.describe(fit.x[:18]) if self.free_antennas else None,
            'accepted_calibration': False, 'columns': self.columns.tolist()}
        if self.free_antennas:
            jacobian = fit.jac[:len(data)]
            antenna, nuisance = jacobian[:, :18], jacobian[:, 18:]
            u, s, _ = np.linalg.svd(nuisance, full_matrices=False)
            q = u[:, s > s[0]*1e-9]
            profiled = antenna-q@(q.T@antenna)
            result['data_information_eigenvalues_after_nuisance'] = np.linalg.eigvalsh(profiled.T@profiled)[::-1].tolist()
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arcs', type=Path, nargs='+', required=True)
    p.add_argument('--geometry', type=Path, default=Path('docs/data/calibration_20261016_sites'))
    p.add_argument('--routes', type=Path, default=Path('build/calibration-20261017/road-routes.json'))
    p.add_argument('--screens', type=Path, default=Path('build/calibration-20261017/road-screen'))
    p.add_argument('--histories', type=Path, default=Path('build/calibration-20261013/shared-seed-histories'))
    p.add_argument('--assignments', nargs='+', default=['02002', '00002'])
    p.add_argument('--preferred-roads', nargs='*', default=['Albert Street', 'Victoria Parade'])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    targets = [t for path in args.arcs for t in json.loads(path.read_text())['targets']]
    routes = json.loads(args.routes.read_text())['routes']
    prior = json.loads((args.geometry/'geometry-prior.json').read_text())
    terrain = RasterTerrain(Path('build/calibration-20261013'), prior, Path('build/calibration-20261013/ga-footprint-dem.tif'))
    result = {'scope': __doc__, 'accepted_calibration': False, 'complete': False,
        'source_arcs': [str(p) for p in args.arcs], 'geometry': str(args.geometry), 'trials': []}
    for code in args.assignments:
        assignment = list(map(int, code))
        sites, offsets, description = licensed_geometry(args.geometry, Path('build/calibration-20261013'), assignment, terrain)
        screens = {t['label']: t['trials'] for t in json.loads((args.screens/f'sites-{code}.json').read_text())['targets']}
        arcs, chosen_roads, associations = [], [], []
        for target in targets:
            label = target['label']; choices = screens[label]
            preferred = [r for r in choices if r['name'] in args.preferred_roads]
            chosen = min(preferred or choices, key=lambda r: r['cost'])
            history = json.loads((args.histories/(label+'.json')).read_text())
            seed = history['channels'][0]; half = (history['end']-history['start'])/2
            # First interval is the strongest training-selected arc. The other
            # arc remains a useful temporal control, not a duplicated target.
            arc = dict(target['arcs'][0]); midpoint = (arc['first_s']+arc['last_s'])/2
            u = (midpoint-seed['epoch'])/half
            state = [np.polynomial.polynomial.polyval(u, np.polynomial.polynomial.polyder(chosen['state'], j))/half**j for j in range(3)]
            arc['initial_road_state'] = [*state, 0., 2.]
            arcs.append(arc); chosen_roads.append(routes[chosen['route']])
            associations.append({'label': label, 'arc': arc['label'], 'road': chosen['name'], 'route': chosen['route']})
        for free_antennas, free_origins, columns in [(False, False, (0, 1, 2)), (True, False, (0, 1, 2)),
                (False, True, (0, 1, 2)), (True, True, (0, 1, 2)), (True, True, (0, 1))]:
            model = RoadGeometry(arcs, chosen_roads, sites[assignment]+offsets, [d['site_id'] for d in description],
                np.array(prior['ecef_to_enu']), free_origins=free_origins, free_antennas=free_antennas, columns=columns)
            fit = model.fit(); row = model.describe(fit)
            row.update(assignment=assignment, associations=associations, free_antennas=free_antennas, free_origins=free_origins)
            result['trials'].append(row)
            (args.output/'fits.json').write_text(json.dumps(result, indent=2, allow_nan=False))
            print(code, 'antennas', free_antennas, 'origins', free_origins, 'columns', columns,
                'data', round(row['data_squared_residual'], 2), 'gradient', round(fit.optimality, 5),
                'origins m', np.round(row['range_origins_m'], 2).tolist(), flush=True)
    result['complete'] = True
    (args.output/'fits.json').write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
