"""Check whether fitted static phase slopes resemble plausible antenna errors.

Use the existing correlated location prior. Fit within three standard units on
each whitened prior coordinate, then evaluate the finite-offset phase gradient
exactly. This reference-point diagnostic is not a calibration or uncertainty
estimate, and does not fit the received ship data again.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from ais import read_csv, utc_seconds
from antenna_motion import Antennas, exact_spatial_phase, spatial_phase_mapping
from imaging import physical_state, sites
from motion import C
from ship_site_signature import licensed_options
from spline_motion import SplineMotion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('motion', 'ais', 'geometry-directory', 'work', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    model = json.loads(args.motion.read_text())
    curve = SplineMotion(model)
    track = read_csv(args.ais, reference_height_ellipsoid_m=model['reference_height_ellipsoid_m'], mmsi=563052200)
    nominal, rx, _ = sites(args.geometry_directory)
    geometry = json.loads((args.geometry_directory/'geometry-prior.json').read_text())
    options = licensed_options(args.geometry_directory)
    k = 2*np.pi*np.array([177.5e6, 184.5e6, 191.625e6, 219.5e6, 226.5e6])/C
    output = {'scope': __doc__, 'source_motion': str(args.motion), 'correction_adopted': False, 'trials': []}
    for assignment in ('main', 'selected'):
        for span in ('full', 'first', 'second'):
            source = args.work/f'spatial-phase-{assignment}-{span}-64.json'
            fit = json.loads(source.read_text())
            interval = fit['identity'].get('interval', model['interval_s'])
            time = np.array([np.mean(interval)])
            prior = physical_state(track, utc_seconds(model['nominal_start_utc'])+time, model['attitude_prior'])
            state = curve.apply(time, prior, nominal[3], rx)
            position, frame = state['position_ecef_m'][0], state['body_to_ecef'][0]
            sites_used = fit['trials'][0]['sites']
            tx = np.array([next(o['transmitter_ecef_m'] for o in options if o['channel'] == c and
                o['site_id'] == site) for c, site in enumerate(sites_used)])
            _, groups = np.unique(sites_used, return_inverse=True)
            antennas = Antennas(geometry['ecef_to_enu'], groups)
            response = spatial_phase_mapping(position, frame, tx, rx, k, antennas, 3)
            measured = np.array(fit['trials'][0]['fit']['spatial_phase_gradient_radians_per_m'])
            a, y = response.reshape(-1, antennas.count), measured.ravel()
            unbounded = np.linalg.lstsq(a, y, rcond=1e-10)[0]
            bounded = lsq_linear(a, y, bounds=(-3., 3.), tol=1e-13, max_iter=500)
            exact = exact_spatial_phase(position, frame, tx, rx, k, antennas, bounded.x, 3)
            output['trials'].append({'source': str(source), 'sites': sites_used, 'epoch_s': float(time[0]),
                'unbounded_minimum_prior_norm': float(np.linalg.norm(unbounded)),
                'bounded_prior': antennas.describe(bounded.x),
                'bounded_solver_optimality': float(bounded.optimality),
                'bounded_linear_residual_fraction': float(np.linalg.norm(a@bounded.x-y)/np.linalg.norm(y)),
                'bounded_exact_residual_fraction': float(np.linalg.norm(exact-measured)/np.linalg.norm(measured)),
                'linear_exact_difference_degrees_per_m': float(np.degrees(np.max(abs(exact-(a@bounded.x).reshape(5, 2))))),
                'measured_phase_gradient_radians_per_m': measured.tolist(),
                'bounded_exact_phase_gradient_radians_per_m': exact.tolist()})
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    for t in output['trials']:
        print(json.dumps({k: t[k] for k in ('source', 'unbounded_minimum_prior_norm',
            'bounded_exact_residual_fraction', 'linear_exact_difference_degrees_per_m')}))


if __name__ == '__main__':
    main()
