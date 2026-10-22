"""Spatial phase-gradient sampling check for the declared near-field aperture.

A complex scene can absorb a constant spatial carrier. Its sampling interval
is therefore governed by the span of observed spatial frequencies. The
curvature margin bounds local gradient changes over a ball around the AIS
reference. This is a necessary grid-resolution check, not a proof of finite
support convergence or an exactly bandlimited near-field kernel.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ais import read_csv, utc_seconds
from imaging import physical_state, sites
from motion import C, excess_path
from spline_motion import SplineMotion


def bounds(position, frame, transmitters, receiver, carriers, bandwidth, radius):
    if not (bandwidth > 0 and radius >= 0 and len(transmitters) == len(carriers)):
        raise ValueError('Invalid band or spatial support')
    wavevectors, margins = [], []
    for transmitter, carrier in zip(transmitters, carriers):
        if carrier <= bandwidth/2:
            raise ValueError('Carrier must exceed half the bandwidth')
        _, gradient = excess_path(position, transmitter, receiver)
        body_gradient = np.einsum('nji,nj->ni', frame, gradient)[:, :2]
        k = 2*np.pi*(carrier+np.array([-bandwidth/2, bandwidth/2]))/C
        wavevectors.append(k[:, None, None]*body_gradient)
        distance = [np.min(np.linalg.norm(position-antenna, axis=1)) for antenna in (transmitter, receiver)]
        if min(distance) <= radius:
            raise ValueError('Support ball intersects an antenna')
        # Each leg's path Hessian has spectral norm 1/d. The mean-value bound
        # applies to every body axis; both endpoints of the observed span move.
        margins.append(2*max(k)*radius*sum(1/(d-radius) for d in distance))
    samples = np.concatenate(wavevectors).reshape(-1, 2)
    span = np.ptp(samples, axis=0)
    margin = max(margins)
    return {'wavevector_span_rad_per_m': span.tolist(),
            'conservative_nearfield_span_margin_rad_per_m': float(margin),
            'maximum_axis_grid_spacing_m': (2*np.pi/(span+margin)).tolist(),
            'support_ball_radius_m': radius, 'channel_bandwidth_hz': bandwidth,
            'scope': __doc__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion', type=Path, required=True)
    parser.add_argument('--alignment', type=Path, required=True)
    parser.add_argument('--ais', type=Path, required=True)
    parser.add_argument('--geometry-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--radius-m', type=float, default=220.)
    args = parser.parse_args()
    motion, alignment = [json.loads(p.read_text()) for p in (args.motion, args.alignment)]
    if motion.get('antenna_fit') or motion.get('rock_spacing_seconds'):
        raise ValueError('This control uses nominal sites with roll and pitch fixed')
    tx, rx, _ = sites(args.geometry_directory)
    track = read_csv(args.ais, reference_height_ellipsoid_m=motion['reference_height_ellipsoid_m'], mmsi=563052200)
    t = np.linspace(*motion['interval_s'], 601)
    prior = physical_state(track, utc_seconds(motion['nominal_start_utc'])+t, motion['attitude_prior'])
    state = SplineMotion(motion).apply(t, prior, tx[3], rx)
    carriers = []
    for c in range(5):
        with np.load(Path(alignment['input_fields'])/f'channel-{c}.npz') as f:
            carriers.append(json.loads(str(f['metadata_json']))['rf_hz'])
    result = bounds(state['position_ecef_m'], state['body_to_ecef'], tx, rx, carriers, 7e6, args.radius_m)
    result.update(source_motion=str(args.motion), carrier_hz=carriers, sampled_epochs=len(t))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
