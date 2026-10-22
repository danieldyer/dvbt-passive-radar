"""Conditional point-location information for alternative fitted apertures.

This separates angular-aperture resolution from the fit of an extended ship.
Motion, geometry, covariance and the point's height are treated as known. An
unknown complex point amplitude is eliminated, independently in each band or
jointly with the declared relative calibration. These are local bounds, not
real-data motion error bars, measured image widths, or hull-shape validation.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ais import read_csv, utc_seconds
from compression_information import point_information
from field import Field
from imaging import motion_state, sites
from spline_motion import SplineMotion
from stationary import Stationary, make_design


def combined_information(responses):
    """Equal unit amplitude before each supplied, calibrated band response."""
    per_band = [point_information(u, d) for u, d in responses]
    energy = sum(e for e, _ in per_band)
    independent = sum(j for _, j in per_band)
    _, coherent = point_information(np.concatenate([u for u, _ in responses]),
                                    np.concatenate([d for _, d in responses]))
    return energy, {'independent_complex_band_amplitudes': independent,
                    'known_relative_band_calibration': coherent}


def summarize(information, energy, snr_db):
    values, vectors = np.linalg.eigh(information)
    if values[0] <= 0:
        raise ValueError('Unresolved conditional point-location null')
    return {'information_per_unit_integrated_snr': (information/energy).tolist(),
        'principal_axis_standard_deviation_m': np.sqrt(energy/values/10**(snr_db/10)).tolist(),
        'principal_axes_east_north': vectors.T.tolist(),
        'information_condition_number': float(values[-1]/values[0])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fields', 'ais', 'geometry-directory', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--motions', nargs='+', type=Path, required=True)
    parser.add_argument('--interval', nargs=2, type=float, required=True)
    parser.add_argument('--point-body-m', nargs=3, type=float, default=[0., 0., 20.])
    parser.add_argument('--integrated-snr-db', type=float, default=30.)
    args = parser.parse_args()
    tx, rx, _ = sites(args.geometry_directory)
    fields, nuisances = [], []
    for c in range(5):
        with np.load(args.fields/f'channel-{c}.npz') as f:
            field = Field(dict(f), f['pulse_samples']).windowed(*args.interval)
        design, _ = make_design(field)
        fields.append(field)
        nuisances.append(Stationary(design, np.ones(field.y.size, bool)))
    report = {'scope': __doc__, 'requested_interval_s': args.interval,
        'actual_retained_interval_s': [float(fields[0].time[0]), float(fields[0].time[-1])],
        'point_body_m': args.point_body_m, 'integrated_snr_db': args.integrated_snr_db,
        'snr_definition': 'Sum of original whitened point energy across bands before stationary projection; same imposed SNR for each alternative aperture.',
        'results': []}
    for path in args.motions:
        model = json.loads(path.read_text())
        if model.get('antenna_fit') or model.get('rock_spacing_seconds'):
            raise ValueError('This comparison requires the declared nominal sites and fixed roll/pitch')
        track = read_csv(args.ais, reference_height_ellipsoid_m=model['reference_height_ellipsoid_m'], mmsi=563052200)
        epoch = utc_seconds(model['nominal_start_utc'])
        curve = SplineMotion(model)
        def state(t):
            return curve.apply(t, motion_state(track, epoch+t, model), tx[model['reference_channel']], rx)
        sample = state(np.array([*args.interval, np.mean(args.interval)]))
        frame = sample['body_to_ecef'][-1]
        # With roll/pitch fixed, the body's vertical axis is local Up. Define
        # East and North in that same tangent plane; no display rotation enters
        # formation or the conditional information calculation.
        east = np.cross([0., 0., 1.], frame[:, 2]); east /= np.linalg.norm(east)
        local = np.column_stack([east, np.cross(frame[:, 2], east)])
        body_from_en = frame.T@local
        projected_heading = sample['body_to_ecef'][:, :, 0]@local
        headings = np.arctan2(projected_heading[:, 0], projected_heading[:, 1])
        turn = np.angle(np.exp(1j*(headings[1]-headings[0])))
        gains = np.array(model['channel_gains_real_imag'])@np.array([1., 1j])
        responses, projected = [], []
        for c, (field, nuisance) in enumerate(zip(fields, nuisances)):
            value, derivative = field.point(args.point_body_m, derivatives=True,
                motion=state(field.data['row_time']), range_bias_m=model['range_bias_m'][c])
            u = gains[c]*value.ravel()
            d = gains[c]*derivative.reshape(-1, 3)@body_from_en
            responses.append((u, d))
            projected.append((nuisance.remove(u), nuisance.remove(d)))
        energy, raw_information = combined_information(responses)
        retained_energy, clean_information = combined_information(projected)
        result = {'motion': str(path), 'net_heading_change_degrees': float(np.degrees(turn)),
            'stationary_projection_point_energy_retention': retained_energy/energy,
            'without_stationary_nuisance': {k: summarize(v, energy, args.integrated_snr_db) for k, v in raw_information.items()},
            'with_stationary_nuisance': {k: summarize(v, energy, args.integrated_snr_db) for k, v in clean_information.items()}}
        report['results'].append(result)
        print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
